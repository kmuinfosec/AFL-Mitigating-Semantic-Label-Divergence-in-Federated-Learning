import os, argparse
import numpy as np
import pandas as pd
import yaml

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from utils.encoding import contents2count, key_from_hex
from utils.clustering import prototypeClustering


# ---------------- Model ----------------
class FFN(nn.Module):
    def __init__(self, input_dim=768, hidden=(512,256,128)):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(1)


# --------------- Utils -----------------
def make_loader(X, y, batch=1024, shuffle=True):
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(X, y), batch_size=batch, shuffle=shuffle, drop_last=False)

def score_from_logits(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).float()
    eps = 1e-12
    tp = ((y_pred==1)&(y_true==1)).sum().item()
    fp = ((y_pred==1)&(y_true==0)).sum().item()
    fn = ((y_pred==0)&(y_true==1)).sum().item()
    prec = tp/(tp+fp+eps); rec = tp/(tp+fn+eps)
    f1 = 2*prec*rec/(prec+rec+eps)
    return prec, rec, f1

def average_state_dicts(state_dicts):
    """단순 평균 FedAvg"""
    avg = {k: v.clone() for k, v in state_dicts[0].items()}
    for k in avg.keys():
        for sd in state_dicts[1:]:
            avg[k] += sd[k]
        avg[k] /= len(state_dicts)
    return avg

def clone_and_train(global_model, loader, lr=1e-3, epochs=1, device="cpu", wd=0.0):
    """글로벌 모델 복제 → 로컬 데이터로 학습 → state_dict 반환"""
    m = type(global_model)()
    m.load_state_dict({k: v.detach().clone() for k, v in global_model.state_dict().items()})
    m.to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    bce = nn.BCELoss()
    m.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            p = m(xb)
            loss = bce(p, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return m.state_dict()

def eval_loss(model, loader, device="cpu"):
    """검증 손실(평균)"""
    bce = nn.BCELoss(reduction="mean")
    model.eval()
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            p = model(xb)
            tot += bce(p, yb).item() * len(yb)
            cnt += len(yb)
    return tot / max(1, cnt)

def eval_f1(model, loader, thr=0.5, device="cpu"):
    """필요시 사용할 F1(평가 전용). 본 스크립트에서는 Early Stopping 지표로 쓰지 않음."""
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            p = model(xb).cpu()
            yt.append(yb)
            yp.append(p)
    if not yt:
        return 0.0, 0.0, 0.0
    yt, yp = torch.cat(yt), torch.cat(yp)
    return score_from_logits(yt, yp, thr=thr)

def encode_hex(hex_list, vec_size, win_size, key_hex):
    key = key_from_hex(key_hex) if key_hex else None
    return contents2count(hex_list, vec_size, win_size, key=key)

def load_dataset(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".parquet", ".pq"]:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df = df[["payload", "analyResult", "orgIDX"]].copy()
    df["payload"] = df["payload"].astype(str)
    df["analyResult"] = df["analyResult"].astype(int)  # 이미 0/1이라 가정
    df["orgIDX"] = df["orgIDX"].astype(int)
    return df

def split_org3(df_org, val_ratio=0.2, test_ratio=0.2, seed=42):
    """기관 데이터 → train/valid/test (층화 분할)"""
    idx = np.arange(len(df_org))
    y = df_org["analyResult"].values
    train_idx, test_idx = train_test_split(
        idx, test_size=test_ratio, stratify=y, random_state=seed
    )
    y_train = y[train_idx]
    train_idx2, valid_idx = train_test_split(
        train_idx, test_size=val_ratio, stratify=y_train, random_state=seed
    )
    return df_org.iloc[train_idx2], df_org.iloc[valid_idx], df_org.iloc[test_idx]

def load_yaml(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

# -------- AFL helpers (post-hoc exclusion seeds) --------
def build_exclusion_from_valid(
    X_val, y_val, theta=0.95, min_size=0, min_purity=0.80
):
    """
    규칙:
      1) len(cluster) < min_size  → 패스
      2) purity(cluster) < min_purity → centroid를 제외 시드로 저장
         purity = max(#pos, #neg) / |cluster|
      3) 아웃라이어(label==-1)는 사용하지 않음
    """
    labels = prototypeClustering(X_val, th=theta, precompute=True, return_seed=False)
    labels = np.asarray(labels)
    if labels.size == 0 or labels.max() < 0:
        return np.zeros((0, X_val.shape[1]), dtype=np.float64)

    y_val = np.asarray(y_val).astype(int)
    cents = []
    for cid in range(labels.max() + 1):
        idx = np.where(labels == cid)[0]
        if len(idx) == 0:
            continue
        if len(idx) < min_size:
            continue
        y_c = y_val[idx]
        cnt_pos = int((y_c == 1).sum())
        cnt_neg = len(idx) - cnt_pos
        purity = max(cnt_pos, cnt_neg) / float(len(idx))
        if purity < min_purity:
            cents.append(X_val[idx].mean(axis=0))

    return np.vstack(cents) if cents else np.zeros((0, X_val.shape[1]), dtype=np.float64)


# -------- Main --------
def main(args, parser):
    # YAML → 기본값 덮어쓰기(동일 키에서만), 이후 CLI가 최종 우선
    ycfg = load_yaml(args.config)
    for k, v in ycfg.items():
        if hasattr(args, k) and getattr(args, k) == parser.get_default(k):
            setattr(args, k, v)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df_all = load_dataset(args.data_path)
    orgs = sorted(df_all["orgIDX"].unique().tolist())

    clients = []
    for org in orgs:
        df_org = df_all[df_all["orgIDX"] == org]
        train_df, valid_df, test_df = split_org3(
            df_org,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        )

        train_X = encode_hex(train_df["payload"].tolist(), args.vec_size, args.win_size, args.key_hex)
        train_y = train_df["analyResult"].values.astype(int)

        valid_X = encode_hex(valid_df["payload"].tolist(), args.vec_size, args.win_size, args.key_hex)
        valid_y = valid_df["analyResult"].values.astype(int)

        test_X  = encode_hex(test_df["payload"].tolist(),  args.vec_size, args.win_size, args.key_hex)
        test_y  = test_df["analyResult"].values.astype(int)

        clients.append({
            "org": org,
            "train_loader": make_loader(train_X, train_y, args.batch, True),
            "valid_loader": make_loader(valid_X, valid_y, args.batch, False),
            # AFL 제외영역 생성용
            "valid_X": valid_X, "valid_y": valid_y,
            # 평가 스크립트에서 쓸 수 있도록 test도 보관(이 스크립트에서는 사용하지 않음)
            "test_X": test_X, "test_y": test_y,
        })

    model = FFN(args.vec_size, tuple(args.hidden)).to(device)
    best_metric, best_sd, wait = None, None, 0

    # ---- Federated training with Early Stopping (metric: val_loss) ----
    for r in range(args.rounds):
        sds = []
        for c in clients:
            sd = clone_and_train(
                model, c['train_loader'],
                lr=args.lr, epochs=args.local_epochs,
                device=device, wd=args.weight_decay
            )
            sds.append(sd)

        model.load_state_dict(average_state_dicts(sds))

        vals = [eval_loss(model, c['valid_loader'], device) for c in clients]
        metric = float(np.mean(vals)) if vals else np.inf
        improve = (best_metric is None) or (metric < best_metric - 1e-8)

        print(f"[Round {r+1}/{args.rounds}] val_loss(avg)={metric:.6f}")

        if improve:
            best_metric, best_sd, wait = metric, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"[Early Stopping] stop at round {r+1}")
                break

    if best_sd is not None:
        model.load_state_dict(best_sd)

    # ---- AFL: 기관별 valid에서 제외영역 centroid 생성(기관별로만 사용) ----
    afl_centroids = {}
    for c in clients:
        cents = build_exclusion_from_valid(
            c["valid_X"], c["valid_y"],
            theta=args.afl_theta,
            min_size=args.afl_min_size,
            min_purity=args.afl_min_purity
        )
        afl_centroids[c["org"]] = cents  # dict: org -> (M_org, D)

    # ---- 저장 ----
    os.makedirs(args.out_dir, exist_ok=True)
    fname = f"fed_round{args.rounds}_lr{args.lr}_seed{args.seed}.pt"
    save_path = os.path.join(args.out_dir, fname)

    torch.save({
        "model_state": model.state_dict(),
        "config": vars(args),
        "afl_centroids": afl_centroids,
    }, save_path)

    print(f"[Done] saved -> {fname}")
    if best_metric is not None:
        print(f"Best val_loss(avg) = {best_metric:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data_path", default="data/dataset.csv")
    parser.add_argument("--vec_size", type=int, default=768)
    parser.add_argument("--win_size", type=int, default=64)
    parser.add_argument("--key_hex", default="")
    parser.add_argument("--hidden", nargs="+", type=int, default=[512,256,128])
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.5)   # 평가 스크립트에서 사용
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out_dir", default="outputs")
    # AFL 하이퍼파라미터
    parser.add_argument("--afl_theta", type=float, default=0.95)
    parser.add_argument("--afl_min_size", type=int, default=0)
    parser.add_argument("--afl_min_purity", type=float, default=0.80)

    args = parser.parse_args()
    main(args, parser)
