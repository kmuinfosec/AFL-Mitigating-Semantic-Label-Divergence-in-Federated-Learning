import os, numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset

from main import FFN, load_dataset, split_org3   # 재사용
from utils.encoding import contents2count, key_from_hex


def f1_from_logits(y_true, y_prob, thr=0.5, eps=1e-12):
    y_pred = (y_prob >= thr).float()
    tp = ((y_pred==1)&(y_true==1)).sum().item()
    fp = ((y_pred==1)&(y_true==0)).sum().item()
    fn = ((y_pred==0)&(y_true==1)).sum().item()
    prec = tp/(tp+fp+eps); rec = tp/(tp+fn+eps)
    f1 = 2*prec*rec/(prec+rec+eps)
    return prec, rec, f1

def evaluate_f1(model, X, y, thr=0.5, batch=1024):
    if len(X) == 0:
        return 0.0, 0.0, 0.0
    dl = DataLoader(TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)),
        batch_size=batch, shuffle=False)
    yt, yp = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in dl:
            p = model(xb).cpu()
            yt.append(yb); yp.append(p)
    yt, yp = torch.cat(yt), torch.cat(yp)
    return f1_from_logits(yt, yp, thr)

def _safe_norm_rows(A: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    return n

def apply_exclusion_per_org(X: np.ndarray, cents: np.ndarray | None, theta: float):
    """같은 org의 centroid만으로 'cosine >= theta' 인 샘플 제외."""
    n = len(X)
    if n == 0 or cents is None or (hasattr(cents, "shape") and cents.shape[0] == 0):
        return X, np.arange(n, dtype=int), 1.0

    X = np.asarray(X, dtype=np.float64)
    C = np.asarray(cents, dtype=np.float64)

    Xn = X / _safe_norm_rows(X)
    Cn = C / _safe_norm_rows(C)

    S = Xn @ Cn.T  # (N, M)
    excl_mask = (S >= theta).any(axis=1)
    kept_mask = ~excl_mask

    kept_idx = np.where(kept_mask)[0]
    kept_X = X[kept_idx]
    coverage = float(kept_mask.sum()) / float(n) if n > 0 else 1.0
    return kept_X, kept_idx, coverage


def main():
    # 1) 체크포인트 경로
    ckpt_path = os.path.join("outputs", "fed_round10_lr0.001_seed42.pt")  # ← 파일명에 맞게 수정
    ckpt = torch.load(ckpt_path, map_location="cpu")

    cfg = ckpt["config"]
    afl_centroids = ckpt["afl_centroids"]

    # 2) 모델 복원
    model = FFN(cfg["vec_size"], tuple(cfg["hidden"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # 3) 데이터 로드 및 기관별 test 재구성(동일 seed/ratio)
    df_all = load_dataset(cfg["data_path"])
    orgs = sorted(df_all["orgIDX"].unique().tolist())
    key = key_from_hex(cfg["key_hex"]) if cfg["key_hex"] else None

    theta = cfg.get("afl_theta", 0.95)
    thr   = cfg.get("threshold", 0.5)
    batch = cfg.get("batch", 1024)

    f1_fl, f1_afl, covs = [], [], []

    for org in orgs:
        df_org = df_all[df_all["orgIDX"] == org]
        # train/valid/test 분할은 모델 학습과 동일한 규칙으로 재현
        _, _, test_df = split_org3(
            df_org,
            val_ratio=cfg["val_ratio"],
            test_ratio=cfg["test_ratio"],
            seed=cfg["seed"]
        )

        Xte = contents2count(test_df["payload"].tolist(), cfg["vec_size"], cfg["win_size"], key)
        yte = test_df["analyResult"].values.astype(int)

        # (A) 필터링 없이 평가 (FL)
        _, _, f1_no = evaluate_f1(model, Xte, yte, thr=thr, batch=batch)
        f1_fl.append(f1_no)

        # (B) AFL: 같은 org의 centroid만 사용해 제외 후 평가
        cents = afl_centroids.get(org, None)
        X_keep, idx_keep, cov = apply_exclusion_per_org(Xte, cents, theta)
        y_keep = yte[idx_keep]
        _, _, f1_with = evaluate_f1(model, X_keep, y_keep, thr=thr, batch=batch) if len(idx_keep) > 0 else (0.0, 0.0, 0.0)

        f1_afl.append(f1_with)
        covs.append(cov)

        print(f"[Org {org}]  FL-F1={f1_no:.4f} | AFL-F1={f1_with:.4f} | coverage={cov:.4f}")

    print("-----")
    print(f"Macro FL  F1 : {np.mean(f1_fl):.4f}")
    print(f"Macro AFL F1 : {np.mean(f1_afl):.4f}")
    print(f"AFL coverage : {np.mean(covs):.4f} (min={np.min(covs):.4f}, max={np.max(covs):.4f})")


if __name__ == "__main__":
    main()
