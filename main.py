import argparse, os
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

def load_cfg(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def save_cfg(cfg, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "config_resolved.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

def read_df(path):
    p = Path(path)
    if p.is_dir() or p.suffix.lower() in (".parquet", ".parq"):
        return pd.read_parquet(p, engine="pyarrow")
    if p.suffix.lower() in (".csv"):
        return pd.read_csv(p)

def main():
    ap = argparse.ArgumentParser(description="AFL")
    ap.add_argument("--config", required=True, help="YAML path")
    ap.add_argument("--mode", default="all", choices=["train", "filter", "eval", "all"])
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    np.random.seed(cfg.get("seed", 42))

    paths = cfg.get("paths", {})
    data_path = paths.get("data_path")
    if not data_path:
        raise ValueError("Please set paths.data_path in YAML")
    artifacts = Path(paths.get("artifacts", "artifacts"))
    artifacts.mkdir(parents=True, exist_ok=True)

    df = read_df(data_path)
    if "org" not in df.columns:
        raise ValueError("Dataset must contain column 'org'.")
    
    fl = cfg.get("fl", {})
    orgs = df["org"].unique().tolist()
    cfg["fl"] = {**fl, "n_clients": len(orgs), "org_list": orgs}

    test_ratio = float(cfg.get("split", {}).get("test_ratio", 0.2))
    m = np.random.rand(len(df)) >= test_ratio
    train_df, test_df = df[m].reset_index(drop=True), df[~m].reset_index(drop=True)

    clients = {o: train_df[train_df["org"] == o].reset_index(drop=True)
               for o in orgs if len(train_df[train_df["org"] == o]) > 0}
    
    save_cfg(cfg, artifacts)

    print(f"[INFO] orgs={orgs} (n={len(orgs)}) | train={len(train_df)} | test={len(test_df)}")

    if args.mode in ("train", "all"):
        try:
            from afl.core import train_fl
            train_fl(clients, cfg, artifacts)
        except Exception as e:
            print("[WARN] train skipped:", e)
        
    if args.mode in ("filter", "all"):
        try:
            from afl.core import build_filter
            build_filter(train_df, cfg, artifacts)
        except Exception as e:
            print("[WARN] filter skipped:", e)
        
    if args.mode in ("eval", "all"):
        try:
            from afl.core import evaluate
            evaluate(test_df, cfg, artifacts)
        except Exception as e:
            print("[WARN] eval skipped:", e)
    
    print("[DONE]")

if __name__ == "__main__":
    main()