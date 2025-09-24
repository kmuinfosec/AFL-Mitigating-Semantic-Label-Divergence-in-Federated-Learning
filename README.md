# Mitigating Semantic Label Divergence in Federated Learning
### Obfuscated Encoding and Alert Filtering for Security Monitoring

This repository contains the official implementation accompanying the paper:

> **Mitigating Semantic Label Divergence in Federated Learning: Obfuscated Encoding and Alert Filtering for Security Monitoring** (submitted to *PLOS One*).

The code provides a reproducible framework to address **Semantic Label Divergence** in federated security monitoring by combining:
- **Keyed Feature Hashing (KFH)** for privacy-preserving, consistent feature extraction across organizations; and
- **Alert Filtering (AFL)** that removes alerts prone to cross-organization label disagreement using centroid-based cosine similarity.

> **Note:** This README intentionally omits performance numbers and experimental results.

---

## 1. Problem & Approach (High-level)

In security operations, multiple organizations train a shared model without exchanging raw data (**Federated Learning, FL**). However, **policy differences** and **local context** often cause the *same or highly similar alert* to receive different labels across entities (a phenomenon we call **Inconsistent Labeling Among Different Entities, ILADE**). This semantic mismatch degrades global model stability and accuracy.

This project mitigates ILADE with two components:

1) **Keyed Feature Hashing (KFH)** — converts packet payloads into fixed-length vectors using feature hashing keyed with a shared secret. This obfuscates raw content while aligning representations across entities.

2) **Alert Filtering (AFL)** — clusters KFH vectors and builds **entity-specific centroids** for confusing regions of the space. At inference, alerts too close (by cosine) to these centroids are *skipped* from automatic labeling, preventing error propagation due to semantic disagreement.

---

## 2. Repository Layout

```
.
├─ configs/
│  └─ default.yaml         # Experiment configuration and hyperparameters
├─ utils/
│  ├─ clustering.py        # Prototype clustering (cosine-based)
│  ├─ cosine.py            # Pairwise and single-vector cosine similarity
│  ├─ encoding.py          # KFH-style obfuscated encoding and helpers
│  └─ eval.py              # F1 utilities, per-org exclusion, AFL evaluation
├─ main.py                 # End-to-end training / evaluation entrypoint
├─ requirements.txt        # Python dependencies
├─ README.md               # This file
└─ .gitignore
```

---

## 3. Installation

Python ≥ 3.9 is recommended.

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

**Key dependencies:** `numpy`, `pandas`, `scikit-learn`, `pyyaml`, `torch`, `pyarrow`.

---

## 4. Quick Start

1) **Configure the experiment**: edit `configs/default.yaml` (see §5).  
2) **Run**:
```bash
python main.py
```
The script will train a global FL model (feed-forward network) and evaluate with/without AFL filtering using the same configuration and seed policy used in training.

> The dataset used in the paper is private. Prepare a dataset with payloads and labels in a compatible schema. See §6 for data notes.

---

## 5. Configuration Reference (`configs/default.yaml`)

Below are the commonly used fields. Names mirror those stored in checkpoints and read at evaluation time.

- `data_path`: Path to your dataset.  
- `vec_size`: Integer dimension of the hashed feature vector (e.g., 768).  
- `win_size`: Content-defined chunking window size for the encoder.  
- `key_hex`: Hex-encoded secret used for keyed hashing (leave empty for unkeyed SHA-256).  
- `hidden`: List of hidden layer sizes for the FFN (e.g., `[512, 256, 128]`).  
- `val_ratio`: Validation split ratio per entity.  
- `test_ratio`: Test split ratio per entity.  
- `seed`: Global random seed used for splits and reproducibility.  
- `threshold`: Decision threshold for positive class during evaluation (default 0.5).  
- `batch`: Evaluation batch size (e.g., 1024).  
- `afl_theta`: Cosine similarity threshold for AFL centroid matching (e.g., 0.95).  
- `outputs`: Directory for checkpoints (e.g., `outputs/…round10_….pt`).

> Checkpoints save `config`, the trained `model_state`, and (for AFL) per-entity centroids used during filtering.

---

## 6. Data Expectations

- **Input**: network security alerts with *hex-encoded payloads* and binary labels (e.g., `Attack` / `Benign`).  
- **Per-entity splits**: each organization’s data is split into train / validation / test with the same `seed`, `val_ratio`, and `test_ratio`.  
- **Privacy**: raw payloads are never shared across entities; KFH provides obfuscated, key-dependent representations for training and filtering.

---

## 7. Key Components (Code Guide)

### 7.1 Obfuscated Encoding (`utils/encoding.py`)
- `AE2(s, window_size)`: content-defined chunking on a hex string.
- `contents2count(hex_list, vec_size, win_size, key=None)`: encodes payloads into count vectors using SHA-256 or HMAC (if `key` provided).
- `key_from_hex(s)`: convenience to load the keyed-hash secret from hex.

### 7.2 Similarity Utilities (`utils/cosine.py`)
- `getCosinePairwise(X)`: pairwise cosine similarity with safe normalization.
- `getCosineSimilarity(v1, v2)`: scalar cosine similarity for two vectors.

### 7.3 Prototype Clustering (`utils/clustering.py`)
- `prototypeClustering(X, th=0.95, precompute=True, return_seed=False)`: cosine-based prototype clustering. Seeds track cluster exemplars; labels can be returned alone or with seed mapping.

### 7.4 AFL Evaluation Utilities (`utils/eval.py`)
- `f1_from_logits(...)` / `evaluate_f1(...)`: precision/recall/F1 computation from logits.
- `apply_exclusion_per_org(X, cents, theta)`: remove samples whose cosine sim. to any same-org centroid ≥ `theta`; returns kept indices and coverage.
- `main()`: loads a saved checkpoint, reconstructs per-entity test sets with the same split policy, applies per-org filtering, and reports metrics (coverage and F1).

---

## 8. Reproducibility Tips

- Use the same `seed`, `val_ratio`, and `test_ratio` across training and evaluation.  
- Set `key_hex` consistently for all parties so that KFH vectors align across entities.  
- Keep `afl_theta` conservative at first (e.g., 0.95) to preserve high coverage; tune later if needed.  
- Ensure your dataset schema supplies (at minimum): entity identifier, payload (hex), and binary label.

---

## 9. Security & Privacy Considerations

- **Obfuscation**: KFH reduces the interpretability of feature vectors and helps limit information recovery if a global model is leaked. Keep the key confidential and rotate it as needed.  
- **Operational use**: AFL can be deployed as a *guardrail*—routes alerts near confusing centroids to manual review while auto-labeling the rest.  
- **Threat model**: KFH is complementary to FL security measures (e.g., secure aggregation). Use in combination with established FL defenses according to your risk profile.

---

## 10. FAQ

- **Q1. Can I disable keyed hashing?**  
  Yes—leave `key_hex` empty. This falls back to unkeyed SHA-256 feature hashing.

- **Q2. Does AFL require retraining?**  
  AFL relies on per-entity clustering of encoded vectors. You can regenerate centroids from a filter-generation split without changing the model architecture.

- **Q3. How do I adjust aggressiveness of filtering?**  
  Tune `afl_theta`. Higher values filter fewer alerts (higher coverage), lower values filter more (stricter).

---

## 11. Citation

If you use this code, please cite the accompanying manuscript:

> **Mitigating Semantic Label Divergence in Federated Learning: Obfuscated Encoding and Alert Filtering for Security Monitoring** (*PLOS One*, under review).

---

## 12. License

Released under the MIT License. See `LICENSE` for details.
