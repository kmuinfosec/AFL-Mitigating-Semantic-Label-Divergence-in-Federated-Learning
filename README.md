# Mitigating Semantic Label Divergence in Federated Learning  
### Obfuscated Encoding and Alert Filtering for Security Monitoring

This repository provides the official implementation of the paper  
**“Mitigating Semantic Label Divergence in Federated Learning: Obfuscated Encoding and Alert Filtering for Security Monitoring”**  
submitted to *PLOS ONE*.

The code enables reproduction of the proposed methods to address **Semantic Label Divergence (SLD)** in federated learning for security monitoring tasks.  
Our approach combines **Obfuscated Encoding** for privacy-preserving feature extraction and **Alert Filtering (AFL)** for reducing label inconsistency across organizations.

---

## Overview

In federated security monitoring, multiple organizations collaboratively train a global model without sharing raw data.  
However, differences in security policies and labeling practices often lead to **Semantic Label Divergence**, where identical network traffic can receive inconsistent labels across organizations.  
This divergence degrades model accuracy and stability.

To mitigate this issue, we propose:

1. **Obfuscated Encoding**  
   - Converts packet payloads into fixed-length vectors using hash-based encoding (SHA-256 or HMAC) to protect sensitive information while retaining structural patterns.

2. **Alert Filtering (AFL)**  
   - Identifies and removes samples that are too similar to organization-specific centroids (based on cosine similarity) to reduce label conflicts during training.

---

## Project Structure

```
.
├─ configs/
│  └─ default.yaml         # Experiment configuration file
├─ utils/
│  ├─ clustering.py        # Prototype clustering using cosine similarity
│  ├─ cosine.py            # Pairwise and single cosine similarity calculations
│  ├─ encoding.py          # Obfuscated Encoding implementation
│  └─ eval.py              # AFL filtering, evaluation metrics, and utility functions
├─ main.py                 # Main training and evaluation script
├─ requirements.txt        # Python dependencies
├─ README.md               # Project documentation (this file)
└─ .gitignore
```

---

## Requirements

Install the required packages with:

```bash
pip install -r requirements.txt
```

Key dependencies:
- numpy>=1.24
- pandas>=2.0
- scikit-learn>=1.4
- torch>=2.3
- pyyaml>=6.0
- pyarrow>=15.0

---

## Usage

1. **Prepare Configuration**  
   Edit `configs/default.yaml` to specify:
   - Dataset paths
   - Vector size and window size for encoding
   - Training parameters (learning rate, batch size, etc.)

2. **Run Training and Evaluation**  
   Execute the main script:
   ```bash
   python main.py
   ```
   The script will:
   - Load the dataset
   - Train a federated learning model (`FFN`)
   - Apply the proposed Alert Filtering method for evaluation

---

## Data

The code assumes network security event data with packet payloads and labels.  
Due to privacy restrictions, the dataset used in the paper is not publicly released.  
Users should prepare their own datasets in a similar format to reproduce experiments.

---

## Citation

If you use this code or build upon it, please cite the following paper:

> **Mitigating Semantic Label Divergence in Federated Learning:  
> Obfuscated Encoding and Alert Filtering for Security Monitoring**  
> *PLOS One*, 2025 (under review).

---

## License

This project is released under the MIT License.  
See the [LICENSE](LICENSE) file for details.
