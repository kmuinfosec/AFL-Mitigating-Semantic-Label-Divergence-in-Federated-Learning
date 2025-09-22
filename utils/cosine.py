import numpy as np

def _safe_norms(A):
    n = np.linalg.norm(A, axis=1)
    n[n == 0] = 1.0
    return n

def getCosinePairwise(X):
    A = np.asarray(X, dtype=np.float64)
    S = A @ A.T
    n = _safe_norms(A)
    S *= 1.0 / n[:, None]
    S *= 1.0 / n[None, :]
    np.clip(S, -1.0, 1.0, out=S)
    return S

def getCosineSimilarity(v1, v2):
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    s = float(v1 @ v2) / float(n1 * n2)
    return max(-1.0, min(1.0, s))