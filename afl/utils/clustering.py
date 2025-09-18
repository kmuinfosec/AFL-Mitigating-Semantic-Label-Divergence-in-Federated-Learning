import numpy as np
from .cosine import getCosinePairwise, getCosineSimilarity

def prototypeClustering(X, th=0.95, precompute=True, return_seed=False):
    X = np.asarray(X)
    n = len(X)
    if n == 0:
        return ([], {}) if return_seed else []
    
    ready = list(range(n))
    labels = [-1] * n
    cid, nxt = 0, 0
    seeds = {}

    S = getCosinePairwise(X) if precompute else None

    while ready:
        src = nxt
        tmp, next_score, cnt = [], 1.0, 0
        for trg in ready:
            score = S[src, trg] if precompute else getCosineSimilarity(X[src], X[trg])
            if score >= th:
                labels[trg] = cid
                cnt += 1
            else:
                tmp.append(trg)
            if score < next_score:
                next_score, nxt = score, trg
        if cnt == 1:
            labels[src] = -1
            cid -= 1
        else:
            seeds[cid] = src
        ready = tmp
        cid += 1
    
    return (labels, seeds) if return_seed else labels