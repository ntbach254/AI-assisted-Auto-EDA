import numpy as np


def benjamini_hochberg(pvals):
    pvals = np.asarray(pvals, dtype=float)
    m = int(len(pvals))
    if m == 0:
        return pvals

    order = np.argsort(pvals)
    p_sorted = pvals[order]

    q_sorted = np.empty(m, dtype=float)
    prev = 1.0

    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = (p_sorted[i] * m) / rank
        if val > 1.0:
            val = 1.0
        prev = min(prev, val)
        q_sorted[i] = prev

    q = np.empty(m, dtype=float)
    q[order] = q_sorted
    return q