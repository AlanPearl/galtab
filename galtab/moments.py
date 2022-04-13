# Functions which calculate the "standardized" moments.
# For simplicity, I'm not going to try implementing unbiased estimators
import numpy as np


def moments_from_samples(samples, k_vals, weights=None):
    kmax = np.max(k_vals)
    weights = np.ones_like(samples) if weights is None else weights
    weights = np.array(weights) / np.sum(weights)
    mu1 = np.sum(samples * weights) if kmax > 0 else 0
    mu2 = (np.sum(weights * (samples - mu1)**2)) ** 0.5 if kmax > 1 else 0

    moments = []
    for k in k_vals:
        if k == 1:
            moment = mu1
        elif k == 2:
            moment = mu2
        else:
            moment = np.sum(weights * (samples - mu1)**k) / mu2**k
        moments.append(moment)
    return np.array(moments)


def moments_from_binned_pmf(bin_edges, pmf, k_vals):
    bin_edges = np.sort(bin_edges)
    ceil_edges = np.ceil(bin_edges).astype(int)
    assert not np.any(bin_edges == ceil_edges)

    ints = np.arange(ceil_edges[0], ceil_edges[-1]).astype(int)
    lengths = np.diff(ceil_edges)

    pdf = sum([x*[y] for x, y in zip(lengths, pmf)], [])
    return moments_from_samples(ints, k_vals, weights=pdf)
