# Functions which calculate the "standardized" moments.
# For simplicity, I'm not going to try implementing unbiased estimators
import numpy as np


def moments_from_samples(samples, k_vals, weights=None):
    kmax = np.max(k_vals)
    weights = np.ones_like(samples) if weights is None else weights
    weights = weights / np.sum(weights)
    mu1 = np.average(samples, weights=weights) if kmax > 0 else 0
    mu2 = np.sqrt(np.sum(weights * (samples - mu1)**2)) if kmax > 1 else 0

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


def moments_from_pdf(x, pdf, k_vals):
    kmax = np.max(k_vals)
    pdf = pdf / np.sum(pdf)
    mu1 = np.sum(x * pdf) if kmax > 0 else 0
    mu2 = np.sqrt(np.sum((x - mu1)**2 * pdf)) if kmax > 1 else 0

    moments = []
    for k in k_vals:
        if k == 1:
            moment = mu1
        elif k == 2:
            moment = mu2
        else:
            moment = np.sum((x - mu1)**k) / mu2**k
        moments.append(moment)
    return np.array(moments)
