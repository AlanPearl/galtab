# Functions which calculate the "standardized" moments.
# For simplicity, I'm not going to try implementing unbiased estimators
import jax.numpy as jnp


def moments_from_samples(samples, k_vals, weights=None):
    kmax = jnp.max(k_vals)
    weights = jnp.ones_like(samples) if weights is None else weights
    weights = weights / jnp.sum(weights)
    mu1 = jnp.average(samples, weights=weights) if kmax > 0 else 0
    mu2 = jnp.sqrt(jnp.sum(weights * (samples - mu1)**2)) if kmax > 1 else 0

    moments = []
    for k in k_vals:
        if k == 1:
            moment = mu1
        elif k == 2:
            moment = mu2
        else:
            moment = jnp.sum(weights * (samples - mu1)**k) / mu2**k
        moments.append(moment)
    return jnp.array(moments)


def moments_from_pdf(x, pdf, k_vals):
    kmax = jnp.max(k_vals)
    pdf = pdf / jnp.sum(pdf)
    mu1 = jnp.sum(x * pdf) if kmax > 0 else 0
    mu2 = jnp.sqrt(jnp.sum((x - mu1)**2 * pdf)) if kmax > 1 else 0

    moments = []
    for k in k_vals:
        if k == 1:
            moment = mu1
        elif k == 2:
            moment = mu2
        else:
            moment = jnp.sum((x - mu1)**k) / mu2**k
        moments.append(moment)
    return jnp.array(moments)
