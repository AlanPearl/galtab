# Functions which calculate the "standardized" moments.
# For simplicity, I'm not going to try implementing unbiased estimators
import math
from functools import partial

import numpy as np
import jax
from jax import numpy as jnp


@partial(jax.jit, static_argnums=(3, 4))
def jit_sum_at(arr_in, ind_in, ind_out, len_out=None, ind_out_is_sorted=False):
    if len_out is None:
        len_out = len(arr_in)
    arr_out = jnp.zeros(len_out, dtype=arr_in.dtype)
    arr_out = arr_out.at[ind_out].add(
        arr_in[ind_in], indices_are_sorted=ind_out_is_sorted)
    return arr_out


def numpy_sum_at(arr_in, ind_in, ind_out, len_out=None):
    if len_out is None:
        len_out = len(arr_in)
    arr_out = np.zeros(len_out, dtype=arr_in.dtype)
    np.add.at(arr_out, ind_out, arr_in[ind_in])
    return arr_out


def moments_from_samples(samples, k_vals, weights=None):
    kmax = np.max(k_vals)
    weights = np.ones_like(samples) if weights is None else weights
    weights = np.array(weights) / np.sum(weights)
    mu1 = np.sum(samples * weights) if kmax > 0 else np.nan
    mu2 = (np.sum(weights * (samples - mu1)**2)) ** 0.5 if kmax > 1 else np.nan

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
    assert not np.any(bin_edges == ceil_edges), \
        "bin_edges are not allowed to be integers"

    ints = np.arange(ceil_edges[0], ceil_edges[-1]).astype(int)
    lengths = np.diff(ceil_edges)

    pdf = sum([x*[y] for x, y in zip(lengths, pmf)], [])
    return moments_from_samples(ints, k_vals, weights=pdf)


def standardized_moments_from_cumulants(kappa):
    m = raw_moments_from_cumulants(kappa)
    return standardized_moments_from_raw_moments(m)


def standardized_moments_from_raw_moments(m):
    # Prepend the 0th raw moment, which is always 1
    m = [1, *m]
    # Don't calculate 0th central moment - it is always zero
    # Start with the 1st standardized moment (aka the mean - same as raw moment)
    mu = [m[1]]
    for k in range(2, len(m)):
        mu_k = np.zeros_like(m[1])
        if not np.ndim(mu_k):
            mu_k = mu_k.tolist()
        # Calculate central moments
        for i in range(k+1):
            mu_k += math.comb(k, i) * pow(-1, k-i) * m[i] * pow(m[1], k-i)
        # Standardized 2nd moment is the standard deviation
        if k == 2:
            mu_k = np.sqrt(mu_k)
        # Standardized k(>2)th moment is central moment / kth power of std dev
        else:
            assert k > 2
            mu_k = mu_k / np.power(mu[1], k)
        mu.append(mu_k)
    return mu


def raw_moments_from_cumulants(kappa):
    # Prepend the 0th cumulant as NaN
    kappa = [np.nan, *kappa]
    # Start with the 0th raw moment for bookkeeping; remove it before returning
    m = [np.nan]
    for n in range(1, len(kappa)):
        m_n = kappa[n]
        if np.ndim(m_n):
            m_n = np.array(m_n)
        for i in range(1, n):
            m_n += math.comb(n - 1, i - 1) * kappa[i]*m[n-i]
        m.append(m_n)
    return m[1:]


@partial(jax.jit, static_argnums=(1,))
def bernoulli_cumulant(p, k):
    assert k > 0, "There is no 0th Bernoulli cumulant"
    if k == 1:
        ans = p
    elif k == 2:
        ans = p * (1 - p)
    elif k == 3:
        ans = p * (p - 1) * (2 * p - 1)
    elif k == 4:
        ans = p * (-6 * p ** 3 + 12 * p ** 2 - 7 * p + 1)
    elif k == 5:
        ans = p * (24 * p ** 4 - 60 * p ** 3 + 50 * p ** 2 - 15 * p + 1)
    elif k == 6:
        ans = p * (-120 * p ** 5 + 360 * p ** 4 - 390 * p ** 3
                   + 180 * p ** 2 - 31 * p + 1)
    elif k == 7:
        ans = p * (720 * p ** 6 - 2520 * p ** 5 + 3360 * p ** 4
                   - 2100 * p ** 3 + 602 * p ** 2 - 63 * p + 1)
    elif k == 8:
        ans = p * (-5040 * p ** 7 + 20160 * p ** 6 - 31920 * p ** 5
                   + 25200 * p ** 4 - 10206 * p ** 3 + 1932 * p ** 2
                   - 127 * p + 1)
    elif k == 9:
        ans = p * (40320 * p ** 8 - 181440 * p ** 7 + 332640 * p ** 6
                   - 317520 * p ** 5 + 166824 * p ** 4 - 46620 * p ** 3
                   + 6050 * p ** 2 - 255 * p + 1)
    elif k == 10:
        ans = p * (-362880 * p ** 9 + 1814400 * p ** 8 - 3780000 * p ** 7
                   + 4233600 * p ** 6 - 2739240 * p ** 5 + 1020600 * p ** 4
                   - 204630 * p ** 3 + 18660 * p ** 2 - 511 * p + 1)
    else:
        raise NotImplementedError(
            """
            Analytic expression for k>10 not yet implemented. Run the
            BernoulliCumulantGenerator to implement higher cumulants:

            # Example (going up to k=11):
            gen = BernoulliCumulantGenerator()
            gen.generate(kmax=11)
            print(*gen.get_cumulants(), sep="\n\n")
            """
        )
    return ans


class BernoulliCumulantGenerator:
    def __init__(self):
        import sympy
        p, t = sympy.symbols("p t")
        self.generator = sympy.log(1 - p + p*sympy.exp(t))
        self.known_derivatives = [
            sympy.Derivative(self.generator, t).simplify()]

    def calc_next_deriv(self):
        import sympy
        exp = self.known_derivatives[-1]
        deriv = sympy.Derivative(exp, "t").simplify()
        self.known_derivatives.append(deriv)

    def generate(self, kmax):
        n = kmax - len(self.known_derivatives)
        if n <= 0:
            return
        for _ in range(n):
            self.calc_next_deriv()

    def get_cumulants(self):
        return [x.subs("t", 0).simplify() for x in self.known_derivatives]


"""
Precomputed Bernoulli cumulants
===============================

gen = BernoulliCumulantGenerator()
gen.generate(kmax=10)
print(*gen.get_cumulants(), sep="\n\n")
---------------------------------------
p

p*(1 - p)

p*(p - 1)*(2*p - 1)

p*(-6*p**3 + 12*p**2 - 7*p + 1)

p*(24*p**4 - 60*p**3 + 50*p**2 - 15*p + 1)

p*(-120*p**5 + 360*p**4 - 390*p**3 + 180*p**2 - 31*p + 1)

p*(720*p**6 - 2520*p**5 + 3360*p**4 - 2100*p**3 + 602*p**2 - 63*p + 1)

p*(-5040*p**7 + 20160*p**6 - 31920*p**5 + 25200*p**4 - 10206*p**3 + 1932*p**2 - 127*p + 1)

p*(40320*p**8 - 181440*p**7 + 332640*p**6 - 317520*p**5 + 166824*p**4 - 46620*p**3 + 6050*p**2 - 255*p + 1)

p*(-362880*p**9 + 1814400*p**8 - 3780000*p**7 + 4233600*p**6 - 2739240*p**5 + 1020600*p**4 - 204630*p**3 + 18660*p**2 - 511*p + 1)
"""
