import numpy as np
import scipy.optimize
import scipy.stats


class PoissonTabulator:
    def __init__(self):
        self.saved = {}

    def save(self, k):
        low, high = get_low_high_grid(k)
        x = np.linspace(low, high, 10_000)
        y = poisson_cdf(x, k)
        self.saved[k] = x, y

    def interp_cdf(self, mu, k):
        x, y = self.saved[int(np.floor(k))]
        return np.interp(mu, x, y, left=1.0, right=0.0)

    def check_if_saved(self, k):
        k = int(np.floor(k))
        if k not in self.saved:
            self.save(k)


def poisson_cdf(lam, k):
    return scipy.stats.poisson(mu=lam).cdf(k)


def lam_from_cdf(k, cdf):
    return scipy.optimize.root_scalar(
        lambda x: poisson_cdf(x, k) - cdf, xtol=(1-cdf)/10,
        x0=(k+1)*10, x1=(k+1)/10, bracket=[0, (k+1)*1e5]).root


def get_low_high_grid(k):
    if k < 0:
        return 0.0, 1.0
    return lam_from_cdf(k, 1 - 1e-16), lam_from_cdf(k, 1e-16)


def poisson(mu, k):
    poisson.tabulator.check_if_saved(k)
    return poisson.tabulator.interp_cdf(mu, k)


poisson.tabulator = PoissonTabulator()
