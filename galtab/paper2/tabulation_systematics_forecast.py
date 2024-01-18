import gc

import numpy as np
from tqdm import tqdm

from .param_sampler import ParamSampler


class FisherSystematicsForecast:
    def __init__(self):
        sampler_kw = dict(
            simname="smdpl",
            obs_dir="../desi_observations/",
            OBS_FILENAME="desi_obs_20p5_kmax5.npz",
            SAVE_DIR="../desi_results/results_20p0/",
            use_default_halotools_catalogs=False,
            seed=None,
            N=10,
            verbose=True,
            temp_cictab=True,
            n_mc=1,
            min_quant=1e-4,
            max_weight=0.05,
            sqiomw=False,
            start_without_assembias=True,
            tabulate_at_starting_params=True,
            sampler_name="nautilus",  # (so it doesn't try making a backend.h5)
        )
        self.sampler = ParamSampler(**sampler_kw)
        self.sampler_kw = dict(halocat=self.sampler.halocat, **sampler_kw)

    def calc_cic_systematics_cov(self, num_trials=100):
        gt_results = []
        desc = "CiCTabulator realizations"
        for _ in tqdm(range(num_trials), desc=desc):
            self.sampler = ParamSampler(**self.sampler_kw)
            gc.collect()
            cic_val = self.sampler.predict_cic(self.sampler.model)
            gt_results.append([*cic_val])

        return np.cov(gt_results, rowvar=False)

    def calc_jacobian_dcic_dparam(self, delta_params=0.01):
        """Using finite difference, return J where J_ij = dcic_i/dparam_j"""
        model = self.sampler.model
        fid_params = model.param_dict.copy()

        nparams = len(fid_params)
        if np.ndim(delta_params) < 1:
            delta_params = np.array([delta_params] * nparams)

        jac_transpose = []
        desc = "Finite difference over each parameter"
        for i in tqdm(range(nparams), desc=desc):
            param = list(fid_params.keys())[i]
            fid_value = list(fid_params.values())[i]
            delta_param = delta_params[i]

            model.param_dict.update({param: fid_value + delta_param})
            cic_up = self.sampler.predict_cic(model)
            model.param_dict.update({param: fid_value - delta_param})
            cic_down = self.sampler.predict_cic(model)
            model.param_dict.update(fid_params)
            jac_transpose.append((cic_up - cic_down) / (2 * delta_param))

        return np.array(jac_transpose).T

    def fisher_forecast(self, cov, jac):
        """Returns inv(fisher_matrix) which estimates cov(params)"""
        fisher_matrix = jac.T @ np.linalg.inv(cov) @ jac
        # terms = (cic_errs[:, None, None] ** -2
        #          * jac[:, None, :]
        #          * jac[:, :, None])
        # fisher_matrix = np.sum(terms, axis=0)
        return fisher_matrix


if __name__ == "__main__":
    forecast = FisherSystematicsForecast()
    cic_cov = forecast.calc_cic_systematics_cov()
    print(f"CiC cov matrix = {repr(cic_cov)}")
    jacobian = forecast.calc_jacobian_dcic_dparam()
    print(f"Jacobian = {repr(jacobian)}")
    fisher_matrix = forecast.fisher_forecast(cic_cov, jacobian)
    print(f"Forecasted Fisher matrix = {repr(fisher_matrix)}")
    param_errs = 1/np.sqrt(np.diag(fisher_matrix))
    print(f"Errors for params ({forecast.sampler.model.param_dict.keys()})"
          f" = {repr(param_errs)}")
