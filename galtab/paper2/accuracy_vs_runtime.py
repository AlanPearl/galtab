from time import time

from tqdm import tqdm
import numpy as np
import pandas as pd

import galtab.paper2.param_sampler


class AccuracyRuntimeTester:
    def __init__(self):
        self.bolplanck_sampler = galtab.paper2.param_sampler.ParamSampler(
            obs_dir="../desi_observations/",
            OBS_FILENAME="desi_obs_20p0_kmax5.npz",
            SAVE_DIR="../desi_results/results_20p0/",
            simname="bolplanck",
            use_default_halotools_catalogs=False,
            seed=None,
            n_live=10,
            verbose=True,
            temp_cictab=True,
            n_mc=10,
            min_quant=0.001,
            max_quant=0.9999,
        )
        self.smdpl_sampler = galtab.paper2.param_sampler.ParamSampler(
            obs_dir="../desi_observations/",
            OBS_FILENAME="desi_obs_20p0_kmax5.npz",
            SAVE_DIR="../desi_results/results_20p0/",
            simname="smdpl",
            use_default_halotools_catalogs=False,
            seed=None,
            n_live=10,
            verbose=True,
            temp_cictab=True,
            n_mc=10,
            min_quant=0.001,
            max_quant=0.9999,
        )
        self.bolplanck_halocat = self.bolplanck_sampler.halocat
        self.smdpl_halocat = self.smdpl_sampler.halocat

        self.gt_results = None
        self.ht_results = None

    def save(self):
        arr = np.array([self.gt_results, self.ht_results], dtype=object)
        np.save("accuracy_runtime_results.npy", arr)

    def run_ht_trials(self):
        num_ht_trials = 500
        ht_results = []
        for simname in tqdm(["bolplanck", "smdpl"]):
            for delmock in tqdm([False, True], leave=None):
                for i in tqdm(range(num_ht_trials)):
                    if simname == "bolplanck":
                        sampler = self.bolplanck_sampler
                    else:
                        sampler = self.smdpl_sampler
                    t0 = time()
                    if delmock:
                        del sampler.model.mock  # ~3 sec
                    val, n = sampler.predict_cic_halotools(
                        sampler.model, return_number_density=True)
                    t = time() - t0

                    results = {}
                    for k_i in range(len(val)):
                        results[f"k{k_i + 1}"] = val[k_i]
                    results["n"] = n
                    results["time"] = t
                    results["simname"] = simname
                    results["delmock"] = delmock
                    results["trial_num"] = i
                    ht_results.append(results)

        self.ht_results = pd.DataFrame(ht_results)
        return self.ht_results

    def run_gt_trials(self):
        gt_results = []
        for simname in tqdm(["bolplanck", "smdpl"]):
            for min_quant in tqdm([0.001, 0.01, 0.1], leave=None):
                for max_quant in tqdm([0.9999, 0.999, 0.99, 0.9], leave=None):
                    t0 = time()
                    if simname == "smdpl":
                        halocat = self.smdpl_halocat
                    else:
                        halocat = self.bolplanck_halocat
                    sampler = galtab.paper2.param_sampler.ParamSampler(
                        obs_dir="../desi_observations/",
                        OBS_FILENAME="desi_obs_20p0_kmax5.npz",
                        SAVE_DIR="../desi_results/results_20p0/",
                        simname=simname,
                        halocat=halocat,
                        use_default_halotools_catalogs=False,
                        seed=None,
                        n_live=10,
                        verbose=True,
                        temp_cictab=True,
                        n_mc=10,
                        min_quant=min_quant,
                        max_quant=max_quant,
                    )
                    tabtime = time() - t0
                    for n_mc in tqdm([10, 5, 2], leave=None):
                        t0 = time()
                        val, n1, n2 = sampler.predict_cic(
                            sampler.model, n_mc=n_mc,
                            return_number_densities=True)
                        t = time() - t0

                        results = {}
                        for i in range(len(val)):
                            results[f"k{i + 1}"] = val[i]
                        results["n1"] = n1
                        results["n2"] = n2
                        results["time"] = t
                        results["simname"] = simname
                        results["min_quant"] = min_quant
                        results["max_quant"] = max_quant
                        results["n_mc"] = n_mc
                        results["tabtime"] = tabtime
                        gt_results.append(results)
        self.gt_results = pd.DataFrame(gt_results)
        return gt_results


if __name__ == "__main__":
    tester = AccuracyRuntimeTester()
    tester.run_gt_trials()
    tester.run_ht_trials()
    tester.save()
