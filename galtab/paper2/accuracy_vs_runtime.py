from time import time

from tqdm import tqdm
import numpy as np
import pandas as pd

import galtab.paper2.param_sampler


class AccuracyRuntimeTester:
    def __init__(self):
        self.sampler_kw = dict(
            obs_dir="../desi_observations/",
            OBS_FILENAME="desi_obs_20p0_kmax5.npz",
            SAVE_DIR="../desi_results/results_20p0/",
            use_default_halotools_catalogs=False,
            seed=None,
            n_live=10,
            verbose=True,
            temp_cictab=True,
            n_mc=10,
            min_quant=0.001,
            max_quant=0.9999,
        )
        self.bolplanck_sampler = galtab.paper2.param_sampler.ParamSampler(
            simname="bolplanck", **self.sampler_kw
        )
        self.smdpl_sampler = galtab.paper2.param_sampler.ParamSampler(
            simname="smdpl", **self.sampler_kw
        )
        self.bolplanck_halocat = self.bolplanck_sampler.halocat
        self.smdpl_halocat = self.smdpl_sampler.halocat

        self.gt_results = None
        self.ht_results = None

    def save(self, file):
        arr = np.array([self.gt_results, self.ht_results], dtype=object)
        np.save(file, arr)

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
        num_trials = 10
        simnames = ["bolplanck"]
        min_quants = [0.001, 0.0001]
        max_quants = 1 - np.logspace(-1, -7, 14)
        n_mcs = [1, 2, 3, 4, 5]

        for simname in tqdm(simnames):
            for min_quant in tqdm(min_quants, leave=None):
                for max_quant in tqdm(max_quants, leave=None):
                    for i_trial in tqdm(range(num_trials)):
                        t0 = time()
                        if simname == "smdpl":
                            halocat = self.smdpl_halocat
                        else:
                            halocat = self.bolplanck_halocat
                        kw = self.sampler_kw.copy()
                        kw.update(dict(
                            simname=simname,
                            min_quant=min_quant,
                            max_quant=max_quant,
                            halocat=halocat,
                            n_mc=max(n_mcs)
                        ))
                        sampler = galtab.paper2.param_sampler.ParamSampler(
                            **kw)
                        tabtime = time() - t0
                        for n_mc in tqdm(n_mcs, leave=None):
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
                            results["n_placeholders"] = \
                                len(sampler.cictab.galtabulator.galaxies)
                            results["trial_num"] = i_trial
                            gt_results.append(results)
        self.gt_results = pd.DataFrame(gt_results)
        return gt_results


if __name__ == "__main__":
    output = "accuracy_runtime_results.npy"
    try:
        past_gt_results, past_ht_results = np.load(output, allow_pickle=True)
    except FileNotFoundError:
        past_gt_results, past_ht_results = None, None

    tester = AccuracyRuntimeTester()
    tester.run_gt_trials()
    tester.ht_results = past_ht_results  # tester.run_ht_trials()
    tester.save(output)
