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
            n_mc=1,
            min_quant=1e-4,
            max_weight=0.05,
            sqiomw=False,
        )
        self.bolplanck_sampler = galtab.paper2.param_sampler.ParamSampler(
            simname="bolplanck", **self.sampler_kw
        )
        self.smdpl_sampler = galtab.paper2.param_sampler.ParamSampler(
            simname="smdpl", **self.sampler_kw
        )
        self.bolplanck_trimmed_halocat = \
            self.bolplanck_sampler.cictab.galtabulator.halocat
        self.smdpl_trimmed_halocat = \
            self.smdpl_sampler.cictab.galtabulator.halocat

        self.gt_results = None
        self.ht_results = None

    def save(self, file):
        arr = np.array([self.gt_results.to_dict(), self.ht_results.to_dict()],
                       dtype=object)
        np.save(file, arr)

    def run_ht_trials(self):
        self.bolplanck_sampler.model.populate_mock(
            self.bolplanck_trimmed_halocat
        )
        self.smdpl_sampler.model.populate_mock(
            self.smdpl_trimmed_halocat
        )

        num_ht_trials = 250
        simnames = ["bolplanck"]  # , "smdpl"]
        ht_results = []
        for simname in tqdm(simnames, leave=None, desc="simname (ht)"):
            for delmock in tqdm([False, True], leave=None, desc="delmock"):
                for i in tqdm(range(num_ht_trials), leave=None, desc="trial_num"):
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
        num_trials = 5
        simnames = ["bolplanck"]  # , "smdpl"]
        min_quants = np.logspace(-2, -5, 4)
        max_weights = np.geomspace(0.9, 0.01, 8)
        # max_quants = [0.9, 0.999, 0.99999]
        sat_1quants = np.logspace(-3, -15, 8)
        sqiomws = [False]*len(max_weights) + [True]*len(sat_1quants)
        max_weights = [*max_weights, *sat_1quants]

        for simname in tqdm(simnames, leave=None, desc="simname (gt)"):
            for min_quant in tqdm(min_quants, leave=None, desc="min_quant"):
                for j in tqdm(range(len(max_weights)), leave=None, desc="max_weight"):
                    max_weight = max_weights[j]
                    sqiomw = sqiomws[j]
                    for i_trial in tqdm(range(num_trials), leave=None):
                        t0 = time()
                        if simname == "smdpl":
                            halocat = self.smdpl_sampler.halocat
                        else:
                            halocat = self.bolplanck_sampler.halocat
                        kw = self.sampler_kw.copy()
                        kw.update(dict(
                            simname=simname,
                            min_quant=min_quant,
                            max_weight=max_weight,
                            halocat=halocat,
                            sqiomw=sqiomw,
                        ))
                        sampler = galtab.paper2.param_sampler.ParamSampler(
                            **kw)
                        tabtime = time() - t0
                        t0 = time()
                        val, n1, n2 = sampler.predict_cic(
                            sampler.model,
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
                        results["max_weight"] = max_weight
                        results["sqiomw"] = sqiomw
                        results["tabtime"] = tabtime
                        results["n_placeholders"] = \
                            len(sampler.cictab.galtabulator.galaxies)
                        results["trial_num"] = i_trial
                        gt_results.append(results)
        self.gt_results = pd.DataFrame(gt_results)
        return gt_results


if __name__ == "__main__":
    import jax
    jax.config.update("jax_platform_name", "cpu")

    output = "accuracy_runtime_results.npy"
    try:
        past_gt_results, past_ht_results = np.load(output, allow_pickle=True)
        past_gt_results = pd.DataFrame(past_gt_results)
        past_ht_results = pd.DataFrame(past_ht_results)
    except FileNotFoundError:
        past_gt_results, past_ht_results = None, None

    tester = AccuracyRuntimeTester()
    # Always run the galtab trials
    tester.run_gt_trials()
    if past_ht_results is None:
        # Don't rerun halotools trials if they've already been run
        tester.run_ht_trials()
    else:
        tester.ht_results = past_ht_results
    tester.save(output)
