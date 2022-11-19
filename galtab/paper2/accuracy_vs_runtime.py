from time import time
import argparse

from tqdm import tqdm
import numpy as np
import pandas as pd

import galtab.paper2.param_sampler


default_num_ht_trials = 250
default_num_gt_trials = 5
default_num_min_quants = 4
default_num_max_weights = 8
default_also_test_smdpl = False


class AccuracyRuntimeTester:
    def __init__(self, **kwargs):
        self.num_ht_trials = kwargs.get(
            "num_ht_trials", default_num_ht_trials)
        self.num_gt_trials = kwargs.get(
            "num_gt_trials", default_num_gt_trials)
        self.num_min_quants = kwargs.get(
            "num_min_quants", default_num_min_quants)
        self.num_max_weights = kwargs.get(
            "num_max_weights", default_num_max_weights)
        self.also_test_smdpl = kwargs.get(
            "also_test_smdpl", default_also_test_smdpl)

        self.simnames = ["bolplanck"]
        if self.also_test_smdpl:
            self.simnames.append("smdpl")
        self.gt_results = None
        self.ht_results = None

        self.sampler_kw = dict(
            obs_dir="../desi_observations/",
            OBS_FILENAME="desi_obs_20p5_kmax5.npz",
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
            start_without_assembias=False,
        )

        self.bolplanck_sampler = galtab.paper2.param_sampler.ParamSampler(
            simname="bolplanck", **self.sampler_kw
        )
        self.bolplanck_trimmed_halocat = (
            self.bolplanck_sampler.cictab.galtabulator.halocat)

        self.smdpl_sampler = galtab.paper2.param_sampler.ParamSampler(
            simname="smdpl", **self.sampler_kw
        ) if self.also_test_smdpl else None
        self.smdpl_trimmed_halocat = (
            self.smdpl_sampler.cictab.galtabulator.halocat
            if self.also_test_smdpl else None)

    def save(self, file):
        arr = np.array([self.gt_results.to_dict(), self.ht_results.to_dict()],
                       dtype=object)
        np.save(file, arr)

    def run_ht_trials(self):
        self.bolplanck_sampler.model.populate_mock(
            # self.bolplanck_trimmed_halocat
            self.bolplanck_sampler.halocat
        )
        if self.also_test_smdpl:
            self.smdpl_sampler.model.populate_mock(
                # self.smdpl_trimmed_halocat
                self.smdpl_sampler.halocat
            )

        ht_results = []
        for simname in tqdm(self.simnames, leave=None, desc="simname (ht)"):
            for delmock in tqdm([False, True], leave=None, desc="delmock"):
                for i in tqdm(range(self.num_ht_trials),
                              leave=None, desc="trial_num"):
                    if simname == "bolplanck":
                        sampler = self.bolplanck_sampler
                    else:
                        sampler = self.smdpl_sampler
                    t0 = time()
                    if delmock:
                        halocat = sampler.halocat
                    else:
                        halocat = None
                    val, n = sampler.predict_cic_halotools(
                        sampler.model, return_number_density=True,
                        halocat=halocat)
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
        min_quants = np.logspace(-2, -5, self.num_min_quants)
        max_weights = np.geomspace(0.9, 0.01, self.num_max_weights)
        # max_quants = [0.9, 0.999, 0.99999]
        sat_1quants = np.logspace(-3, -15, self.num_max_weights)
        sqiomws = [False]*len(max_weights) + [True]*len(sat_1quants)
        max_weights = [*max_weights, *sat_1quants]

        for simname in tqdm(self.simnames, leave=None, desc="simname (gt)"):
            for min_quant in tqdm(min_quants, leave=None, desc="min_quant"):
                for j in tqdm(range(len(max_weights)),
                              leave=None, desc="max_weight"):
                    max_weight = max_weights[j]
                    sqiomw = sqiomws[j]
                    for i_trial in tqdm(range(self.num_gt_trials), leave=None):
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

    parser = argparse.ArgumentParser(prog="accuracy_vs_runtime")
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser.add_argument(
        "--num-ht-trials", type=int, default=default_num_ht_trials,
        help="Number of halotools trials per hyperparam search"
    )
    parser.add_argument(
        "--num-gt-trials", type=int, default=default_num_gt_trials,
        help="Number of galtab trials per hyperparam search"
    )
    parser.add_argument(
        "--num-min-quants", type=int, default=default_num_min_quants,
        help="Number of min_quant values to search"
    )
    parser.add_argument(
        "--num-max-weights", type=int, default=default_num_max_weights,
        help="Number of max_weight/sat_quant values to search"
    )
    parser.add_argument(
        "--also-test-smdpl", type=int, default=default_also_test_smdpl,
        help="Test 'smdpl' in addition to 'bolplanck'"
    )
    parser.add_argument(
        "--use-past-ht-results", action="store_true",
        help="Don't rerun halotools trials if already ran"
    )
    a = parser.parse_args()

    output = "accuracy_runtime_results.npy"

    past_gt_results, past_ht_results = None, None
    if a.__dict__.pop("use_past_ht_results"):
        try:
            past_gt_results, past_ht_results = np.load(
                output, allow_pickle=True)
        except FileNotFoundError:
            pass
        else:
            past_gt_results = pd.DataFrame(past_gt_results)
            past_ht_results = pd.DataFrame(past_ht_results)

    tester = AccuracyRuntimeTester(**a.__dict__)
    # Always run the galtab trials
    tester.run_gt_trials()
    if past_ht_results is None:
        tester.run_ht_trials()
    else:
        tester.ht_results = past_ht_results
    tester.save(output)
