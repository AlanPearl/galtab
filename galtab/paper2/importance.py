import argparse

import numpy as np
from tqdm import tqdm
from smt.sampling_methods import LHS

import halotools.mock_observables as htmo

import galtab.moments
from .param_sampler import ParamSampler
from . import param_config

default_num_samples = 1000
default_outfile = "importance_results.npz"


class ImportanceCalculator:
    save_names = ["hod_samples", "obs_samples", "nobs_samples", "cic_moment_numbers"]

    def __init__(self):
        self.cic_moment_numbers = np.arange(1, 21)
        self.sampler_kw = dict(
            obs_dir="../desi_observations/",
            OBS_FILENAME="desi_obs_20p5_kmax5.npz",
            SAVE_DIR="../desi_results/results_20p5_kmax/",
            use_default_halotools_catalogs=False,
            seed=None,
            N=10,
            verbose=True,
            temp_cictab=True,
            n_mc=1,
            min_quant=1e-2,
            max_weight=0.25,
            sqiomw=False,
            start_without_assembias=False,
            tabulate_at_starting_params=True,
            simname="bolplanck",
            cictab="none",
            sampler_name="nautilus",  # (so it doesn't try making a backend.h5 file)
        )

        self.sampler = ParamSampler(**self.sampler_kw)
        self.sampler.make_halocat()
        self.magthresh = self.sampler.magthresh
        self.hod_samples = self.obs_samples = self.nobs_samples = None

    def calc_observables(self, ht_model):
        if hasattr(ht_model, "mock"):
            ht_model.mock.populate()
        else:
            ht_model.populate_mock(self.sampler.halocat)
        vz = ht_model.mock.galaxy_table["vz"]
        xyz = [ht_model.mock.galaxy_table[x] for x in "xyz"]
        xyz = htmo.return_xyz_formatted_array(
            *xyz, period=self.sampler.halocat.Lbox, velocity=vz,
            velocity_distortion_dimension="z",
            cosmology=self.sampler.cosmo, redshift=self.sampler.redshift)

        # Calculate ngal (number density)
        volume = np.product(self.sampler.halocat.Lbox)
        ngal = len(xyz) / volume

        # Calculate wp (projected two-point correlation function)
        wp = htmo.wp(xyz, self.sampler.rp_edges, pi_max=self.sampler.pimax,
                     period=self.sampler.halocat.Lbox)

        # Calculate dP(N_CIC)/dN_CIC (counts-in-cylinders)
        cic_counts = htmo.counts_in_cylinders(
            xyz, xyz, proj_search_radius=self.sampler.proj_search_radius,
            cylinder_half_length=self.sampler.cylinder_half_length,
            period=self.sampler.halocat.Lbox)

        cic_moments = galtab.moments.moments_from_samples(
            cic_counts, self.cic_moment_numbers)

        return np.array([ngal, *wp, *cic_moments])

    def sample(self, n):
        model = self.sampler.model

        param_names = list(param_config.kuan_params[self.magthresh].keys())
        param_means = [param_config.kuan_params[self.magthresh][x]
                       for x in param_names]
        param_highs = [param_config.kuan_err_high[self.magthresh][x]
                       for x in param_names]
        param_lows = [param_config.kuan_err_low[self.magthresh][x]
                      for x in param_names]
        param_lims = np.array([[x - y, x + z] for x, y, z
                              in zip(param_means, param_lows, param_highs)])
        hod_samples = LHS(xlimits=param_lims)(n)

        obs_samples = []
        for params in tqdm(hod_samples):
            params = dict(zip(param_names, params))
            model.param_dict.update(params)
            obs_samples.append(self.calc_observables(model))
        obs_samples = np.array(obs_samples)

        self.hod_samples, self.obs_samples = hod_samples, obs_samples
        self.nobs_samples = self.normalize_observables(self.obs_samples)

    @property
    def obs_mean(self):
        return np.mean(self.obs_samples, axis=0)

    @property
    def obs_std(self):
        return np.std(self.obs_samples, ddof=1, axis=0)

    def normalize_observables(self, observables):
        return (observables - self.obs_mean) / self.obs_std

    def denormalize_observables(self, normalized_observables):
        return normalized_observables * self.obs_std + self.obs_mean

    def save(self, file):
        kw = {name: getattr(self, name) for name in self.save_names}
        np.savez(file, **kw)

    @classmethod
    def load(cls, file):
        npzip = np.load(file, allow_pickle=True)
        new_obj = cls()
        for name in cls.save_names:
            setattr(new_obj, name, npzip[name])
        return new_obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="importance")
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser.add_argument(
        "--num-samples", type=int, default=default_num_samples,
        help="Number of points to search in parameter-space"
    )
    parser.add_argument(
        "--outfile", type=str, default=default_outfile,
        help="Name of file to output"
    )

    a = parser.parse_args()
    num_samples = a.__dict__.pop("num_samples")
    outfile = a.__dict__.pop("outfile")

    calc = ImportanceCalculator()
    calc.sample(num_samples)
    calc.save(outfile)
