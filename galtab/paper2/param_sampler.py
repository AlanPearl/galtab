import numpy as np
import scipy.stats
import argparse
import pathlib

import halotools.empirical_models as htem
import halotools.sim_manager as htsm
import halotools.mock_observables as htmo
import tabcorr
import nautilus

import galtab
from .param_config import pimax, rp_edges, simname


class ParamSampler:
    def __init__(self, **kwargs):
        self.obs_dir = pathlib.Path(kwargs["obs_dir"])
        self.obs_filename = kwargs["obs_filename"]
        self.save_dir = pathlib.Path(kwargs["save_dir"])
        self.simname = kwargs["simname"]
        if kwargs["use_default_halotools_catalogs"]:
            self.version_name = htsm.sim_defaults.default_version_name
        else:
            self.version_name = "my_cosmosim_halos"

        self.n_live = kwargs["n_live"]
        self.verbose = kwargs["verbose"]

        self.fiducial_params = None
        self.halocat, self.model = None, None
        self.obs = self.load_obs()
        self.redshift = np.mean([self.obs["zmin"], self.obs["zmax"]])
        self.magthresh = self.obs["abs_mr_max"]
        self.model = self.make_model()
        self.cictab, self.wptab = self.load_tabulators()
        self.prior = self.make_prior()
        self.logpdf = self.make_logpdf()
        self.sampler = self.make_sampler()

        self.parameter_samples = None
        self.log_weights = None
        self.log_likelihoods = None
        self.blob = []

        n_start = self.obs["slice_n"].tolist().start
        wp_start = self.obs["slice_wp"].tolist().start
        cic_start = self.obs["slice_cic"].tolist().start
        assert n_start < wp_start < cic_start

    def run(self, verbose=None):
        """
        Runs the nested sampler

        Returns parameter_samples; also saves log_weights, and log_likelihoods
        """
        verbose = self.verbose if verbose is None else verbose
        sampler.run(verbose=verbose)
        self.parameter_samples, self.log_weights, \
            self.log_likelihoods = sampler.posterior()
        return self.parameter_samples

    def save(self, filename):
        np.save(filename, np.array([self], dtype=object))

    @classmethod
    def load(cls, filename):
        obj = np.load(filename, allow_pickle=True)[0]
        return obj

    def make_sampler(self):
        return nautilus.Sampler(
            self.prior, self.likelihood, n_live=self.n_live)

    def load_tabulators(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        cictab_file = self.save_dir / "cictab.npy"
        wptab_file = self.save_dir / "wptab.hdf5"
        if cictab_file.is_file():
            cictab = galtab.CICTabulator.load(cictab_file)
        else:
            cictab = self.make_cictab()
            cictab.save(cictab_file)
        if wptab_file.is_file():
            wptab = tabcorr.TabCorr.read(wptab_file)
        else:
            wptab = self.make_wptab()
            wptab.write(wptab_file)
        return cictab, wptab

    def load_obs(self):
        path = self.obs_dir / self.obs_filename
        return np.load(path, allow_pickle=True)

    def make_model(self):
        # Construct the assembly bias-added Zheng 2007 HOD model
        redshift = self.redshift
        magthresh = self.magthresh

        model = htem.HodModelFactory(
            centrals_occupation=htem.AssembiasZheng07Cens(
                threshold=magthresh, redshift=redshift),
            satellites_occupation=htem.AssembiasZheng07Sats(
                threshold=magthresh, redshift=redshift),
            centrals_profile=htem.TrivialPhaseSpace(redshift=redshift),
            satellites_profile=htem.NFWPhaseSpace(redshift=redshift)
        )
        # self.fiducial_params = self.model.param_dict.copy()
        return model

    def make_halocat(self):
        if self.halocat is None:
            self.halocat = htsm.CachedHaloCatalog(
                simname=self.simname, redshift=self.redshift,
                version_name=self.version_name)
        # model.populate_mock(halocat)

    def make_wptab(self):
        self.make_halocat()

        halotab = tabcorr.TabCorr.tabulate(
            self.halocat, htmo.wp, rp_edges, pi_max=pimax,
            prim_haloprop_key="halo_mvir", prim_haloprop_bins=100,
            sec_haloprop_key="halo_nfw_conc",
            sec_haloprop_percentile_bins=2, project_xyz=True)
        return halotab

    def make_cictab(self):
        self.make_halocat()

        kmax = self.obs.get("cic_kmax")
        kvals = None if kmax is None else np.arange(kmax) + 1
        gtab = galtab.GalaxyTabulator(self.halocat, self.model)
        return gtab.tabulate_cic(
            proj_search_radius=self.obs["proj_search_radius"],
            cylinder_half_length=self.obs["cylinder_half_length"],
            k_vals=kvals)

    @staticmethod
    def make_prior():
        prior = nautilus.Prior()
        prior.add_parameter("logMmin", dist=(9, 16))
        prior.add_parameter("sigma_logM", dist=(1e-5, 5))
        prior.add_parameter("logM0", dist=(9, 16))
        prior.add_parameter("logM1", dist=(11, 16))
        prior.add_parameter("alpha", dist=(1e-5, 5))
        prior.add_parameter(
            "mean_occupation_centrals_assembias_param1", dist=(0, 1))
        prior.add_parameter(
            "mean_occupation_satellites_assembias_param1", dist=(0, 1))
        return prior

    def make_logpdf(self):
        mean, cov = self.obs["mean"], self.obs["cov"]
        return scipy.stats.multivariate_normal(
            mean=mean, cov=cov, allow_singular=True).logpdf

    def likelihood(self, param_dict):
        x = self.predict_observables(param_dict)
        loglike = self.logpdf(x)
        self.blob[-1]["loglike"] = loglike
        return loglike

    def predict_observables(self, param_dict):
        self.model.param_dict.update(param_dict)
        n, wp = self.wptab.predict(self.model)
        cic, n2, n3 = self.cictab.predict(
            self.model, return_number_densities=True)
        if self.obs.get("cic_kmax") is None:
            cic = np.histogram(cic, bins=self.obs["cic_edges"])

        self.blob.append({"param_dict": param_dict,
                          "n": n, "n2": n2, "n3": n3,
                          "wp": wp, "cic": cic})
        return np.array([n, *wp, *cic])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="param_sampler")
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser.add_argument(
        "--obs-dir", type=str, metavar="PATH",
        help="Directory the observation is saved"
    )
    parser.add_argument(
        "--obs-filename", type=str, metavar="NAME",
        help="Name of the observation file (should end in .npz)"
    )
    parser.add_argument(
        "--save-dir", type=str, metavar="PATH",
        help="Directory to save outputs and tabulations"
    )
    parser.add_argument(
        "--simname", type=str, metavar="NAME", default=simname,
        help="Name of dark matter simulation"
    )
    parser.add_argument(
        "-n", "--n-live", type=int, metavar="N", default=1000,
        help="Number of 'live points' to sample"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true"
    )
    parser.add_argument(
        "--use-default-halotools-catalogs", action="store_true"
    )

    a = parser.parse_args()
    sampler = ParamSampler(**a.__dict__)
    sampler.run()
    sampler.save(sampler.save_dir / "sampler.npy")
