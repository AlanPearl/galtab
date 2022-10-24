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
from .param_config import simname


class ParamSampler:
    def __init__(self, **kwargs):
        self.obs_dir = pathlib.Path(kwargs["obs_dir"])
        self.obs_filename = kwargs["OBS_FILENAME"]
        self.save_dir = pathlib.Path(kwargs["SAVE_DIR"])
        self.simname = kwargs["simname"]
        if kwargs["use_default_halotools_catalogs"]:
            self.version_name = htsm.sim_defaults.default_version_name
        else:
            self.version_name = "my_cosmosim_halos"

        self.n_live = kwargs["n_live"]
        self.verbose = kwargs["verbose"]
        self.temp_cictab = kwargs.get("temp_cictab", False)

        self.fiducial_params = None
        self.halocat, self.model = None, None
        self.obs = self.load_obs()
        self.proj_search_radius = self.obs["proj_search_radius"].tolist()
        self.cylinder_half_length = self.obs["cylinder_half_length"].tolist()
        self.pimax = self.obs["pimax"].tolist()
        self.cic_edges = self.obs["cic_edges"]
        self.rp_edges = self.obs["rp_edges"]
        self.kmax = self.obs.get("cic_kmax", np.array(None)).tolist()
        self.redshift = np.mean([self.obs["zmin"], self.obs["zmax"]])
        self.magthresh = self.obs["abs_mr_max"].tolist()

        self.model = self.make_model()
        self.cictab, self.wptab = self.load_tabulators()
        self.prior = self.make_prior()
        self.logpdf = self.make_logpdf()
        self.sampler = self.make_sampler()

        self.parameter_samples = None
        self.log_weights = None
        self.log_likelihoods = None
        self.blob = []

        n_slice = self.obs["slice_n"].tolist()
        wp_slice = self.obs["slice_wp"].tolist()
        cic_slice = self.obs["slice_cic"].tolist()
        self.len_cic = cic_slice.stop - cic_slice.start
        assert n_slice.start < wp_slice.start < cic_slice.start

    def run(self, verbose=None):
        """
        Runs the nested sampler

        Returns parameter_samples; also saves log_weights, and log_likelihoods
        """
        verbose = self.verbose if verbose is None else verbose
        self.sampler.run(verbose=verbose)
        self.parameter_samples, self.log_weights, \
            self.log_likelihoods = sampler.posterior()
        return self.parameter_samples

    def save(self, filename):
        self.model = None
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

        if cictab_file.is_file() and not self.temp_cictab:
            cictab = galtab.CICTabulator.load(cictab_file)
        else:
            cictab = self.make_cictab()
            if not self.temp_cictab:
                cictab.save(cictab_file)

        wptab = None
        if not self.temp_cictab:
            if wptab_file.is_file():
                wptab = tabcorr.TabCorr.read(wptab_file)
            else:
                wptab = self.make_wptab()
                if not self.temp_cictab:
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
            self.halocat.cosmology = self.obs["cosmo"].tolist()
        # model.populate_mock(halocat)

    def make_wptab(self):
        self.make_halocat()

        halotab = tabcorr.TabCorr.tabulate(
            self.halocat, htmo.wp, self.rp_edges, pi_max=self.pimax,
            prim_haloprop_key="halo_mvir", prim_haloprop_bins=100,
            sec_haloprop_key="halo_nfw_conc",
            sec_haloprop_percentile_bins=2, project_xyz=True)
        return halotab

    def make_cictab(self):
        self.make_halocat()

        kvals = None if self.kmax is None else np.arange(self.kmax) + 1
        gtab = galtab.GalaxyTabulator(self.halocat, self.model)
        return gtab.tabulate_cic(
            proj_search_radius=self.proj_search_radius,
            cylinder_half_length=self.cylinder_half_length,
            k_vals=kvals, bin_edges=self.cic_edges)

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
        n, wp = self.predict_wp(
            self.model, return_number_density=True)
        cic, n2, n3 = self.predict_cic(
            self.model, return_number_densities=True)

        if np.isnan(cic[0]):
            cic = np.zeros(self.len_cic)

        self.blob.append({"param_dict": param_dict,
                          "n": n, "n2": n2, "n3": n3,
                          "wp": wp, "cic": cic})
        return np.array([n, *wp, *cic])

    def predict_wp(self, model, return_number_density=False):
        n, wp = self.wptab.predict(model)
        if return_number_density:
            return n, wp
        else:
            return wp

    def predict_cic(self, model, return_number_densities=False):
        return self.cictab.predict(
            model, return_number_densities=return_number_densities)

    def predict_cic_halotools(self, model, num_threads=1):
        if "mock" in model.__dict__:
            model.mock.populate(self.halocat)
        else:
            model.populate_mock()

        gal = model.mock.galaxies
        xyz = htmo.return_xyz_formatted_array(
            gal["x"], gal["y"], gal["z"], period=self.halocat.Lbox,
            cosmology=self.halocat.cosmology, redshift=self.redshift,
            velocity=gal["vz"], velocity_distortion_dimension="z")
        counts = htmo.counts_in_cylinders(
            xyz, xyz, self.proj_search_radius, self.cylinder_half_length,
            period=self.halocat.Lbox, num_threads=num_threads)

        if self.kmax is None:
            hist = np.histogram(counts, bins=self.cic_edges)[0]
            return hist / len(xyz) / np.diff(self.cic_edges)
        else:
            return galtab.moments.moments_from_samples(
                counts, np.arange(self.kmax) + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="param_sampler")
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser.add_argument(
        "OBS_FILENAME", type=str,
        help="Name of the observation file (should end in .npz)"
    )
    parser.add_argument(
        "SAVE_DIR", type=str,
        help="Directory to save outputs and tabulations"
    )
    parser.add_argument(
        "--obs-dir", type=str, metavar="PATH", default=".",
        help="Directory the observation is saved"
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
        "-t", "--tabulate-only", action="store_true",
        help="Don't run the sampler"
    )
    parser.add_argument(
        "--temp-cictab", action="store_true",
        help="Create a temporary CICTabulator, and don't save it"
    )
    parser.add_argument(
        "--use-default-halotools-catalogs", action="store_true"
    )

    a = parser.parse_args()
    tabulate_only = a.__dict__.pop("tabulate_only")

    sampler = ParamSampler(**a.__dict__)

    if not tabulate_only:
        sampler.run()
        sampler.save(sampler.save_dir / "sampler.npy")
