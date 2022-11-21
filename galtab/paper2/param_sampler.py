import numpy as np
import scipy.stats
import argparse
import pathlib

from astropy.cosmology import Cosmology
import halotools.empirical_models as htem
import halotools.sim_manager as htsm
import halotools.sim_manager.sim_defaults
import halotools.mock_observables as htmo
import tabcorr
import nautilus

import galtab
from . import param_config


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
        self.temp_cictab = kwargs["temp_cictab"]
        self.n_mc = kwargs["n_mc"]
        self.min_quant = kwargs["min_quant"]
        self.max_weight = kwargs["max_weight"]
        self.seed = kwargs["seed"]
        self.sqiomw = kwargs["sqiomw"]
        self.start_without_assembias = kwargs["start_without_assembias"]

        self.halocat = kwargs.get("halocat")
        self.use_numpy = kwargs.get("use_numpy", False)
        self.obs = self.load_obs()
        self.cosmo = self.obs["cosmo"].tolist()
        if not isinstance(self.cosmo, Cosmology):
            self.cosmo = Cosmology.from_format(self.cosmo, format="mapping")
        self.proj_search_radius = self.obs["proj_search_radius"].tolist()
        self.cylinder_half_length = self.obs["cylinder_half_length"].tolist()
        self.pimax = self.obs["pimax"].tolist()
        self.cic_edges = self.obs["cic_edges"]
        self.rp_edges = self.obs["rp_edges"]
        self.kmax = self.obs.get("cic_kmax", np.array(None)).tolist()
        self.redshift = np.mean([self.obs["zmin"], self.obs["zmax"]])
        self.magthresh = self.obs["abs_mr_max"].tolist()

        self.fiducial_params = None
        self.gt_params = None
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

        # TODO: Why doesn't this work?
        # self.parameter_samples, self.log_weights, \
        #     self.log_likelihoods = sampler.posterior()
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
        if wptab_file.is_file():
            wptab = tabcorr.TabCorr.read(str(wptab_file))
        elif not self.temp_cictab:
            wptab = self.make_wptab()
            wptab.write(str(wptab_file))
        return cictab, wptab

    def load_obs(self):
        path = self.obs_dir / self.obs_filename
        return np.load(path, allow_pickle=True)

    def make_model(self):
        # Construct the assembly bias-added Zheng 2007 HOD model
        redshift = self.redshift
        magthresh = self.magthresh

        self.fiducial_params = param_config.kuan_params[self.magthresh]
        self.gt_params = self.fiducial_params.copy()

        self.gt_params["mean_occupation_centrals_assembias_param1"] = 0
        self.gt_params["mean_occupation_satellites_assembias_param1"] = 0
        for name in ["logMmin", "logM1", "logM0"]:
            self.gt_params[name] -= param_config.kuan_err_low[magthresh][name]

        if self.start_without_assembias:
            self.fiducial_params[
                "mean_occupation_centrals_assembias_param1"] = 0
            self.fiducial_params[
                "mean_occupation_satellites_assembias_param1"] = 0

        model = htem.HodModelFactory(
            centrals_occupation=htem.AssembiasZheng07Cens(
                threshold=magthresh, redshift=redshift),
            satellites_occupation=htem.AssembiasZheng07Sats(
                threshold=magthresh, redshift=redshift),
            centrals_profile=htem.TrivialPhaseSpace(redshift=redshift),
            satellites_profile=htem.NFWPhaseSpace(redshift=redshift)
        )
        model.param_dict.update(self.gt_params)
        return model

    def make_halocat(self):
        if self.halocat is None:
            self.halocat = htsm.CachedHaloCatalog(
                simname=self.simname, redshift=self.redshift,
                version_name=self.version_name)
            self.halocat.cosmology = self.cosmo
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
        gtab = galtab.GalaxyTabulator(
            self.halocat, self.model, n_mc=self.n_mc, min_quant=self.min_quant,
            max_weight=self.max_weight, seed=self.seed,
            sat_quant_instead_of_max_weight=self.sqiomw)
        return gtab.tabulate_cic(
            proj_search_radius=self.proj_search_radius,
            cylinder_half_length=self.cylinder_half_length,
            k_vals=kvals, bin_edges=self.cic_edges, analytic_moments=True)

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

    def predict_cic(self, model, return_number_densities=False, n_mc=None):
        return self.cictab.predict(
            model, return_number_densities=return_number_densities,
            n_mc=n_mc, use_numpy=self.use_numpy)

    def predict_wp_halotools(self, model, return_number_density=False,
                             num_threads=1):
        xyz = self.populate_halotools(model)
        wp = htmo.wp(
            xyz, self.rp_edges, self.pimax,
            period=self.halocat.Lbox, num_threads=num_threads)

        if return_number_density:
            return wp, len(xyz) / np.product(self.halocat.Lbox)
        else:
            return wp

    def predict_cic_halotools(self, model, return_number_density=False,
                              num_threads=1, halocat=None):
        xyz = self.populate_halotools(model, halocat=halocat)
        counts = htmo.counts_in_cylinders(
            xyz, xyz, self.proj_search_radius, self.cylinder_half_length,
            period=self.halocat.Lbox, num_threads=num_threads) - 1

        if self.kmax is None:
            hist = np.histogram(counts, bins=self.cic_edges)[0]
            cic = hist / len(xyz) / np.diff(self.cic_edges)
        else:
            cic = galtab.moments.moments_from_samples(
                counts, np.arange(self.kmax) + 1)
            assert np.isclose(cic[0], np.mean(counts)), \
                f"{cic[0]} != {np.mean(counts)}"

        if return_number_density:
            return cic, len(xyz) / np.product(self.halocat.Lbox)
        else:
            return cic

    def populate_halotools(self, model, halocat=None):
        if halocat is None and "mock" in model.__dict__:
            model.mock.populate()
        elif halocat is None:
            model.populate_mock(self.cictab.galtabulator.halocat)
        else:
            model.populate_mock(halocat)

        gal = model.mock.galaxy_table
        xyz = htmo.return_xyz_formatted_array(
            gal["x"], gal["y"], gal["z"], period=self.halocat.Lbox,
            cosmology=self.cosmo, redshift=self.redshift,
            velocity=gal["vz"], velocity_distortion_dimension="z")
        return xyz.astype(np.float64)


if __name__ == "__main__":
    import jax
    jax.config.update("jax_platform_name", "cpu")

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
        "--simname", type=str, metavar="NAME", default=param_config.simname,
        help="Name of dark matter simulation"
    )
    parser.add_argument(
        "-n", "--n-live", type=int, metavar="N", default=1000,
        help="Number of 'live points' to sample"
    )
    parser.add_argument(
        "--n-mc", type=int, metavar="N", default=10,
        help="GalaxyTabulator parameter"
    )
    parser.add_argument(
        "--min-quant", type=float, metavar="X", default=0.001,
        help="GalaxyTabulator parameter"
    )
    parser.add_argument(
        "--max-weight", type=float, metavar="X", default=0.9999,
        help="GalaxyTabulator parameter"
    )
    parser.add_argument(
        "--seed", type=int, metavar="N", default=None,
        help="GalaxyTabulator parameter"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true"
    )
    parser.add_argument(
        "--start-without-assembias", action="store_true",
        help="Start sampling parameter-space at Acen=Asat=0"
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
        "--sqiomw", action="store_true",
        help="sat_quant instead of max_weight (CICTabulator parameter)"
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
