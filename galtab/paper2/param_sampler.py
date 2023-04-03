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
import emcee

import galtab
from . import param_config


default_simname = param_config.simname
default_sampler_name = "emcee"
default_nwalkers = 20


class BetterMultivariateNormal:
    def __init__(self, mean, cov, allow_singular=False):
        # Factor to multiply observables by
        # For float64, don't scale where std is less than ~ 10^(-308)
        inv_norm = np.sqrt(np.diag(cov))
        inv_norm[inv_norm < 2 ** np.finfo(inv_norm.dtype.type).minexp] = 1
        self.norm = 1 / inv_norm
        # Factor to *add* to log(PDF)
        self.logpdf_factor = np.sum(np.log(self.norm))

        # Construct a normalized multivariate normal distribution
        norm_mean = mean * self.norm
        corr_matrix = cov * self.norm[:, None] * self.norm[None, :]
        self.normalized_dist = scipy.stats.multivariate_normal(
            mean=norm_mean, cov=corr_matrix, allow_singular=allow_singular)

        if hasattr(self.normalized_dist, "cov_object"):
            # cov_info changes to cov_object in scipy 1.10
            self.cov_object = self.normalized_dist.cov_object
        else:
            self.cov_object = self.normalized_dist.cov_info

    def logpdf(self, x):
        x = self.norm * x
        return self.normalized_dist.logpdf(x) + self.logpdf_factor


class ParamSampler:
    def __init__(self, **kwargs):
        self.obs_dir = pathlib.Path(kwargs["obs_dir"])
        self.n = kwargs["N"]
        self.obs_filename = kwargs["OBS_FILENAME"]
        self.save_dir = pathlib.Path(kwargs["SAVE_DIR"])
        self.simname = kwargs.get("simname", default_simname).lower()
        self.reset_sampler = kwargs.get("reset_sampler", False)
        self.sampler_name = kwargs.get("sampler_name", default_sampler_name)
        if kwargs["use_default_halotools_catalogs"]:
            self.version_name = htsm.sim_defaults.default_version_name
        else:
            self.version_name = "my_cosmosim_halos"

        self.nwalkers = kwargs.get("nwalkers", default_nwalkers)
        self.verbose = kwargs["verbose"]
        self.temp_cictab = kwargs["temp_cictab"]
        self.n_mc = kwargs["n_mc"]
        self.min_quant = kwargs["min_quant"]
        self.max_weight = kwargs["max_weight"]
        self.seed = kwargs["seed"]
        self.sqiomw = kwargs["sqiomw"]
        self.no_assembias = kwargs.get("no_assembias", False)
        self.start_without_assembias = kwargs["start_without_assembias"]
        self.tabulate_at_starting_params = kwargs[
            "tabulate_at_starting_params"]

        self.halocat = kwargs.get("halocat")
        self.cictab = kwargs.get("cictab")
        self.use_numpy = kwargs.get("use_numpy", False)
        self.use_halotools = kwargs.get("use_halotools", False)
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

        self.starting_params = {}
        self.starting_bounds = None
        self.emcee_init_params = None
        self.make_halocat()
        self.model = self.make_model()
        self.param_names = list(self.starting_params.keys())
        self.ndim = len(self.param_names)

        self.cictab, self.wptab = self.load_tabulators()
        self.prior = self.make_prior()
        self.likelihood_dist = self.make_likelihood_dist()
        self.sampler = self.make_sampler()

        self.parameter_samples = None
        self.log_weights = None
        self.log_likelihoods = None
        self.blob = []
        saved_filename = kwargs.get("saved_filename", "sampler.npy")
        if pathlib.Path(saved_filename).is_file():
            self.blob = ParamSampler.load(saved_filename).blob

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
        if self.sampler_name == "emcee":
            self.sampler.run_mcmc(
                self.emcee_init_params, nsteps=self.n, progress=verbose)
        elif self.sampler_name == "nautilus":
            self.sampler.run(verbose=verbose)

            # TODO: Why doesn't this work?
            # self.parameter_samples, self.log_weights, \
            #     self.log_likelihoods = sampler.posterior()
            # return self.parameter_samples

    def save(self, filename):
        self.obs = self.cosmo = self.model = self.halocat = None
        self.cictab = self.sampler = None
        np.save(filename, np.array([self], dtype=object))

    @classmethod
    def load(cls, filename):
        obj = np.load(filename, allow_pickle=True)[0]
        return obj

    def make_sampler(self):
        if self.sampler_name == "emcee":
            backend = emcee.backends.HDFBackend(
                str(self.save_dir / "emcee_backend.h5"))
            if self.reset_sampler or not (backend.initialized
                                          and backend.iteration):
                backend.reset(self.nwalkers, self.ndim)
                rng = np.random.RandomState(self.seed)
                self.emcee_init_params = rng.uniform(
                    *self.starting_bounds.T,
                    (self.nwalkers, self.ndim))
            return emcee.EnsembleSampler(
                self.nwalkers, self.ndim,
                self.emcee_prob, backend=backend)
        elif self.sampler_name == "nautilus":
            return nautilus.Sampler(
                self.prior, self.likelihood, n_live=self.n)
        else:
            raise ValueError(f"Invalid sampler_name: {self.sampler_name}")

    def load_tabulators(self):
        if not self.temp_cictab:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        cictab_file = self.save_dir / "cictab.pickle"
        wptab_file = self.save_dir / "wptab.hdf5"

        if self.cictab is None:
            if cictab_file.is_file() and not self.temp_cictab:
                cictab = galtab.CICTabulator.load(cictab_file)
            else:
                cictab = self.make_cictab()
                if not self.temp_cictab:
                    cictab.save(cictab_file)
        else:
            cictab = self.cictab

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

        self.starting_params = param_config.kuan_params[magthresh].copy()
        gt_params = self.starting_params.copy()

        gt_params["mean_occupation_centrals_assembias_param1"] = 0
        gt_params["mean_occupation_satellites_assembias_param1"] = 0
        if not self.tabulate_at_starting_params:
            for name in ["logMmin", "logM1", "logM0"]:
                gt_params[name] -= param_config.kuan_err_low[magthresh][name]
            for name in ["sigma_logM"]:
                gt_params[name] += param_config.kuan_err_high[magthresh][name]

        err_low = param_config.kuan_err_low[magthresh].copy()
        err_high = param_config.kuan_err_high[magthresh].copy()
        if self.start_without_assembias or self.no_assembias:
            for galtype in ("centrals", "satellites"):
                name = f"mean_occupation_{galtype}_assembias_param1"
                self.starting_params[name] = 0
                err_low[name] = err_high[name] = 0.5

        model = htem.HodModelFactory(
            centrals_occupation=htem.AssembiasZheng07Cens(
                threshold=magthresh, redshift=redshift),
            satellites_occupation=htem.AssembiasZheng07Sats(
                threshold=magthresh, redshift=redshift),
            centrals_profile=htem.TrivialPhaseSpace(redshift=redshift),
            satellites_profile=htem.NFWPhaseSpace(redshift=redshift)
        )
        # Initialize model parameters - used by galtab as fiducial model
        # Assembias parameters should always be zero here
        model.param_dict.update(gt_params)

        if self.no_assembias:
            for galtype in ("centrals", "satellites"):
                name = f"mean_occupation_{galtype}_assembias_param1"
                del self.starting_params[name]
        self.starting_params = convert_params_model_to_sampler(
            self.starting_params)
        err_low["logM0_quant"] = err_low["logM0"]
        err_high["logM0_quant"] = err_high["logM0"]
        self.starting_bounds = np.array(
            [[self.starting_params[name] - 0.01 * err_low[name],
              self.starting_params[name] + 0.01 * err_high[name]]
             for name in self.starting_params.keys()]
        )
        return model

    def make_halocat(self):
        if self.halocat is None:
            self.halocat = htsm.CachedHaloCatalog(
                simname=self.simname, redshift=self.redshift,
                version_name=self.version_name)
            self.halocat.cosmology = self.cosmo
        # model.populate_mock(halocat)
        return self.halocat

    def make_wptab(self):
        halotab = tabcorr.TabCorr.tabulate(
            self.halocat, htmo.wp, self.rp_edges, pi_max=self.pimax,
            prim_haloprop_key="halo_mvir", prim_haloprop_bins=100,
            sec_haloprop_key="halo_nfw_conc",
            sec_haloprop_percentile_bins=2, project_xyz=True)
        return halotab

    def make_cictab(self):
        kvals = None if self.kmax is None else np.arange(self.kmax) + 1
        gtab = galtab.GalaxyTabulator(
            self.halocat, self.model, n_mc=self.n_mc, min_quant=self.min_quant,
            max_weight=self.max_weight, seed=self.seed,
            sat_quant_instead_of_max_weight=self.sqiomw)
        return gtab.tabulate_cic(
            proj_search_radius=self.proj_search_radius,
            cylinder_half_length=self.cylinder_half_length,
            k_vals=kvals, bin_edges=self.cic_edges, analytic_moments=True)

    def make_prior(self):
        prior = nautilus.Prior()
        prior.add_parameter("logMmin", dist=(9, 16))
        prior.add_parameter("sigma_logM", dist=(1e-5, 5))
        prior.add_parameter("logM1", dist=(10, 16))
        prior.add_parameter("logM0_quant", dist=(0, 1))
        prior.add_parameter("alpha", dist=(1e-5, 5))
        if not self.no_assembias:
            prior.add_parameter(
                "mean_occupation_centrals_assembias_param1", dist=(-1, 1))
            prior.add_parameter(
                "mean_occupation_satellites_assembias_param1", dist=(-1, 1))
        return prior

    def make_likelihood_dist(self):
        mean, cov = self.obs["mean"], self.obs["cov"]
        return BetterMultivariateNormal(
            mean=mean, cov=cov, allow_singular=True)

    def likelihood(self, param_dict):
        x = self.predict_observables(param_dict)
        loglike = self.likelihood_dist.logpdf(x)
        if np.isnan(loglike):
            loglike = -np.inf  # assume overflow error -> zero likelihood
        self.blob[-1]["loglike"] = loglike
        return loglike

    def emcee_prob(self, theta):
        param_dict = dict(zip(self.param_names, theta))
        prior = 0
        for i in range(self.ndim):
            prior += self.prior.dists[i].logpdf(param_dict[self.prior.keys[i]])
        if not np.isfinite(prior):
            return prior
        else:
            param_dict = convert_params_sampler_to_model(param_dict)
            return prior + self.likelihood(param_dict)

    def predict_observables(self, param_dict):
        # Update model parameters
        self.model.param_dict.update(param_dict)

        # Calculate wp
        n, wp = self.predict_wp(
            self.model, return_number_density=True)

        # Calculate CiC
        warn_status = {}
        if self.kmax is None or self.kmax:
            if self.use_halotools:
                cic, n2 = self.predict_cic_halotools(
                    self.model, return_number_density=True)
                n3 = n2
            else:
                cic, n2, n3, warn_status = self.predict_cic(
                    self.model, return_number_densities=True,
                    warn_p_over_1="return_warning")
        else:
            # No calculations necessary for kmax = 0
            cic, n2, n3 = [], None, None

        # Concatenate observables into array and blobs into dictionary
        self.blob.append({"param_dict": param_dict,
                          "n": n, "n2": n2, "n3": n3,
                          "wp": wp, "cic": cic, **warn_status})
        return np.array([n, *wp, *cic])

    def predict_wp(self, model, return_number_density=False):
        n, wp = self.wptab.predict(model)
        if return_number_density:
            return n, wp
        else:
            return wp

    def predict_cic(self, model, return_number_densities=False,
                    n_mc=None, warn_p_over_1=True):
        """
        :type model: object
        :type return_number_densities: bool
        :type n_mc: None | int
        :type warn_p_over_1: bool | str
        """
        return self.cictab.predict(
            model, return_number_densities=return_number_densities,
            n_mc=n_mc, use_numpy=self.use_numpy, warn_p_over_1=warn_p_over_1)

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


def convert_params_sampler_to_model(param_dict):
    logm0_range = param_dict["logM1"] - (logm0_min := param_dict["logMmin"])
    logm0 = logm0_min + param_dict["logM0_quant"] * logm0_range
    return {"logM0" if name == "logM0_quant" else name:
            logm0 if name == "logM0_quant" else param_dict[name]
            for name in param_dict}


def convert_params_model_to_sampler(param_dict):
    logm0_range = param_dict["logM1"] - (logm0_min := param_dict["logMmin"])
    logm0_quant = (param_dict["logM0"] - logm0_min) / logm0_range
    return {"logM0_quant" if name == "logM0" else name:
            logm0_quant if name == "logM0" else param_dict[name]
            for name in param_dict}

if __name__ == "__main__":
    import jax
    jax.config.update("jax_platform_name", "cpu")

    parser = argparse.ArgumentParser(prog="param_sampler")
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser.add_argument(
        "N", type=int,
        help="Number of MCMC (or 'live') points to sample"
    )
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
        "--simname", type=str, metavar="NAME", default=default_simname,
        help="Name of dark matter simulation"
    )
    parser.add_argument(
        "--sampler-name", type=str, default=default_sampler_name,
        help="Name of the MCMC/nested sampler to use"
    )
    parser.add_argument(
        "--reset-sampler", action="store_true",
        help="Restart sampler if it has already begun"
    )
    parser.add_argument(
        "--nwalkers", type=int, default=default_nwalkers,
        help="Number of emcee walkers"
    )
    parser.add_argument(
        "--n-mc", type=int, metavar="N", default=10,
        help="GalaxyTabulator parameter"
    )
    parser.add_argument(
        "--min-quant", type=float, metavar="X", default=1e-4,
        help="GalaxyTabulator parameter"
    )
    parser.add_argument(
        "--max-weight", type=float, metavar="X", default=0.05,
        help="GalaxyTabulator parameter"
    )
    parser.add_argument(
        "--seed", type=int, metavar="N", default=None,
        help="GalaxyTabulator parameter"
    )
    parser.add_argument(
        "--sqiomw", action="store_true",
        help="GalaxyTabulator parameter: sat_quant_instead_of_max_weight"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true"
    )
    parser.add_argument(
        "--no-assembias", action="store_true",
        help="Always set Acen=Asat=0"
    )
    parser.add_argument(
        "--start-without-assembias", action="store_true",
        help="Start sampling parameter-space at Acen=Asat=0"
    )
    parser.add_argument(
        "--tabulate-at-starting-params", action="store_true",
        help="Don't decrease logM1/0/min or increase sigma for tabulation"
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
        "--use-numpy", action="store_true",
        help="Use NumPy instead of JAX (slower, but it never seg faults)"
    )
    parser.add_argument(
        "--use-halotools", action="store_true",
        help="Use halotools to predict CiC (inefficient for MCMC sampling)"
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
