import numpy as np
import scipy.stats
import uncertainties
import argparse

import emcee
import halotools.empirical_models as htem
import halotools.sim_manager as htsm
# import halotools.mock_observables as htmo
# import tabcorr

import mocksurvey as ms
import galtab

# Make sure Corrfunc is installed, or else this will run VERY slow
assert ms.cf.corrfunc_works

parser = argparse.ArgumentParser(prog="kuan-mcmc")
parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
parser.add_argument(
    "-c", "--continue-mcmc", action="store_true",
    help="Continue MCMC chain from where it left off")
parser.add_argument(
    "-s", "--nsteps", type=int, default=600,
    help="Number of MCMC chains to run")
parser.add_argument(
    "-w", "--nwalkers", type=int, default=50,
    help="Number of MCMC walkers"
)
parser.add_argument(
    "-k", "--start-from-kuan-best-fit", action="store_true",
    help="Start from Kuan's best-fit parameters (ignored if continue-mcmc=True)"
)
parser.add_argument(
    "-d", "--full-cic-distribution", action="store_true",
    help="Use the distribution itself, rather than moments of CiC"
)
parser.add_argument(
    "-b", "--use-old-moment-binning", action="store_true",
    help="Use the old (and even more naive) moment-binning method"
)
a = parser.parse_args()
start_new_mcmc = not a.continue_mcmc
nsteps = a.nsteps
nwalkers = a.nwalkers
full_cic_distribution = a.full_cic_distribution
start_from_kuan_best_fit = a.start_from_kuan_best_fit
use_old_moment_binning = a.use_old_moment_binning
kuan_best_fits = [12.265, 0.218, 0.950, 13.454, 12.617, 0.811, -0.147]

# Load Kuan's data
# ================
obs_val = np.load("data_stat.npz")["stat20p5"]
obs_cov = np.load("total_cov.npz")["cov20p5"]

obs_std = np.sqrt(np.diag(obs_cov))
obs_correl = obs_cov / obs_std[:, None] / obs_std[None, :]

# Define important parameters for wp(rp) and CiC
# ==============================================
n_mask = slice(0, 1)
wp_mask = slice(1, 13)
cic_mask = slice(13, None)
not_cic_mask = slice(None, 13)

pimax = 60.0
proj_search_radius = 2.0
cylinder_half_length = 10.0
# This corresponds to 1000 km/s at z=0 (Kuan used 10 Mpc/h)
# =========================================================

rp_edges = np.logspace(-0.8, 1.6, 13)
rp_cens = np.sqrt(rp_edges[:-1] * rp_edges[1:])
cic_edges = np.concatenate([np.arange(-0.5, 9),
                            np.round(np.geomspace(10, 150, 20)) - 0.5])
cic_cens = 0.5 * (cic_edges[:-1] + cic_edges[1:])
cic_bin_inds = np.repeat(np.arange(len(cic_edges)-1),
                         np.diff(cic_edges).astype(int))
cic_kmax = 5
cic_num_obs = np.sum(obs_val[cic_mask] != 0) if full_cic_distribution else cic_kmax

kuan_uncerts = np.array(uncertainties.correlated_values(obs_val, obs_cov))
# Make sure P(Ncic) is normalized exactly
kuan_uncerts[cic_mask] /= np.sum(kuan_uncerts[cic_mask])
cic_pdf = kuan_uncerts[cic_mask]

# Standardized by dividing by sigma^k for k>3
# (This seems to be the more common way to standardize moments, so I guess I'll do this)
if use_old_moment_binning:
    cic_moments = galtab.moments.moments_from_samples(
        cic_cens, np.arange(1, cic_kmax + 1), cic_pdf)
else:
    # Convert P(N) to dP(N)/dN and then calculate moments
    cic_pdf /= np.diff(cic_edges)
    cic_moments = galtab.moments.moments_from_binned_pmf(
        cic_edges, cic_pdf, np.arange(1, cic_kmax + 1))

if full_cic_distribution:
    my_uncerts = kuan_uncerts
else:
    my_uncerts = [*kuan_uncerts[not_cic_mask],
                  *cic_moments]
my_vals = np.array([x.nominal_value for x in my_uncerts])
my_covar = np.array(uncertainties.covariance_matrix(my_uncerts))

# Construct the assembly bias-added Zheng 2007 HOD model
redshift = 0
magthresh = -20.5
model = htem.HodModelFactory(
    centrals_occupation=htem.AssembiasZheng07Cens(threshold=magthresh, redshift=redshift),
    satellites_occupation=htem.AssembiasZheng07Sats(threshold=magthresh, redshift=redshift),
    centrals_profile=htem.TrivialPhaseSpace(redshift=redshift),
    satellites_profile=htem.NFWPhaseSpace(redshift=redshift)
)
fiducial_params = model.param_dict.copy()

halocat = htsm.CachedHaloCatalog(simname="bolplanck", redshift=redshift)
model.populate_mock(halocat)


# Sadly, we can't make use of TabCorr because it doesn't work
# for assembly bias models :(((((((((((
# ===========================================================
# halotab_file = "halotab-kuan-mcmc.hdf5"
# try:
#     halotab = tabcorr.TabCorr.read(halotab_file)
# except OSError:
#     halotab = tabcorr.TabCorr.tabulate(
#         halocat, htmo.wp, rp_edges, pi_max=pimax)
#     halotab.write(halotab_file)
#
# Instead, we have to calculate wp(rp) on runtime for each of the 10
# galtab realizations (using Corrfunc, this takes ~1.7 sec to run,
# which is fine considering the galtab CiC prediction time is ~2.7 sec)
# =====================================================================
def predict_wp(galaxy_tabulator, weights=None, new_model=None):
    if weights is None:
        assert model is not None
        weights = galaxy_tabulator.calc_weights(new_model)

    galaxies = galaxy_tabulator.galaxies
    rands: np.ndarray = galaxy_tabulator.predictor.mc_rands
    obs_xyz = np.array([galaxies[f"obs_{x}"] for x in "xyz"]).T
    masks = (rands < weights[:, None]).T
    wp_reals = []
    for mask in masks:
        wp = ms.cf.wp_rp(obs_xyz[mask], None, rp_edges,
                         pimax, halocat.Lbox[0])
        wp_reals.append(wp)
    return np.mean(wp_reals, axis=0)


# Set up GalTab to calculate CiC deterministically
k_vals = None if full_cic_distribution else np.arange(cic_kmax) + 1
gtab = galtab.GalaxyTabulator(halocat, model)
gtab.tabulate_cic(k_vals=k_vals,
                  proj_search_radius=proj_search_radius,
                  cylinder_half_length=cylinder_half_length)


# Model predictions of n, wp, and cic
def observables_from_hod_params(hod_params):
    param_dict = dict(zip(param_names, hod_params))
    model.param_dict.update(param_dict)
    cic, n1, _ = gtab.predict(model, return_number_densities=True)
    if full_cic_distribution:
        binned_cic = np.zeros(num_cic_observables)
        np.add.at(binned_cic, cic_bin_inds[:len(cic)], cic)
        cic = binned_cic
    # n, wp = halotab.predict(model)
    wp = predict_wp(gtab, gtab.weights)
    return n1, wp, cic


# When calculating likelihood from covariance matrix,
# multiply number density by 10^3, so it is a similar order of magnitude
# to the other observables. This helps with machine precision.
# ======================================================================
observable_nonzero_mask = my_vals != 0
num_cic_observables = np.sum(observable_nonzero_mask[cic_mask])
assert np.all(observable_nonzero_mask[not_cic_mask]), \
    "Found a zero value for n or wp"
my_vals_copy = my_vals.copy()[observable_nonzero_mask]
my_covar_copy = my_covar.copy()[observable_nonzero_mask, :][:, observable_nonzero_mask]

my_vals_copy[0] *= 1e3
my_covar_copy[:, 0] *= 1e3
my_covar_copy[0, :] *= 1e3
# We must allow singular covariance matrices if using the full CiC distribution
# because there is a degeneracy by definition, where all bins must sum to 1.
# allow_singular=True uses the Moore-Penrose pseudo-inverse, which effectively
# ignores deviations along the direction of this degeneracy.
loglike = scipy.stats.multivariate_normal(mean=my_vals_copy, cov=my_covar_copy,
                                          allow_singular=full_cic_distribution).logpdf


def loglike_from_observables(observables):
    observables = observables.copy()
    # observables *= observable_scaling
    observables[0] *= 1e3
    return loglike(observables)


def logprior(p):
    # param_names = ["logMmin", "sigma_logM", "alpha", "logM1", "logM0",
    #                "mean_occupation_centrals_assembias_param1",
    #                "mean_occupation_satellites_assembias_param1"]

    # Central parameters
    if (9 > p[0]) or (p[0] > 16):
        return -np.inf
    if (1e-5 > p[1]) or (p[1] > 5):
        return -np.inf

    # Satellite parameters
    if (1e-5 > p[2]) or (p[2] > 5):
        return -np.inf
    if (9 > p[3]) or (p[3] > 16):
        return -np.inf
    if (9 > p[4]) or (p[4] > 16):
        return -np.inf

    # Assembly bias parameters
    if np.any(p[-2:] > 1):
        return -np.inf
    if np.any(p[-2:] < -1):
        return -np.inf
    return 0.0


def logprob(hod_params):
    # Default NaN values for n_galtab, wp, and cic
    nan_blobs = [np.nan, [np.nan] * len(rp_cens), [np.nan] * cic_num_obs]

    logprior_here = logprior(hod_params)
    if not np.isfinite(logprior_here):
        return logprior_here, *nan_blobs

    n_gt, wp, cic = observables_from_hod_params(hod_params)
    observables = np.array([n_gt, *wp, *cic])
    loglike_here = loglike_from_observables(observables)
    return loglike_here + logprior_here, n_gt, wp, cic


blobs_dtype = [("n", float),
               ("wp", float, len(rp_cens)),
               ("cic", float, cic_num_obs)]

param_names = ["logMmin", "sigma_logM", "alpha", "logM1", "logM0",
               "mean_occupation_centrals_assembias_param1",
               "mean_occupation_satellites_assembias_param1"]
if start_from_kuan_best_fit:
    initial_state = kuan_best_fits
else:
    initial_state = [fiducial_params[x] for x in param_names]
ndim = len(initial_state)
backend = emcee.backends.HDFBackend("kuan-mcmc-backend.h5")
if start_new_mcmc:
    backend.reset(nwalkers, ndim)
sampler = emcee.EnsembleSampler(nwalkers=nwalkers, ndim=ndim, log_prob_fn=logprob,
                                backend=backend, blobs_dtype=blobs_dtype)

if start_new_mcmc:
    initial_walker_states = scipy.stats.norm.rvs(
        loc=[initial_state], scale=[[0.001]] * nwalkers)
else:
    initial_walker_states = None

sampler.run_mcmc(initial_walker_states, nsteps=nsteps, progress=True)
