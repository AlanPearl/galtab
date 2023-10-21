import numpy as np
import matplotlib.pyplot as plt
import jax
from jax import numpy as jnp

from astropy import cosmology
import halotools.empirical_models as htem
import halotools.sim_manager as htsm
import halotools.mock_observables as htmo
from halotools.custom_exceptions import HalotoolsError

import galtab


class JaxZheng07Cens(htem.Zheng07Cens):
    def mean_occupation(self, **kwargs):
        # Retrieve the array storing the mass-like variable
        if 'table' in list(kwargs.keys()):
            mass = kwargs['table'][self.prim_haloprop_key]
        elif 'prim_haloprop' in list(kwargs.keys()):
            mass = jnp.atleast_1d(kwargs['prim_haloprop'])
        else:
            msg = ("\nYou must pass either a ``table`` or ``prim_haloprop`` argument \n"
                   "to the ``mean_occupation`` function of the ``Zheng07Cens`` class.\n")
            raise HalotoolsError(msg)

        return zheng07_cenocc(mass, self.param_dict["logMmin"],
                              self.param_dict["sigma_logM"])


class JaxZheng07Sats(htem.Zheng07Sats):
    def mean_occupation(self, **kwargs):
        if self.modulate_with_cenocc:
            for key, value in list(self.param_dict.items()):
                if key in self.central_occupation_model.param_dict:
                    self.central_occupation_model.param_dict[key] = value

        # Retrieve the array storing the mass-like variable
        if 'table' in list(kwargs.keys()):
            mass = kwargs['table'][self.prim_haloprop_key]
        elif 'prim_haloprop' in list(kwargs.keys()):
            mass = jnp.atleast_1d(kwargs['prim_haloprop'])
        else:
            msg = ("\nYou must pass either a ``table`` or ``prim_haloprop`` argument \n"
                   "to the ``mean_occupation`` function of the ``Zheng07Sats`` class.\n")
            raise HalotoolsError(msg)

        mean_nsat = zheng07_satocc(mass, self.param_dict["logM0"],
                                   self.param_dict["logM1"], self.param_dict["alpha"])

        # If a central occupation model was passed to the constructor,
        # multiply mean_nsat by an overall factor of mean_ncen
        if self.modulate_with_cenocc:
            # compatible with AB models
            mean_ncen = getattr(self.central_occupation_model, "baseline_mean_occupation",
                                self.central_occupation_model.mean_occupation)(**kwargs)
            mean_nsat *= mean_ncen

        return mean_nsat


def vectorized_cond(pred, true_fun, false_fun, operand, safe_operand_value=0):
    # Taken from https://github.com/google/jax/issues/1052
    # ====================================================
    # true_fun and false_fun must act elementwise (i.e. be vectorized)
    true_op = jnp.where(pred, operand, safe_operand_value)
    false_op = jnp.where(pred, safe_operand_value, operand)
    return jnp.where(pred, true_fun(true_op), false_fun(false_op))


@jax.jit
def zheng07_cenocc(mass, logmmin, sigma_logm):
    logm = jnp.log10(mass)
    return 0.5 * (1.0 + jax.scipy.special.erf((logm - logmmin) / sigma_logm))


@jax.jit
def zheng07_satocc(mass, logm0, logm1, alpha):
    m0 = 10. ** logm0
    m1 = 10. ** logm1
    is_nonzero = mass > m0

    def nonzero_func(x):
        return ((x - m0) / m1) ** alpha

    def zero_func(x):
        return 0

    mean_nsat = vectorized_cond(is_nonzero, nonzero_func, zero_func, mass,
                                safe_operand_value=m0 + m1)
    return mean_nsat

print("1. Set our CiC parameters (all lengths are in Mpc/h)")
proj_search_radius = 2.0
cylinder_half_length = 10.0
cic_edges = np.arange(-0.5, 16)

print("2. Set our cosmology and HOD model")
cosmo = cosmology.Planck13
hod = htem.PrebuiltHodModelFactory("zheng07", threshold=-21)

print("3. Load Bolshoi-Planck simulation halos at z=0")
halocat = htsm.CachedHaloCatalog(simname="bolplanck", redshift=0)

print("4. Give the Tabulator the halo catalog and a fiducial HOD model")
gtab = galtab.GalaxyTabulator(halocat, hod)

print("5. Prepare the CICTabulator to make predictions")
cictab = galtab.CICTabulator(gtab, proj_search_radius, cylinder_half_length,
                             k_vals=[1, 2])  # [1, 2] -> [mean, std dev]


print("6. Construct a differentiable HOD model with our JAX-compatible mean occupation functions")
hod_jax = htem.HodModelFactory(
    centrals_occupation=JaxZheng07Cens(threshold=-21),
    satellites_occupation=JaxZheng07Sats(threshold=-21),
    centrals_profile=htem.TrivialPhaseSpace(),
    satellites_profile=htem.NFWPhaseSpace()
)

def calc_cic1_halotools(logMmin=12.79):
    hod_jax.param_dict.update({"logMmin": logMmin})

    # Populated model galaxies and get their Cartesian coordinates
    hod_jax.populate_mock(halocat)
    galaxies = hod_jax.mock.galaxy_table
    xyz = htmo.return_xyz_formatted_array(
        galaxies["x"], galaxies["y"], galaxies["z"], velocity=galaxies["vz"],
        velocity_distortion_dimension="z", period=halocat.Lbox, cosmology=cosmo
    )

    # Compute CiC (self-counting subtracted by the `-1`)
    cic_counts = htmo.counts_in_cylinders(
        xyz, xyz, proj_search_radius, cylinder_half_length) - 1
    cic = np.histogram(cic_counts, bins=cic_edges, density=True)[0]
    return galtab.moments.moments_from_binned_pmf(
        cic_edges, cic, [1, 2])[1]

def calc_cic1(logMmin=12.79):
    hod_jax.param_dict.update({"logMmin": logMmin})
    return cictab.predict(hod_jax, warn_p_over_1=False)[1]


diff_cic1 = jax.grad(calc_cic1)

# Note that we shouldn't make logMmin too much lower than that of our fiducial
# model. If desired, make more conservative choices for the fiducial parameters.
# i.e., low logMmin / logM1 / logM0 values and large sigma_logM values
for logmmin in np.linspace(11.0, 15.0, 20):
    value_ht = calc_cic1_halotools(logmmin)
    value = calc_cic1(logmmin)
    derivative = diff_cic1(logmmin)

    plt.plot(logmmin, value, "bo")
    plt.quiver(logmmin, value, 1, derivative, angles="xy")
    plt.plot(logmmin, value_ht, "gx")

plt.xlabel("$\\log M_{\\rm min}$")
plt.ylabel("$\\langle N_{\\rm CiC} \\rangle$")

plt.plot([], [], "gx", label="halotools")
plt.plot([], [], "bo", label="galtab")
plt.legend(frameon=False)
plt.show()
