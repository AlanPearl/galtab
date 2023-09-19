import jax
import jax.scipy
from jax import numpy as jnp

import halotools.empirical_models as htem


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
            raise htem.HalotoolsError(msg)

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
            raise htem.HalotoolsError(msg)

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
