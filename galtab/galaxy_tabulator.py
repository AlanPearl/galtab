import types
from copy import copy

import numpy as np
import scipy.stats

import halotools.mock_observables as htmo


def placeholder_occupation(self, **kwargs):
    """Overwrite the fiducial model's `mc_occupation` methods for populating placeholders"""
    mean_occ = self.mean_occupation(**kwargs)

    if self._upper_occupation_bound > 1:
        if self.sat_quant_instead_of_max_weight:
            assert self.max_weight >= 6e-17, \
                "sat_quant too low for scipy.stats.poisson.isf"
            occupation = scipy.stats.poisson.isf(self.max_weight, mu=mean_occ).astype(int)
        else:
            occupation = np.ceil(mean_occ / self.max_weight).astype(int)
    else:
        occupation = np.where((mean_occ >= self.min_prob), 1, 0)

    return occupation


# noinspection PyProtectedMember
def make_placeholder_model(galtab):

    model = galtab.fiducial_model
    ph_model = copy(model)
    min_quant = galtab.min_quant
    max_weight = galtab.max_weight

    # Access the occupation component models
    occupation_model_names = [x for x in ph_model._input_model_dictionary
                              if x.endswith("_occupation")]
    occupation_models = [ph_model._input_model_dictionary[x]
                         for x in occupation_model_names]
    copied_models = [copy(x) for x in occupation_models]

    # Hack the occupation models to make them do what we want
    # before returning them to their original state
    try:
        for name, occ_model in zip(occupation_model_names, occupation_models):
            gal_type = occ_model.gal_type
            assert gal_type == name.replace("_occupation", "")
            method_name = "mc_occupation_" + gal_type
            assert method_name in model._mock_generation_calling_sequence

            occ_model.min_prob = get_min_prob(galtab, occ_model, min_quant)
            occ_model.max_weight = max_weight
            occ_model.sat_quant_instead_of_max_weight = \
                galtab.sat_quant_instead_of_max_weight
            occ_model.mc_occupation = types.MethodType(
                placeholder_occupation, occ_model)
            setattr(ph_model, method_name, occ_model.mc_occupation)

        kwargs = {}
        if galtab.num_ptcl_requirement is not None:
            kwargs["Num_ptcl_requirement"] = galtab.num_ptcl_requirement
        if galtab.seed is not None:
            kwargs["seed"] = galtab.seed
        ph_model.populate_mock(galtab.halocat, **kwargs)

        # Get rid of useless halos (no placeholders populated)
        # ====================================================
        trimmed_halocat = copy(galtab.halocat)
        trimmed_halotable = ph_model.mock.halo_table
        useless_conditions = []
        for name in trimmed_halotable.keys():
            if not name.startswith("halo_num_"):
                continue
            useless_conditions.append(trimmed_halotable[name] < 1)
        useless_halos = np.all(useless_conditions, axis=0)
        trimmed_halotable = trimmed_halotable[~useless_halos]

        # Repopulate the mock, this time without the useless halos
        # ========================================================
        trimmed_halocat._halo_table = trimmed_halotable
        ph_model.populate_mock(trimmed_halocat, **kwargs)
        galaxies = ph_model.mock.galaxy_table

    finally:
        # Placeholders have been populated, so now we can
        # return the model to its original state
        for name, copied_model in zip(occupation_model_names, copied_models):
            model._input_model_dictionary[name] = copied_model

    # Add velocity-distorted position columns 'obs_x/y/z'
    real_xyz = np.array([galaxies[x] for x in "xyz"]).T
    obs_xyz = htmo.return_xyz_formatted_array(
        *real_xyz.T, period=galtab.halocat.Lbox, cosmology=galtab.cosmo,
        redshift=galtab.halocat.redshift, velocity_distortion_dimension="z",
        velocity=galaxies["vz"])
    galaxies.add_columns(obs_xyz.T, names=[f"obs_{x}" for x in "xyz"])

    return galaxies, ph_model, trimmed_halocat


# noinspection PyProtectedMember
def calc_weights(halos, galaxies, halo_inds, model):
    # TODO: Speed this up with num_ptcl_requirement???
    weights = np.full(len(galaxies), np.nan, dtype=np.float32)

    # Access the occupation component models
    names = [x for x in model._input_model_dictionary
             if x.endswith("_occupation")]
    gal_types = [x[:-11] for x in names]
    methods = [getattr(model, "mean_occupation_" + x) for x in gal_types]

    for gal_type, method in zip(gal_types, methods):
        mask = galaxies["gal_type"] == gal_type
        mean_occ = method(table=halos)[halo_inds][mask]
        num_placeholders = galaxies["halo_num_" + gal_type][mask]
        weights[mask] = mean_occ / num_placeholders

    return weights


# Old version made a histogram of centrals as function of prim_haloprop
# The current version makes the histogram as function of mean_occupation
# ======================================================================
# def get_min_prob(galtab, occ_model, min_quantile):
#     haloprop = galtab.halo_table[occ_model.prim_haloprop_key]
#     prop_bins = np.geomspace(haloprop.min(), haloprop.max(), 100)
#     prop_hist = np.histogram(haloprop, prop_bins)[0]
#     prop_cens = np.sqrt(prop_bins[:-1] * prop_bins[1:])
#
#     meanocc = np.mean(
#         [occ_model.mean_occupation(prim_haloprop=prop_cens, sec_haloprop_percentile=0),
#          occ_model.mean_occupation(prim_haloprop=prop_cens, sec_haloprop_percentile=1)],
#         axis=0)
#     gal_hist = meanocc * prop_hist
#     gal_cumsum = np.cumsum(gal_hist)
#     gal_cdf = np.concatenate([[0.0], gal_cumsum / gal_cumsum[-1]])
#
#     min_prop = np.interp(min_quantile, gal_cdf, prop_bins)
#     min_prob = np.mean(
#         [occ_model.mean_occupation(prim_haloprop=min_prop, sec_haloprop_percentile=0),
#          occ_model.mean_occupation(prim_haloprop=min_prop, sec_haloprop_percentile=1)])
#     return min_prob


def get_min_prob(galtab, occ_model, min_quantile, numbins=1000):
    meanocc = occ_model.mean_occupation(table=galtab.halo_table)
    occ_bins = np.geomspace(meanocc[meanocc > 0].min(), 1.0, numbins)
    occ_bins = np.concatenate([[0.0], occ_bins])
    occ_hist = np.histogram(meanocc, occ_bins)[0]
    occ_cens = (occ_bins[:-1] + occ_bins[1:]) / 2

    gal_cumsum = np.cumsum(occ_cens * occ_hist)
    gal_cdf = np.concatenate([[0.0], gal_cumsum / gal_cumsum[-1]])

    min_meanocc = np.interp(min_quantile, gal_cdf, occ_bins)
    return min_meanocc  # mean occupation of central = prob of central
