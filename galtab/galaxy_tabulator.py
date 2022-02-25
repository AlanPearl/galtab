import types
from copy import copy
import numpy as np

import halotools.mock_observables as htmo


def placeholder_occupation(self, **kwargs):
    """This will overwrite the occupation models to populate placeholders"""
    mean_occ = self.mean_occupation(**kwargs)

    occupation = np.ceil(mean_occ / self.max_prob).astype(int)
    return np.where(mean_occ > self.min_prob, occupation, 0)


# noinspection PyProtectedMember
def make_placeholder_model(galtab):
    model = galtab.fiducial_model
    ph_model = copy(model)
    min_quantile_dict = galtab.min_quantile_dict
    max_weight_dict = galtab.max_weight_dict

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

            # Default min_quantile for halos to be assigned a placeholder
            default_min_quantile = 0.001
            min_quantile = min_quantile_dict.get(gal_type, default_min_quantile)
            occ_model.min_prob = get_min_prob(galtab, occ_model, min_quantile)

            # Default max_prob for satellite-like gal_types
            default_max_prob = 0.25
            if occ_model._upper_occupation_bound == 1:
                # Default max_prob for central-like gal_types
                default_max_prob = 1
            occ_model.max_prob = max_weight_dict.get(gal_type, default_max_prob)
            occ_model.mc_occupation = types.MethodType(placeholder_occupation,
                                                       occ_model)
            setattr(ph_model, method_name, occ_model.mc_occupation)

        kwargs = {}
        if galtab.num_ptcl_requirement is not None:
            kwargs["Num_ptcl_requirement"] = galtab.num_ptcl_requirement
        if galtab.seed is not None:
            kwargs["seed"] = galtab.seed
        ph_model.populate_mock(galtab.halocat, **kwargs)
        galaxies = ph_model.mock.galaxy_table
    finally:
        for name, copied_model in zip(occupation_model_names, copied_models):
            model._input_model_dictionary[name] = copied_model

            # Check that the models were returned to their original state
            assert copied_model.mc_occupation.__doc__ != (
                "This will overwrite the occupation models to "
                "populate placeholders")
            assert not hasattr(copied_model, "min_prob")
            assert not hasattr(copied_model, "max_prob")

    # Add velocity-distorted position columns 'obs_x/y/z'
    real_xyz = np.array([galaxies[x] for x in "xyz"]).T
    obs_xyz = htmo.return_xyz_formatted_array(
        *real_xyz.T, period=galtab.halocat.Lbox, cosmology=galtab.cosmo,
        redshift=galtab.halocat.redshift, velocity_distortion_dimension="z",
        velocity=galaxies["vz"])
    galaxies.add_columns(obs_xyz.T, names=[f"obs_{x}" for x in "xyz"])

    return galaxies, ph_model


# noinspection PyProtectedMember
def calc_weights(halos, galaxies, halo_inds, model):  # TODO: Speed this up with num_ptcl_requirement???
    weights = np.empty_like(galaxies["x"])

    # Access the occupation component models
    names = [x for x in model._input_model_dictionary
             if x.endswith("_occupation")]
    gal_types = [x.replace("_occupation", "") for x in names]
    methods = [getattr(model, "mean_occupation_" + x) for x in gal_types]

    for gal_type, method in zip(gal_types, methods):
        mask = galaxies["gal_type"] == gal_type
        mean_occ = method(table=halos)[halo_inds][mask]
        num_placeholders = galaxies["halo_num_" + gal_type][mask]
        weights[mask] = mean_occ / num_placeholders

    return weights


def get_min_prob(galtab, occ_model, min_quantile):
    haloprop = galtab.halo_table[occ_model.prim_haloprop_key]
    prop_bins = np.geomspace(haloprop.min(), haloprop.max(), 100)
    prop_hist = np.histogram(haloprop, prop_bins)[0]
    prop_cens = np.sqrt(prop_bins[:-1] * prop_bins[1:])

    meanocc = np.mean(
        [occ_model.mean_occupation(prim_haloprop=prop_cens, sec_haloprop_percentile=0),
         occ_model.mean_occupation(prim_haloprop=prop_cens, sec_haloprop_percentile=1)],
        axis=0)
    gal_hist = meanocc * prop_hist
    gal_cumsum = np.cumsum(gal_hist)
    gal_cdf = np.concatenate([[0], gal_cumsum / gal_cumsum[-1]])

    min_prop = np.interp(min_quantile, gal_cdf, prop_bins)
    min_prob = np.mean(
        [occ_model.mean_occupation(prim_haloprop=min_prop, sec_haloprop_percentile=0),
         occ_model.mean_occupation(prim_haloprop=min_prop, sec_haloprop_percentile=1)])
    return min_prob
