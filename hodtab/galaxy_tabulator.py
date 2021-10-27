import types
from copy import copy
import numpy as np


def placeholder_occupation(self, **kwargs):
    """This will overwrite the occupation models to populate placeholders"""
    mean_occ = self.mean_occupation(**kwargs)

    ans = np.ceil(mean_occ / self.max_prob).astype(int)
    return np.where(mean_occ > self.min_prob, ans, 0)


# noinspection PyProtectedMember
def make_placeholder_model(galtab):
    model = galtab.fiducial_model
    ph_model = copy(model)
    min_weight_dict = galtab.min_weight_dict
    max_weight_dict = galtab.max_weight_dict

    # Access the occupation component models
    occupation_model_names = [x for x in model._input_model_dictionary
                              if x.endswith("_occupation")]
    occupation_models = [model._input_model_dictionary[x]
                         for x in occupation_model_names]
    copied_models = [copy(x) for x in occupation_models]

    # Hack the occupation models to make them do what we want
    # before returning them to their original state :)
    try:
        for name, occ_model in zip(occupation_model_names, occupation_models):
            gal_type = occ_model.gal_type
            assert gal_type == name.replace("_occupation", "")
            method_name = "mc_occupation_" + gal_type
            assert method_name in model._mock_generation_calling_sequence

            default_min_prob = 0.01
            # Default max_prob for satellite-like gal_types
            default_max_prob = 0.25
            if occ_model._upper_occupation_bound == 1:
                # Default max_prob for central-like gal_types
                default_max_prob = 1
            occ_model.min_prob = min_weight_dict.get(gal_type, default_min_prob)
            occ_model.max_prob = max_weight_dict.get(gal_type, default_max_prob)
            occ_model.mc_occupation = types.MethodType(placeholder_occupation,
                                                       occ_model)
            setattr(ph_model, method_name, occ_model.mc_occupation)

        kwargs = {"Num_ptcl_requirement": galtab.num_ptcl_requirement}
        if galtab.num_ptcl_requirement is None:
            kwargs = {}
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

    galaxies.add_column(np.empty(len(galaxies), dtype=float), name="weights")
    return galaxies, ph_model


# noinspection PyProtectedMember
def calc_weights(galaxies, model, inplace=False):
    if inplace:
        weights = galaxies["weights"]
    else:
        weights = np.empty_like(galaxies["weights"])

    # Access the occupation component models
    names = [x for x in model._input_model_dictionary
             if x.endswith("_occupation")]
    gal_types = [x.replace("_occupation", "") for x in names]
    methods = [getattr(model, "mean_occupation_" + x) for x in gal_types]

    for gal_type, method in zip(gal_types, methods):
        mask = galaxies["gal_type"] == gal_type
        mean_occ = method(table=galaxies[mask])
        num_placeholders = galaxies["halo_num_" + gal_type][mask]
        weights[mask] = mean_occ / num_placeholders

    return weights
