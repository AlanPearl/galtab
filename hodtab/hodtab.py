import numpy as np
# import jax.numpy as jnp

# import halotools.empirical_models as htem
# import halotools.mock_observables as htmo
# import halotools.simulation_manager as htsm

from . import galaxy_tabulator as gt
from . import counts_in_cylinders as cic


class GalaxyTabulator:
    """
    This should be the only class a casual user needs to use.

    It is responsible for populating place holder galaxies which will be given a
    `probability_weight` variable according to the probability that
    it would exist in its given halo
    """

    def __init__(self, halocat, fiducial_model,
                 min_weight_dict=None,
                 max_weight_dict=None,
                 num_ptcl_requirement=None):
        """
        Parameters
        ----------
        halocat : htsm.CachedHaloCatalog
            Catalog containing the halos to populate galaxies on top of
        fiducial_model : htem.ModelFactory
            The model used to calculate the number of placeholder galaxies
        min_weight_dict : Optional[dict]
            pass
        max_weight_dict : Optional[dict]
            pass
        min_central_prob : float (default = 0.01)
            Central galaxies lower than this probability (calculated according
            to the `fiducial_model`) will have no tabulated placeholder
        min_satellite_prob : Optional[float]
            Halos with lower than this mean satellite occupation (calculated
            according to the `fiducial_model`) will have no placeholder. By
            default, assign at least one if halo has a central placeholder
        max_satellite_prob : float (default = 0.25)
            Populate just enough placeholders such that this is >= to the mean
            occupation (calculated according to the `fiducial_model`) divided by
            the number of placeholder satellite galaxies
        num_ptcl_requirement : Optional[int]
            Passed to model.populate_mock()

        Examples
        --------
        TODO
        """
        self.halocat = halocat
        self.fiducial_model = fiducial_model
        self.min_weight_dict = {} if min_weight_dict is None else min_weight_dict
        self.max_weight_dict = {} if max_weight_dict is None else max_weight_dict
        self.num_ptcl_requirement = num_ptcl_requirement
        self.predictor = None

        self.galaxies, self._placeholder_model = self.populate_placeholders()
        self.calc_weights(self._placeholder_model, inplace=True)

    def populate_placeholders(self):
        return gt.make_placeholder_model(self)

    def calc_weights(self, model, inplace=False):
        return gt.calc_weights(self.galaxies, model, inplace=inplace)

    def tabulate_cic(self):
        self.predictor = CICTabulator(self)
        return self.predictor

    def predict(self, model):
        if self.predictor is None:
            raise RuntimeError("You must tabulate a statistic "
                               "before predicting it")
        return self.predictor.predict(model)


class BaseStatisticTabulator:
    def __init__(self, galtab):
        self.galtab = galtab
        self.galaxies = galtab.galaxies
        # List of array of indices used in each pair
        # Zeroth array in the list will be the first galaxy in the pair
        # Next array in the list will have the second galaxy in the pair
        self.indices = []
        # A weighted sum of the prediction matrix will yield the prediction
        self.prediction_matrix = None
        self.tabulate()

    def tabulate(self):
        raise NotImplementedError("tabulate must be implemented in child class")

    def predict(self, model):
        raise NotImplementedError("predict must be implemented in child class")


class CICTabulator(BaseStatisticTabulator):
    def tabulate(self):
        self.indices, self.prediction_matrix = cic.tabulate(self)

    def predict(self, model):
        indices = self.indices
        prediction_matrix = self.prediction_matrix

        weights = gt.calc_weights(self.galaxies, model)
        matrix_product = np.ones(len(indices[0]))
        for index_array in indices:
            matrix_product *= weights[index_array]

        return np.sum(matrix_product * prediction_matrix, axis=0)
