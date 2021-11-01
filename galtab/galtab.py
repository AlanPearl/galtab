import numpy as np
import scipy.stats
# import jax.numpy as jnp

# import halotools.empirical_models as htem
import halotools.mock_observables as htmo
# import halotools.simulation_manager as htsm

from . import galaxy_tabulator as gt


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
            Dictionary whose keys must be names of gal_types of fiducial_model,
            and values specify the minimum probabilty to populate a placeholder
            - default is 0.01 for all galaxy types
        max_weight_dict : Optional[dict]
            Same as min_weight_dict, but the values specify the maximum weight
            to assign to any placeholder galaxy before populating additional
            placeholder(s) - default is 1 for centrals and 0.25 for satellites
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

    def tabulate_cic(self, **kwargs):
        self.predictor = CICTabulator(self, **kwargs)
        return self.predictor

    def predict(self, model):
        if self.predictor is None:
            raise RuntimeError("You must tabulate a statistic "
                               "before predicting it")
        return self.predictor.predict(model)


class CICTabulator:
    def __init__(self, galtab, bin_edges,
                 sample1_selector=None, sample2_selector=None,
                 **kwargs):
        self.galtab = galtab
        self.bin_edges = bin_edges
        self.sample1_selector = sample1_selector
        self.sample2_selector = sample2_selector

        self.sample1_inds = slice(None)
        self.sample2_inds = slice(None)
        if sample1_selector is not None:
            self.sample1_inds = np.where(sample1_selector(galtab.galaxies))[0]
        if sample2_selector is not None:
            self.sample2_inds = np.where(sample2_selector(galtab.galaxies))[0]

        self.sample1 = galtab.galaxies[self.sample1_inds]
        self.sample2 = galtab.galaxies[self.sample2_inds]
        self.indices = None
        self.tabulate(**kwargs)

    def tabulate(self, **kwargs):
        assert "proj_search_radius" in kwargs
        assert "cylinder_half_length" in kwargs

        if "sample1" in kwargs:
            print("Warning: overwriting 'sample1' kwarg")
        if "sample2" in kwargs:
            print("Warning: overwriting 'sample2' kwarg")
        if "period" in kwargs:
            print("Warning: overwriting 'period' kwarg")
        if "return_indexes" in kwargs:
            print("Warning: overwriting 'return_indexes' kwarg")

        kwargs["sample1"] = np.array([self.sample1[x] for x in "xyz"]).T
        kwargs["sample2"] = np.array([self.sample2[x] for x in "xyz"]).T
        kwargs["period"] = self.galtab.halocat.Lbox
        kwargs["return_indexes"] = True
        self.indices = htmo.counts_in_cylinders(**kwargs)[1]

    def predict(self, model):
        counts, weights = self.calc_counts_and_weights(model)
        return self.cic_prob_poisson(counts, weights)

    def calc_counts_and_weights(self, model):
        indices = self.indices
        weights = gt.calc_weights(self.galtab.galaxies, model)
        weights1 = weights[self.sample1_inds]
        weights2 = weights[self.sample2_inds]

        cic_galaxies = np.zeros((len(weights)), dtype=float)
        np.add.at(cic_galaxies, indices["i1"], weights2[indices["i2"]])

        return cic_galaxies, weights1

    def cic_prob_poisson(self, galaxy_counts, galaxy_weights):
        """Returns P(N_CIC)/dN_CIC, assuming Poisson distributions"""
        bin_edges = self.bin_edges
        poisson_dist = scipy.stats.poisson(mu=galaxy_counts[:, None])
        hist = np.sum(np.diff(poisson_dist.cdf(bin_edges[None, :]), axis=-1)
                      * galaxy_weights[:, None], axis=0)
        return hist / np.sum(galaxy_weights) / np.diff(np.ceil(bin_edges))
