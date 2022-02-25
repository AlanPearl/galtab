import numpy as np
import jax.scipy
import jax.numpy as jnp

import halotools.mock_observables as htmo

from . import galaxy_tabulator as gt
from . import cdf_interp
from . import moments
from .. import galtab


class GalaxyTabulator:
    """
    This object populates placeholder galaxies according to a fiducial halo
    occupation model. Then, given any occupation model, each placeholder which
    will be assigned a "weight" according to the mean occupation of
    """

    def __init__(self, halocat, fiducial_model,
                 min_quantile_dict=None,
                 max_weight_dict=None,
                 num_ptcl_requirement=None,
                 seed=None):
        """
        Parameters
        ----------
        halocat : htsm.CachedHaloCatalog
            Catalog containing the halos to populate galaxies on top of
        fiducial_model : htem.ModelFactory
            The model used to calculate the number of placeholder galaxies
        min_quantile_dict : Optional[dict]
            Dictionary whose keys must be names of gal_types of fiducial_model,
            and values specify the minimum quantile of galaxies as a function
            of the model's prim_haloprop to populate a placeholder
            - default is 0.001 for all galaxy types
        max_weight_dict : Optional[dict]
            Same as min_weight_dict, but the values specify the maximum weight
            to assign to any placeholder galaxy before populating additional
            placeholder(s) - default is 1 for centrals and 0.25 for satellites
        num_ptcl_requirement : Optional[int]
            Passed to model.populate_mock()
        seed : Optional[int]
            Passed to model.populate_mock()

        Examples
        --------
        TODO
        """
        self.halocat = halocat
        self.fiducial_model = fiducial_model
        self.min_quantile_dict = {} if min_quantile_dict is None else min_quantile_dict
        self.max_weight_dict = {} if max_weight_dict is None else max_weight_dict
        self.num_ptcl_requirement = num_ptcl_requirement
        self.seed = seed
        self.predictor = None

        if not hasattr(fiducial_model, "mock"):
            fiducial_model.populate_mock(halocat)
        self.halo_table = fiducial_model.mock.halo_table
        self.galaxies, self._placeholder_model = self.populate_placeholders()
        # self.calc_weights(self._placeholder_model, inplace=True)

    def populate_placeholders(self):
        return gt.make_placeholder_model(self)

    def calc_weights(self, model):
        return gt.calc_weights(self.galaxies, model)

    def tabulate_cic(self, *args, **kwargs):
        self.predictor = CICTabulator(self, *args, **kwargs)
        return self.predictor

    def predict(self, model):
        if self.predictor is None:
            raise RuntimeError("You must tabulate a statistic "
                               "before predicting it")
        return self.predictor.predict(model)


class CICTabulator(galtab.CICTabulator):
    def __init__(self, galtabulator, bin_edges_or_moments,
                 sample1_selector=None, sample2_selector=None,
                 return_moments=False, interp_bins=True, **kwargs):
        galtab.CICTabulator.__init__(
            self, galtabulator, bin_edges_or_moments, sample1_selector,
            sample2_selector, return_moments, interp_bins, **kwargs)
        if sample1_selector is not None:
            self.sample1_inds = jnp.array(
                np.where(sample1_selector(galtabulator.galaxies))[0])
        if sample2_selector is not None:
            self.sample2_inds = jnp.array(
                np.where(sample2_selector(galtabulator.galaxies))[0])

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

        kwargs["sample1"] = jnp.array([self.sample1[x] for x in "xyz"]).T
        kwargs["sample2"] = jnp.array([self.sample2[x] for x in "xyz"]).T
        kwargs["period"] = self.galtabulator.halocat.Lbox
        kwargs["return_indexes"] = True
        self.indices = htmo.counts_in_cylinders(**kwargs)[1]

    def calc_counts_and_weights(self, model):
        indices = self.indices
        weights = gt.calc_weights(self.galtabulator.galaxies, model)
        weights1 = weights[self.sample1_inds]
        weights2 = weights[self.sample2_inds]

        cic = jnp.zeros((len(weights)), dtype=float)
        cic = cic.at[indices["i1"]].add(weights2[indices["i2"]])

        return cic, weights1

    def cic_moments(self, galaxy_counts, galaxy_weights):
        return moments.moments_from_samples(
            galaxy_counts, k_vals=self.bin_edges, weights=galaxy_weights)

    def cic_prob_poisson(self, galaxy_counts, galaxy_weights):
        """Returns dP(N_CIC)/dN_CIC, assuming Poisson distributions"""
        bin_edges = self.bin_edges
        poisson_cdf = jax.scipy.stats.poisson.cdf(
            mu=galaxy_counts[:, None], k=bin_edges[None, :])
        hist = jnp.sum(jnp.diff(poisson_cdf, axis=-1)
                       * galaxy_weights[:, None], axis=0)
        return hist / jnp.sum(galaxy_weights) / jnp.diff(jnp.floor(bin_edges))

    def cic_prob_poisson_interp(self, galaxy_counts, galaxy_weights):
        """Same, but interpolates pretabulated Poisson CDF grid"""
        bin_edges = self.bin_edges
        poisson_cdf = jnp.array([cdf_interp.poisson(
            mu=galaxy_counts, k=k) for k in bin_edges])
        hist = jnp.sum(jnp.diff(poisson_cdf, axis=0)
                       * galaxy_weights[None, :], axis=1)
        return hist / jnp.sum(galaxy_weights) / jnp.diff(jnp.floor(bin_edges))


CICTabulator.__init__.__doc__ = galtab.CICTabulator.__init__.__doc__
GalaxyTabulator.tabulate_cic.__doc__ = CICTabulator.__init__.__doc__
