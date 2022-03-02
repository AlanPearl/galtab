import numpy as np
import pandas as pd

from astropy import cosmology
# import halotools.empirical_models as htem
import halotools.mock_observables as htmo
import halotools.sim_manager as htsm

from . import galaxy_tabulator as gt
from . import moments

bplcosmo = cosmology.FlatLambdaCDM(name="Bolshoi-Planck",
                                   H0=67.8,
                                   Om0=0.307,
                                   Ob0=0.048)


class GalaxyTabulator:
    """
    This object populates placeholder galaxies according to a fiducial halo
    occupation model. Then, given any occupation model, each placeholder which
    will be assigned a "weight" according to the mean occupation of
    """

    def __init__(self, halocat, fiducial_model, n_mc=10,
                 min_quantile_dict=None, max_weight_dict=None,
                 num_ptcl_requirement=None, seed=None, cosmo=None):
        """
        Parameters
        ----------
        halocat : htsm.CachedHaloCatalog
            Catalog containing the halos to populate galaxies on top of
        fiducial_model : htem.ModelFactory
            The model used to calculate the number of placeholder galaxies
        n_mc : int
            Number of Monte-Carlo realizations to combine
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
            Monte-Carlo realization seeds; also passed to model.populate_mock()
        cosmo : astropy.Cosmology object
            Used for redshift-space distortions; default is Bolshoi-Planck

        Examples
        --------
        TODO
        """
        self.halocat = halocat
        self.fiducial_model = fiducial_model
        self.n_mc = n_mc
        self.min_quantile_dict = {} if min_quantile_dict is None else min_quantile_dict
        self.max_weight_dict = {} if max_weight_dict is None else max_weight_dict
        if num_ptcl_requirement is None:
            self.num_ptcl_requirement = htsm.sim_defaults.Num_ptcl_requirement
        else:
            self.num_ptcl_requirement = num_ptcl_requirement
        self.seed = seed
        self.cosmo = bplcosmo if cosmo is None else cosmo
        self.predictor = None

        if not hasattr(fiducial_model, "mock"):
            fiducial_model.populate_mock(
                halocat, seed=seed,
                Num_ptcl_requirement=self.num_ptcl_requirement)
        self.halo_table = fiducial_model.mock.halo_table
        self.galaxies, self._placeholder_model = self.populate_placeholders()
        self.halo_inds = self.tabulate_halo_inds()
        # self.calc_weights(self._placeholder_model, inplace=True)

    def populate_placeholders(self):
        return gt.make_placeholder_model(self)

    def calc_weights(self, model):
        return gt.calc_weights(self.halo_table, self.galaxies, self.halo_inds, model)

    def tabulate_cic(self, **kwargs):
        self.predictor = CICTabulator(self, **kwargs)
        return self.predictor

    def predict(self, model, *args, **kwargs):
        if self.predictor is None:
            raise RuntimeError("You must tabulate a statistic "
                               "before predicting it")
        return self.predictor.predict(model, *args, **kwargs)

    def tabulate_halo_inds(self):
        gal_df = pd.DataFrame(dict(halo_id=self.galaxies["halo_id"]))
        halo_df = pd.DataFrame(dict(halo_id=self.halo_table["halo_id"],
                                    halo_ind=np.arange(len(self.halo_table))))
        return pd.merge(gal_df, halo_df, on="halo_id", how="left")["halo_ind"].values


class CICTabulator:
    def __init__(self, galtabulator, proj_search_radius,
                 cylinder_half_length, k_vals=None, sample1_selector=None,
                 sample2_selector=None, seed=None, **kwargs):
        """
        Initialize a CICTabulator

        This object tabulates the cylinder counts of each placeholder galaxy to
        quickly, deterministically, and differentiably predict dP(N_CIC)/dN_CIC
        for any given occupation model.

        Parameters
        ----------
        galtabulator : GalaxyTabulator
        proj_search_radius : float
        cylinder_half_length : float
        k_vals : Optional[np.ndarray]
        sample1_selector : Optional[callable]
        sample2_selector : Optional[callable]
        seed : Optional[int]

        Note: Remaining keyword arguments are passed to halotools' counts-in-
        cylinders function.
        """
        self.galtabulator = galtabulator
        self.k_vals = k_vals
        self.sample1_selector = sample1_selector
        self.sample2_selector = sample2_selector
        self.rs = np.random.RandomState(seed)

        self.sample1_inds = slice(None)
        self.sample2_inds = slice(None)
        if sample1_selector is not None:
            self.sample1_inds = np.where(sample1_selector(galtabulator.galaxies))[0]
        if sample2_selector is not None:
            self.sample2_inds = np.where(sample2_selector(galtabulator.galaxies))[0]
        self.sample1 = galtabulator.galaxies[self.sample1_inds]
        self.sample2 = galtabulator.galaxies[self.sample2_inds]

        self.n = len(galtabulator.galaxies)
        self.n1 = len(self.sample1)
        self.n2 = len(self.sample2)
        self.n_mc = galtabulator.n_mc

        self.mc_rands = self.rs.random((self.n, self.n_mc))
        self.indices = None

        self.tabulate(proj_search_radius=proj_search_radius,
                      cylinder_half_length=cylinder_half_length, **kwargs)

    def tabulate(self, **kwargs):
        if "sample1" in kwargs:
            raise ValueError("Attempting to overwrite 'sample1' kwarg")
        if "sample2" in kwargs:
            raise ValueError("Attempting to overwrite 'sample2' kwarg")
        if "period" in kwargs:
            raise ValueError("Attempting to overwrite 'period' kwarg")
        if "return_indexes" in kwargs:
            raise ValueError("Attempting to overwrite 'return_indexes' kwarg")

        kwargs["sample1"] = np.array([self.sample1[f"obs_{x}"] for x in "xyz"],
                                     dtype=np.float64).T
        kwargs["sample2"] = np.array([self.sample2[f"obs_{x}"] for x in "xyz"],
                                     dtype=np.float64).T
        kwargs["period"] = self.galtabulator.halocat.Lbox
        kwargs["return_indexes"] = True
        self.indices = htmo.counts_in_cylinders(**kwargs)[1]

        # Remove self-counting!
        self.indices = self.indices[self.indices["i1"] != self.indices["i2"]]

    def predict(self, model, return_number_densities=False):
        cic = self.calc_cic(
            model, return_number_densities=return_number_densities)
        n1 = n2 = None
        if return_number_densities:
            cic, n1, n2 = cic
        if self.k_vals is not None:
            cic = moments.moments_from_pdf(
                np.arange(len(cic)), cic, self.k_vals)
        if return_number_densities:
            return cic, n1, n2
        else:
            return cic

    def calc_cic(self, model, return_number_densities=False):
        n1, n_mc = self.n1, self.n_mc
        indices = self.indices
        weights = self.galtabulator.calc_weights(model)

        mc_bool_arrays = self.mc_rands < weights[:, None]

        ncic_arrays = np.zeros((n1, n_mc), dtype=int)
        np.add.at(ncic_arrays, indices["i1"],
                  mc_bool_arrays[self.sample2_inds][indices["i2"]])

        numbins = np.max(ncic_arrays) + 1
        ncic_arrays1 = ncic_arrays + (numbins * np.arange(n_mc))
        ncic_hists = np.bincount(ncic_arrays1.ravel()[mc_bool_arrays.ravel()],
                                 minlength=numbins*n_mc).reshape(n_mc, -1)
        p_ncic_configs = ncic_hists / np.sum(mc_bool_arrays, axis=0)[:, None]
        p_ncic = np.nanmean(p_ncic_configs, axis=0)

        if return_number_densities:
            vol = np.product(self.galtabulator.halocat.Lbox)
            weights1 = weights[self.sample1_inds]
            weights2 = weights[self.sample2_inds]
            return p_ncic, np.sum(weights1)/vol, np.sum(weights2)/vol
        else:
            return p_ncic


GalaxyTabulator.tabulate_cic.__doc__ = CICTabulator.__init__.__doc__
