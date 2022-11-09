import numpy as np
import pandas as pd

# from astropy import cosmology
# import halotools.empirical_models as htem
import halotools.mock_observables as htmo
import halotools.sim_manager as htsm

from . import galaxy_tabulator as gt
from . import moments


class GalaxyTabulator:
    """
    This object populates placeholder galaxies according to a fiducial halo
    occupation model. Then, given any occupation model, each placeholder which
    will be assigned a "weight" according to the mean occupation of
    """

    def __init__(self, halocat, fiducial_model, n_mc=10,
                 min_quant=0.001, max_weight=0.01, sample_fraction=1.0,
                 num_ptcl_requirement=None, seed=None, cosmo=None,
                 sat_quant_instead_of_max_weight=False):
        """
        Parameters
        ----------
        halocat : htsm.CachedHaloCatalog
            Catalog containing the halos to populate galaxies on top of
        fiducial_model : htem.ModelFactory
            The model used to calculate the number of placeholder galaxies
        n_mc : int (default = 10)
            Number of Monte-Carlo realizations to combine
        min_quant : float (default = 0.001)
            The minimum quantile of galaxies as a function of the model's
            prim_haloprop to populate at least one placeholder in such a halo
        max_weight : float (default = 0.01)
            The quantile of the Poisson distribution centered around <N_sat>
            to use as the number of satellite placeholders per halo
        sample_fraction : float (default = 1.0)
            The fraction of galaxies to keep - useful for modeling surveys
            whose targeting fraction is <100%, but not spatially correlated
        num_ptcl_requirement : Optional[int]
            Passed to the initial call of model.populate_mock()
        seed : Optional[int]
            Monte-Carlo realization seeds; also passed to model.populate_mock()
        cosmo : Optional[astropy.Cosmology object]
            Used for redshift-space distortions; default is Bolshoi-Planck
        sat_quant_instead_of_max_weight : Optional[bool]
            If true, max_weight is interpretted as 1 - sat_quant, where
            sat_quant specifies the number of placeholder satellites by the
            quantile of the (Poisson) occupation distribution of each halo

        Examples
        --------
        TODO: Add example usage of galtab
        """
        self.halocat = halocat
        self.fiducial_model = fiducial_model
        self.n_mc = n_mc
        self.min_quant = min_quant
        self.max_weight = max_weight
        self.sample_fraction = sample_fraction
        self.sat_quant_instead_of_max_weight = sat_quant_instead_of_max_weight
        if num_ptcl_requirement is None:
            self.num_ptcl_requirement = htsm.sim_defaults.Num_ptcl_requirement
        else:
            self.num_ptcl_requirement = num_ptcl_requirement
        self.seed = seed
        self.cosmo = halocat.cosmology if cosmo is None else cosmo
        self.predictor = None
        self.weights = None

        if (hasattr(fiducial_model, "mock")
                and fiducial_model.mock.Num_ptcl_requirement
                == self.num_ptcl_requirement
                and fiducial_model.mock.redshift == halocat.redshift
                and fiducial_model.mock.simname == halocat.simname
                and fiducial_model.mock.version_name == halocat.version_name):
            pass
        else:
            fiducial_model.populate_mock(
                halocat, seed=seed,
                Num_ptcl_requirement=self.num_ptcl_requirement)

        self.halo_table = fiducial_model.mock.halo_table
        self.galaxies, self._placeholder_model = self.populate_placeholders()
        self.halo_table = self._placeholder_model.mock.halo_table
        useless_halos = ((self.halo_table["halo_num_satellites"] < 1) &
                         (self.halo_table["halo_num_centrals"] < 1))
        self.halo_table = self.halo_table[~useless_halos]
        self.halo_inds = self.tabulate_halo_inds()

        # Unassign the halotools models to preserve pickle-ability :(
        self.fiducial_model = None
        self._placeholder_model = None

    def populate_placeholders(self):
        return gt.make_placeholder_model(self)

    def calc_weights(self, model):
        self.weights = gt.calc_weights(
            self.halo_table, self.galaxies, self.halo_inds,
            model)
        # TODO: More efficient if I had just thrown out (1 - sample_fraction)
        # TODO: of the tabulated galaxy sample at the start
        return self.weights * self.sample_fraction

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
                 cylinder_half_length, k_vals=None, bin_edges=None,
                 sample1_selector=None, sample2_selector=None,
                 analytic_moments=False, max_ncic=int(1e5),
                 seed=None, **kwargs):
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
        bin_edges : Optional[np.ndarray]
        sample1_selector : Optional[callable]
        sample2_selector : Optional[callable]
        analytic_moments : Optional[bool]
        max_ncic : Optional[int]
        seed : Optional[int]

        Note: Remaining keyword arguments are passed to halotools' counts-in-
        cylinders function.
        """
        assert sample1_selector is None, "Parameter not implemented"
        assert sample2_selector is None, "Parameter not implemented"
        self.galtabulator = galtabulator
        self.k_vals = np.array(k_vals, dtype=int)
        assert np.all(self.k_vals == k_vals), "k_vals must be a list of ints"
        self.kmax = np.max(k_vals)
        self.bin_edges = bin_edges
        self.sample1_selector = sample1_selector
        self.sample2_selector = sample2_selector
        self.analytic_moments = analytic_moments
        self.max_ncic = max_ncic
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

        self.mc_rands = self.seed_monte_carlo()
        self.indices = None

        self.tabulate(proj_search_radius=proj_search_radius,
                      cylinder_half_length=cylinder_half_length, **kwargs)

    def seed_monte_carlo(self):
        self.mc_rands = self.rs.random((self.n, self.n_mc))
        return self.mc_rands

    def tabulate(self, **kwargs):
        if "sample1" in kwargs:
            raise ValueError("Cannot overwrite 'sample1' kwarg")
        if "sample2" in kwargs:
            raise ValueError("Cannot overwrite 'sample2' kwarg")
        if "period" in kwargs:
            raise ValueError("Cannot overwrite 'period' kwarg")
        if "return_indexes" in kwargs:
            raise ValueError("Cannot overwrite 'return_indexes' kwarg")

        kwargs["sample1"] = np.array([self.sample1[f"obs_{x}"] for x in "xyz"],
                                     dtype=np.float64).T
        kwargs["sample2"] = np.array([self.sample2[f"obs_{x}"] for x in "xyz"],
                                     dtype=np.float64).T
        kwargs["period"] = self.galtabulator.halocat.Lbox
        kwargs["return_indexes"] = True
        self.indices = htmo.counts_in_cylinders(**kwargs)[1]

        # Remove self-counting!
        # TODO: Do this in a way that doesn't break if sample1 != sample2
        self.indices = self.indices[self.indices["i1"] != self.indices["i2"]]

    def predict(self, model, return_number_densities=False, n_mc=None,
                reseed_mc=False):
        cic = self.calc_cic(
            model, return_number_densities=return_number_densities,
            n_mc=n_mc, reseed_mc=reseed_mc)
        n1 = n2 = None
        if return_number_densities:
            cic, n1, n2 = cic

        if self.analytic_moments and self.k_vals is None:
            pass
        elif self.k_vals is not None:
            cic = moments.moments_from_samples(
                np.arange(len(cic)), self.k_vals, weights=cic)
        elif self.bin_edges is not None:
            hist = np.histogram(
                np.arange(len(cic)), self.bin_edges, weights=cic)[0]
            cic = hist / hist.sum() / np.diff(self.bin_edges)
        if return_number_densities:
            return cic, n1, n2
        else:
            return cic

    def calc_cic(self, model, return_number_densities=False, n_mc=None,
                 reseed_mc=False, warn_p_over_1=True):
        if reseed_mc:
            self.seed_monte_carlo()
        if n_mc is None:
            n_mc = self.n_mc
            mc_rands = self.mc_rands
        else:
            mc_rands = self.mc_rands[:, :n_mc]
        n1 = self.n1
        indices = self.indices
        weights = self.galtabulator.calc_weights(model)
        previous_ints = np.ceil(weights).astype(int) - 1
        previous_ints[previous_ints < 0] = 0
        next_int_probs = weights - previous_ints
        if self.analytic_moments and self.kmax is not None:
            if warn_p_over_1 and np.any(previous_ints):
                bad_weights = weights[previous_ints != 0]
                print(f"WARNING: There are {len(bad_weights)} placeholders "
                      f"with weight>1, averaging: {bad_weights.mean()}")
            pb_cumulants = []
            for k in range(1, self.kmax + 1):
                # Bernoulli cumulant
                p = weights if k == 1 else next_int_probs
                bc = moments.bernoulli_cumulant(p, k)

                # Sum of Bernoulli cumulants --> Poisson Binomial cumulants
                pc = np.zeros(n1, dtype=float)
                np.add.at(pc, indices["i1"], bc[indices["i2"]])
                pb_cumulants.append(pc)

            pb_raw_moments = moments.raw_moments_from_cumulants(pb_cumulants)
            avg_raw_moments = np.average(
                pb_raw_moments, weights=weights, axis=1)
            cic = moments.standardized_moments_from_raw_moments(
                avg_raw_moments)
            cic = np.array([cic[k-1] if k > 0 else 1 for k in self.k_vals])

        else:
            mc_num_arrays = previous_ints[:, None] + (mc_rands <
                                                      next_int_probs[:, None])

            ncic_arrays = np.zeros((n1, n_mc), dtype=int)
            np.add.at(ncic_arrays, indices["i1"],
                      mc_num_arrays[indices["i2"]])
            # mc_num_arrays[self.sample2_inds][indices["i2"]])

            numbins = np.max(ncic_arrays) + 1
            if numbins < self.max_ncic:
                ncic_arrays1 = ncic_arrays + (numbins * np.arange(n_mc))

                ncic_hists = np.bincount(
                    ncic_arrays1.ravel(), weights=mc_num_arrays.ravel(),
                    minlength=numbins * n_mc).reshape(n_mc, -1)
                p_ncic_configs = ncic_hists / np.sum(mc_num_arrays, axis=0)[:, None]
                cic = np.nanmean(p_ncic_configs, axis=0)
            else:
                cic = np.array([np.nan])

        if return_number_densities:
            vol = np.product(self.galtabulator.halocat.Lbox)
            weights1 = weights  # [self.sample1_inds]
            n1 = np.sum(weights1) / vol
            # weights2 = weights[self.sample2_inds]
            n2 = n1  # np.sum(weights2) / vol

            return cic, n1, n2
        else:
            return cic

    def save(self, filename):
        np.save(filename, np.array([self], dtype=object))

    @classmethod
    def load(cls, filename):
        obj = np.load(filename, allow_pickle=True)[0]
        return obj


GalaxyTabulator.tabulate_cic.__doc__ = CICTabulator.__init__.__doc__
