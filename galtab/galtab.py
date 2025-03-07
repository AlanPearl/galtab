from copy import copy
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd

# from astropy import cosmology
# import halotools.empirical_models as htem
import halotools.mock_observables as htmo
import halotools.sim_manager as htsm

from . import _galaxy_tabulator as gt
from . import moments


class GalaxyTabulator:
    """
    This object populates placeholder galaxies according to a fiducial halo
    occupation model. Then, given any occupation model, each placeholder which
    will be assigned a "weight" according to the mean occupation of
    """

    def __init__(self, halocat, fiducial_model, n_mc=10,
                 min_quant=1e-4, max_weight=0.05, sample_fraction=1.0,
                 num_ptcl_requirement=None, seed=None, cosmo=None,
                 sat_quant_instead_of_max_weight=False):
        """
        Parameters
        ----------
        halocat : htsm.CachedHaloCatalog
            Catalog containing the halos to populate galaxies on top of
        fiducial_model : htem.ModelFactory
            Used to calculate the number of placeholder galaxies per halo
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
            Used for redshift-space distortions; default taken from halocat
        sat_quant_instead_of_max_weight : Optional[bool]
            If true, max_weight is interpretted as 1 - sat_quant, where
            sat_quant specifies the number of placeholder satellites by the
            quantile of the (Poisson) occupation distribution of each halo

        Examples
        --------
        Choose HOD model and load halos

        >>> hod = halotools.empirical_models.PrebuiltHodModelFactory("zheng07")
        >>> halocat = halotools.sim_manager.CachedHaloCatalog(simname="bolplanck")

        Instantiate the tabulators

        >>> gtab = GalaxyTabulator(halocat, hod)
        >>> cictab = CICTabulator(
        ...     gtab, proj_search_radius=2.0,
        ...     cylinder_half_length=10.0, bin_edges=np.arange(-0.5, 16))

        Update HOD parameters to your liking and perform CiC prediction

        >>> hod.param_dict.update({})
        >>> cictab.predict(hod)

        """
        self.halocat = copy(halocat)
        self.fiducial_model = copy(fiducial_model)
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
        if cosmo is None:
            if hasattr(halocat, "cosmology"):
                cosmo = halocat.cosmology
            else:
                raise ValueError(
                    "There is no halocat.cosmology - please specify it manually")
        self.cosmo = cosmo
        self.predictor = None
        self.weights = None

        # Populate_mock is my lazy way of making sure the halo_table
        # doesn't include subhalos if the model doesn't need them
        fiducial_model.populate_mock(
            halocat, seed=seed,
            Num_ptcl_requirement=self.num_ptcl_requirement)
        self.halo_table = fiducial_model.mock.halo_table

        self.galaxies, self._placeholder_model, self.halocat = \
            self.populate_placeholders()
        self.halo_table = self._placeholder_model.mock.halo_table
        self.halo_inds = self.tabulate_halo_inds()

        # Unassign the halotools models to preserve pickle-ability
        self.fiducial_model = None
        self._placeholder_model = None

    def populate_placeholders(self):
        return gt.make_placeholder_model(self)

    def calc_weights(self, model):
        self.weights = gt.calc_weights(
            self.halo_table, self.galaxies, self.halo_inds,
            model) * self.sample_fraction
        return self.weights

    def tabulate_cic(self, **kwargs):
        # TODO: Replace kwargs with actual arguments for tab-complete-ability
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
                 analytic_moments=True, sort_tabulated_indices=False,
                 max_ncic=int(1e5),
                 seed=None, **kwargs):
        """
        Initialize a CICTabulator

        This object tabulates the cylinder counts of each placeholder galaxy to
        quickly, deterministically, and differentiably predict dP(N_CIC)/dN_CIC
        for any given occupation model.

        Parameters
        ----------
        galtabulator : GalaxyTabulator
            Object containing the tabulated placeholder galaxies
        proj_search_radius : float
            Perpendicular radius of circle in which to count pairs
        cylinder_half_length : float
            Half length of cylinder in which to count pairs
        k_vals : Optional[np.ndarray]
            Array of moment numbers (i.e. [1, 2] for mean, std)
        bin_edges : Optional[np.ndarray]
            Bin edges of P(Ncic); ignored if k_vals are provided
        sample1_selector : Optional[callable]
            Not implemented
        sample2_selector : Optional[callable]
            Not implemented
        analytic_moments : Optional[bool]
            Less noisy than Monte-Carlo approximation and quicker
            if only calculating a few k_vals
        sort_tabulated_indices : Optional[bool]
            This will cause longer tabulation time, but shorter
            subsequent prediction calls
        max_ncic : Optional[int]
            If any galaxies have more Ncic than this, return Ncic=0
        seed : Optional[int]
            Random seed to produce reproducible results, even if
            not using analytic_moments=True

        Note: Remaining keyword arguments are passed to halotools' counts-in-
        cylinders function.
        """
        assert sample1_selector is None, "Parameter not implemented"
        assert sample2_selector is None, "Parameter not implemented"
        self.galtabulator = galtabulator
        self.k_vals = k_vals
        self.kmax = None
        if k_vals is not None:
            self.k_vals = np.array(k_vals, dtype=int)
            self.kmax = np.max(k_vals, initial=0)
            assert np.min(k_vals, initial=1) > 0, "Lowest allowed k_val is 1"
            assert np.all(self.k_vals == k_vals), "k_vals must be list of ints"
        self.bin_edges = bin_edges
        self.sample1_selector = sample1_selector
        self.sample2_selector = sample2_selector
        self.analytic_moments = analytic_moments
        self.sort_tabulated_indices = sort_tabulated_indices
        self.max_ncic = max_ncic
        self.rs = np.random.RandomState(seed)

        self.sample1_inds = slice(None)
        self.sample2_inds = slice(None)
        if sample1_selector is not None:
            self.sample1_inds = np.where(
                sample1_selector(galtabulator.galaxies))[0]
        if sample2_selector is not None:
            self.sample2_inds = np.where(
                sample2_selector(galtabulator.galaxies))[0]
        self.sample1 = galtabulator.galaxies[self.sample1_inds]
        self.sample2 = galtabulator.galaxies[self.sample2_inds]

        self.n = len(galtabulator.galaxies)
        self.n1 = len(self.sample1)
        self.n2 = len(self.sample2)
        self.n_mc = galtabulator.n_mc

        self.mc_rands = self.seed_monte_carlo()
        self.indices = None

        self.proj_search_radius = proj_search_radius
        self.cylinder_half_length = cylinder_half_length
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
        self.indices = htmo.counts_in_cylinders(**kwargs)[1].astype(
            [("i1", "<u8"), ("i2", "<u8")])

        # Remove self-counting!
        # TODO: Is there a way that allows sample1 and sample2 to differ?
        self.indices = self.indices[self.indices["i1"] != self.indices["i2"]]
        if self.sort_tabulated_indices:
            self.indices.sort(order="i1")

    def predict(self, model, return_number_densities=False, n_mc=None,
                reseed_mc=False, warn_p_over_1=True):
        """
        Perform tabulation-accelerated prediction

        Parameters
        ----------
        model : halotools.empirical_models.ModelFactory
            Halotools model we want to evaluate CiC for
        return_number_densities : bool [Optional]
            Return number densities of both samples n1 and n2
        n_mc : int [Optional]
            Number of Monte-Carlo realizations (ignored in analytic moments)
        reseed_mc : bool [Optional]
            Reseed the CICTabulator's Monte Carlo realization generator
        warn_p_over_1 : bool | str [Optional]
            If true (default), print a warning if any placeholder weights > 1
            If string starting with "return", return warn_status, don't print

        Returns
        -------
        cic : np.ndarray
            Values of P(Ncic) or moments specified by k_vals
        [n1] : float
            Number density of sample1. Only returned if return_number_densities
        [n2] : float
            n2 always = n1 for now. Only returned if return_number_densities
        [warn_status] : dict
            Dictionary specifying the warning status. Only returned if
            warn_p_over_1 is a string starting with "return"
        """
        result = self.calc_cic(
            model, return_number_densities=return_number_densities,
            n_mc=n_mc, reseed_mc=reseed_mc, warn_p_over_1=warn_p_over_1)
        n1 = n2 = None
        if return_number_densities:
            cic, n1, n2, warn_status = result
        else:
            cic, warn_status = result

        if self.analytic_moments and self.k_vals is not None:
            pass
        elif self.k_vals is not None:
            cic = moments.moments_from_samples(
                jnp.arange(len(cic)), self.k_vals, weights=cic)
        elif self.bin_edges is not None:
            hist = jnp.histogram(
                jnp.arange(len(cic)), self.bin_edges, weights=cic)[0]
            cic = hist / hist.sum() / jnp.diff(self.bin_edges)

        return_warn_status = (hasattr(warn_p_over_1, "lower") and
                              warn_p_over_1.lower().startswith("return"))
        if return_number_densities:
            if return_warn_status:
                return cic, n1, n2, warn_status
            else:
                return cic, n1, n2
        elif return_warn_status:
            return cic, warn_status
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
        previous_ints = jnp.ceil(weights) - 1
        # previous_ints[previous_ints < 0] = 0
        previous_ints = jnp.where(previous_ints < 0, 0, previous_ints)

        warn_raised = False
        n_bad_weights, mean_bad_weight = None, None
        if warn_p_over_1 and jnp.any(previous_ints):
            warn_raised = True
            bad_weights = weights[previous_ints != 0]
            n_bad_weights = len(bad_weights)
            mean_bad_weight = bad_weights.mean()
            if hasattr(warn_p_over_1, "lower") and warn_p_over_1.lower(
            ).startswith("return"):
                pass
            else:
                jax.debug.print(
                    "WARNING: There are {x} placeholders with weight>1, averaging: {y}",
                    x=n_bad_weights, y=mean_bad_weight)

        if self.analytic_moments and self.kmax is not None:
            # It was spending ~99% of computational time in np.add.at
            # before replacing np.add.at with jax at[].add()
            pb_cumulants = []
            for k in range(1, self.kmax + 1):
                # Bernoulli cumulant
                bc = moments.bernoulli_cumulant(weights, k)

                # Sum of Bernoulli cumulants --> Poisson Binomial cumulants
                pc = moments.jit_sum_at(
                    bc, indices["i2"], indices["i1"], len_out=n1,
                    ind_out_is_sorted=self.sort_tabulated_indices)
                # TODO: replace bc with bc[self.sample2_inds] ???
                pb_cumulants.append(pc)

            pb_raw_moments = moments.raw_moments_from_cumulants(pb_cumulants)
            avg_raw_moments = jnp.average(
                jnp.array(pb_raw_moments), weights=weights, axis=1)
            cic = moments.standardized_moments_from_raw_moments(
                avg_raw_moments)
            cic = jnp.array([cic[k-1] if k > 0 else 1 for k in self.k_vals])

        else:
            next_int_probs = weights - previous_ints.astype(jnp.float32)
            mc_num_arrays = previous_ints[:, None] + (mc_rands <
                                                      next_int_probs[:, None])

            ncic_arrays = moments.jit_sum_at(
                mc_num_arrays, indices["i2"], indices["i1"],
                len_out=(n1, n_mc),
                ind_out_is_sorted=self.sort_tabulated_indices)
            # replace mc_num_arrays with mc_num_arrays[self.sample2_inds] ?

            numbins = int(jnp.max(ncic_arrays) + 1)
            if numbins < self.max_ncic:
                ncic_arrays1 = ncic_arrays.astype(int) + (
                    numbins * jnp.arange(n_mc))

                ncic_hists = jnp.bincount(
                    ncic_arrays1.ravel(), weights=jnp.repeat(weights, n_mc),
                    minlength=numbins * n_mc).reshape(n_mc, -1)
                p_ncic_configs = ncic_hists / jnp.sum(ncic_hists,
                                                      axis=1)[:, None]
                cic = jnp.nanmean(p_ncic_configs, axis=0)
            else:
                cic = jnp.array([jnp.nan])

        warn_status = {"warn_raised": warn_raised,
                       "n_bad_weights": n_bad_weights,
                       "mean_bad_weight": mean_bad_weight}
        if return_number_densities:
            vol = jnp.product(self.galtabulator.halocat.Lbox)
            weights1 = weights  # [self.sample1_inds]
            n_density1 = jnp.sum(weights1) / vol
            # weights2 = weights[self.sample2_inds]
            n_density2 = n_density1  # np.sum(weights2) / vol

            return cic, n_density1, n_density2, warn_status
        else:
            return cic, warn_status

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f, protocol=4)

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            obj = pickle.load(f)
        assert isinstance(obj, cls)
        return obj


GalaxyTabulator.tabulate_cic.__doc__ = CICTabulator.__init__.__doc__
