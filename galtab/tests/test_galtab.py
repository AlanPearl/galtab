
import numpy as np

import halotools.empirical_models as htem
import halotools.sim_manager as htsm
import galtab
from galtab import jaxhalotools


class TestsWithFakeSim:
    def init_if_needed(self, use_jax_model=False, seed=123):
        if hasattr(self, "gt"):
            return

        self.redshift = 0
        self.threshold = -22
        self.halocat = htsm.FakeSim(seed=seed, redshift=self.redshift)
        if use_jax_model:
            self.model = htem.HodModelFactory(
                centrals_occupation=jaxhalotools.JaxZheng07Cens(
                    threshold=self.threshold, redshift=self.redshift),
                satellites_occupation=jaxhalotools.JaxZheng07Sats(
                    threshold=self.threshold, redshift=self.redshift),
                centrals_profile=htem.TrivialPhaseSpace(
                    redshift=self.redshift),
                satellites_profile=htem.NFWPhaseSpace(
                    redshift=self.redshift)
            )
        else:
            self.model = htem.PrebuiltHodModelFactory(
                "zheng07", threshold=self.threshold, redshift=self.redshift)

        self.gt = galtab.GalaxyTabulator(
            self.halocat, self.model, n_mc=10, min_quant=0.0001,
            max_weight=0.05, seed=seed)

    def test_bolshoi_zheng_placeholder_weights(self):
        self.init_if_needed()
        self.model.restore_init_param_dict()
        # Test that the histogram of central weights and satellite weights
        # matches known values exactly
        is_central = self.gt.galaxies["gal_type"] == "centrals"
        weights = self.gt.calc_weights(self.model)
        weights_hist_cens = np.histogram(weights[is_central], bins=10)[0]
        weights_hist_sats = np.histogram(weights[~is_central], bins=10)[0]

        assert np.all(np.isclose(
            weights_hist_cens,
            [178, 0, 0, 94, 0, 0, 0, 94, 0, 175],
            atol=0)), f"known values != {repr(weights_hist_cens)}"
        assert np.all(np.isclose(
            weights_hist_sats,
            [1504, 0, 0, 0, 0, 0, 0, 0, 6160, 23838],
            atol=0)), f"known values != {repr(weights_hist_sats)}"

        # Test that the correct number of halos from the
        # original halo catalog get assigned placeholders
        nhalo_original = len(self.halocat.halo_table)
        nhalo_galtab = len(self.gt.halo_table)
        nhalo_with_ph = len(set(self.gt.galaxies["halo_id"]))
        nhalo_with_cen_ph = len(set(self.gt.galaxies["halo_id"][is_central]))
        nhalo_with_sat_ph = len(set(self.gt.galaxies["halo_id"][~is_central]))

        nums = np.array([nhalo_original, nhalo_galtab, nhalo_with_ph,
                        nhalo_with_cen_ph, nhalo_with_sat_ph])
        assert np.all(
            nums == [1000, 808, 541, 541, 269]
        ), f"known values != {repr(nums)}"

        # Test histogram of satellite weights after changing logM1
        self.model.param_dict.update(dict(alpha=0.9))
        weights = self.gt.calc_weights(self.model)
        self.model.restore_init_param_dict()
        new_weights_hist_sats = np.histogram(weights[~is_central], bins=10)[0]
        assert np.all(
            new_weights_hist_sats ==
            [1504, 0, 0, 0, 0, 0, 6160, 0, 0, 23838]
        ), f"known values != {repr(new_weights_hist_sats)}"

    def test_bolshoi_zheng_cic(self):
        self.init_if_needed()
        self.model.restore_init_param_dict()
        bin_edges = np.concatenate([[-0.5, 2.5, 5.5],
                                    np.geomspace(9.5, 100.5, 4)])
        cic_kwargs = dict(proj_search_radius=2.0, cylinder_half_length=10.0)
        predictor = self.gt.tabulate_cic(bin_edges=bin_edges, **cic_kwargs)

        cic = predictor.predict(self.model)

        # Low precision because setting the random seed only partially works
        assert np.all(np.isclose(
            cic, [0.1569901, 0.12223569, 0.03778022, 0.00098653, 0., 0.],
            rtol=0.25, atol=1e-4
        )), f"known values != {repr(cic)}"


if __name__ == "__main__":
    tests = TestsWithFakeSim()
    method_names = [name for name in dir(TestsWithFakeSim)
                    if name.lower().startswith("test")
                    and callable(getattr(TestsWithFakeSim, name))]
    for name in method_names:
        getattr(tests, name)()
        print(f"{name} passed.")
    print("All tests passed!")
