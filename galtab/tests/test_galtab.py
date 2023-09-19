
import numpy as np

import halotools.empirical_models as htem
import halotools.sim_manager as htsm
import galtab
from galtab import jaxhalotools


def test_bolshoi_zheng_placeholder_weights(use_jax=False, seed=123):
    redshift = 0
    threshold = -21
    halocat = htsm.CachedHaloCatalog(simname="bolshoi", redshift=redshift,
                                     halo_finder="rockstar")
    if use_jax:
        model = htem.HodModelFactory(
            centrals_occupation=jaxhalotools.JaxZheng07Cens(
                threshold=threshold, redshift=redshift),
            satellites_occupation=jaxhalotools.JaxZheng07Sats(
                threshold=threshold, redshift=redshift),
            centrals_profile=htem.TrivialPhaseSpace(redshift=redshift),
            satellites_profile=htem.NFWPhaseSpace(redshift=redshift)
        )
    else:
        model = htem.PrebuiltHodModelFactory("zheng07", threshold=threshold,
                                             redshift=redshift)
    gt = galtab.GalaxyTabulator(
        halocat, model, n_mc=10, min_quant=0.0001, max_weight=0.05, seed=seed)
    # Test that the histogram of central weights and satellite weights
    # matches known values exactly
    is_central = gt.galaxies["gal_type"] == "centrals"
    weights = gt.calc_weights(model)
    weights_hist_cens = np.histogram(weights[is_central], bins=10)[0]
    weights_hist_sats = np.histogram(weights[~is_central], bins=10)[0]

    assert np.all(np.isclose(
        weights_hist_cens,
        [73736,  5206,  2980,  2145,  1760,  1483,  1336,  1300,  1460,  4042],
        atol=2)), f"known values != {weights_hist_cens}"
    assert np.all(np.isclose(
        weights_hist_sats,
        [32417,  9728,  5238,  3235,  2258,  3871,  3576,  4487,  7676, 35641],
        atol=2)), f"known values != {weights_hist_sats}"

    # Test that the correct number of halos from the
    # original halo catalog get assigned placeholders
    nhalo_original = len(halocat.halo_table)
    nhalo_galtab = len(gt.halo_table)
    nhalo_with_ph = len(set(gt.galaxies["halo_id"]))
    nhalo_with_cen_ph = len(set(gt.galaxies["halo_id"][is_central]))
    nhalo_with_sat_ph = len(set(gt.galaxies["halo_id"][~is_central]))

    nums = np.array([nhalo_original, nhalo_galtab, nhalo_with_ph,
                     nhalo_with_cen_ph, nhalo_with_sat_ph])
    assert np.all(
        nums==[1367493, 95448, 95448, 95448, 66617]
    ), f"known values != {nums}"

    # Test histogram of satellite weights after changing logM1
    model.param_dict.update(dict(alpha=0.9))
    weights = gt.calc_weights(model)
    new_weights_hist_sats = np.histogram(weights[~is_central], bins=10)[0]
    assert np.all(
        new_weights_hist_sats ==
        [23454, 12534, 7745, 10831, 13434, 17953, 14254, 5083, 1914, 925]
    ), f"known values != {new_weights_hist_sats}"


def test_bolshoi_zheng_cic(use_jax=False, seed=123):
    redshift = 0
    threshold = -21.0
    halocat = htsm.CachedHaloCatalog(simname="bolshoi", redshift=redshift,
                                     halo_finder="rockstar")

    if use_jax:
        model = htem.HodModelFactory(
            centrals_occupation=jaxhalotools.JaxZheng07Cens(
                threshold=threshold, redshift=redshift),
            satellites_occupation=jaxhalotools.JaxZheng07Sats(
                threshold=threshold, redshift=redshift),
            centrals_profile=htem.TrivialPhaseSpace(redshift=redshift),
            satellites_profile=htem.NFWPhaseSpace(redshift=redshift)
        )
    else:
        model = htem.PrebuiltHodModelFactory("zheng07", threshold=threshold)
    gt = galtab.GalaxyTabulator(
        halocat, model, n_mc=10, min_quant=0.0001, max_weight=0.05, seed=seed)

    bin_edges = np.concatenate([[-0.5, 2.5, 5.5], np.geomspace(9.5, 100.5, 4)])
    cic_kwargs = dict(proj_search_radius=2.0, cylinder_half_length=10.0)
    predictor = gt.tabulate_cic(bin_edges=bin_edges, **cic_kwargs)

    cic = predictor.predict(model)

    # Low precision because setting the random seed only partially works
    assert np.all(np.isclose(
        cic, [2.6474011e-01, 5.3108532e-02, 9.6307192e-03,
              6.6112762e-04, 1.7021703e-05, 0.0000000e+00],
        rtol=0.1, atol=1e-4
    )), f"known values != {cic}"
