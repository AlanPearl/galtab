
import numpy as np

import halotools.empirical_models as htem
import halotools.sim_manager as htsm
import galtab


def test_bolshoi_zheng_placeholder_weights(use_jax=False):
    redshift = 0
    threshold = -21
    halocat = htsm.CachedHaloCatalog(simname="bolshoi", redshift=redshift,
                                     halo_finder="rockstar")
    if use_jax:
        model = htem.HodModelFactory(
            centrals_occupation=galtab.jax.JaxZheng07Cens(
                threshold=threshold, redshift=redshift),
            satellites_occupation=galtab.jax.JaxZheng07Sats(
                threshold=threshold, redshift=redshift),
            centrals_profile=htem.TrivialPhaseSpace(redshift=redshift),
            satellites_profile=htem.NFWPhaseSpace(redshift=redshift)
        )
        gt = galtab.jax.GalaxyTabulator(halocat, model)
    else:
        model = htem.PrebuiltHodModelFactory("zheng07", threshold=threshold,
                                             redshift=redshift)
        gt = galtab.jax.GalaxyTabulator(halocat, model)
    # Test that the histogram of central weights and satellite weights
    # matches known values exactly
    is_central = gt.galaxies["gal_type"] == "centrals"
    weights = gt.calc_weights(model)
    weights_hist_cens = np.histogram(weights[is_central], bins=10)[0]
    weights_hist_sats = np.histogram(weights[~is_central], bins=10)[0]

    assert np.all(np.isclose(
        weights_hist_cens,
        [19756, 4879, 2868, 2099, 1714, 1461, 1323, 1279, 1452, 4025],
        atol=2)), f"known values != {weights_hist_cens}"
    assert np.all(np.isclose(
        weights_hist_sats,
        [13505, 3774, 1796, 1137, 837, 1099, 1002, 1173, 1811, 3912],
        atol=2)), f"known values != {weights_hist_sats}"

    # Test that the fraction of halos with central/satellite placeholders
    # matches known values exactly
    nhalo = len(gt.halocat.halo_table)
    ph_frac = len(set(gt.galaxies["halo_id"])) / nhalo
    cen_ph_frac = len(set(gt.galaxies["halo_id"][is_central])) / nhalo
    sat_ph_frac = len(set(gt.galaxies["halo_id"][~is_central])) / nhalo

    assert ph_frac == cen_ph_frac == 0.029876569752093796, \
        f"known value != {ph_frac} or {cen_ph_frac}"
    assert sat_ph_frac == 0.017895521220218313, \
        f"known value != {sat_ph_frac}"

    # Test histogram of satellite weights after changing logM1
    model.param_dict.update(dict(alpha=0.9))
    weights = gt.calc_weights(model)
    new_weights_hist_sats = np.histogram(weights[~is_central], bins=10)[0]
    assert np.all(np.isclose(
        new_weights_hist_sats,
        [11297, 4471, 2310, 1545, 2438, 3890, 2324, 1027, 475, 269],
        atol=2)), f"known values != {new_weights_hist_sats}"


def test_bolshoi_zheng_cic(use_jax=False):
    redshift = 0
    halocat = htsm.CachedHaloCatalog(simname="bolshoi", redshift=redshift,
                                     halo_finder="rockstar")

    if use_jax:
        model19 = htem.HodModelFactory(
            centrals_occupation=galtab.jax.JaxZheng07Cens(
                threshold=-19, redshift=redshift),
            satellites_occupation=galtab.jax.JaxZheng07Sats(
                threshold=-19, redshift=redshift),
            centrals_profile=htem.TrivialPhaseSpace(redshift=redshift),
            satellites_profile=htem.NFWPhaseSpace(redshift=redshift)
        )
        model205 = htem.HodModelFactory(
            centrals_occupation=galtab.jax.JaxZheng07Cens(
                threshold=-20.5, redshift=redshift),
            satellites_occupation=galtab.jax.JaxZheng07Sats(
                threshold=-20.5, redshift=redshift),
            centrals_profile=htem.TrivialPhaseSpace(redshift=redshift),
            satellites_profile=htem.NFWPhaseSpace(redshift=redshift)
        )
        gt = galtab.jax.GalaxyTabulator(halocat, model19, seed=1)
    else:
        model19 = htem.PrebuiltHodModelFactory("zheng07", threshold=-19)
        model205 = htem.PrebuiltHodModelFactory("zheng07", threshold=-20.5)
        gt = galtab.GalaxyTabulator(halocat, model19, seed=1)

    bin_edges = np.concatenate([[-0.5, 2.5, 5.5], np.geomspace(9.5, 100.5, 4)])
    cic_kwargs = dict(proj_search_radius=2.0, cylinder_half_length=10.0)
    predictor = gt.tabulate_cic(bin_edges=bin_edges, **cic_kwargs)

    cic19 = predictor.predict(model19)
    cic205 = predictor.predict(model205)

    assert np.all(np.isclose(
        cic19, [0.05488518, 0.06851228, 0.05155291,
                0.02285464, 0.00497551, 0.00077832],
    )), f"known values != {cic19}"

    assert np.all(np.isclose(
        cic205, [1.63377928e-01, 9.66691666e-02, 3.42786433e-02,
                 6.50972054e-03, 3.83935374e-04, 2.79730971e-05],
    )), f"known values != {cic205}"