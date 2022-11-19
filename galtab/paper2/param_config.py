import numpy as np
import astropy.cosmology

cosmo = astropy.cosmology.Planck13
simname = "smdpl"
proj_search_radius = 2.0
cylinder_half_length = 10.0
pimax = 40.0

rp_edges = np.logspace(-0.8, 1.6, 13)
cic_edges = np.concatenate([np.arange(-0.5, 9),
                            np.round(np.geomspace(10, 150, 20)) - 0.5])

zmin = 0.1
zmax = 0.2

# Best-fit params from Wang+ 2022 (Table 4)
# dHOD results (assembly bias included)
# =========================================

# Median from
# ngal + wp + P(Ncic) dHOD, because it fits lower M1, Mmin, M0
kuan_params = {
    -21.0: {
        "logMmin": 12.671,
        "sigma_logM": 0.185,
        "alpha": 0.892,
        "logM1": 13.844,
        "logM0": 13.086,
        "mean_occupation_centrals_assembias_param1": 0.105,
        "mean_occupation_satellites_assembias_param1": -0.235,
    },
    -20.5: {
        "logMmin": 12.265,
        "sigma_logM": 0.218,
        "alpha": 0.950,
        "logM1": 13.454,
        "logM0": 12.617,
        "mean_occupation_centrals_assembias_param1": 0.811,
        "mean_occupation_satellites_assembias_param1": -0.147,
    },
    -20.0: {
        "logMmin": 11.973,
        "sigma_logM": 0.298,
        "alpha": 0.928,
        "logM1": 13.118,
        "logM0": 12.420,
        "mean_occupation_centrals_assembias_param1": 0.922,
        "mean_occupation_satellites_assembias_param1": -0.166,
    },
    -19.5: {
        "logMmin": 11.709,
        "sigma_logM": 0.367,
        "alpha": 0.890,
        "logM1": 12.785,
        "logM0": 12.527,
        "mean_occupation_centrals_assembias_param1": 0.711,
        "mean_occupation_satellites_assembias_param1": -0.087,
    },
    -19.0: {
        "logMmin": 11.565,
        "sigma_logM": 0.356,
        "alpha": 0.842,
        "logM1": 12.649,
        "logM0": 12.316,
        "mean_occupation_centrals_assembias_param1": 0.559,
        "mean_occupation_satellites_assembias_param1": -0.634,
    },
}

# Lower errorbar value from
# ngal + wp dHOD, because it has larger errorbars
# (replace lower bound of M0 with lower bound of M1
# since M0 doesn't have any lower constraints)
kuan_err_low = {
    -21.0: {
        "logMmin": 0.054,
        "sigma_logM": 0.209,
        "alpha": 0.188,
        "logM1": 0.118,
        "logM0": 0.118,
        "mean_occupation_centrals_assembias_param1": 0.686,
        "mean_occupation_satellites_assembias_param1": 0.515,
    },
    -20.5: {
        "logMmin": 0.075,
        "sigma_logM": 0.254,
        "alpha": 0.073,
        "logM1": 0.065,
        "logM0": 0.065,
        "mean_occupation_centrals_assembias_param1": 0.453,
        "mean_occupation_satellites_assembias_param1": 0.284,
    },
    -20.0: {
        "logMmin": 0.204,
        "sigma_logM": 0.396,
        "alpha": 0.080,
        "logM1": 0.110,
        "logM0": 0.110,
        "mean_occupation_centrals_assembias_param1": 0.301,
        "mean_occupation_satellites_assembias_param1": 0.319,
    },
    -19.5: {
        "logMmin": 0.262,
        "sigma_logM": 0.530,
        "alpha": 0.071,
        "logM1": 0.098,
        "logM0": 0.098,
        "mean_occupation_centrals_assembias_param1": 0.516,
        "mean_occupation_satellites_assembias_param1": 0.409,
    },
    -19.0: {
        "logMmin": 0.218,
        "sigma_logM": 0.477,
        "alpha": 0.049,
        "logM1": 0.062,
        "logM0": 0.062,
        "mean_occupation_centrals_assembias_param1": 0.593,
        "mean_occupation_satellites_assembias_param1": 0.420,
    },
}
