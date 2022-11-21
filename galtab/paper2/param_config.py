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
# Note: Acen/sat errors and sigma lower errors are taken
# from ngal + wp + P(Ncic) to ensure physical bounds
kuan_err_low = {
    -21.0: {
        "logMmin": 0.054,
        "sigma_logM": 0.123,
        "alpha": 0.188,
        "logM1": 0.118,
        "logM0": 0.118,
        "mean_occupation_centrals_assembias_param1": 0.702,
        "mean_occupation_satellites_assembias_param1": 0.512,
    },
    -20.5: {
        "logMmin": 0.075,
        "sigma_logM": 0.104,
        "alpha": 0.073,
        "logM1": 0.065,
        "logM0": 0.065,
        "mean_occupation_centrals_assembias_param1": 0.295,
        "mean_occupation_satellites_assembias_param1": 0.226,
    },
    -20.0: {
        "logMmin": 0.204,
        "sigma_logM": 0.062,
        "alpha": 0.080,
        "logM1": 0.110,
        "logM0": 0.110,
        "mean_occupation_centrals_assembias_param1": 0.138,
        "mean_occupation_satellites_assembias_param1": 0.184,
    },
    -19.5: {
        "logMmin": 0.262,
        "sigma_logM": 0.064,
        "alpha": 0.071,
        "logM1": 0.098,
        "logM0": 0.098,
        "mean_occupation_centrals_assembias_param1": 0.243,
        "mean_occupation_satellites_assembias_param1": 0.369,
    },
    -19.0: {
        "logMmin": 0.218,
        "sigma_logM": 0.114,
        "alpha": 0.049,
        "logM1": 0.062,
        "logM0": 0.062,
        "mean_occupation_centrals_assembias_param1": 0.349,
        "mean_occupation_satellites_assembias_param1": 0.203,
    },
}

kuan_err_high = {
    -21.0: {
        "logMmin": 0.123,
        "sigma_logM": 0.242,
        "alpha": 0.115,
        "logM1": 0.049,
        "logM0": 0.049,
        "mean_occupation_centrals_assembias_param1": 0.619,
        "mean_occupation_satellites_assembias_param1": 0.444,
    },
    -20.5: {
        "logMmin": 0.166,
        "sigma_logM": 0.278,
        "alpha": 0.070,
        "logM1": 0.054,
        "logM0": 0.054,
        "mean_occupation_centrals_assembias_param1": 0.141,
        "mean_occupation_satellites_assembias_param1": 0.158,
    },
    -20.0: {
        "logMmin": 0.401,
        "sigma_logM": 0.450,
        "alpha": 0.069,
        "logM1": 0.081,
        "logM0": 0.081,
        "mean_occupation_centrals_assembias_param1": 0.060,
        "mean_occupation_satellites_assembias_param1": 0.140,
    },
    -19.5: {
        "logMmin": 0.432,
        "sigma_logM": 0.510,
        "alpha": 0.065,
        "logM1": 0.078,
        "logM0": 0.078,
        "mean_occupation_centrals_assembias_param1": 0.194,
        "mean_occupation_satellites_assembias_param1": 0.185,
    },
    -19.0: {
        "logMmin": 0.390,
        "sigma_logM": 0.525,
        "alpha": 0.038,
        "logM1": 0.061,
        "logM0": 0.061,
        "mean_occupation_centrals_assembias_param1": 0.292,
        "mean_occupation_satellites_assembias_param1": 0.564,
    },
}
