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
cic_kmax = 5

zmin = 0.1
zmax = 0.2
