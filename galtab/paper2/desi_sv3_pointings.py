import numpy as np

# Each element is a different region
# Format for single region: [(xlim), (ylim), [(X0, Y0, M, C), ...]]
# where M(x-X0) < C*(y-Y0)
# (Note: x = RA and y = DEC in degrees)
# With these limits, each galaxy is assigned to 1 region,
# except for 9 galaxies which are assigned to 0 regions
lims = [
    [(147, 153), (0, 5), []],
    [(177, 181.37), (-2, 2), []],
    [(181.37, 185), (-2, 2), []],
    [(211, 214.55), (-2.5, 1.2), []],
    [(214.55, 218.03), (-2.5, 1.2), [(217, 0.97, 0.5, -1)]],
    [(218.03, 222), (-2.5, 1.2), [(219.5, 1.18, -0.5, -1)]],
    [(216, 220), (0.5, 4.5), [(217, 0.97, -0.5, 1), (219.5, 1.18, 0.5, 1)]],
    [(208, 212), (3, 7), []],
    [(192.5, 197), (23, 26.43), []],
    [(192.5, 197), (26.43, 30), []],
    [(215, 220), (32, 37), []],
    [(250, 255), (32, 37), []],
    [(233, 238.5), (41, 46), []],
    [(238.5, 243.5), (41, 46), []],
    [(243.5, 249), (41, 46), []],
    [(186, 194), (60, 64), []],
    [(212.5, 218.5), (50.5, 54.5), []],
    [(239.5, 246), (53, 57), []],
    [(265, 275), (60.5, 64.3), []],
    [(265, 275), (64.3, 68), []],
]


def select_region(region, ra, dec):
    assert region in range(len(lims))
    ra, dec = np.asarray(ra), np.asarray(dec)
    xlim, ylim, inequals = lims[region]

    selection = (xlim[0] < ra) & (ra <= xlim[1])
    selection &= (ylim[0] < dec) & (dec <= ylim[1])
    for x0, y0, m, c in inequals:
        selection &= m * (ra - x0) < c * (dec - y0)
    return selection
