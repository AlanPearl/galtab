import os
import argparse

import numpy as np
from mpi4py import MPI
from astropy.io import fits
import tqdm

import galtab.obs
from galtab.paper2 import desi_sv3_pointings
import mocksurvey as ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="count_desi_randoms")
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser.add_argument(
        "-o", "--output", type=str, default="desi_rand_counts.npy",
        help="Specify the output filename")
    parser.add_argument(
        "-s", "--small-region", action="store_true",
        help="Test this code on only a small region of the data")
    parser.add_argument(
        "-p", "--progress", action="store_true",
        help="Show progress with tqdm")
    parser.add_argument(
        "-f", "--first-n", type=int, default=None, metavar="N",
        help="Run this code on the first N data only")
    parser.add_argument(
        "--rand-dir", type=str, default="/home/alan/data/DESI/SV3/rands_fuji/",
        help="Directory containing the randoms (rancomb_... files)")
    parser.add_argument(
        "--data-dir", type=str, default="/home/alan/data/DESI/SV3/",
        help="Directory containing the data (stellar_mass_specz_ztile... file)")
    parser.add_argument(
        "--num-rand-files", type=int, default=4,
        help="Number of random catalogs to concatenate (up to 18)")

    a = parser.parse_args()
    output_file = a.output
    make_small_region_cut = a.small_region
    progress = a.progress
    first_n = a.first_n
    rand_dir = a.rand_dir
    data_dir = a.data_dir
    num_rand_files = a.num_rand_files

    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()
    if comm_rank != 0:
        progress = False

    # Load in DESI data and corresponding randoms
    # ===========================================
    rand_cats = []
    for i in range(num_rand_files):

        randfile = os.path.join(
            rand_dir, f"rancomb_{i}brightwdupspec_Alltiles.fits")
        rands = fits.open(randfile)[1].data

        randcut = rands["ZWARN"] == 0
        if make_small_region_cut:
            randcut &= (233 < rands["RA"]) & (rands["RA"] < 238.5)
            randcut &= (41.5 < rands["DEC"]) & (rands["DEC"] < 45.3)

        rands = rands[randcut]
        rand_cats.append(rands)
    rands = np.concatenate(rand_cats)

    datafile = os.path.join(
        data_dir, "stellar_mass_specz_ztile-sv3-bright-cumulative.fits")
    data = fits.open(datafile)[1].data

    datacut = data["ZWARN"] == 0
    datacut &= data["Z"] > 0
    if make_small_region_cut:
        datacut &= (233 < data["TARGET_RA"]) & (data["TARGET_RA"] < 238.5)
        datacut &= (41.5 < data["TARGET_DEC"]) & (data["TARGET_DEC"] < 45.3)

    data = data[datacut]

    proj_search_radius = 2.0
    cylinder_half_length = 10.0

    cosmo = ms.bplcosmo
    cylinder_half_length = cylinder_half_length
    proj_search_radius = proj_search_radius

    ra = data["TARGET_RA"]
    dec = data["TARGET_DEC"]
    dist = ms.util.comoving_disth(data["Z"], cosmo)

    dist_max, dist_min = np.max(dist), np.min(dist)
    dist_center = (dist_max + dist_min)/2
    dist_range = (dist_max - dist_min)

    rand_ra = rands["RA"]
    rand_dec = rands["DEC"]
    rand_dist = np.full_like(rand_ra, dist_center)

    sample1 = np.array([ra, dec, dist]).T
    sample2 = np.array([rand_ra, rand_dec, rand_dist]).T


    def job(job_index):
        job_data = sample1[desi_sv3_pointings.select_region(
            job_index, sample1[:, 0], sample1[:, 1])]
        job_rands = sample2[desi_sv3_pointings.select_region(
            job_index, sample2[:, 0], sample2[:, 1])]
        if first_n is not None:
            job_data = job_data[:first_n]
        rands_in_cylinders = galtab.obs.cic_obs_data(
            job_data, job_rands, proj_search_radius,
            dist_range, progress=progress, tqdm_kwargs=dict(leave=False))
        return rands_in_cylinders

    num_jobs = len(desi_sv3_pointings.lims)
    job_assignments = np.arange(comm_rank, num_jobs, comm_size)
    if progress:
        job_assignments = tqdm.tqdm(job_assignments,
                                    desc="Completed pointings")

    job_results = [None for _ in range(num_jobs)]
    for i in job_assignments:
        job_results[i] = job(i)

    job_results_gathered = comm.allgather(job_results)
    if comm_rank == 0:
        job_results_reassembled = [job_results_gathered[i % comm_size][i]
                                   for i in range(num_jobs)]
        results = np.array(job_results_reassembled, dtype=object)

        # search_angles = (proj_search_radius / (dist - cylinder_half_length)) * 180 / np.pi
        # rand_density_in_cylinders = rands_in_cylinders / (np.pi * search_angles**2)
        np.save(output_file, results)
