import os
import argparse

import numpy as np
from astropy.io import fits
import astropy.cosmology
import tqdm

import galtab.obs
from galtab.paper2 import desi_sv3_pointings

cosmo = astropy.cosmology.Planck13
proj_search_radius = 2.0
cylinder_half_length = 10.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="count_desi_randoms")
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser.add_argument(
        "-o", "--output", type=str, default="desi_rand_counts.npy",
        help="Specify the output filename")
    parser.add_argument(
        "-p", "--progress", action="store_true",
        help="Show progress with tqdm")
    parser.add_argument(
        "-f", "--first-n", type=int, default=None, metavar="N",
        help="Run this code only on the first N data per region")
    parser.add_argument(
        "-r", "--first-regions", type=int, default=None, metavar="N",
        help="Run this code only on the first N regions"
    )
    parser.add_argument(
        "--rand-dir", type=str, default="/home/alan/data/DESI/SV3/rands_fuji/",
        help="Directory containing the randoms (rancomb_... files)")
    parser.add_argument(
        "--data-dir", type=str, default="/home/alan/data/DESI/SV3/",
        help="Directory containing the data (stellar_mass_specz_ztile... file)")
    parser.add_argument(
        "--num-rand-files", type=int, default=4,
        help="Number of random catalogs to concatenate (up to 18)")
    parser.add_argument(
        "-n", "--num-threads", type=int, default=1,
        help="Number of multiprocessing threads for each CiC process")
    parser.add_argument(
        "--force-no-mpi", action="store_true",
        help="Prevent even attempting to import the mpi4py module")

    a = parser.parse_args()
    output_file = a.output
    progress = a.progress
    first_n = a.first_n
    first_regions = a.first_regions
    rand_dir = a.rand_dir
    data_dir = a.data_dir
    num_rand_files = a.num_rand_files
    num_threads = a.num_threads

    if a.force_no_mpi:
        MPI, comm = None, None
        comm_rank, comm_size = 0, 1
    else:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        comm_rank = comm.Get_rank()
        comm_size = comm.Get_size()

    if comm_rank != 0:
        progress = False

    def load_data_and_rands(region_index):
        """Load in DESI data and corresponding randoms"""
        preprocessed_dir = os.path.join(
            data_dir, f"preprocess_with_{num_rand_files}_rand"
                      f"files_at_region_{region_index}")
        if not os.path.isdir(preprocessed_dir):
            preprocess_region(region_index, preprocessed_dir)
        data = np.load(os.path.join(preprocessed_dir, "data.npy"))
        rand = np.load(os.path.join(preprocessed_dir, "rand.npy"))

        return data, rand

    def preprocess_region(region_index, save_dir):
        """Save only the data used for the analysis in a given region"""
        rand_cats = []
        for i in range(num_rand_files):
            randfile = os.path.join(
                rand_dir, f"rancomb_{i}brightwdupspec_Alltiles.fits")
            rands = fits.open(randfile)[1].data

            randcut = rands["ZWARN"] == 0

            rands = rands[randcut]
            rand_cats.append(rands)
        rands = np.concatenate(rand_cats)

        datafile = os.path.join(
            data_dir, "stellar_mass_specz_ztile-sv3-bright-cumulative.fits")
        data = fits.open(datafile)[1].data

        datacut = data["ZWARN"] == 0
        datacut &= data["Z"] > 0

        data = data[datacut]

        data = data[desi_sv3_pointings.select_region(
            region_index, data["TARGET_RA"], data["TARGET_DEC"])]
        rands = rands[desi_sv3_pointings.select_region(
            region_index, rands["RA"], rands["DEC"])]

        ra = data["TARGET_RA"]
        dec = data["TARGET_DEC"]
        dist = cosmo.comoving_distance(data["Z"]).value * cosmo.h

        dist_max, dist_min = np.max(dist), np.min(dist)
        dist_center = (dist_max + dist_min) / 2

        rand_ra = rands["RA"]
        rand_dec = rands["DEC"]
        rand_dist = np.full_like(rand_ra, dist_center)

        sample1 = np.array([ra, dec, dist]).T
        sample2 = np.array([rand_ra, rand_dec, rand_dist]).T

        os.mkdir(save_dir)
        np.save(os.path.join(save_dir, "data.npy"), sample1)
        np.save(os.path.join(save_dir, "rand.npy"), sample2)


    def job(job_index):
        """Define a job for each MPI process, split into 20 sky regions"""
        job_data, job_rands = load_data_and_rands(job_index)
        dist_range = np.max(job_data[:, 2]) - np.min(job_data[:, 0])
        leave = len(job_assignments) < 2
        if first_n is not None:
            job_data = job_data[:first_n]
        rands_in_cylinders = galtab.obs.cic_obs_data(
            job_data, job_rands, proj_search_radius,
            dist_range, progress=progress, tqdm_kwargs=dict(leave=leave),
            num_threads=num_threads)
        return rands_in_cylinders


    num_jobs = len(desi_sv3_pointings.lims)
    if first_regions:
        assert first_regions <= num_jobs
        num_jobs = first_regions
    job_assignments = np.arange(comm_rank, num_jobs, comm_size)
    if progress:
        job_assignments = tqdm.tqdm(job_assignments,
                                    desc="Completed pointings")

    job_results = [None for _ in range(num_jobs)]
    for job_i in job_assignments:
        job_results[job_i] = job(job_i)

    if comm_size == 1:
        job_results_gathered = [job_results]
    else:
        job_results_gathered = comm.allgather(job_results)

    if comm_rank == 0:
        job_results_reassembled = [job_results_gathered[i % comm_size][i]
                                   for i in range(num_jobs)]
        results = np.array(job_results_reassembled, dtype=object)

        # search_angles = (proj_search_radius / (dist - cylinder_half_length)) * 180 / np.pi
        # rand_density_in_cylinders = rands_in_cylinders / (np.pi * search_angles**2)
        np.save(output_file, results)
