import argparse

import numpy as np
import tqdm
import pathlib

import galtab.obs
from galtab.paper2 import desi_sv3_pointings
from .param_config import cosmo, proj_search_radius, cylinder_half_length


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="desi_cic")
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser.add_argument(
        "-o", "--output", type=str, default="desi_cic.npy",
        help="Specify the output filename")
    parser.add_argument(
        "-p", "--progress", action="store_true",
        help="Show progress with tqdm")
    parser.add_argument(
        "-f", "--first-n", type=int, default=None, metavar="N",
        help="Run this code only on the first N data per region")
    parser.add_argument(
        "-r", "--first-regions", type=int, default=None, metavar="N",
        help="Run this code only on the first N regions")
    parser.add_argument(
        "--data-dir", type=str,
        default=pathlib.Path.home() / "data" / "DESI" / "SV3" / "clean_fuji",
        help="Directory containing the data (fastphot.npy file)")
    parser.add_argument(
        "-n", "--num-threads", type=int, default=1,
        help="Number of multiprocessing threads for each CiC process")
    parser.add_argument(
        "--force-no-mpi", action="store_true",
        help="Prevent even attempting to import the mpi4py module")
    parser.add_argument(
        "--zmax", type=float, default=0.3,
        help="Upper limit on redshift of the sample")
    parser.add_argument(
        "--logmmin", type=float, default=9.9,
        help="Lower limit on log stellar mass of the sample")
    parser.add_argument(
        "--abs-mr-max", type=float, default=np.inf,
        help="Upper limit on absolute R-mand magnitude (e.g. -19.5)")
    parser.add_argument(
        "--passive-evolved-mags", action="store_true",
        help="Apply Q=1.62 passive evolution for the M_R threshold cut")

    a = parser.parse_args()
    output_file = a.output
    progress = a.progress
    first_n = a.first_n
    first_regions = a.first_regions
    data_dir = pathlib.Path(a.data_dir)
    zmax = a.zmax
    logmmin = a.logmmin
    abs_mr_max = a.abs_mr_max
    num_threads = a.num_threads
    passive_evolved_mags = a.passive_evolved_mags

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

    def load_data(region_index):
        """Save only the data used for the analysis in a given region"""
        data = np.load(str(data_dir / "fastphot.npy"))
        if passive_evolved_mags:
            abs_mr = data["abs_rmag_0p1_evolved"]
        else:
            abs_mr = data["abs_rmag_0p1"]

        # Threshold cuts here: logm > logmmin, z < zmax, magnitude cut
        cut = data["Z"] <= zmax
        if logmmin > -np.inf:
            cut &= data["logmass"] >= logmmin
        if abs_mr_max < np.inf:
            cut &= abs_mr <= abs_mr_max

        data = data[cut]

        data = data[desi_sv3_pointings.select_region(
            region_index, data["RA"], data["DEC"])]

        ra = data["RA"]
        dec = data["DEC"]
        dist = cosmo.comoving_distance(data["Z"]).value * cosmo.h

        return np.array([ra, dec, dist]).T


    def job(job_index):
        """Define a job for each MPI process, split into 20 sky regions"""
        sample1 = sample2 = load_data(job_index)
        if first_n is not None:
            sample1 = sample1[:first_n]
        cic = galtab.obs.cic_obs_data(
            sample1, sample2, proj_search_radius, cylinder_half_length,
            progress=progress, tqdm_kwargs=dict(leave=False),
            num_threads=num_threads)
        return cic


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
