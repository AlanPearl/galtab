from mpi4py import MPI
import numpy as np
import argparse
import tqdm
import time

parser = argparse.ArgumentParser(prog="test_mpi")
parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

parser.add_argument(
    "-p", "--progress", action="store_true")
parser.add_argument(
    "-s", "--save", action="store_true")
parser.add_argument(
    "-o", "--outfile", type=str, default="out_array.npy")
parser.add_argument(
    "-w", "--wait-time", type=float, default=0.0)

a = parser.parse_args()
progress = a.progress
save = a.save
outfile = a.outfile
wait_time = a.wait_time


def job(job_index):
    a_size = 1 + job_index
    a_start = job_index * a_size
    a_stop = a_start + a_size
    arr = np.arange(a_start, a_stop)
    if progress:
        arr = tqdm.tqdm(arr, leave=False)
    ans = []
    for a_i in arr:
        time.sleep(wait_time)
        ans.append(a_i)
    return ans


comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()
num_jobs = 10

if comm_rank != 0:
    progress = False

job_assignments = np.arange(comm_rank, num_jobs, comm_size)
print(f"Worker {comm_rank} performing jobs: {job_assignments}")
if progress:
    job_assignments = tqdm.tqdm(job_assignments)

job_results = [None for _ in range(num_jobs)]
for i in job_assignments:
    job_results[i] = job(i)

print(f"Worker {comm_rank} job results: {job_results}")
job_results_gathered = comm.allgather(job_results)
if comm_rank == 0:
    job_results_reassembled = [job_results_gathered[i % comm_size][i]
                               for i in range(num_jobs)]
    results = np.concatenate(job_results_reassembled)
    print(results)
    if save:
        np.save(outfile, results)
