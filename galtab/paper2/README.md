# Instructions to run all calculations from the command line

(Full notebook for reproducing plots at: https://github.com/AlanPearl/galtab/tree/main/galtab/paper2/plots/PaperPlots.ipynb)

## Downloading SV3 Fuji Data

All clustering catalogs can be downloaded from https://data.desi.lbl.gov/desi/survey/catalogs/SV3/LSS/fuji/LSScats/EDAbeta/

NOTE: use `/global/cfs/cdirs/` instead of https://data.desi.lbl.gov/ on NERSC

### Clustering catalog data (north and south fields)
- `BGS_BRIGHT_N_clustering.dat.fits`
- `BGS_BRIGHT_S_clustering.dat.fits`
### Clustering catalog randoms (18 realizations, north and south fields)
- `BGS_BRIGHT_N_{i}_clustering.ran.fits` for `{i}` in 0-17
- `BGS_BRIGHT_S_{i}_clustering.ran.fits` for `{i}` in 0-17

The fastspecphot catalog I used can be downloaded from

**Warning**: If using a version of fastspecphot newer than ~Feb 2023, I think they switched their assumption from h=0.7 to h=1.0

## Data cleaning

Generate clean DESI data files with: `python -m galtab.paper2.clean_desi_data`

This generates new data files which have been "cleaned" of stars, z<0 spectra, duplicates, non BGS bright targets, and sources with DELTACHI2 < 25. All cleaned data is placed in `/home/alan/data/DESI/SV3/clean_fuji/` by default

## Counting Randoms in Cylinders (for sky-completeness masking)

```
job.sh (executed using `sbatch job.sh`) -> Output: desi_rand_counts.npy
---------------------------------------
#!/bin/bash
#SBATCH --qos=shared
#SBATCH --job-name=rand-cic
#SBATCH --constraint=haswell
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=0:40:00
#SBATCH --account=desi

# half the 64 total CPUs in a haswell node = 32 (2 CPUs per core => 16 physical cores)

srun -n 1 -c 16 --cpu-bind=cores python -m galtab.paper2.count_desi_randoms --progress --output desi_rand_counts.npy --data-dir /global/homes/a/apearl/data/clean_fuji/ --num-threads 16 --force-no-mpi
```

Download `desi_rand_counts.npy` and place it in `/home/alan/data/DESI/SV3/clean_fuji`

## Jackknife Observations (n + wp + CiC)

From this directory, `cd ../desi_observables/`

Perform all calculations with:
```
# Full CiC Distribution
# =====================
python -m galtab.paper2.desi_observables -vpn4 -o desi_obs_20p0.npz --abs-mr-max " -20.0" --wp-rand-frac 0.1
python -m galtab.paper2.desi_observables -vpn4 -o desi_obs_20p5.npz --abs-mr-max " -20.5" --wp-rand-frac 0.1
python -m galtab.paper2.desi_observables -vpn4 -o desi_obs_21p0.npz --abs-mr-max " -21.0" --wp-rand-frac 0.1
python -m galtab.paper2.desi_observables -vpn4 -o desi_obs_21p0_z0p2-0p3.npz --abs-mr-max " -21.0" --wp-rand-frac 0.1 --zmin 0.2 --zmax 0.3

# CiC Moments 1-5
# ===============
python -m galtab.paper2.desi_observables -vpn4 -o desi_obs_20p0_kmax5.npz --abs-mr-max " -20.0" --wp-rand-frac 0.1 --cic-kmax 5
python -m galtab.paper2.desi_observables -vpn4 -o desi_obs_20p5_kmax5.npz --abs-mr-max " -20.5" --wp-rand-frac 0.1 --cic-kmax 5
python -m galtab.paper2.desi_observables -vpn4 -o desi_obs_21p0_kmax5.npz --abs-mr-max " -21.0" --wp-rand-frac 0.1 --cic-kmax 5
python -m galtab.paper2.desi_observables -vpn4 -o desi_obs_21p0_z0p2-0p3_kmax5.npz --abs-mr-max " -21.0" --wp-rand-frac 0.1 --zmin 0.2 --zmax 0.3 --cic-kmax 5

# No CiC
# ======
python -m galtab.paper2.desi_observables -vpn4 -o desi_obs_20p0_kmax0.npz --abs-mr-max " -20.0" --wp-rand-frac 0.1 --cic-kmax 0
python -m galtab.paper2.desi_observables -vpn4 -o desi_obs_20p5_kmax0.npz --abs-mr-max " -20.5" --wp-rand-frac 0.1 --cic-kmax 0
python -m galtab.paper2.desi_observables -vpn4 -o desi_obs_21p0_kmax0.npz --abs-mr-max " -21.0" --wp-rand-frac 0.1 --cic-kmax 0
python -m galtab.paper2.desi_observables -vpn4 -o desi_obs_21p0_z0p2-0p3_kmax0.npz --abs-mr-max " -21.0" --wp-rand-frac 0.1 --zmin 0.2 --zmax 0.3 --cic-kmax 0

# cylinder_half_length = pimax = 40.0 (CiC Moments 1-5)
# =====================================================
python -m galtab.paper2.desi_observables --cylinder-half-length 40.0 --pimax 40.0 -vpn4 -o desi_obs_20p0_hl40_kmax5.npz --abs-mr-max  -20.0 --wp-rand-frac 0.1 --cic-kmax 5
python -m galtab.paper2.desi_observables --cylinder-half-length 40.0 --pimax 40.0 -vpn4 -o desi_obs_20p5_hl40_kmax5.npz --abs-mr-max  -20.5 --wp-rand-frac 0.1 --cic-kmax 5
python -m galtab.paper2.desi_observables --cylinder-half-length 40.0 --pimax 40.0 -vpn4 -o desi_obs_21p0_hl40_kmax5.npz --abs-mr-max  -21.0 --wp-rand-frac 0.1 --cic-kmax 5
python -m galtab.paper2.desi_observables --cylinder-half-length 40.0 --pimax 40.0 -vpn4 -o desi_obs_21p0_z0p2-0p3_hl40_kmax5.npz --abs-mr-max  -21.0 --wp-rand-frac 0.1 --zmin 0.2 --zmax 0.3 --cic-kmax 5

# For comparison to Kuan:
# =======================
python -m galtab.paper2.desi_observables -vpn4 -o desi_obs_20p0_kuansample.npz --abs-mr-max " -20.0" --wp-rand-frac 0.1 --kuan-mags --zmin 0.02 --zmax 0.106
python -m galtab.paper2.desi_observables -vpn4 -o desi_obs_21p0_kuansample.npz --abs-mr-max " -21.0" --wp-rand-frac 0.1 --kuan-mags --zmin 0.02 --zmax 0.159
```

## Testing the parameter importance of n, wp, and CiC moments

From this directory, `cd ../desi_results/`

Run `python -m galtab.paper2.importance --num-samples 10_000`

## Testing galtab accuracy vs runtime performance parameters

From this directory, `cd ../desi_results/`

Run `python -m galtab.paper2.accuracy_vs_runtime --num-gt-trials 25 --num-ht-trials 2500`

## Fitting HOD parameters

For example, from `~/Paper2Data/desi_results/21p0_results` (or replace `Paper2Data` with `paper2` on Osiris), run 3000 x 20 samples with:

`python -m galtab.paper2.param_sampler -v 3000 ../../desi_observations/desi_obs_20p0.npz .`

