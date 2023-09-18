# galtab
Galaxy tabulation: HOD counts-in-cells statistics with JAX

Documentation at https://galtab.readthedocs.org/

# Author
- Alan Pearl

# Prerequisites
- `python>=3.8`
- `mpi4py` with MPI backend (several implementations available - see https://pypi.org/project/mpi4py)
<!-- - At the time of writing, a prerequisite (`halotools>=0.8`) is incompatible with `python>=3.10`. -->
### Example to automatically install prerequisites using a conda environment:
```
conda create -n py310-galtab python=3.10
conda activate py310-galtab
conda install -c conda-forge mpi4py 
```
<!-- conda install -c conda-forge openmpi=4.1.4=ha1ae619_100  # (no longer needed, automatically installs with mpi4py on conda-forge) -->
### Optional dependencies to run scripts in the paper2 subpackage
- mocksurvey and pycorr (instructions at https://github.com/AlanPearl/mocksurvey)
- tabcorr (`pip install tabcorr`)
- emcee (`pip install emcee`)
- nautilus (`pip install nautilus-sampler`)
- uncertainties (`pip install uncertainties`)

# Installation
```
pip install galtab
```
