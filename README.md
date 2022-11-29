# galtab
A general approach to tabulating HOD statistics

# Author
- Alan Pearl

# Prerequisites
- `python=3.9.*`
- MPI (several implementations available - see https://pypi.org/project/mpi4py)
- At the time of writing, a prerequisite (`halotools>=0.8`) is incompatible with `python>=3.10`.
### Example to automatically install prerequisites using a conda environment:
```
conda create -n py39-galtab python=3.9
conda activate py39-galtab
conda install -c conda-forge mpi4py openmpi=4.1.4=ha1ae619_100
```
### Optional dependencies to run scripts in the paper2 subpackage
- mocksurvey and pycorr (instructions at https://github.com/AlanPearl/mocksurvey)
- tabcorr (`pip install tabcorr`)
- emcee (`pip install emcee`)
- nautilus (`pip install nautilus-sampler`)
- uncertainties (`pip install uncertainties`)

# Installation
```
pip install --upgrade git+https://github.com/AlanPearl/galtab.git
```
