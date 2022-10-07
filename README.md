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
conda install -c conda-forge mpi4py openmpi
```

# Installation
```
pip install --upgrade git+https://github.com/AlanPearl/galtab.git
```
