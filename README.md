# galtab
Use JAX-accelerated galaxy tabulation to compute HOD counts-in-cells statistics

- Install: `pip install galtab`
- Read the docs: https://galtab.readthedocs.org
- To cite `galtab`, learn more implementation details, and explore an example science use case, check out https://arxiv.org/abs/2309.08675.
- Source code: https://github.com/AlanPearl/galtab

# Author
- Alan Pearl

# Prerequisites
- `python>=3.9`
- `jax`
### Example to automatically install prerequisites using a conda environment:
```
conda create -n py39 python=3.9 jax
conda activate py39 
```
<!-- conda install -c conda-forge openmpi=4.1.4=ha1ae619_100  # (no longer needed, automatically installs with mpi4py on conda-forge) -->
### Optional dependencies for the [paper2](https://github.com/AlanPearl/galtab/tree/main/galtab/paper2) subpackage
- mocksurvey and pycorr (instructions at https://github.com/AlanPearl/mocksurvey)
- tabcorr (`pip install tabcorr`)
- emcee (`pip install emcee`)
- nautilus (`pip install nautilus-sampler`)
- uncertainties (`pip install uncertainties`)
- mpi4py (`conda install -c conda-forge mpi4py`)
  - with MPI backend (several implementations available - see https://pypi.org/project/mpi4py)
