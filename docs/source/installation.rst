Installation Instructions
=========================

Installation
------------
``pip install --upgrade git+https://github.com/AlanPearl/galtab.git``

Prerequisites
-------------
- Tested on Python versions: ``python=3.9`` and ``python=3.10``
- MPI (several implementations available - see https://pypi.org/project/mpi4py)

Example installation with conda env:
++++++++++++++++++++++++++++++++++++

.. code-block:: console

    conda create -n py310-galtab python=3.10
    conda activate py310-galtab
    conda install -c conda-forge mpi4py  # openmpi=4.1.4=ha1ae619_100

Optional dependencies for ``galtab.paper2`` subpackage
++++++++++++++++++++++++++++++++++++++++++++++++++++++

Install the following dependencies in order to reproduce Pearl et al. (2023)
[link to paper coming soon]

- mocksurvey and pycorr (instructions at https://github.com/AlanPearl/mocksurvey)
- tabcorr (``pip install tabcorr``)
- emcee (``pip install emcee``)
- nautilus (``pip install nautilus-sampler``)
- uncertainties (``pip install uncertainties``)
- corner (``pip install corner``)