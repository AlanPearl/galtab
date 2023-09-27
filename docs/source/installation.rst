Installation Instructions
=========================

Installation
------------
``pip install galtab``

Prerequisites
-------------
- Tested on Python versions: ``python=3.9`` and ``python=3.10``
- MPI (several implementations available - see https://pypi.org/project/mpi4py)

Example installation with conda env:
++++++++++++++++++++++++++++++++++++

.. code-block:: bash

    conda create -n py39 python=3.9
    conda activate py39
    conda install -c conda-forge mpi4py

Optional dependencies for ``galtab.paper2`` subpackage
++++++++++++++++++++++++++++++++++++++++++++++++++++++

Install the following dependencies in order to reproduce `Pearl et al. (2023)`_.

- mocksurvey and pycorr (instructions at https://github.com/AlanPearl/mocksurvey)
- tabcorr (``pip install tabcorr``)
- emcee (``pip install emcee``)
- nautilus (``pip install nautilus-sampler``)
- uncertainties (``pip install uncertainties``)
- corner (``pip install corner``)

.. _Pearl et al. (2023): https://arxiv.org/abs/2309.08675
