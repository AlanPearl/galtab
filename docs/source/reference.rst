API reference
=============

``galtab``
----------

.. autoclass:: galtab.GalaxyTabulator
    :members:
    :special-members: __init__

.. autoclass:: galtab.CICTabulator
    :members:
    :special-members: __init__


``galtab.obs``
--------------

**Note:** this package must be loaded separately with ``import galtab.obs``

.. autofunction:: galtab.obs.cic_obs_data

.. autofunction:: galtab.obs.get_search_angle

.. autofunction:: galtab.obs.fuzzy_histogram

``galtab.moments``
------------------

.. autofunction:: galtab.moments.jit_sum_at

.. autofunction:: galtab.moments.moments_from_samples

.. autofunction:: galtab.moments.moments_from_binned_pmf

.. autoclass:: galtab.moments.BernoulliCumulantGenerator