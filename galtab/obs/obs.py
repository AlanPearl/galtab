import numpy as np
import fast3tree
import mocksurvey as ms
import tqdm
import multiprocessing
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

global _global_tree_xy, _global_counter_args


def _counter(pt):
    global _global_tree_xy, _global_counter_args
    tree_xy = _global_tree_xy
    (companions, r_cyl, cyl_half_length, weigh_companions, companion_weights,
     search_angle_at_near_end_of_cylinder,
     perform_additional_angle_selection_at_companion_dist
     ) = _global_counter_args

    if search_angle_at_near_end_of_cylinder:
        pt_dist = pt[2] - cyl_half_length
    else:
        pt_dist = pt[2]
    ang_radius = r_cyl / pt_dist  # radians
    idx_xy = tree_xy.query_radius(pt[:2], ang_radius / np.cos(pt[1]))
    cnt = 0
    for j in idx_xy:
        if perform_additional_angle_selection_at_companion_dist:
            ang_radius = r_cyl / companions[j, 2]
        if (np.abs(pt[2] - companions[j, 2]) < cyl_half_length and
           ((pt[0] - companions[j, 0]) * np.cos(pt[1])) ** 2 +
           (pt[1] - companions[j, 1]) ** 2 < ang_radius ** 2):
            # and not np.allclose(pt, self.companions[j], 0, 1e-7)):
            if weigh_companions:
                cnt += companion_weights[j]
            else:
                cnt += 1
    return cnt


def _counter_init():
    global _global_tree_xy, _global_counter_args

    companions = _global_counter_args[0]
    _global_tree_xy = fast3tree.fast3tree(companions[:, :2])
    _global_tree_xy.__enter__()
    return _global_tree_xy, _global_counter_args


def _counter_exit():
    global _global_tree_xy, _global_counter_args
    _global_tree_xy.free()
    _global_tree_xy = None
    _global_counter_args = None
    return _global_tree_xy, _global_counter_args


def cic_obs_data(centers, companions, r_cyl, cyl_half_length, cosmo=None,
                 weigh_companions=False, companion_weights=None,
                 weigh_counts=False, count_weights=None, progress=False,
                 search_angle_at_near_end_of_cylinder=False,
                 perform_additional_angle_selection_at_companion_dist=False,
                 num_threads=1, use_mpi=False, tqdm_kwargs=None):
    """
    Calculate counts-in-cylinders from observed celestial data
    Based off Kuan Wang's Ncic function

    Notes: All units must be in degrees and Mpc/h

    Self-counting is not removed, so make sure to subtract
    by 1 (or by the count_weights array) if desired

    `centers` and `companions` must be (N, 3) arrays with
    columns [ra, dec, redshift] if `cosmo` is specified.
    Otherwise, columns are [ra, dec, comoving_dist].

    `r_cyl` and `cyl_half_length` are comoving distances
    to search in the transverse and line-of-sight directions.

    `cosmo` is an astropy.cosmology.Cosmology object for
    converting redshift to distance.
    """
    global _global_tree_xy, _global_counter_args

    centers = np.array(centers)
    centers[:, 0] = np.radians(centers[:, 0])
    centers[:, 1] = np.radians(centers[:, 1])

    companions = np.array(companions)
    companions[:, 0] = np.radians(companions[:, 0])
    companions[:, 1] = np.radians(companions[:, 1])

    tqdm_default_kwargs = {"smoothing": 0.15, "total": len(centers)}
    if tqdm_kwargs is not None:
        tqdm_kwargs = {**tqdm_default_kwargs, **tqdm_kwargs}

    if cosmo is not None:
        centers[:, 2] = ms.util.comoving_disth(centers[:, 2], cosmo)
        companions[:, 2] = ms.util.comoving_disth(companions[:, 2], cosmo)

    _global_counter_args = (
        companions, r_cyl, cyl_half_length, weigh_companions,
        companion_weights, search_angle_at_near_end_of_cylinder,
        perform_additional_angle_selection_at_companion_dist)

    pool_class = pool_args = pool_kwargs = None
    if num_threads > 1:
        # if MPI.COMM_WORLD.Get_size() > 1:
        # if MPI.UNIVERSE_SIZE > 1:
        if use_mpi:
            # This is usually 1, but is 2 if I do mpiexec -n 2 ...
            print("world size =", MPI.COMM_WORLD.Get_size())
            # This is always 6.
            print("universe size =", MPI.UNIVERSE_SIZE)
            pool_class = MPIPoolExecutor
            pool_args = ()
            pool_kwargs = dict(initializer=_counter_init,
                               globals=dict(
                                   _global_tree_xy=None,
                                   _global_counter_args=None
                               ))
        else:
            pool_class = multiprocessing.Pool
            pool_args = (num_threads,)
            pool_kwargs = dict(initializer=_counter_init)

    if pool_class is None:
        _counter_init()
        iterator = centers
        if progress:
            iterator = tqdm.tqdm(iterator, **tqdm_kwargs)
        cnts = [_counter(point) for point in iterator]
    else:
        with pool_class(*pool_args, **pool_kwargs) as pool:
            if use_mpi:
                # future = pool.submit(_counter_init)
                # print(repr(future.result()))
                # assert future.done()
                if progress:
                    print("Sadly, I don't think it's possible to "
                          "make a tqdm progress bar with MPI pools :(")
                cnts = list(pool.map(_counter, centers))
            else:
                iterator = pool.imap(_counter, centers)
                if progress:

                    iterator = tqdm.tqdm(iterator, **tqdm_kwargs)
                cnts = list(iterator)

    # TODO: Move this into the multiprocessing Finalizer registry
    _counter_exit()
    if weigh_counts:
        cnts = np.array(cnts) * count_weights
    else:
        cnts = np.array(cnts)

    # To remove self-counting, subtract count_weights * companion_weights[center_indices]
    # In the case of no weights, simply subtract 1
    return cnts
