import numpy as np
import fast3tree
import tqdm
import multiprocessing

global _global_tree_xy, _global_counter_args


def _counter(pt):
    global _global_tree_xy, _global_counter_args
    tree_xy = _global_tree_xy
    (companions, r_cyl, cyl_half_length, weigh_companions, companion_weights,
     infinite_distance, search_angle_at_near_end_of_cylinder,
     perform_additional_angle_selection_at_companion_dist,
     return_indices) = _global_counter_args

    if search_angle_at_near_end_of_cylinder:
        pt_dist = pt[2] - cyl_half_length
    else:
        pt_dist = pt[2]
    # ang_radius = r_cyl / pt_dist
    ang_radius = get_search_angle(r_cyl, cyl_half_length, pt_dist)
    idx_xy = tree_xy.query_radius(pt[:2], ang_radius / np.cos(pt[1]))
    cnt = 0
    indices = []
    for j in idx_xy:
        if perform_additional_angle_selection_at_companion_dist:
            ang_radius = r_cyl / companions[j, 2]
        dist_check = True
        if not infinite_distance:
            dist_check = np.abs(pt[2] - companions[j, 2]) < cyl_half_length
        if dist_check and (((pt[0] - companions[j, 0]) * np.cos(pt[1])) ** 2 +
                           (pt[1] - companions[j, 1]) ** 2 < ang_radius ** 2):
            # and not np.allclose(pt, self.companions[j], 0, 1e-7)):
            indices.append(j)
            if weigh_companions:
                cnt += companion_weights[j]
            else:
                cnt += 1
    if return_indices:
        return cnt, indices
    else:
        return cnt


def _counter_init():
    global _global_tree_xy, _global_counter_args

    companions = _global_counter_args[0]
    _global_tree_xy = fast3tree.fast3tree(companions[:, :2])
    _global_tree_xy.__enter__()
    return _global_tree_xy, _global_counter_args


def _counter_exit():
    global _global_tree_xy, _global_counter_args
    try:
        _global_tree_xy.free()
        _global_tree_xy = None
        _global_counter_args = None
    except NameError:
        pass


def cic_obs_data(centers, companions, r_cyl, cyl_half_length, cosmo=None,
                 weigh_companions=False, return_indices=False,
                 companion_weights=None, weigh_counts=False,
                 count_weights=None, progress=False, infinite_distance=False,
                 search_angle_at_near_end_of_cylinder=False,
                 perform_additional_angle_selection_at_companion_dist=False,
                 num_threads=1, tqdm_kwargs=None):
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
    if tqdm_kwargs is None:
        tqdm_kwargs = {}
    tqdm_kwargs = {**tqdm_default_kwargs, **tqdm_kwargs}

    if cosmo is not None:
        centers[:, 2] = cosmo.comoving_distance(
            centers[:, 2]).value * cosmo.h
        companions[:, 2] = cosmo.comoving_distance(
            companions[:, 2]).value * cosmo.h

    _global_counter_args = (
        companions, r_cyl, cyl_half_length, weigh_companions,
        companion_weights, infinite_distance,
        search_angle_at_near_end_of_cylinder,
        perform_additional_angle_selection_at_companion_dist,
        return_indices)

    pool_class = pool_args = pool_kwargs = None
    if num_threads > 1:
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
            iterator = pool.imap(_counter, centers)
            if progress:

                iterator = tqdm.tqdm(iterator, **tqdm_kwargs)
            cnts = list(iterator)
    # TODO: Move this into the multiprocessing Finalizer registry
    _counter_exit()
    indices = None
    if return_indices:
        cnts, indices = zip(*cnts)

    if weigh_counts:
        cnts = np.array(cnts) * count_weights
    else:
        cnts = np.array(cnts)

    # To remove self-counting, subtract count_weights * companion_weights[center_indices]
    # In the case of no weights, simply subtract 1
    if return_indices:
        return cnts, np.array(indices, dtype=object)
    else:
        return cnts


def get_search_angle(r_cyl, cyl_half_length, point_dist):
    """Calculate the search angle in radians"""
    # volume = 2*pi*r_cyl^2*cyl_half_length = 2/3*pi*(1-cos(search_angle))*diff_r3
    diff_r3 = (point_dist + cyl_half_length)**3 - (point_dist - cyl_half_length)**3
    return np.arccos(1 - 3 * r_cyl**2 * cyl_half_length / diff_r3)


def fuzzy_histogram(x, centroids, weights=None):
    x = np.asarray(x)
    if weights is None:
        weights = np.ones_like(x, dtype=float)
    else:
        weights = np.asarray(weights)
    sorted_centroid_inds = np.argsort(centroids)
    sorted_centroids = np.asarray(centroids)[sorted_centroid_inds]
    ans = np.zeros_like(centroids, dtype=float)

    # Handle edge cases
    # =================
    mask_under = x <= sorted_centroids[0]
    mask_over = x >= sorted_centroids[-1]
    ans[0] = np.sum(weights[mask_under])
    ans[-1] = np.sum(weights[mask_over])

    # Handle non-edge cases
    # =====================
    x = x[~(mask_under | mask_over)]
    weights = weights[~(mask_under | mask_over)]
    bin_inds = np.digitize(x, sorted_centroids) - 1

    dist_left = x - sorted_centroids[bin_inds]
    dist_right = sorted_centroids[bin_inds + 1] - x
    bin_width = dist_left + dist_right
    weight_left = weights * dist_right / bin_width
    weight_right = weights * dist_left / bin_width

    np.add.at(ans, bin_inds, weight_left)
    np.add.at(ans, bin_inds + 1, weight_right)

    # Return centroids and corresponding counts to their original order
    # =================================================================
    orig_order_ans = np.zeros_like(ans)
    orig_order_ans[sorted_centroid_inds] = ans
    return orig_order_ans
