import argparse

import numpy as np
import scipy.special
import scipy.optimize
from tqdm import tqdm
import pathlib

import mocksurvey as ms

import galtab.obs
from galtab.paper2 import desi_sv3_pointings
from .param_config import cosmo, proj_search_radius, cylinder_half_length


class ObservableCalculator:
    def __init__(self, **kwargs):
        # Files and performance arguments
        self.data_dir = pathlib.Path(kwargs["data_dir"])
        self.rand_dir = pathlib.Path(kwargs["rand_dir"])
        self.num_rand_files = kwargs["num_rand_files"]
        self.num_threads = kwargs["num_threads"]
        self.progress = kwargs["progress"]
        self.wp_rand_frac = kwargs["wp_rand_frac"]
        self.verbose = kwargs["verbose"]
        self.first_n = kwargs["first_n"]
        self.apply_pip_weights_wp = kwargs["apply_pip_weights_wp"]
        self.apply_pip_weights_cic = kwargs["apply_pip_weights_cic"]
        # Cosmology, sample, and metric parameters
        self.cosmo = kwargs["cosmo"]
        self.zmin = kwargs["zmin"]
        self.zmax = kwargs["zmax"]
        self.logmmin = kwargs["logmmin"]
        self.abs_mr_max = kwargs["abs_mr_max"]
        self.passive_evolved_mags = kwargs["passive_evolved_mags"]
        self.rp_edges = kwargs["rp_edges"]
        self.pimax = kwargs["pimax"]
        self.cic_edges = kwargs["cic_edges"]
        self.proj_search_radius = kwargs["proj_search_radius"]
        self.cylinder_half_length = kwargs["cylinder_half_length"]
        self.effective_area_sqdeg = kwargs["effective_area_sqdeg"]

        # Load the data, prepare it, and calculate masks
        self.slice_n, self.slice_wp, self.slice_cic = None, None, None
        self.fastphot, self.rands, self.randcyl, self.region_masks, \
            self.mask_z, self.mask_thresh, self.rand_region_masks, \
            self.nrand_per_region = self.load_data()

        self.bitmasks = self.fastphot["bitweights"]
        self.numjack = len(self.region_masks)
        self.data_rdx, self.rands_rdx = self.make_rdx_arrays()
        self.randcyl_density, self.randcic_cut = self.calc_randcic_cut()
        self.randcic_mask = self.randcyl_density >= self.randcic_cut
        self.effective_area_sqdeg, self.effective_volume, \
            self.average_cylinder_completeness = self.calc_area_and_completeness()

    def __call__(self):
        n_iterator = range(self.numjack)
        wp_iterator = range(self.numjack)
        cic_iterator = range(self.numjack)
        if self.progress:
            n_iterator = tqdm(
                n_iterator, desc="Completed n jackknife samples")
            wp_iterator = tqdm(
                wp_iterator, desc="Completed wp jackknife samples")
            cic_iterator = tqdm(
                cic_iterator, desc="Completed CiC jackknife samples")

        n_jacks = np.array([self.jack_n(x) for x in n_iterator])
        wp_jacks = np.array([self.jack_wp(x) for x in wp_iterator])
        cic_jacks = [self.jack_cic(x) for x in cic_iterator]
        cic_lengths = [len(x) for x in cic_jacks]
        cic_pads = [max(cic_lengths) - x for x in cic_lengths]
        cic_jacks = np.array([np.pad(x, (0, y)) for (x, y)
                              in zip(cic_jacks, cic_pads)])

        self.slice_n = slice(0, 1)
        self.slice_wp = slice(self.slice_n.stop,
                              self.slice_n.stop + wp_jacks.shape[1])
        self.slice_cic = slice(self.slice_wp.stop,
                               self.slice_wp.stop + cic_jacks.shape[1])

        obs_jacks = np.vstack([n_jacks, wp_jacks.T, cic_jacks.T]).T
        obs_mean = np.mean(obs_jacks, axis=0)
        obs_cov = (self.numjack - 1)**2 / self.numjack * np.cov(
            obs_jacks, rowvar=False)
        return obs_mean, obs_cov

    def load_data(self):
        fastphot_fn = str(self.data_dir / "fastphot.npy")
        randcyl_fn = str(self.data_dir / "desi_rand_counts.npy")
        rand_fn = str(self.rand_dir / "rands.npy")

        fastphot = np.load(fastphot_fn)
        rands = np.load(rand_fn)
        randcyl = np.concatenate(np.load(randcyl_fn, allow_pickle=True))
        region_masks = [galtab.paper2.desi_sv3_pointings.select_region(
            i, fastphot["RA"], fastphot["DEC"]) for i in range(20)]

        mask_z = (self.zmin <= fastphot["Z"]) & (
                fastphot["Z"] <= self.zmax)
        mask_thresh = np.ones_like(mask_z)
        if self.logmmin > -np.inf:
            mask_thresh = fastphot["logmass"] >= self.logmmin
        elif self.abs_mr_max < np.inf:
            if self.passive_evolved_mags:
                abs_mr = fastphot["abs_rmag_0p1_evolved"]
            else:
                abs_mr = fastphot["abs_rmag_0p1"]
            mask_thresh = abs_mr <= self.abs_mr_max

        # We only want to use BGS_BRIGHT
        assert np.all(fastphot["SV3_BGS_TARGET"] & 2 != 0)

        rand_region_masks = [galtab.paper2.desi_sv3_pointings.select_region(
            i, rands["RA"], rands["DEC"]) for i in range(20)]
        nrand_per_region = np.array([np.sum(x) for x in rand_region_masks])
        return fastphot, rands, randcyl, region_masks, mask_z, mask_thresh, \
            rand_region_masks, nrand_per_region

    def make_rdx_arrays(self):
        """Make (N,3) arrays of (RA, DEC, comoving_dist*h)"""
        data_r = self.fastphot["RA"]
        data_d = self.fastphot["DEC"]
        data_x = ms.util.comoving_disth(self.fastphot["Z"], self.cosmo)

        rand_r = self.rands["RA"]
        rand_d = self.rands["DEC"]
        rand_x = ms.util.rand_rdz(
            len(rand_r), [-1, 1], [-1, 1],
            ms.util.comoving_disth([self.zmin, self.zmax], cosmo))[:, 2]

        data_rdx = np.array([data_r, data_d, data_x]).T
        rands_rdx = np.array([rand_r, rand_d, rand_x]).T
        return data_rdx, rands_rdx

    def calc_randcic_cut(self):
        angles = 180 / np.pi * galtab.obs.get_search_angle(
            proj_search_radius, cylinder_half_length, self.data_rdx[:, 2])
        randcyl_density = self.randcyl / (np.pi * angles ** 2)
        model = RandDensityModelCut(randcyl_density)
        randcic_cut = model.optimal_cut()
        if self.verbose:
            print("Optimal rand density cut:", randcic_cut)
        return randcyl_density, randcic_cut

    def calc_area_and_completeness(self):
        rands_per_sqdeg = 2500.0
        effective_area_sqdeg = self.effective_area_sqdeg
        if effective_area_sqdeg is None:
            num_rands_per_file = len(self.rands_rdx) / self.num_rand_files
            effective_area_sqdeg = num_rands_per_file / rands_per_sqdeg
        cut = (self.randcic_cut & self.mask_z)
        average_completeness = (self.randcyl_density[cut].mean() /
                                self.num_rand_files / rands_per_sqdeg)
        if self.verbose:
            print("Effective area =", effective_area_sqdeg, "sq deg")
            print("Average cylinder completeness =", average_completeness)

        effective_volume = ms.util.volume(
            effective_area_sqdeg, [0.1, 0.25], self.cosmo)

        return effective_area_sqdeg, effective_volume, average_completeness

    def jack_cic(self, njack):
        non_spatial_cut = (self.mask_thresh & ~self.region_masks[njack])
        sample_cut = self.mask_z & self.randcic_mask & non_spatial_cut
        sample2 = self.data_rdx[non_spatial_cut]
        sample1 = self.data_rdx[sample_cut]
        if self.first_n is not None:
            sample1 = sample1[:self.first_n]
        cic, indices = galtab.obs.cic_obs_data(
            sample1, sample2, self.proj_search_radius,
            self.cylinder_half_length, return_indices=True,
            progress=self.progress, tqdm_kwargs=dict(leave=False),
            num_threads=self.num_threads)
        return self.bin_raw_cic_counts(cic, indices, sample_cut)

    def jack_wp(self, njack):
        if self.apply_pip_weights_wp:
            raise NotImplementedError()
        sample = self.data_rdx[self.mask_thresh & self.mask_z &
                               ~self.region_masks[njack]]
        rands = self.rands_rdx[~self.rand_region_masks[njack]]
        wp = ms.cf.wp_rp(
            sample, rands, self.rp_edges, self.pimax, is_celestial_data=True)
        return wp

    def jack_n(self, njack):
        jack_vol_factor = (1 - self.nrand_per_region[njack] /
                           np.sum(self.nrand_per_region))
        n_sample = np.sum(self.mask_thresh & self.mask_z &
                          ~self.region_masks[njack])
        return n_sample / self.effective_volume / jack_vol_factor

    def bin_raw_cic_counts(self, cic, indices, sample_cut):
        iip_weights = None
        if self.apply_pip_weights_cic:
            bitmasks = self.bitmasks[sample_cut]
            numbits = bitmasks.shape[1] * bitmasks.itemsize * 8
            iip_weights = numbits / np.sum(
                ms.util.bitsum_hamming_weight(bitmasks), axis=1)
            counts = []
            for i in range(len(bitmasks)):
                pip_weights = numbits / np.sum(ms.util.bitsum_hamming_weight(
                    bitmasks[i] & bitmasks[indices[i]]), axis=1)
                counts.append(pip_weights)
            cic = counts

        if self.cic_edges is None:
            return np.bincount(cic)
        elif self.apply_pip_weights_cic:
            np.histogram(cic, bins=self.cic_edges, weights=iip_weights)
        else:
            np.histogram(cic, bins=self.cic_edges)


class RandDensityModelCut:
    # Functions controlling the norm + erf-tail model, and its purity/completeness
    def __init__(self, randcyl_density):
        samples = randcyl_density
        bins = np.linspace(np.median(samples) / 2, np.median(samples) * 2, 300)
        self.bin_cens = 0.5 * (bins[:-1] + bins[1:])
        self.hist = np.histogram(samples, bins=bins)[0]
        self.p0 = [np.median(samples) * 1.1,
                   np.std(samples) / 3,
                   np.max(self.hist) * 0.8,
                   np.max(self.hist) / 5,
                   np.std(samples) / 10,
                   0]
        self.bounds = ([0, 0, 0, 0, 0, -10],
                       [np.max(bins),
                        np.max(bins) - np.min(bins),
                        np.max(self.hist) * 1.5,
                        np.max(self.hist) * 0.5,
                        np.max(bins) - np.min(bins),
                        10])
        self.bestp, self.bestp_cov = scipy.optimize.curve_fit(
            self.model_pdf, self.bin_cens, self.hist, p0=self.p0, bounds=self.bounds)

    def optimal_cut(self):
        comp = np.array([self.model_selection_completeness(x, *self.bestp) for x in self.bin_cens])
        purt = np.array([self.model_selection_purity(x, *self.bestp) for x in self.bin_cens])
        opt_arg = np.argmin((1 - comp) ** 2 + (5 - 5 * purt) ** 2)
        opt_cut = self.bin_cens[opt_arg]
        return opt_cut

    @staticmethod
    def model_pdf_norm_component(x, truth, std, true_mag):
        return true_mag * np.exp(-0.5 * ((x-truth)/std)**2)

    @staticmethod
    def model_pdf_erf_component(x, truth, tail_mag, tail_std, tail_slope):
        y_intercept = (1 + tail_slope) * tail_mag
        # line between two points: (0, y_intercept) and (truth, tail_mag)
        sloped_tail_mag = np.abs(y_intercept + (tail_mag - y_intercept) / truth * x)
        return sloped_tail_mag * 0.5*(1 - scipy.special.erf((x - truth - tail_std)/tail_std))

    def model_pdf(self, x, truth, std, true_mag, tail_mag, tail_std, tail_slope):
        """Model: true_mag * N(truth, std) + sloped_tail_mag * (0.5 - erf(x-truth))"""
        ans = self.model_pdf_norm_component(x, truth, std, true_mag)
        ans += self.model_pdf_erf_component(x, truth, tail_mag, tail_std, tail_slope)
        return ans

    def model_selection_purity(self, x, truth, std, true_mag, tail_mag, tail_std, tail_slope):
        """Model: true_mag * N(truth, std) + sloped_tail_mag * (0.5 - erf(x-truth))"""
        good_hist = self.model_pdf_norm_component(self.bin_cens, truth, std, true_mag)
        bad_hist = self.model_pdf_erf_component(self.bin_cens, truth, tail_mag, tail_std, tail_slope)
        tot_good = good_hist[self.bin_cens >= x].sum()
        tot_bad = bad_hist[self.bin_cens >= x].sum()
        tot = tot_good + tot_bad
        assert tot >= tot_bad
        assert tot >= tot_good
        return tot_good / tot

    def model_selection_completeness(self, x, truth, std, true_mag, *_args):
        """Model: true_mag * N(truth, std) + sloped_tail_mag * (0.5 - erf(x-truth))"""
        good_hist = self.model_pdf_norm_component(self.bin_cens, truth, std, true_mag)
        return good_hist[self.bin_cens >= x].sum() / good_hist.sum()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="desi_cic")
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser.add_argument(
        "-o", "--output", type=str, default="desi_cic.npy",
        help="Specify the output filename")
    parser.add_argument(
        "-p", "--progress", action="store_true",
        help="Show progress with tqdm")
    parser.add_argument(
        "-f", "--first-n", type=int, default=None, metavar="N",
        help="Run this code only on the first N data per region")
    parser.add_argument(
        "-r", "--first-regions", type=int, default=None, metavar="N",
        help="Run this code only on the first N regions")
    parser.add_argument(
        "--data-dir", type=str,
        default=pathlib.Path.home() / "data" / "DESI" / "SV3" / "clean_fuji",
        help="Directory containing the data (fastphot.npy file)")
    parser.add_argument(
        "-n", "--num-threads", type=int, default=1,
        help="Number of multiprocessing threads for each CiC process")
    parser.add_argument(
        "--force-no-mpi", action="store_true",
        help="Prevent even attempting to import the mpi4py module")
    parser.add_argument(
        "--zmin", type=float, default=0.1,
        help="Lower limit on redshift of the sample"
    )
    parser.add_argument(
        "--zmax", type=float, default=0.2,
        help="Upper limit on redshift of the sample")
    parser.add_argument(
        "--logmmin", type=float, default=-np.inf,
        help="Lower limit on log stellar mass of the sample")
    parser.add_argument(
        "--abs-mr-max", type=float, default=np.inf,
        help="Upper limit on absolute R-mand magnitude (e.g. -19.5)")
    parser.add_argument(
        "--passive-evolved-mags", action="store_true",
        help="Apply Q=1.62 passive evolution for the M_R threshold cut")

    a = parser.parse_args()
    output_file = a.output

    calc = ObservableCalculator(**a.__dict__)
    mean, cov = calc()

    np.savez(output_file,
             mean=mean,
             cov=cov,
             slice_n=calc.slice_n,
             slice_wp=calc.slice_wp,
             slice_cic=calc.slice_cic,
             average_cylinder_completeness=calc.average_cylinder_completeness,
             effective_area_sqdeg=calc.effective_area_sqdeg,
             effective_volume=calc.effective_volume)
