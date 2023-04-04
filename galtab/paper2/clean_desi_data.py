import argparse
import glob
import json

import numpy as np
import pandas as pd
from astropy.io import fits
import pathlib

from galtab.paper2 import desi_sv3_pointings
import mocksurvey as ms


def clean_data():
    meta = fits.open(data_dir / fastphot_filename)[2].data
    fastphot = fits.open(data_dir / fastphot_filename)[1].data
    biprateep_masses = fits.open(data_dir / biprateep_filename)[1].data
    clustering_cat_n = fits.open(data_dir / clustering_cat_n_filename)[1].data
    clustering_cat_s = fits.open(data_dir / clustering_cat_s_filename)[1].data
    clustering_cat = np.concatenate([clustering_cat_n, clustering_cat_s])

    # Remove bad data
    # JK, all necessary cuts are applied to the clustering catalogs
    # and this cut would throw out more data than we want
    # ===============
    # meta_cut = (  # (meta["ZWARN"] == 0) &
    #             (meta["Z"] > 0) &
    #             (meta["SPECTYPE"] == "GALAXY") & (meta["DELTACHI2"] > 25))
    # meta = meta[meta_cut]
    # fastphot = fastphot[meta_cut]

    # Remove BGS_FAINT (keep BGS_BRIGHT)
    # ==================================
    bright_bit = 1
    bgs_bright_cut = meta["SV3_BGS_TARGET"] & 2 ** bright_bit != 0
    meta = meta[bgs_bright_cut]
    fastphot = fastphot[bgs_bright_cut]

    # Merge catalogs by TARGETID and remove duplicates
    # ================================================
    meta_df = pd.DataFrame(
        {"target_id": meta["TARGETID"].astype(np.int64),
         "meta_ind": np.arange(len(meta))}
    )
    biprateep_df = pd.DataFrame(
        {"target_id": biprateep_masses["TARGETID"].astype(np.int64),
         "biprateep_ind": np.arange(len(biprateep_masses))}
    )
    merged_df = pd.merge(meta_df, biprateep_df, on="target_id"
                         ).drop_duplicates("target_id")

    clustering_df = pd.DataFrame(
        {"target_id": clustering_cat["TARGETID"].astype(np.int64),
         "clustering_ind": np.arange(len(clustering_cat))}
    )
    merged_df = pd.merge(merged_df, clustering_df, on="target_id"
                         ).drop_duplicates("target_id")

    meta = meta[merged_df["meta_ind"].values]
    fastphot = fastphot[merged_df["meta_ind"].values]
    biprateep_masses = biprateep_masses[merged_df["biprateep_ind"].values]
    clustering_cat = clustering_cat[merged_df["clustering_ind"].values]

    # Add only the necessary data from fastphot and biprateep_masses
    # ==============================================================
    # No need to subtract KCORR_SDSS_R -- it's already k-corrected to z=0.1
    # Store M_R - 5log(h) (i.e. h=1, but FastSpecFit used h=0.7)
    abs_rmag_0p1 = fastphot["ABSMAG_SDSS_R"] - 5 * np.log10(0.7)

    # New column using passive evolution to z=0.1
    # q = 1.62  # table 3 - Blanton+ 2003 (The Galaxy Luminosity Function...)
    # abs_rmag_0p1_evolved = abs_rmag_0p1 + q * (meta["Z"] - 0.1)

    # Parameters for evolution correction to match Kuan (new version)
    q0, qz0, q1 = 2, 0.1, -1
    ec = q0 * (1 + q1 * (meta["Z"] - qz0)) * (meta["Z"] - qz0)
    abs_rmag_0p1_evolved = abs_rmag_0p1 + ec

    # Empirical correction for the Petrosian magnitudes Kuan used
    petro_mag_corr = 0.23 * meta["Z"] + 0.10
    abs_rmag_0p1_kuan = abs_rmag_0p1_evolved + petro_mag_corr

    # Convert bitweights from signed to unsigned ints to make myself happy
    # ====================================================================
    unsigned_dtype = clustering_cat["BITWEIGHTS"].dtype.descr[0][1]
    unsigned_dtype = unsigned_dtype.replace("i", "u")
    bitweights = clustering_cat["BITWEIGHTS"].astype(unsigned_dtype)
    weights = clustering_cat["WEIGHT"]

    # Put all data into a single catalog
    # ==================================
    names = list(meta.dtype.names) + ["abs_rmag_0p1",
                                      "abs_rmag_0p1_evolved",
                                      "abs_rmag_0p1_kuan",
                                      "logmass",
                                      "bitweights",
                                      "weights"]
    values = [meta[name] for name in meta.dtype.names] + \
             [abs_rmag_0p1,
              abs_rmag_0p1_evolved,
              abs_rmag_0p1_kuan,
              biprateep_masses["logmass"],
              bitweights,
              weights]
    dtypes = [value.dtype.descr[0][1] for value in values]
    subshapes = [value.shape[1:] for value in values]
    # names[names.index("RA")] = "TARGET_RA"
    # names[names.index("DEC")] = "TARGET_DEC"
    meta = ms.util.make_struc_array(names, values, dtypes, subshapes)

    del fastphot
    del biprateep_masses
    del clustering_cat

    # Order data by region index (and remove those not in a region)
    # ==========================
    n_region = len(desi_sv3_pointings.lims)
    region_masks = [desi_sv3_pointings.select_region(
        i, meta["RA"], meta["DEC"]) for i in range(n_region)]

    meta = np.concatenate([meta[mask] for mask in region_masks])

    return meta


def clean_rands(n_rand_files=None):
    def filename_n_i(i):
        return f"BGS_BRIGHT_N_{i}_clustering.ran.fits"

    def filename_s_i(i):
        return f"BGS_BRIGHT_S_{i}_clustering.ran.fits"

    n_rand_n_files = n_rand_s_files = n_rand_files
    if n_rand_files is None:
        n_rand_n_files = len(glob.glob(str(rand_dir / filename_n_i("*"))))
        n_rand_s_files = len(glob.glob(str(rand_dir / filename_s_i("*"))))
        assert n_rand_n_files == n_rand_s_files
        n_rand_files = n_rand_n_files
    rand_n_files = [rand_dir / filename_n_i(i) for i in range(n_rand_n_files)]
    rand_s_files = [rand_dir / filename_s_i(i) for i in range(n_rand_s_files)]
    rands = np.concatenate([fits.open(x)[1].data
                            for x in rand_n_files + rand_s_files])

    # Remove "bad" data (these are apparently unnecessary or already done)
    # =================
    # rands = rands[rands["ZWARN"] == 0]
    # rands = rands[rands["COADD_FIBERSTATUS"] == 0]

    # Remove duplicates (the BGS_BRIGHT randoms already did this)
    # =================
    # unique_inds = np.unique(rands["TARGETID"], return_index=True)[1]
    # rands = rands[unique_inds]

    rands_meta_dict = {"num_rand_files": n_rand_files}
    return rands, rands_meta_dict


default_data_dir = pathlib.Path.home() / "data" / "DESI" / "SV3"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="clean_desi_data")
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser.add_argument(
        "--data-dir", type=str, default=default_data_dir,
        help="Directory containing all data files"
    )
    parser.add_argument(
        "--rand-dir", type=str, default=default_data_dir / "rands_fuji",
        help="Directory containing all randoms files"
    )
    parser.add_argument(
        "--num-rand-files", type=int, default=None,
        help="Specify the number of random files to combine"
    )
    parser.add_argument(
        "--everest", action="store_true",
        help="Overrides output-dirname to 'clean_fuji'"
    )
    parser.add_argument(
        "--fastphot-filename", type=str,
        default="fastphot-fuji-sv3-bright.fits",
        help="Filename of the fastphot catalog"
    )
    parser.add_argument(
        "--biprateep-mass-cat-filename", type=str,
        default="stellar_mass_specz_ztile-sv3-bright-cumulative.fits",
        help="Filename of Biprateep's stellar mass catalog"
    )
    parser.add_argument(
        "--clustering-cat-n-filename", type=str,
        default="BGS_BRIGHT_N_clustering.dat.fits",
        help="Filename of the catalog containing fiber assignment bitweights"
    )
    parser.add_argument(
        "--clustering-cat-s-filename", type=str,
        default="BGS_BRIGHT_S_clustering.dat.fits",
        help="Filename of the catalog containing fiber assignment bitweights"
    )
    parser.add_argument(
        "--output-dirname", type=str, default="clean_fuji",
        help="Directory to save the clean data in"
    )

    a = parser.parse_args()
    num_rand_files = a.num_rand_files
    data_dir = pathlib.Path(a.data_dir)
    rand_dir = pathlib.Path(a.rand_dir)
    output_dirname = a.output_dirname
    if a.everest:
        output_dirname = "clean_everest"
    output_dir = pathlib.Path(a.data_dir) / output_dirname
    out_rand_dir = output_dir / "rands"
    fastphot_filename = a.fastphot_filename
    biprateep_filename = a.biprateep_mass_cat_filename
    clustering_cat_n_filename = a.clustering_cat_n_filename
    clustering_cat_s_filename = a.clustering_cat_s_filename

    cleaned_fastphot = clean_data()
    cleaned_rands, rands_meta = clean_rands(n_rand_files=num_rand_files)
    output_dir.mkdir(exist_ok=False)
    out_rand_dir.mkdir()
    np.save(str(output_dir / "fastphot.npy"), cleaned_fastphot)
    np.save(str(out_rand_dir / "rands.npy"), cleaned_rands)
    with open(str(out_rand_dir / "rands_meta.json"), "w") as f:
        json.dump(rands_meta, f, indent=4)
