import argparse
import glob

import numpy as np
import pandas as pd
from astropy.io import fits
import pathlib

from galtab.paper2 import desi_sv3_pointings
import mocksurvey as ms


def clean_data():
    fastphot_filename = "fastphot-everest-sv3-bright.fits"
    desi_filename = "stellar_mass_specz_ztile-sv3-bright-cumulative.fits"

    meta = fits.open(data_dir / fastphot_filename)[2].data
    fastphot = fits.open(data_dir / fastphot_filename)[1].data
    desi = fits.open(data_dir / desi_filename)[1].data

    abs_rmag_rest = fastphot["ABSMAG_SDSS_R"] - fastphot["KCORR_SDSS_R"]

    # Put abs rest magnitudes into the fastphot (meta) catalog
    # ========================================================
    names = list(meta.dtype.names) + ["abs_rmag_rest"]
    values = [meta[name] for name in meta.dtype.names] + [abs_rmag_rest]
    dtypes = [x[1] for x in meta.dtype.descr] + [abs_rmag_rest.dtype.descr[0][1]]
    subshapes = [value.shape[1:] for value in values]
    names[names.index("RA")] = "TARGET_RA"
    names[names.index("DEC")] = "TARGET_DEC"
    meta = ms.util.make_struc_array(names, values, dtypes, subshapes)

    del fastphot
    del abs_rmag_rest

    # Remove bad data
    # ===============
    meta_cut = ((meta["ZWARN"] == 0) & (meta["Z"] > 0) &
                (meta["SPECTYPE"] == b"GALAXY") & (meta["DELTACHI2"] > 25))
    desi_cut = ((desi["ZWARN"] == 0) & (desi["Z"] > 0) &
                (desi["SPECTYPE"] == "GALAXY") & (desi["DELTACHI2"] > 25))
    meta = meta[meta_cut]
    desi = desi[desi_cut]

    # Remove duplicates
    # =================
    meta_df = pd.DataFrame(
        {"target_id": meta["TARGETID"].astype(np.int64),
         "meta_ind": np.arange(len(meta))}
    )
    desi_df = pd.DataFrame(
        {"target_id": desi["TARGETID"].astype(np.int64),
         "desi_ind": np.arange(len(desi))}
    )
    merged_df = pd.merge(meta_df, desi_df, on="target_id"
                         ).drop_duplicates("target_id")
    meta = meta[merged_df["meta_ind"].values]
    desi = desi[merged_df["desi_ind"].values]

    # Order data by region index (and remove those not in a region)
    # ==========================
    n_region = 20
    meta_region_masks = [desi_sv3_pointings.select_region(
        i, meta["TARGET_RA"], meta["TARGET_DEC"]) for i in range(n_region)]
    desi_region_masks = [desi_sv3_pointings.select_region(
        i, desi["TARGET_RA"], desi["TARGET_DEC"]) for i in range(n_region)]

    meta = np.concatenate([meta[mask] for mask in meta_region_masks])
    desi = np.concatenate([desi[mask] for mask in desi_region_masks])

    return meta, desi


def clean_rands(n_rand_files=None):
    if n_rand_files is None:
        n_rand_files = len(glob.glob(
            str(rand_dir / "rancomb_*brightwdupspec_Alltiles.fits")))
    rand_filenames = [f"rancomb_{i}brightwdupspec_Alltiles.fits"
                      for i in range(n_rand_files)]
    rand_files = [rand_dir / x for x in rand_filenames]
    rands = np.concatenate([fits.open(x)[1].data for x in rand_files])

    # Remove duplicates
    # =================
    unique_inds = np.unique(rands["TARGETID"], return_index=True)[1]
    rands = rands[unique_inds]

    # Remove "bad" data
    # =================
    rands = rands[rands["ZWARN"] == 0]

    return rands


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="clean_desi_data")
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser.add_argument(
        "--data-dir", type=str, default=pathlib.Path.cwd(),
        help="Directory containing all data files"
    )
    parser.add_argument(
        "--rand-dir", type=str, default=pathlib.Path.cwd() / "rands_fuji",
        help="Directory containing all randoms files"
    )
    parser.add_argument(
        "--output-dir", type=str, default=pathlib.Path.cwd() / "clean",
        help="Specify the output clean data directory"
    )
    parser.add_argument(
        "--num-rand-files", type=int, default=None,
        help="Specify the number of random files to combine"
    )

    a = parser.parse_args()
    num_rand_files = a.num_rand_files
    data_dir = pathlib.Path(a.data_dir)
    rand_dir = pathlib.Path(a.rand_dir)
    output_dir = pathlib.Path(a.output_dir)
    out_rand_dir = output_dir / "rands_fuji"

    cleaned_fastphot, cleaned_bipmass = clean_data()
    cleaned_rands = clean_rands(n_rand_files=num_rand_files)
    out_rand_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(output_dir / "fastphot.npy"), cleaned_fastphot)
    np.save(str(output_dir / "biprateep_masses.npy"), cleaned_bipmass)
    np.save(str(out_rand_dir / "rands.npy"), cleaned_rands)
