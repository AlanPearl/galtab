import argparse
import glob

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

    # Remove duplicates
    # =================
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
    meta = meta[merged_df["meta_ind"].values]
    fastphot = fastphot[merged_df["meta_ind"].values]
    biprateep_masses = biprateep_masses[merged_df["biprateep_ind"].values]

    # Add only the necessary data from fastphot and biprateep_masses
    # ==============================================================
    # No need to subtract KCORR_SDSS_R - already k-corrected to z=0.1
    h = 0.7  # Use h-scaled magnitude (h=0.7 hard-coded into FastSpecFit)
    abs_rmag_0p1 = fastphot["ABSMAG_SDSS_R"] - 5 * np.log10(h)

    # New column using passive evolution to z=0.1
    q = 1.62  # table 3 - Blanton+ 2003 (The Galaxy Luminosity Function...)
    abs_rmag_0p1_evolved = abs_rmag_0p1 + q * (meta["Z"] - 0.1)

    # Put all data into a single catalog
    # ==================================
    names = list(meta.dtype.names) + ["abs_rmag_0p1",
                                      "abs_rmag_0p1_evolved",
                                      "logmass"]
    values = [meta[name] for name in meta.dtype.names] + \
             [abs_rmag_0p1,
              abs_rmag_0p1_evolved,
              biprateep_masses["logmass"]]
    dtypes = [value.dtype.descr[0][1] for value in values]
    subshapes = [value.shape[1:] for value in values]
    # names[names.index("RA")] = "TARGET_RA"
    # names[names.index("DEC")] = "TARGET_DEC"
    meta = ms.util.make_struc_array(names, values, dtypes, subshapes)

    del fastphot
    del biprateep_masses

    # Remove bad data
    # ===============
    meta_cut = ((meta["ZWARN"] == 0) & (meta["Z"] > 0) &
                (meta["SPECTYPE"] == "GALAXY") & (meta["DELTACHI2"] > 25))
    meta = meta[meta_cut]

    # Order data by region index (and remove those not in a region)
    # ==========================
    n_region = 20
    meta_region_masks = [desi_sv3_pointings.select_region(
        i, meta["RA"], meta["DEC"]) for i in range(n_region)]

    meta = np.concatenate([meta[mask] for mask in meta_region_masks])

    return meta


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
        "--output-dir", type=str, default=pathlib.Path.cwd(),
        help="Specify the output clean data directory"
    )
    parser.add_argument(
        "--num-rand-files", type=int, default=None,
        help="Specify the number of random files to combine"
    )
    parser.add_argument(
        "--everest", action="store_true",
        help="Name the clean output directory 'everest' instead of 'fuji'"
    )
    parser.add_argument(
        "--fastphot-filename", type=str,
        default="fastphot-everest-sv3-bright.fits",
        help="Filename of the fastphot catalog"
    )
    parser.add_argument(
        "--biprateep-mass-cat-filename", type=str,
        default="stellar_mass_specz_ztile-sv3-bright-cumulative.fits",
        help="Filename of Biprateep's stellar mass catalog"
    )

    a = parser.parse_args()
    num_rand_files = a.num_rand_files
    data_dir = pathlib.Path(a.data_dir)
    rand_dir = pathlib.Path(a.rand_dir)
    if a.everest:
        output_dir = pathlib.Path(a.output_dir) / "clean_everest"
    else:
        output_dir = pathlib.Path(a.output_dir) / "clean_fuji"
    out_rand_dir = output_dir / "rands"
    fastphot_filename = a.fastphot_filename
    biprateep_filename = a.biprateep_mass_cat_filename

    cleaned_fastphot = clean_data()
    cleaned_rands = clean_rands(n_rand_files=num_rand_files)
    out_rand_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(output_dir / "fastphot.npy"), cleaned_fastphot)
    np.save(str(out_rand_dir / "rands.npy"), cleaned_rands)
