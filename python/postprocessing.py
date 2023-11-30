"""
Postprocesseses the parquets stored in `samples_dir` by applying the preselections defined
in the top of the script, and stores the new output in `out_dir`.

Author: Farouk Mokhtar
"""

import argparse
import glob
import logging
import os
import warnings

import pandas as pd
import pyarrow

logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore", message="Found duplicate branch ")
pd.set_option("mode.chained_assignment", None)


presel = {
    "mu": {
        "lep_fj_dr": "( ( lep_fj_dr<0.8) )",
    },
    "ele": {
        "lep_fj_dr": "( ( lep_fj_dr<0.8) )",
    },
}


def postprocess(year, channels, samples_dir, out_dir):
    for ch in channels:
        condor_dir = os.listdir(samples_dir)

        for sample in condor_dir:
            logging.info(f"Finding {sample} samples")

            out_files = f"{samples_dir}/{sample}/outfiles/"
            parquet_files = glob.glob(f"{out_files}/*_{ch}.parquet")
            pkl_files = glob.glob(f"{out_files}/*.pkl")

            if not parquet_files:
                logging.info(f"No parquet file for {sample}")
                continue

            try:
                data = pd.read_parquet(parquet_files)

            except pyarrow.lib.ArrowInvalid:
                # empty parquet because no event passed selection
                continue

            if len(data) == 0:
                continue

            for k, v in presel[ch].items():
                logging.info(f"Applying {k} selection on {len(data)} events")
                data = data.query(v)
            logging.info(f"Done - will store the remaining {len(data)} events")

            os.system(f"mkdir -p {out_dir}/{sample}/outfiles/")
            data.to_parquet(f"{out_dir}/{sample}/outfiles/{ch}.parquet")

            for ifile in pkl_files:
                os.system(f"cp {ifile} {out_dir}/{sample}/outfiles/{os.path.basename(ifile)}")


def main(args):
    postprocess(args.year, args.channels.split(","), args.samples_dir, args.out_dir)


if __name__ == "__main__":
    # e.g.
    # noqa: python postprocessing.py --year 2017 --channels ele,mu --samples_dir ../eos/Oct5_hidNeurons_2017 --out_dir ../eos/postprocessOct5_hidNeurons_2018
    # noqa: python postprocessing.py --year 2018 --channels ele,mu --samples_dir ../eos/Nov12_2018 --out_dir ../eos/postprocessNov12_2018

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", dest="year", default="2018", help="year")
    parser.add_argument("--channels", dest="channels", default="ele,mu", help="channels separated by commas")
    parser.add_argument(
        "--samples_dir", dest="samples_dir", default="Oct5_hidNeurons_2018", help="path to parquets", type=str
    )
    parser.add_argument("--out_dir", default="postprocessOct5_hidNeurons_2018", help="path of the output", type=str)

    args = parser.parse_args()

    main(args)
