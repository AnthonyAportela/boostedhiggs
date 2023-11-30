# flake8: noqa
from __future__ import division, print_function

import argparse
import os
import pickle
import sys

import numpy as np
import rhalphalib as rl
import ROOT
import scipy.stats

rl.util.install_roofit_helpers()
rl.ParametericSample.PreferRooParametricHist = True


def main(args):
    years = args.years.split(",")
    channels = args.channels.split(",")

    if len(years) == 4:
        save_as = "Run2"
    else:
        save_as = "_".join(years)

    if len(channels) == 1:
        save_as += "_{}_".format(channels[0])

    model_name = "model_{}".format(save_as)

    tmpdir = "{}/{}.pkl".format(args.outdir, model_name)

    with open(tmpdir, "rb") as fout:
        model = pickle.load(fout)

    model.renderCombine(os.path.join(str("{}".format(args.outdir)), "datacards"))


if __name__ == "__main__":
    # e.g.
    # python produce_datacard.py --years 2016,2016APV,2017,2018 --channels ele,mu --outdir templates/v1
    # python diffNuisances.py fitDiagnosticsBlinded.root --all

    parser = argparse.ArgumentParser()
    parser.add_argument("--years", dest="years", default="2017", help="years separated by commas")
    parser.add_argument("--channels", dest="channels", default="mu", help="channels separated by commas (e.g. mu,ele)")
    parser.add_argument("--outdir", dest="outdir", default="templates/test", type=str, help="path to template directory")

    args = parser.parse_args()

    main(args)
