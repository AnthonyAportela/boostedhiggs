#!/usr/bin/python

from axes import axis_dict, add_samples, color_by_sample, signal_by_ch, data_by_ch
import pickle as pkl
import pyarrow.parquet as pq
import pyarrow as pa
import awkward as ak
import numpy as np
import pandas as pd
import json
import os
import sys
import glob
import shutil
import pathlib
from typing import List, Optional

import argparse
from coffea import processor
from coffea.nanoevents.methods import candidate, vector
from coffea.analysis_tools import Weights, PackedSelection

import hist as hist2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mplhep as hep
from hist.intervals import clopper_pearson_interval

import warnings
warnings.filterwarnings("ignore", message="Found duplicate branch ")


def make_1dhist_cut_comparison(idir, odir, samples, years, channels, var, bins, range):
    '''
    makes and plots 1d histograms of a variable "var" and compares the different cuts
    '''

    hists = {}
    for year in years:
        # Get luminosity of year
        f = open('../fileset/luminosity.json')
        luminosity = json.load(f)
        f.close()
        print(f'Processing samples from year {year} with luminosity {luminosity[year]}')

        hists[year] = {}
        hists_btag[year] = {}
        hists_dr[year] = {}
        hists_btagdr[year] = {}

        for ch in channels:  # initialize the histograms for the different channels and different variables
            hists[year][ch] = hist2.Hist(
                hist2.axis.Regular(bins, range[0], range[1], name=var, label=var, flow=False),
                hist2.axis.StrCategory([], name='samples', growth=True)     # to combine different pt bins of the same process
            )
            hists_btag[year][ch] = hist2.Hist(
                hist2.axis.Regular(bins, range[0], range[1], name=var, label=var, flow=False),
                hist2.axis.StrCategory([], name='samples', growth=True)     # to combine different pt bins of the same process
            )
            hists_dr[year][ch] = hist2.Hist(
                hist2.axis.Regular(bins, range[0], range[1], name=var, label=var, flow=False),
                hist2.axis.StrCategory([], name='samples', growth=True)     # to combine different pt bins of the same process
            )
            hists_btagdr[year][ch] = hist2.Hist(
                hist2.axis.Regular(bins, range[0], range[1], name=var, label=var, flow=False),
                hist2.axis.StrCategory([], name='samples', growth=True)     # to combine different pt bins of the same process
            )

        # make directory to store stuff per year
        if not os.path.exists(f'{odir}/plots_{year}/'):
            os.makedirs(f'{odir}/plots_{year}/')
        if not os.path.exists(f'{odir}/plots_{year}/{var}'):
            os.makedirs(f'{odir}/plots_{year}/{var}')

        # loop over the processed files and fill the histograms
        for ch in channels:
            for sample in samples[year][ch]:
                parquet_files = glob.glob(f'{idir}/{sample}/outfiles/*_{ch}.parquet')  # get list of parquet files that have been processed
                if len(parquet_files) != 0:
                    print(f'Processing {ch} channel of sample', sample)
                else:
                    print(f'No processed files for {sample} are found')

                for i, parquet_file in enumerate(parquet_files):
                    try:
                        data = pq.read_table(parquet_file).to_pandas()
                    except:
                        print('Not able to read data: ', parquet_file, ' should remove evts from scaling/lumi')
                        continue
                    if len(data) == 0:
                        continue

                    # remove events with padded Nulls (e.g. events with no candidate jet will have a value of -1 for fj_pt)
                    data = data[data[var] != -1]

                    if cut == "btag":
                        data = data[data["anti_bjettag"] == 1]
                        cut = 'preselection + btag'
                    elif cut == "dr":
                        data = data[data["leptonInJet"] == 1]
                        cut = 'preselection + leptonInJet'
                    elif cut == "btagdr":
                        data = data[data["anti_bjettag"] == 1]
                        data = data[data["leptonInJet"] == 1]
                        cut = 'preselection + btag + leptonInJet'
                    else:
                        cut = 'preselection'

                    single_sample = None
                    for single_key, key in add_samples.items():
                        if key in sample:
                            single_sample = single_key

                    if single_sample is not None:
                        hists[year][ch].fill(
                            data[var], single_sample,  # combining all events under one name
                        )
                        hists_btag[year][ch].fill(
                            data[var][data["anti_bjettag"] == 1], single_sample,  # combining all events under one name
                        )
                        hists_dr[year][ch].fill(
                            data[var][data["leptonInJet"] == 1], single_sample,  # combining all events under one name
                        )
                        hists_btagdr[year][ch].fill(
                            data[var][data["anti_bjettag"] == 1][data["leptonInJet"] == 1], single_sample,  # combining all events under one name
                        )
                    else:
                        hists[year][ch].fill(
                            data[var], sample,
                        )
                        hists_btag[year][ch].fill(
                            data[var][data["anti_bjettag"] == 1], sample,
                        )
                        hists_dr[year][ch].fill(
                            data[var][data["leptonInJet"] == 1], sample,
                        )
                        hists_btagdr[year][ch].fill(
                            data[var][data["anti_bjettag"] == 1][data["leptonInJet"] == 1], sample,
                        )

            for sample in hists[year][ch].axes[-1]:

                fig, ax = plt.subplots(figsize=(8, 5))
                hep.histplot(hists[year][ch][{'samples': sample}], ax=ax, label='preselection')
                hep.histplot(hists_btag[year][ch][{'samples': sample}], ax=ax, label='preselection + btag')
                hep.histplot(hists_dr[year][ch][{'samples': sample}], ax=ax, label='preselection + dr')
                hep.histplot(hists_btagdr[year][ch][{'samples': sample}], ax=ax, label='preselection + btag + dr')
                ax.set_xlabel(f"{var}")
                ax.set_title(f'{ch} channel for \n {sample} \n with {cut} cut')
                ax.legend()
                hep.cms.lumitext(f"{year} (13 TeV)", ax=ax)
                hep.cms.text("Work in Progress", ax=ax)
                plt.savefig(f'{odir}/plots_{year}/{var}/{ch}_{sample}_{cut}.pdf')
                plt.close()


def main(args):
    if not os.path.exists(args.odir):
        os.makedirs(args.odir)

    years = args.years.split(',')
    channels = args.channels.split(',')

    # get samples to make histograms
    f = open(args.samples)
    json_samples = json.load(f)
    f.close()

    # build samples
    samples = {}
    for year in years:
        samples[year] = {}
        for ch in channels:
            samples[year][ch] = []
            for key, value in json_samples[year][ch].items():
                if value == 1:
                    samples[year][ch].append(key)

    range = [args.start, args.end]
    print(f'Plotting {args.var} histogram')
    make_1dhist_cut_comparison(args.idir, args.odir, samples, years, channels, args.var, args.bins, range)


if __name__ == "__main__":
    # e.g. run locally as
    # lep_pt:        python make_1dhist_cut_comparison.py --year 2017 --odir hists/1dhists --channels ele,mu --var lep_pt        --bins 100 --start 0 --end 500 --idir /eos/uscms/store/user/fmokhtar/boostedhiggs/
    # lep_isolation: python make_1dhist_cut_comparison.py --year 2017 --odir hists/1dhists --channels ele,mu --var lep_isolation --bins 100 --start 0 --end 2 --idir /eos/uscms/store/user/fmokhtar/boostedhiggs/
    # lep_fj_dr:     python make_1dhist_cut_comparison.py --year 2017 --odir hists/1dhists --channels ele,mu --var lep_fj_dr     --bins 100 --start 0 --end 2 --idir /eos/uscms/store/user/fmokhtar/boostedhiggs/

    parser = argparse.ArgumentParser()
    parser.add_argument('--years',           dest='years',       default='2017',                             help="year")
    parser.add_argument('--samples',         dest='samples',     default="plot_configs/samples_pfnano.json", help='path to json with samples to be plotted')
    parser.add_argument('--channels',        dest='channels',    default='ele,mu,had',                       help='channels for which to plot this variable')
    parser.add_argument('--odir',            dest='odir',        default='hists/1dhists',                    help="tag for output directory")
    parser.add_argument('--idir',            dest='idir',        default='../results/',                      help="input directory with results")
    parser.add_argument('--var',             dest='var',         default='lep_pt',                           help='variable to plot')
    parser.add_argument('--bins',            dest='bins',        default=50,                                 help="binning of the first variable passed",                type=int)
    parser.add_argument('--start',           dest='start',       default=0,                                  help="starting range of the first variable passed",         type=int)
    parser.add_argument('--end',             dest='end',         default=1,                                  help="end range of the first variable passed",              type=int)

    args = parser.parse_args()

    main(args)
