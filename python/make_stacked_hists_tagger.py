#!/usr/bin/python

from utils import axis_dict, add_samples, color_by_sample, signal_by_ch, data_by_ch, data_by_ch_2018, label_by_ch
from utils import simplified_labels, get_sum_sumgenweight
import pickle as pkl
import pyarrow.parquet as pq
import pyarrow as pa
import awkward as ak
import numpy as np
import pandas as pd
import json
import os
import glob
import shutil
import pathlib
from typing import List, Optional

import argparse
from coffea import processor
from coffea.nanoevents.methods import candidate, vector
from coffea.analysis_tools import Weights, PackedSelection

import hist as hist2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mplhep as hep
from hist.intervals import clopper_pearson_interval

import warnings
warnings.filterwarnings("ignore", message="Found duplicate branch ")


def make_stacked_hists(year, ch, idir, odir, samples):
    """
    Makes 1D histograms to be plotted as stacked over the different samples
    Args:
        year: string that represents the year the processed samples are from
        ch: string that represents the signal channel to look at... choices are ['ele', 'mu', 'had']
        idir: directory that holds the processed samples (e.g. {idir}/{sample}/outfiles/*_{ch}.parquet)
        odir: output directory to hold the hist object
        samples: the set of samples to run over (by default: the samples defined in plot_configs/samples_pfnano.json)
    """

    max_iso = {'ele': 120, 'mu': 55}

    # Get luminosity of year
    f = open('../fileset/luminosity.json')
    luminosity = json.load(f)[year]
    f.close()
    print(f'Processing samples from year {year} with luminosity {luminosity}')

    # instantiates the histogram object

    hists = hist2.Hist(
        hist2.axis.StrCategory([], name='samples', growth=True),
        hist2.axis.Regular(30, 0, 1, name='tagger_score', label=r'Tagger score', overflow=True),
    )

    # loop over the samples
    # print(samples.keys())
    for yr in samples.keys():
        if yr == '2018':
            data_label = data_by_ch_2018
        else:
            data_label = data_by_ch
        for sample in samples[yr][ch]:
            # check if the sample was processed
            pkl_dir = f'{idir}_{yr}/{sample}/outfiles/*.pkl'
            pkl_files = glob.glob(pkl_dir)  #
            if not pkl_files:  # skip samples which were not processed
                print('- No processed files found...', pkl_dir, 'skipping sample...', sample)
                continue

            # define an isdata bool
            is_data = False
            for key in data_label.values():
                if key in sample:
                    is_data = True

            # get list of parquet files that have been processed
            parquet_files = glob.glob(f'{idir}_{yr}/{sample}/outfiles/*_{ch}.parquet')

            if len(parquet_files) != 0:
                print(f'Processing {ch} channel of sample', sample)

            # print(parquet_files)
            for parquet_file in parquet_files:
                # print('Processing ',parquet_file)

                try:
                    data = pq.read_table(parquet_file).to_pandas()
                except:
                    if is_data:
                        print('Not able to read data: ', parquet_file, ' should remove evts from scaling/lumi')
                    else:
                        print('Not able to read data from ', parquet_file)
                    continue

                try:
                    event_weight = data['tot_weight']
                except:
                    print('No tot_weight variable in parquet - run pre-processing first!')
                    continue

                # make iso and miso cuts (different for each channel)
                # iso_cut = (data['lep_isolation'] < 0.15) & (data['lep_pt'] < max_iso[ch])
                # iso_cut = (data['lep_isolation'] < 0.15) & (data['lep_pt'] < max_iso[ch]) | (data['lep_pt'] > max_iso[ch])
                #
                # if ch == 'ele':
                #     miso_cut = (data['lep_pt'] > max_iso[ch])
                # elif ch == 'mu':
                #     # miso_cut = (data['lep_misolation'] < 0.1) & (data['lep_pt'] > max_iso[ch])
                #     miso_cut = (data['lep_misolation'] < 0.1)
                #
                # if ch == 'mu':
                #     miso_cut = (data['lep_misolation'] < 0.1)
                #     select = (iso_cut & miso_cut)
                # else:
                #     select = iso_cut

                # select the jet pT [400-600] GeV and the mSD [30 -150 ]
                select_fj_pt = (data['fj_pt'] > 400) & (data['fj_pt'] < 600)
                select_fj_msd = (data['fj_msoftdrop'] > 30) & (data['fj_msoftdrop'] < 150)

                select = select_fj_pt & select_fj_msd
                data = data[select]
                event_weight = event_weight[select]

                if ch == 'ele':
                    x = data['fj_isHVV_elenuqq'] / (data['fj_isHVV_elenuqq'] + data['fj_ttbar_bmerged'] + data['fj_ttbar_bsplit'] + data['fj_wjets_label'])
                if ch == 'mu':
                    x = data['fj_isHVV_munuqq'] / (data['fj_isHVV_munuqq'] + data['fj_ttbar_bmerged'] + data['fj_ttbar_bsplit'] + data['fj_wjets_label'])

                # filling histograms
                single_sample = None
                for single_key, key in add_samples.items():
                    if key in sample:
                        single_sample = single_key
                if year == "Run2" and is_data:
                    single_sample = 'Data'
                if single_sample is not None:
                    sample_to_use = single_sample
                else:
                    sample_to_use = sample

                # combining all pt bins of a specefic process under one name
                hists.fill(
                    samples=sample_to_use,
                    tagger_score=x,
                    weight=event_weight,
                )

    # store the hists variable
    with open(f'{odir}/{ch}_tagger_hists.pkl', 'wb') as f:  # saves the hists objects
        pkl.dump(hists, f)


def plot_stacked_hists(year, ch, odir, logy=True, add_data=True):
    """
    Plots the stacked 1D histograms that were made by "make_stacked_hists" individually for each year
    Args:
        year: string that represents the year the processed samples are from
        ch: string that represents the signal channel to look at... choices are ['ele', 'mu', 'had']
        odir: output directory to hold the plots
    """

    # load the hists
    with open(f'{odir}/{ch}_tagger_hists.pkl', 'rb') as f:
        hists = pkl.load(f)
        f.close()

    # make the histogram plots in this directory
    if logy:
        if not os.path.exists(f'{odir}/{ch}_tagger_hists_log'):
            os.makedirs(f'{odir}/{ch}_tagger_hists_log')
    else:
        if not os.path.exists(f'{odir}/{ch}_tagger_hists'):
            os.makedirs(f'{odir}/{ch}_tagger_hists')

    data_label = data_by_ch[ch]
    if year == '2018':
        data_label = data_by_ch_2018[ch]
    elif year == 'Run2':
        data_label = 'Data'

    # get histograms
    h = hists
    if h.shape[0] == 0:     # skip empty histograms (such as lepton_pt for hadronic channel)
        print('Empty histogram')

    # get samples existing in histogram
    samples = [h.axes[0].value(i) for i in range(len(h.axes[0].edges))]

    signal_labels = [label for label in samples if label in signal_by_ch[ch]]
    bkg_labels = [label for label in samples if (label and label != data_label and label not in signal_labels)]
    if 'VBFHToWWToLNuQQ-MH125' in signal_labels:
        signal_labels.remove('VBFHToWWToLNuQQ-MH125')
    # data
    data = None
    if data_label in h.axes[0]:
        data = h[{"samples": data_label}]

    # signal
    signal = [h[{"samples": label}] for label in signal_labels]
    if not logy:
        signal = [s * 10 for s in signal]  # if not log, scale the signal

    # background
    bkg = [h[{"samples": label}] for label in bkg_labels]

    if add_data and data and len(bkg) > 0:
        fig, (ax, rax) = plt.subplots(nrows=2,
                                      ncols=1,
                                      figsize=(8, 8),
                                      gridspec_kw={"height_ratios": (4, 1), 'hspace': 0.07},
                                      sharex=True
                                      )
    else:
        fig, ax = plt.subplots(1, 1)
        rax = None

    errps = {
        'hatch': '////',
        'facecolor': 'none',
        'lw': 0,
        'color': 'k',
        'edgecolor': (0, 0, 0, .5),
        'linewidth': 0,
        'alpha': 0.4
    }

    if len(bkg) > 0:
        hep.histplot(bkg,
                     ax=ax,
                     stack=True,
                     sort='yield',
                     edgecolor='black',
                     linewidth=1,
                     histtype="fill",
                     label=[bkg_label for bkg_label in bkg_labels],
                     color=[color_by_sample[bkg_label] for bkg_label in bkg_labels]
                     )

        tot = bkg[0].copy()
        for i, b in enumerate(bkg):
            if i > 0:
                tot = tot + b
        ax.stairs(
            values=tot.values() + np.sqrt(tot.values()),
            baseline=tot.values() - np.sqrt(tot.values()),
            edges=tot.axes[0].edges, **errps,
            label='Stat. unc.'
        )

    if add_data and data:
        data_err_opts = {
            'linestyle': 'none',
            'marker': '.',
            'markersize': 10.,
            'elinewidth': 1,
        }
        hep.histplot(data,
                     ax=ax,
                     histtype="errorbar",
                     color="k",
                     capsize=4,
                     yerr=True,
                     label=data_label,
                     **data_err_opts
                     )

        if len(bkg) > 0:
            from hist.intervals import ratio_uncertainty
            yerr = ratio_uncertainty(data.values(), tot.values(), 'poisson')
            rax.stairs(1 + yerr[1], edges=tot.axes[0].edges, baseline=1 - yerr[0], **errps)

            if ak.all(tot.values()) > 0:
                hep.histplot(data.values() / tot.values(), tot.axes[0].edges, yerr=np.sqrt(data.values()) / tot.values(),
                             ax=rax, histtype='errorbar', color='k', capsize=4)
            else:
                print(f'Warning: not all bins filled for background histogram for {ch}')
            rax.axhline(1, ls='--', color='k')
            rax.set_ylim(0.2, 1.8)
            # rax.set_ylim(0.7, 1.3)

    if len(signal) > 0:
        sigg = None
        for i, sig in enumerate(signal):
            hep.histplot(sig,
                         ax=ax,
                         label=f'10 * {simplified_labels[signal_labels[i]]}',
                         linewidth=3,
                         color=color_by_sample[signal_labels[i]]
                         )
            if sigg == None:
                sigg = signal[i].copy()
            else:
                sigg = sigg + sig
        ax.stairs(
            values=sigg.values() + np.sqrt(sigg.values()),
            baseline=sigg.values() - np.sqrt(sigg.values()),
            edges=sig.axes[0].edges, **errps,
        )

    if rax != None:
        ax.set_xlabel("")
        rax.set_xlabel(f'Tagger score')

    # sort the legend
    order_dic = {}
    for bkg_label in bkg_labels:
        order_dic[bkg_label] = hists['fj_pt'][{'samples': bkg_label}].sum()

    handles, labels = ax.get_legend_handles_labels()

    summ = []
    for label in labels[:len(bkg_labels)]:
        summ.append(order_dic[label])

    order = []
    for i in range(len(summ)):
        order.append(np.argmax(np.array(summ)))
        summ[np.argmax(np.array(summ))] = -100

    hand = [handles[i] for i in order] + handles[len(bkg):]
    lab = [labels[i] for i in order] + labels[len(bkg):]

    ax.legend([hand[idx] for idx in range(len(hand))], [lab[idx] for idx in range(len(lab))])

    if logy:
        ax.set_yscale('log')
        ax.set_ylim(0.1)
    ax.set_title(f'{label_by_ch[ch]} Channel')

    hep.cms.lumitext(f"{year} (13 TeV)", ax=ax)
    hep.cms.text("Work in Progress", ax=ax)

    if logy:
        print(f'Saving to {odir}/{ch}_tagger_hists_log/tagger_score.pdf')
        plt.savefig(f'{odir}/{ch}_tagger_hists_log/tagger_score.pdf', bbox_inches='tight')
    else:
        print(f'Saving to {odir}/{ch}_tagger_hists/tagger_score.pdf')
        plt.savefig(f'{odir}/{ch}_tagger_hists/tagger_score.pdf', bbox_inches='tight')
    plt.close()


def main(args):
    # append '_year' to the output directory
    odir = args.odir + '_' + args.year
    if not os.path.exists(odir):
        os.makedirs(odir)

    # make subdirectory specefic to this script
    if not os.path.exists(odir + '/stacked_hists/'):
        os.makedirs(odir + '/stacked_hists/')
    odir = odir + '/stacked_hists/'

    channels = args.channels.split(',')

    # get year
    years = ["2016", "2016APV", "2017", "2018"] if args.year == "Run2" else [args.year]

    # get samples to make histograms
    f = open(args.samples)
    json_samples = json.load(f)
    f.close()
    samples = {}
    for year in years:
        samples[year] = {}
        for ch in channels:
            samples[year][ch] = json_samples[year][ch]

    for ch in channels:
        if args.make_hists:
            if len(glob.glob(f'{odir}/{ch}_hists.pkl')) > 0:
                print('Histograms already exist - remaking them')
            print('Making histograms...')
            make_stacked_hists(args.year, ch, args.idir, odir, samples)

        if args.plot_hists:
            print('Plotting...')
            plot_stacked_hists(args.year, ch, odir, logy=args.nology, add_data=args.nodata)


if __name__ == "__main__":
    # e.g.
    # run locally as: python make_stacked_hists.py --year 2017 --odir hists --channels ele,mu --idir /eos/uscms/store/user/cmantill/boostedhiggs/Jun20 --make_hists --plot_hists
    # run locally as: python make_stacked_hists_tagger.py --year 2017 --odir hists --channels ele,mu --idir /eos/uscms/store/user/fmokhtar/boostedhiggs/Jul28 --make_hists --plot_hists

    parser = argparse.ArgumentParser()
    parser.add_argument('--year',                   dest='year', required=True, choices=["2016", "2016APV", "2017", "2018", "Run2"],  help="year")
    parser.add_argument('--samples',                dest='samples',                 default="plot_configs/samples_pfnano.json",   help='path to json with samples to be plotted')
    parser.add_argument('--channels',               dest='channels',                default='ele,mu,had',                         help='channels for which to plot this variable')
    parser.add_argument('--odir',                   dest='odir',                    default='hists',                              help="tag for output directory... will append '_{year}' to it")
    parser.add_argument('--idir',                   dest='idir',                    default='../results/',                        help="input directory with results - without _{year}")
    parser.add_argument("--make_hists",             dest='make_hists',              action='store_true',                          help="Make hists")
    parser.add_argument("--plot_hists",             dest='plot_hists',              action='store_true',                          help="Plot the hists")
    parser.add_argument("--nology",                 dest='nology',                  action='store_false',                         help="No logy scale")
    parser.add_argument("--nodata",                 dest='nodata',                  action='store_false',                         help="No data")

    args = parser.parse_args()

    main(args)