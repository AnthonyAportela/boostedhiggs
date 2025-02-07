#!/usr/bin/python

import glob
import os
import pickle
import pickle as pkl

"""
This script combines the pkl files produced by the lumi processor to a single pkl file
that holds a dictionary with "key=dataset_name" and "value=lumi_set" which has the form (run, lumi).
"""


def main():
    # load the pkl outfiles
    year = "2016APV"
    dir_ = f"/eos/uscms/store/user/fmokhtar/boostedhiggs/lumi_{year}/"

    # datasets = [
    #     "SingleElectron_Run2017B",
    #     "SingleElectron_Run2017D",
    #     "SingleElectron_Run2017F",
    #     "SingleMuon_Run2017C",
    #     "SingleMuon_Run2017E",
    #     "SingleElectron_Run2017C",
    #     "SingleElectron_Run2017E",
    #     "SingleMuon_Run2017B",
    #     "SingleMuon_Run2017D",
    #     "SingleMuon_Run2017F"
    # ]

    datasets = os.listdir(dir_)

    out_all = {}
    for dataset in datasets:
        print(dataset)

        pkl_files = glob.glob(dir_ + dataset + "/outfiles/*")

        for i, pkl_file in enumerate(pkl_files):
            # you can load the output!
            with open(pkl_file, "rb") as f:
                out = pickle.load(f)[dataset][year + "APV"]["lumilist"]

            if i == 0:
                out_all[dataset] = out
            else:
                out_all[dataset] = out_all[dataset] | out

    # save the out_all
    with open(f"{dir_}/lumi_set.pkl", "wb") as handle:
        pkl.dump(out_all, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    # e.g.
    # run locally on lpc as: python combine_lumi.py

    main()
