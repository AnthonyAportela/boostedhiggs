#!/usr/bin/python


import json
import pickle as pkl
import warnings
from typing import List

import numpy as np
import scipy
from hist import Hist

warnings.filterwarnings("ignore", message="Found duplicate branch ")

# (name of sample, name in templates)
combine_samples = {
    # data
    "SingleElectron_": "Data",
    "SingleMuon_": "Data",
    "EGamma_": "Data",
    # signal
    "GluGluHToWW_Pt-200ToInf_M-125": "ggF",
    "HToWW_M-125": "VH",
    "VBFHToWWToLNuQQ_M-125_withDipoleRecoil": "VBF",
    "ttHToNonbb_M125": "ttH",
    # bkg
    "QCD_Pt": "QCD",
    # "QCD_Pt": "WJetsQCD",
    "DYJets": "DYJets",
    "WJetsToLNu_": "WJetsLNu",
    # "WJetsToLNu_": "WJetsQCD",
    "JetsToQQ": "WZQQ",
    "TT": "TTbar",
    "ST_": "SingleTop",
    "WW": "Diboson",
    "WZ": "Diboson",
    "ZZ": "Diboson",
    "GluGluHToTauTau": "HTauTau",
}

# (name in templates, name in cards)
labels = {
    # sigs
    "ggF": "ggF",
    "VBF": "VBF",
    "VH": "VH",
    "ttH": "ttH",
    # BKGS
    "QCD": "qcd",
    "TTbar": "ttbar",
    "WJetsLNu": "wjets",
    # "WJetsQCD": "wjetsqcd",
    "SingleTop": "singletop",
    "DYJets": "zjets",
}

bkgs = [
    "TTbar",
    "WJetsLNu",
    "SingleTop",
    "DYJets",
    "QCD",
]

sigs = [
    "ggF",
    "VBF",
    "VH",
    "ttH",
]
samples = sigs + bkgs


def get_sum_sumgenweight(pkl_files, year, sample):
    sum_sumgenweight = 0
    for ifile in pkl_files:
        # load and sum the sumgenweight of each
        with open(ifile, "rb") as f:
            metadata = pkl.load(f)
        sum_sumgenweight = sum_sumgenweight + metadata[sample][year]["sumgenweight"]
    return sum_sumgenweight


def get_xsecweight(pkl_files, year, sample, is_data, luminosity):
    if not is_data:
        # find xsection
        f = open("../fileset/xsec_pfnano.json")
        xsec = json.load(f)
        f.close()
        try:
            xsec = eval(str((xsec[sample])))
        except ValueError:
            print(f"sample {sample} doesn't have xsecs defined in xsec_pfnano.json so will skip it")
            return None

        # get overall weighting of events.. each event has a genweight...
        # sumgenweight sums over events in a chunk... sum_sumgenweight sums over chunks
        xsec_weight = (xsec * luminosity) / get_sum_sumgenweight(pkl_files, year, sample)
    else:
        xsec_weight = 1
    return xsec_weight


def shape_to_num(var, nom, clip=1.5):
    """
    Estimates the normalized rate from a shape systematic by integrating and dividing by the nominal integrated value.
    """
    nom_rate = np.sum(nom)
    var_rate = np.sum(var)
    # print("nom", nom)
    # print("var", var)
    # print("-----------------------------------")
    # print("nom", nom_rate, "var", var_rate, "ratio", var_rate / nom_rate)
    # print(var)

    if var_rate == nom_rate:  # in cases when both are zero
        return 1

    if abs(var_rate / nom_rate) > clip:
        var_rate = clip * nom_rate

    if var_rate < 0:
        var_rate = 0

    return var_rate / nom_rate


def get_template(h, sample, region):
    # massbins = h.axes["mass_observable"].edges
    # return (h[{"Sample": sample, "Systematic": "nominal", "Category": category}].values(), massbins, "mass_observable")
    return h[{"Sample": sample, "Systematic": "nominal", "Region": region}]


def load_templates(years, lep_channels, outdir):
    """Loads the hist templates that were created using ```make_templates.py```."""

    if len(years) == 4:
        save_as = "Run2"
    else:
        save_as = "_".join(years)

    if len(lep_channels) == 1:
        save_as += f"_{lep_channels[0]}_"

    with open(f"{outdir}/hists_templates_{save_as}.pkl", "rb") as f:
        hists_templates = pkl.load(f)

    return hists_templates


def blindBins(h: Hist, blind_region: List = [90, 160], blind_samples: List[str] = []):
    """
    Blind (i.e. zero) bins in histogram ``h``.
    If ``blind_samples`` specified, only blind those samples, else blinds all.

    CAREFUL: assumes axis=0 is samples, axis=3 is mass_axis

    """

    h = h.copy()

    massbins = h.axes["mass_observable"].edges

    lv = int(np.searchsorted(massbins, blind_region[0], "right"))
    rv = int(np.searchsorted(massbins, blind_region[1], "left") + 1)
    if len(blind_samples) >= 1:
        for blind_sample in blind_samples:
            sample_index = np.argmax(np.array(list(h.axes[0])) == blind_sample)
            # h.view(flow=True)[sample_index, :, :, lv:rv] = 0
            h.view(flow=True)[sample_index, :, :, lv:rv].value = 0
            h.view(flow=True)[sample_index, :, :, lv:rv].variance = 0

    else:
        # h.view(flow=True)[:, :, :, lv:rv] = 0
        h.view(flow=True)[:, :, :, lv:rv].value = 0
        h.view(flow=True)[:, :, :, lv:rv].variance = 0

    return h


def get_finetuned_score(data, model_path):
    import onnx
    import onnxruntime as ort

    input_dict = {
        "highlevel": data.loc[:, "fj_ParT_hidNeuron000":"fj_ParT_hidNeuron127"].values.astype("float32"),
    }

    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)

    ort_sess = ort.InferenceSession(
        model_path,
        providers=["AzureExecutionProvider"],
    )
    outputs = ort_sess.run(None, input_dict)

    return scipy.special.softmax(outputs[0], axis=1)[:, 0]
