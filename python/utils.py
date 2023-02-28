#!/usr/bin/python

import json
import pickle as pkl
import warnings

import hist as hist2

warnings.filterwarnings("ignore", message="Found duplicate branch ")

add_samples = {
    "SingleElectron": "SingleElectron",
    "EGamma": "EGamma",
    "SingleMuon": "SingleMuon",
    "JetHT": "JetHT",
    "QCD": "QCD_Pt",
    "DYJets": "DYJets",
    "WZQQ": "JetsToQQ",
    "SingleTop": "ST",
    "TTbar": "TT",
    "WJetsLNu": "WJetsToLNu",
    "Diboson": ["WW", "WZ", "ZZ"],
    "ttH": ["ttHToNonbb_M125"],
    "WH": ["HWminusJ_HToWW_M-125","HWplusJ_HToWW_M-125"],
    "ZH": ["HZJ_HToWW_M-125"],
    "ggH": "GluGluHToWW_Pt-200ToInf_M-125",
    "VBF": "VBFHToWWToLNuQQ_M-125_withDipoleRecoil",
}


def get_sample_to_use(sample, year, is_data):
    """
    Get name of sample that adds small subsamples
    """
    single_sample = None
    for single_key, key in add_samples.items():
        if type(key) is list:
            for k in key:
                if k in sample:
                    single_sample = single_key
        else:
            if key in sample:
                single_sample = single_key

    if year == "Run2" and is_data:
        single_sample = "Data"

    if single_sample is not None:
        sample_to_use = single_sample
    else:
        sample_to_use = sample
    return sample_to_use

simplified_labels = {
    "SingleElectron": "Data",
    "EGamma": "Data",
    "SingleMuon": "Data",
    "JetHT": "Data",
    "Data": "Data",
    "QCD": "Multijet",
    "DYJets": r"Z($\ell\ell$) + jets",
    "WJetsLNu": r"W($\ell\nu$) + jets",
    "Diboson": r"VV",
    "WH": r"WH$\rightarrow$WW",
    "ZH": r"ZH$\rightarrow$WW",
    "ggH": r"ggH$\rightarrow$WW",
    "VBF": r"VBF$\rightarrow$WW",
    "ttH": r"ttH$\rightarrow$WW",
    "TTbar": r"$t\bar{t}$+jets",
    "SingleTop": r"single-t",
    "WZQQ": r"W/Z(qq) + jets",
}


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


def get_cutflow(cut_keys, pkl_files, yr, sample, xsec_weight, ch):
    evyield = dict.fromkeys(cut_keys, 0)
    for ik, pkl_file in enumerate(pkl_files):
        with open(pkl_file, "rb") as f:
            metadata = pkl.load(f)
            cutflows = metadata[sample][yr]["cutflows"][ch]
            for key in cut_keys:
                if key in cutflows.keys():
                    evyield[key] += cutflows[key] * xsec_weight
    return evyield


# define the axes for the different variables to be plotted
# define samples
signal_by_ch = {
    "ele": [
        "ttH",
        "WH","ZH",
        "ggH",
        "VBF",
    ],
    "mu": [
        "ttH",
        "WH","ZH",
        "ggH",
        "VBF",
    ],
}

# there are many signal samples for the moment:
# - ele,mu: GluGluHToWWToLNuQQ
# - had:
#   - ggHToWWTo4Q-MH125 (produced by Cristina from PKU config files Powheg+JHU) - same xsec as GluGluToHToWWTo4q
#   - GluGluToHToWWTo4q (produced by PKU)
#   - GluGluHToWWTo4q (produced by Cristina w Pythia)
#   - GluGluHToWWTo4q-HpT190 (produced by Cristina w Pythia)
# to come: GluGluHToWW_MINLO (for ele,mu,had)

# this is actually no longer by channel since I don't have channels for VH,
# but leave the name for now, since 2018 is different than other years:w
data_by_ch = {
    "ele": "SingleElectron",
    "mu": "SingleMuon",
    "had": "JetHT",
    "DoubleMuon": "DoubleMuon",
    "MuonEG": "MuonEG",
    "DoubleEG": "DoubleEG",
}
data_by_ch_2018 = {
    "ele": "EGamma",  # i guess there was no single ele for this year, so cristina used eGamma instead of single ele?
    "mu": "SingleMuon",
    "had": "JetHT",
    "DoubleMuon": "DoubleMuon",
    "MuonEG": "MuonEG",
}

color_by_sample = {
    "QCD": "tab:orange",
    "DYJets": "tab:purple",
    "WJetsLNu": "tab:green",
    "TTbar": "tab:blue",
    "WZQQ": "salmon",
    "SingleTop": "tab:cyan",
    "Diboson": "orchid",
    "ttH": "tab:olive",
    "ggH": "coral",
    "WH": "tab:brown",
    "ZH": "darkred",
    "VBF": "tab:gray",
}

label_by_ch = {
    "ele": "Electron",
    "mu": "Muon",
}


axis_dict = {
    "lep_pt": hist2.axis.Regular(40, 30, 450, name="var", label=r"Lepton $p_T$ [GeV]", overflow=True),
    "lep_fj_m": hist2.axis.Regular(35, 0, 280, name="var", label=r"Jet - Lepton mass [GeV]", overflow=True),
    "lep_met_mt": hist2.axis.Regular(35, 0, 400, name="var", label=r"$m_T(lep, p_T^{miss})$ [GeV]", overflow=True),
    "fj_bjets_ophem": hist2.axis.Regular(35, 0, 1, name="var", label=r"max btagFlavB (opphem)", overflow=True),
    "fj_bjets": hist2.axis.Regular(35, 0, 1, name="var", label=r"max btagFlavB", overflow=True),
    "lep_fj_dr": hist2.axis.Regular(35, 0.0, 0.8, name="var", label=r"$\Delta R(l, Jet)$", overflow=True),
    "mu_mvaId": hist2.axis.Variable([0, 1, 2, 3, 4, 5], name="var", label="Muon MVAID", overflow=True),
    "ele_highPtId": hist2.axis.Regular(5, 0, 5, name="var", label="Electron high pT ID", overflow=True),
    "mu_highPtId": hist2.axis.Regular(5, 0, 5, name="var", label="Muon high pT ID", overflow=True),
    "fj_pt": hist2.axis.Regular(30, 200, 1000, name="var", label=r"Jet $p_T$ [GeV]", overflow=True),
    "fj_msoftdrop": hist2.axis.Regular(45, 20, 400, name="var", label=r"Jet $m_{sd}$ [GeV]", overflow=True),
    "score": hist2.axis.Regular(25, 0, 1, name="var", label=r"PN score", overflow=True),
    "ht": hist2.axis.Regular(35, 180, 2000, name="var", label="HT [GeV]", overflow=True),
    "nfj": hist2.axis.Regular(4, 1, 5, name="var", label="Num AK8 jets", overflow=True),
    "nj": hist2.axis.Regular(8, 0, 8, name="var", label="Num AK4 jets outside of AK8", overflow=True),
    "deta": hist2.axis.Regular(35, 0, 7, name="var", label=r"\Delta \eta (j,j)", overflow=True),
    "mjj": hist2.axis.Regular(50, 0, 7500, name="var", label=r"M(j,j) [GeV]", overflow=True),
    "met": hist2.axis.Regular(40, 0, 450, name="var", label="MET [GeV]", overflow=True),
    "met_fj_dphi": hist2.axis.Regular(30, -5, 5, name="var", label=r"$\Delta \Phi(Jet, MET)$", overflow=True),
    "gen_Hpt": hist2.axis.Regular(35, 80, 1000, name="var", label=r"Higgs $p_T$ [GeV]", overflow=True),
    "gen_Hnprongs": hist2.axis.Regular(4, 0, 4, name="var", label=r"num of prongs", overflow=True),
    "gen_iswlepton": hist2.axis.Regular(2, 0, 1, name="var", label=r"lepton from W", overflow=True),
    "gen_iswstarlepton": hist2.axis.Regular(2, 0, 1, name="var", label=r"lepton from W*", overflow=True),
    "gen_isVlep": hist2.axis.Regular(2, 0, 1, name="var", label=r"isWlep", overflow=True),
    "gen_isVqq": hist2.axis.Regular(2, 0, 1, name="var", label=r"isWqq", overflow=True),
    "gen_isTop": hist2.axis.Regular(2, 0, 1, name="var", label=r"isTop", overflow=True),
    "gen_isToplep": hist2.axis.Regular(2, 0, 1, name="var", label=r"isToplep", overflow=True),
    "gen_isTopqq": hist2.axis.Regular(2, 0, 1, name="var", label=r"isTopqq", overflow=True),
}

axis_dict["lep_isolation_lowpt"] = hist2.axis.Regular(20, 0, 2.0, name="var", label=r"Lepton iso (low $p_T$)", overflow=True)
axis_dict["lep_isolation_highpt"] = hist2.axis.Regular(20, 0, 5, name="var", label=r"Lepton iso (high $p_T$)", overflow=True)

# axis_dict["lep_misolation_lowpt"] = hist2.axis.Regular(
#     35, 0, 2.0, name="var", label=r"Lepton mini iso (low $p_T$)", overflow=True
# )
axis_dict["lep_misolation_lowpt"] = hist2.axis.Regular(
    50, 0, 0.1, name="var", label=r"Lepton mini iso (low $p_T$)", overflow=True
)
# axis_dict["lep_misolation_highpt"] = hist2.axis.Regular(
#     35, 0, 0.15, name="var", label=r"Lepton mini iso (high $p_T$)", overflow=True
# )
axis_dict["lep_misolation_highpt"] = hist2.axis.Regular(
    50, 0, 0.025, name="var", label=r"Lepton mini iso (high $p_T$)", overflow=True
)


def get_cutflow_axis(cut_keys):
    return hist2.axis.Regular(
        len(cut_keys),
        0,
        len(cut_keys),
        name="var",
        label=r"Event Cutflow",
        overflow=True,
    )
