from __future__ import division, print_function

import json
import logging
import warnings
from typing import List

import pandas as pd
import rhalphalib as rl
from utils import samples

rl.ParametericSample.PreferRooParametricHist = True
logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore", message="Found duplicate branch ")
pd.set_option("mode.chained_assignment", None)

CMS_PARAMS_LABEL = "CMS_HWW_boosted"


def systs_not_from_parquets(years: List[str], lep_channels: List[str]):
    """
    Define systematics that are NOT stored in the parquets.

    Args
        years: e.g. ["2018", "2017", "2016APV", "2016"]
        lep_channels: e.g. ["mu", "ele"]

    Returns
        systs_dict [dict]:        keys are systematics; values are rl.NuisanceParameters
        systs_dict_values [dict]: keys are same as above; values are tuples (up, down) and if (up, None) then down=up
    """
    # get the LUMI (must average lumi over the lepton channels provided)
    LUMI = {}
    for year in years:
        LUMI[year] = 0.0
        for lep_ch in lep_channels:
            with open("../fileset/luminosity.json") as f:
                LUMI[year] += json.load(f)[lep_ch][year]
        LUMI[year] /= len(lep_channels)

    # get the LUMI covered in the templates
    full_lumi = 0
    for year in years:
        full_lumi += LUMI[year]

    systs_dict, systs_dict_values = {}, {}

    # COMMON SYSTEMATICS
    systs_dict["all_samples"], systs_dict_values["all_samples"] = {}, {}

    # branching ratio systematics
    systs_dict["all_samples"]["BR_hww"] = rl.NuisanceParameter("BR_hww", "lnN")
    systs_dict_values["all_samples"]["BR_hww"] = (1.0153, 0.9848)

    # lumi systematics
    if "2016" in years:
        systs_dict["all_samples"]["lumi_13TeV_2016"] = rl.NuisanceParameter("CMS_lumi_13TeV_2016", "lnN")
        systs_dict_values["all_samples"]["lumi_13TeV_2016"] = (1.01 ** ((LUMI["2016"] + LUMI["2016APV"]) / full_lumi), None)
    if "2017" in years:
        systs_dict["all_samples"]["lumi_13TeV_2017"] = rl.NuisanceParameter("CMS_lumi_13TeV_2017", "lnN")
        systs_dict_values["all_samples"]["lumi_13TeV_2017"] = (1.02 ** (LUMI["2017"] / full_lumi), None)

    if "2018" in years:
        systs_dict["all_samples"]["lumi_13TeV_2018"] = rl.NuisanceParameter("CMS_lumi_13TeV_2018", "lnN")
        systs_dict_values["all_samples"]["lumi_13TeV_2018"] = (1.015 ** (LUMI["2018"] / full_lumi), None)

    if len(years) == 4:
        systs_dict["all_samples"]["lumi_13TeV_correlated"] = rl.NuisanceParameter("CMS_lumi_13TeV_corelated", "lnN")
        systs_dict_values["all_samples"]["lumi_13TeV_correlated"] = (
            (
                (1.006 ** ((LUMI["2016"] + LUMI["2016APV"]) / full_lumi))
                * (1.009 ** (LUMI["2017"] / full_lumi))
                * (1.02 ** (LUMI["2018"] / full_lumi))
            ),
            None,
        )

        systs_dict["all_samples"]["lumi_13TeV_1718"] = rl.NuisanceParameter("CMS_lumi_13TeV_1718", "lnN")
        systs_dict_values["all_samples"]["lumi_13TeV_1718"] = (
            (1.006 ** (LUMI["2017"] / full_lumi)) * (1.002 ** (LUMI["2018"] / full_lumi)),
            None,
        )

    # PER SAMPLE SYSTEMATICS
    for sample in samples:
        systs_dict[sample], systs_dict_values[sample] = {}, {}

    n = rl.NuisanceParameter("taggereff", "lnN")
    for sample in ["ggF", "VBF", "VH", "ttH"]:
        systs_dict[sample]["taggereff"] = n
        systs_dict_values[sample]["taggereff"] = (1.10, None)

    return systs_dict, systs_dict_values


def systs_from_parquets(years):
    """
    Specify systematics that ARE stored in the parquets
    """
    if len(years) == 4:
        year = "Run2"
    else:
        year = years[0]

    systs_from_parquets = {
        "all_samples": {
            # "weight_mu_btagSFlight_2017": rl.NuisanceParameter(
            #     f"{CMS_PARAMS_LABEL}_btagSFlight_{year}", "lnN"
            #     ),
            # "weight_mu_btagSFlight_correlated": rl.NuisanceParameter(
            #     f"{CMS_PARAMS_LABEL}_btagSFlight_correlated", "lnN"
            # ),
            # "weight_mu_btagSFbc_2017": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_btagSFbc_{year}", "lnN"),
            # "weight_mu_btagSFbc_correlated": rl.NuisanceParameter(
            #     f"{CMS_PARAMS_LABEL}_btagSFbc_correlated", "lnN"
            # ),
            # "weight_pileup": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_PU_{year}", "shape"),
            "weight_isolation": rl.NuisanceParameter(f"CMS_iso_{year}", "lnN"),
            "weight_id": rl.NuisanceParameter(f"CMS_id_{year}", "lnN"),
            "weight_reco_ele": rl.NuisanceParameter("CMS_reconstruction_ele", "lnN"),
            "weight_L1Prefiring": rl.NuisanceParameter(f"CMS_L1Prefiring_{year}", "lnN"),
            "weight_trigger_ele": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_trigger_{year}_ele", "lnN"),
            "weight_trigger_iso_mu": rl.NuisanceParameter(f"CMS_mu_trigger_iso_{year}_mu", "lnN"),
            "weight_trigger_noniso_mu": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_mu_trigger_{year}_mu", "lnN"),
        },
        # signal
        "ggF": {
            "weight_aS_weight": rl.NuisanceParameter("aS_Higgs_ggF", "lnN"),
            # "weight_UEPS_FSR": rl.NuisanceParameter("UEPS_FSR_ggF", "shape"),
            # "weight_UEPS_ISR": rl.NuisanceParameter("UEPS_ISR_ggF", "shape"),
            "weight_PDF_weight": rl.NuisanceParameter("pdf_Higgs_ggF", "lnN"),
            "weight_PDFaS_weight": rl.NuisanceParameter("pdfAS_Higgs_ggF", "lnN"),
            "weight_scalevar_3pt": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_scale_pt_3_ggF_{year}", "lnN"),
            "weight_scalevar_7pt": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_scale_pt_7_ggF_{year}", "lnN"),
        },
        "VBF": {
            "weight_aS_weight": rl.NuisanceParameter("aS_Higgs_VBF", "lnN"),
            # "weight_UEPS_FSR": rl.NuisanceParameter("UEPS_FSR_VBF", "shape"),
            # "weight_UEPS_ISR": rl.NuisanceParameter("UEPS_ISR_VBF", "shape"),
            "weight_PDF_weight": rl.NuisanceParameter("pdf_Higgs_VBF", "lnN"),
            "weight_PDFaS_weight": rl.NuisanceParameter("pdfAS_Higgs_VBF", "lnN"),
            "weight_scalevar_3pt": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_scale_pt_3_VBF_{year}", "lnN"),
            "weight_scalevar_7pt": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_scale_pt_7_VBF_{year}", "lnN"),
        },
        "VH": {},
        "ttH": {},
        # bkgs
        "TTbar": {},
        "WJetsLNu": {
            "weight_d1K_NLO": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_d1K_NLO_{year}", "lnN"),
            "weight_d2K_NLO": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_d2K_NLO_{year}", "lnN"),
            "weight_d3K_NLO": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_d3K_NLO_{year}", "lnN"),
            "weight_d1kappa_EW": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_W_d1kappa_EW_{year}", "lnN"),
            "weight_W_d2kappa_EW": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_W_d2kappa_EW_{year}", "lnN"),
            "weight_W_d3kappa_EW": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_W_d3kappa_EW_{year}", "lnN"),
        },
        "SingleTop": {},
        "DYJets": {
            "weight_d1kappa_EW": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_Z_d1kappa_EW_{year}", "lnN"),
            "weight_Z_d2kappa_EW": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_Z_d2kappa_EW_{year}", "lnN"),
            "weight_Z_d3kappa_EW": rl.NuisanceParameter(f"{CMS_PARAMS_LABEL}_Z_d3kappa_EW_{year}", "lnN"),
        },
        "QCD": {},
    }

    return systs_from_parquets
