import importlib.resources
import json
import os
import pathlib
import warnings

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from coffea import processor
from coffea.analysis_tools import PackedSelection, Weights
from coffea.nanoevents.methods import candidate

from boostedhiggs.corrections import (
    add_HiggsEW_kFactors,
    add_lepton_weight,
    add_pdf_weight,
    add_pileup_weight,
    add_ps_weight,
    add_scalevar_3pt,
    add_scalevar_7pt,
    add_VJets_kFactors,
    btagWPs,
    corrected_msoftdrop,
    get_btag_weights_farouk,
    get_jec_jets,
    met_factory,
)
from boostedhiggs.utils import match_H, match_Top, match_V

from .run_tagger_inference import runInferenceTriton

warnings.filterwarnings("ignore", message="Found duplicate branch ")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Missing cross-reference index ")
warnings.filterwarnings("ignore", message="divide by zero encountered in log")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
np.seterr(invalid="ignore")

import logging

logger = logging.getLogger(__name__)


def build_p4(cand):
    return ak.zip(
        {
            "pt": cand.pt,
            "eta": cand.eta,
            "phi": cand.phi,
            "mass": cand.mass,
            "charge": cand.charge,
        },
        with_name="PtEtaPhiMCandidate",
        behavior=candidate.behavior,
    )


class TopProcessor(processor.ProcessorABC):
    def __init__(
        self,
        year="2017",
        yearmod="",
        channels=["ele", "mu"],
        output_location="./outfiles/",
        inference=False,
        systematics=False,
        region="signal",
    ):
        """
        region can take ["signal", "zll", "qcd", "wjets"].
        """
        self._year = year
        self._yearmod = yearmod
        self._channels = channels
        self._region = region
        self._systematics = systematics
        # print(f"Will apply selections applicable to {region} region")

        self._output_location = output_location

        # trigger paths
        with importlib.resources.path("boostedhiggs.data", "triggers_all.json") as path:
            with open(path, "r") as f:
                self._HLTs = json.load(f)[self._year]

        # https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2
        with importlib.resources.path("boostedhiggs.data", "metfilters.json") as path:
            with open(path, "r") as f:
                self._metfilters = json.load(f)[self._year]

        if self._year == "2018":
            self.dataset_per_ch = {
                "ele": "EGamma",
                "mu": "SingleMuon",
            }
        else:
            self.dataset_per_ch = {
                "ele": "SingleElectron",
                "mu": "SingleMuon",
            }


        # do inference
        self.inference = inference
        # for tagger model and preprocessing dict
        self.tagger_resources_path = str(pathlib.Path(__file__).parent.resolve()) + "/tagger_resources/"

    @property
    def accumulator(self):
        return self._accumulator

    def save_dfs_parquet(self, fname, dfs_dict, ch):
        if self._output_location is not None:
            table = pa.Table.from_pandas(dfs_dict)
            if len(table) != 0:  # skip dataframes with empty entries
                pq.write_table(table, self._output_location + ch + "/parquet/" + fname + ".parquet")

    def ak_to_pandas(self, output_collection: ak.Array) -> pd.DataFrame:
        output = pd.DataFrame()
        for field in ak.fields(output_collection):
            output[field] = ak.to_numpy(output_collection[field])
        return output

    def add_selection(self, name: str, sel: np.ndarray, channel: str = "all"):
        """Adds selection to PackedSelection object and the cutflow dictionary"""
        channels = self._channels if channel == "all" else [channel]

        for ch in channels:
            if ch not in self._channels:
                logger.warning(f"Attempted to add selection to unexpected channel: {ch} not in %s" % (self._channels))
                continue

            # add selection
            self.selections[ch].add(name, sel)
            selection_ch = self.selections[ch].all(*self.selections[ch].names)

            if self.isMC:
                weight = self.weights[ch].partial_weight(["genweight"])
                self.cutflows[ch][name] = float(weight[selection_ch].sum())
            else:
                self.cutflows[ch][name] = np.sum(selection_ch)

    def process(self, events: ak.Array):
        """Returns skimmed events which pass preselection cuts and with the branches listed in self._skimvars"""

        dataset = events.metadata["dataset"]
        nevents = len(events)
        self.isMC = hasattr(events, "genWeight")
        self.weights = {ch: Weights(nevents, storeIndividual=True) for ch in self._channels}
        self.selections = {ch: PackedSelection() for ch in self._channels}
        self.cutflows = {ch: {} for ch in self._channels}

        sumgenweight = ak.sum(events.genWeight) if self.isMC else 0

        # trigger
        trigger = {}
        trigger_noiso = {}
        trigger_iso = {}
        for ch in self._channels:
            trigger[ch] = np.zeros(nevents, dtype="bool")
            trigger_noiso[ch] = np.zeros(nevents, dtype="bool")
            trigger_iso[ch] = np.zeros(nevents, dtype="bool")
            for t in self._HLTs[ch]:
                if t in events.HLT.fields:
                    if "Iso" in t or "WPTight_Gsf" in t:
                        trigger_iso[ch] = trigger_iso[ch] | events.HLT[t]
                    else:
                        trigger_noiso[ch] = trigger_noiso[ch] | events.HLT[t]
                    trigger[ch] = trigger[ch] | events.HLT[t]

        # metfilters
        metfilters = np.ones(nevents, dtype="bool")
        metfilterkey = "mc" if self.isMC else "data"
        for mf in self._metfilters[metfilterkey]:
            if mf in events.Flag.fields:
                metfilters = metfilters & events.Flag[mf]

        # taus
        loose_taus_mu = (events.Tau.pt > 20) & (abs(events.Tau.eta) < 2.3) & (events.Tau.idAntiMu >= 1)  # loose antiMu ID
        loose_taus_ele = (
            (events.Tau.pt > 20)
            & (abs(events.Tau.eta) < 2.3)
            & (events.Tau.idAntiEleDeadECal >= 2)  # loose Anti-electron MVA discriminator V6 (2018) ?
        )
        n_loose_taus_mu = ak.sum(loose_taus_mu, axis=1)
        n_loose_taus_ele = ak.sum(loose_taus_ele, axis=1)

        muons = ak.with_field(events.Muon, 0, "flavor")
        electrons = ak.with_field(events.Electron, 1, "flavor")

        # muons
        loose_muons = (
            (((muons.pt > 30) & (muons.pfRelIso04_all < 0.25)) | (muons.pt > 55))
            & (np.abs(muons.eta) < 2.4)
            & (muons.looseId)
        )
        n_loose_muons = ak.sum(loose_muons, axis=1)

        good_muons = (
            (muons.pt > 30)
            & (np.abs(muons.eta) < 2.4)
            & (np.abs(muons.dz) < 0.1)
            & (np.abs(muons.dxy) < 0.05)
            & (muons.sip3d <= 4.0)
            & muons.mediumId
        )
        n_good_muons = ak.sum(good_muons, axis=1)

        # electrons
        loose_electrons = (
            (((electrons.pt > 38) & (electrons.pfRelIso03_all < 0.25)) | (electrons.pt > 120))
            & (np.abs(electrons.eta) < 2.4)
            & ((np.abs(electrons.eta) < 1.44) | (np.abs(electrons.eta) > 1.57))
            & (electrons.cutBased >= electrons.LOOSE)
        )
        n_loose_electrons = ak.sum(loose_electrons, axis=1)

        good_electrons = (
            (electrons.pt > 38)
            & (np.abs(electrons.eta) < 2.4)
            & ((np.abs(electrons.eta) < 1.44) | (np.abs(electrons.eta) > 1.57))
            & (np.abs(electrons.dz) < 0.1)
            & (np.abs(electrons.dxy) < 0.05)
            & (electrons.sip3d <= 4.0)
            & (electrons.mvaFall17V2noIso_WP90)
        )
        n_good_electrons = ak.sum(good_electrons, axis=1)

        # get candidate lepton
        goodleptons = ak.concatenate([muons[good_muons], electrons[good_electrons]], axis=1)  # concat muons and electrons
        # goodleptons = ak.concatenate(muons, electrons, axis=1)
        goodleptons = goodleptons[ak.argsort(goodleptons.pt, ascending=False)]  # sort by pt

        candidatelep = ak.firsts(goodleptons)  # pick highest pt

        # ak4 jets
        ak4_jet_selector_no_btag = (
            (events.Jet.pt > 30) & (abs(events.Jet.eta) < 5.0) & events.Jet.isTight & (events.Jet.puId > 0)
        )
        # reject EE noisy jets for 2017
        if self._year == "2017":
            ak4_jet_selector_no_btag = ak4_jet_selector_no_btag & (
                (events.Jet.pt > 50) | (abs(events.Jet.eta) < 2.65) | (abs(events.Jet.eta) > 3.139)
            )

        goodjets = events.Jet[ak4_jet_selector_no_btag]

        ht = ak.sum(goodjets.pt, axis=1)

        # fatjets
        fatjets = events.FatJet
        fatjets["msdcorr"] = corrected_msoftdrop(fatjets)

        good_fatjets = (fatjets.pt > 200) & (abs(fatjets.eta) < 2.5) & fatjets.isTight # kinematic cut
        bjet_mask = (
                        (ak.sum(fatjets.btagDeepB > btagWPs["deepCSV"][self._year]["T"], axis=1) == 2) | \
                        (ak.sum(goodjets.btagDeepB > btagWPs["deepCSV"][self._year]["T"], axis=1) == 2)
                    ) | \
                    (
                        (ak.sum(fatjets.btagDeepB > btagWPs["deepCSV"][self._year]["T"], axis=1) == 1) & \
                        (ak.sum(goodjets.btagDeepB > btagWPs["deepCSV"][self._year]["T"], axis=1) == 1)
                    )

        good_fatjets = good_fatjets & bjet_mask
        
        good_fatjets = fatjets[good_fatjets]  # select good fatjets
        good_fatjets = good_fatjets[ak.argsort(good_fatjets.pt, ascending=False)]  # sort them by pt

              
        first_fatjet = ak.firsts(good_fatjets[:,0:1]) # keep the first and second highest pt jets
        second_fatjet = ak.firsts(good_fatjets[:,1:2])

        two_fatjets = ~ak.is_none(first_fatjet) & ~ak.is_none(second_fatjet) # only keep events with twp jets
        one_lep = ~ak.is_none(candidatelep)

        all_cond = two_fatjets & one_lep
        
        first_fatjet = first_fatjet.mask[all_cond]        
        second_fatjet = second_fatjet.mask[all_cond]
        candidatelep = candidatelep.mask[all_cond]


        delta_r_first = first_fatjet.delta_r(candidatelep)
        delta_r_second = second_fatjet.delta_r(candidatelep)


        closer_fatjet = delta_r_first < delta_r_second
        lep_fatjet = ak.where(closer_fatjet, first_fatjet, second_fatjet)
        lep_fatjet_dR = ak.where(closer_fatjet, delta_r_first, delta_r_second)
        
        had_fatjet = ak.where(closer_fatjet, second_fatjet, first_fatjet)

        
        variables = {
            'lep_fatjet_dR': lep_fatjet_dR
        }

        lep_fatjetvars = {
            "lep_fatjetPt": lep_fatjet.pt,
            "lep_fatjetEta": lep_fatjet.eta,
            "lep_fatjetPhi": lep_fatjet.phi,
            "lep_fatjetMass": lep_fatjet.msdcorr,
        }

        had_fatjetvars = {
            "had_fatjetPt": had_fatjet.pt,
            "had_fatjetEta": had_fatjet.eta,
            "had_fatjetPhi": had_fatjet.phi,
            "had_fatjetMass": had_fatjet.msdcorr,
        }

        lepvars = {
            "lepPt": candidatelep.pt,
            "lepEta": candidatelep.eta,
            "lepPhi": candidatelep.phi,
            "lepMass": candidatelep.mass,
        }

        variables = {**variables, **lep_fatjetvars, **had_fatjetvars, **lepvars}

        """
        HEM issue: Hadronic calorimeter Endcaps Minus (HEM) issue.
        The endcaps of the hadron calorimeter failed to cover the phase space at -3 < eta < -1.3 and -1.57 < phi < -0.87
        during the 2018 data C and D.
        The transverse momentum of the jets in this region is typically under-measured, this results in over-measured MET.
        It could also result on new electrons.
        We must veto the jets and met in this region.
        Should we veto on AK8 jets or electrons too?
        Let's add this as a cut to check first.
        """
        if self._year == "2018":
            hem_cleaning = (
                ((events.run >= 319077) & (not self.isMC))  # if data check if in Runs C or D
                # else for MC randomly cut based on lumi fraction of C&D
                | ((np.random.rand(len(events)) < 0.632) & self.isMC)
            ) & (
                ak.any(
                    (
                        (events.Jet.pt > 30.0)
                        & (events.Jet.eta > -3.2)
                        & (events.Jet.eta < -1.3)
                        & (events.Jet.phi > -1.57)
                        & (events.Jet.phi < -0.87)
                    ),
                    -1,
                )
                | ((events.MET.phi > -1.62) & (events.MET.pt < 470.0) & (events.MET.phi < -0.62))
            )
            self.add_selection(name="HEMCleaning", sel=~hem_cleaning)

        # apply trigger
        for ch in self._channels:
            self.add_selection(name="Trigger", sel=trigger[ch], channel=ch)

        # apply selections

        self.add_selection(name="LepMatch", sel=(lep_fatjet_dR<.8))
        
        self.add_selection(
            name="OneLep",
            sel=(n_good_muons == 1)
            & (n_good_electrons == 0)
            & (n_loose_electrons == 0)
            & ~ak.any(loose_muons & ~good_muons, 1)
            & (n_loose_taus_mu == 0),
            channel="mu",
        )
        self.add_selection(
            name="OneLep",
            sel=(n_good_muons == 0)
            & (n_loose_muons == 0)
            & (n_good_electrons == 1)
            & ~ak.any(loose_electrons & ~good_electrons, 1)
            & (n_loose_taus_ele == 0),
            channel="ele",
            )

        

        # gen-level matching


        # initialize pandas dataframe
        output = {}

        for ch in self._channels:
            selection_ch = self.selections[ch].all(*self.selections[ch].names)

            fill_output = True
            # for data, only fill output for the dataset needed
            if not self.isMC and self.dataset_per_ch[ch] not in dataset:
                fill_output = False
            # only fill output for that channel if the selections yield any events
            if np.sum(selection_ch) <= 0:
                fill_output = False

            if fill_output:
                out = {}
                for var, item in variables.items():
                    # pad all the variables that are not a cut with -1
                    # pad_item = item if ("cut" in var or "weight" in var) else pad_val(item, -1)
                    # fill out dictionary
                    out[var] = item

                # fill the output dictionary after selections
                output[ch] = {key: value[selection_ch] for (key, value) in out.items()}

                # fill inference
                if self.inference:
                    for model_name in [
                        # "particlenet_hww_inclv2_pre2",
                        # "ak8_MD_vminclv2ParT_manual_fixwrap",
                        "ak8_MD_vminclv2ParT_manual_fixwrap_all_nodes",
                    ]:
                        pnet_vars = runInferenceTriton(
                            self.tagger_resources_path,
                            events[selection_ch],
                            fj_idx_lep[selection_ch],
                            model_name=model_name,
                        )
                        pnet_df = self.ak_to_pandas(pnet_vars)

                        hwwev = [
                            "fj_ParT_probHWqqWev0c",
                            "fj_ParT_probHWqqWev1c",
                            "fj_ParT_probHWqqWtauev0c",
                            "fj_ParT_probHWqqWtauev1c",
                        ]
                        hwwmv = [
                            "fj_ParT_probHWqqWmv0c",
                            "fj_ParT_probHWqqWmv1c",
                            "fj_ParT_probHWqqWtaumv0c",
                            "fj_ParT_probHWqqWtaumv1c",
                        ]
                        hwwhad = [
                            "fj_ParT_probHWqqWqq0c",
                            "fj_ParT_probHWqqWqq1c",
                            "fj_ParT_probHWqqWqq2c",
                            "fj_ParT_probHWqqWq0c",
                            "fj_ParT_probHWqqWq1c",
                            "fj_ParT_probHWqqWq2c",
                            "fj_ParT_probHWqqWtauhv0c",
                            "fj_ParT_probHWqqWtauhv1c",
                        ]
                        sigs = hwwev + hwwmv + hwwhad

                        scores = {"fj_ParT_score": pnet_df[sigs].sum(axis=1).values}

                        hidNeurons = {}
                        for key in pnet_vars:
                            if "hidNeuron" in key:
                                hidNeurons[key] = pnet_vars[key]

                        reg_mass = {"fj_ParT_mass": pnet_vars["fj_ParT_mass"]}
                        output[ch] = {**output[ch], **scores, **reg_mass, **hidNeurons}

            else:
                output[ch] = {}

            # convert arrays to pandas
            if not isinstance(output[ch], pd.DataFrame):
                output[ch] = self.ak_to_pandas(output[ch])

            if "rec_higgs_m" in output[ch].keys():
                output[ch]["rec_higgs_m"] = np.nan_to_num(output[ch]["rec_higgs_m"], nan=-1)

        # now save pandas dataframes
        fname = events.behavior["__events_factory__"]._partition_key.replace("/", "_")
        fname = "condor_" + fname

        for ch in self._channels:  # creating directories for each channel
            if not os.path.exists(self._output_location + ch):
                os.makedirs(self._output_location + ch)
            if not os.path.exists(self._output_location + ch + "/parquet"):
                os.makedirs(self._output_location + ch + "/parquet")
            self.save_dfs_parquet(fname, output[ch], ch)

        # return dictionary with cutflows
        return {
            dataset: {
                "mc": self.isMC,
                self._year
                + self._yearmod: {
                    "sumgenweight": sumgenweight,
                    "cutflows": self.cutflows,
                },
            }
        }

    def postprocess(self, accumulator):
        return accumulator
