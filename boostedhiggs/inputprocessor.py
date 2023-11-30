"""
Skimmer for ParticleNet tagger inputs.

Author(s): Cristina Mantilla Suarez, Raghav Kansal
"""
import os
import pathlib
import warnings
from typing import Dict

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import uproot
from coffea.analysis_tools import PackedSelection
from coffea.nanoevents.methods import candidate
from coffea.processor import ProcessorABC, dict_accumulator

from .corrections import btagWPs
from .run_tagger_inference import runInferenceTriton

# from .run_tagger_inference import runInferenceTriton
from .utils import FILL_NONE_VALUE, add_selection_no_cutflow, sigs, tagger_gen_matching

warnings.filterwarnings("ignore", message="Found duplicate branch ")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Missing cross-reference index ")
warnings.filterwarnings("ignore", message="divide by zero encountered in log")
np.seterr(invalid="ignore")


class InputProcessor(ProcessorABC):
    """
    Produces a flat training ntuple from PFNano.
    """

    def __init__(self, year, label, inference, output_location="./outfiles/"):
        self._year = year
        self.label = label
        self.inference = inference
        self._output_location = output_location

        self.skim_vars = {
            "GenPart": [
                "fj_genjetmass",
                "fj_genRes_pt",
                "fj_genRes_eta",
                "fj_genRes_phi",
                "fj_genRes_mass",
                "fj_nprongs",
                "fj_ncquarks",
                "fj_lepinprongs",
                "fj_nquarks",
                "fj_H_VV",
                "fj_H_VV_isMatched",
                "fj_H_VV_4q",
                "fj_H_VV_elenuqq",
                "fj_H_VV_munuqq",
                "fj_H_VV_leptauelvqq",
                "fj_H_VV_leptaumuvqq",
                "fj_H_VV_hadtauvqq",
                "fj_QCDb",
                "fj_QCDbb",
                "fj_QCDc",
                "fj_QCDcc",
                "fj_QCDothers",
                "fj_V_2q",
                "fj_V_elenu",
                "fj_V_munu",
                "fj_V_taunu",
                "fj_Top_nquarksnob",
                "fj_Top_nbquarks",
                "fj_Top_ncquarks",
                "fj_Top_nleptons",
                "fj_Top_nele",
                "fj_Top_nmu",
                "fj_Top_ntau",
                "fj_Top_taudecay",
            ],
            "FatJet": [
                "eta",
                "phi",
                "mass",
                "pt",
                "msoftdrop",
            ],
        }

        self.tagger_resources_path = str(pathlib.Path(__file__).parent.resolve()) + "/tagger_resources/"

        self.fatjet_label = "FatJet"
        self.pfcands_label = "FatJetPFCands"
        self.svs_label = "FatJetSVs"

        self._accumulator = dict_accumulator({})

    @property
    def accumulator(self):
        return self._accumulator

    def save_dfs_parquet(self, df, fname):
        if self._output_location is not None:
            PATH = f"{self._output_location}/parquet/"
            if not os.path.exists(PATH):
                os.makedirs(PATH)

            table = pa.Table.from_pandas(df)
            if len(table) != 0:  # skip dataframes with empty entries
                pq.write_table(table, f"{PATH}/{fname}.parquet")

    def ak_to_pandas(self, output_collection: ak.Array) -> pd.DataFrame:
        output = pd.DataFrame()
        for field in ak.fields(output_collection):
            output[field] = ak.to_numpy(output_collection[field])
        return output

    def dump_root(self, skimmed_vars: Dict[str, np.array], fname: str) -> None:
        """
        Saves ``jet_vars`` dict as a rootfile to './outroot'
        """
        local_dir = os.path.abspath(os.path.join(self._output_location, "outroot"))
        os.system(f"mkdir -p {local_dir}")

        with uproot.recreate(f"{local_dir}/{fname}.root", compression=uproot.LZ4(4)) as rfile:
            rfile["Events"] = ak.Array(skimmed_vars)
            rfile["Events"].show()

    def process(self, events: ak.Array):
        import time

        start = time.time()

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

        met = events.MET

        # lepton
        electrons = events["Electron"][events["Electron"].pt > 40]
        muons = events["Muon"][events["Muon"].pt > 30]
        leptons = ak.concatenate([electrons, muons], axis=1)
        leptons = leptons[ak.argsort(leptons.pt, ascending=False)]
        candidatelep = ak.firsts(leptons)
        candidatelep_p4 = build_p4(candidatelep)

        lep_reliso = (
            candidatelep.pfRelIso04_all if hasattr(candidatelep, "pfRelIso04_all") else candidatelep.pfRelIso03_all
        )  # reliso for candidate lepton
        lep_miso = candidatelep.miniPFRelIso_all  # miniso for candidate lepton

        # fatjet
        fatjets = events[self.fatjet_label]
        good_fatjets = (fatjets.pt > 200) & (abs(fatjets.eta) < 2.5) & fatjets.isTight
        good_fatjets = fatjets[good_fatjets]  # select good fatjets
        good_fatjets = good_fatjets[ak.argsort(good_fatjets.pt, ascending=False)]  # sort them by pt

        # candidatefj
        fj_idx_lep = ak.argmin(good_fatjets.delta_r(candidatelep_p4), axis=1, keepdims=True)
        candidatefj = ak.firsts(good_fatjets[fj_idx_lep])

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

        dr_jet_lepfj = goodjets.delta_r(candidatefj)
        ak4_outside_ak8 = goodjets[dr_jet_lepfj > 0.8]

        # VBF variables
        jet1 = ak4_outside_ak8[:, 0:1]
        jet2 = ak4_outside_ak8[:, 1:2]
        deta = abs(ak.firsts(jet1).eta - ak.firsts(jet2).eta)
        mjj = (ak.firsts(jet1) + ak.firsts(jet2)).mass

        # rec_higgs
        candidateNeutrino = ak.zip(
            {
                "pt": met.pt,
                "eta": candidatelep_p4.eta,
                "phi": met.phi,
                "mass": 0,
                "charge": 0,
            },
            with_name="PtEtaPhiMCandidate",
            behavior=candidate.behavior,
        )
        rec_W_lnu = candidatelep_p4 + candidateNeutrino
        rec_W_qq = candidatefj - candidatelep_p4
        rec_higgs = rec_W_qq + rec_W_lnu

        # selection
        selection = PackedSelection()
        add_selection_no_cutflow("fjselection", (candidatefj.pt > 200), selection)

        if np.sum(selection.all(*selection.names)) == 0:
            return {}

        # variables
        genparts = events.GenPart
        matched_mask, genVars = tagger_gen_matching(
            events,
            genparts,
            candidatefj,
            self.skim_vars["GenPart"],
            label=self.label,
        )

        FatJetVars = {f"fj_{var}": ak.fill_none(candidatefj[var], FILL_NONE_VALUE) for var in self.skim_vars["FatJet"]}

        LepVars = {}
        LepVars["lep_dR_fj"] = candidatelep_p4.delta_r(candidatefj).to_numpy().filled(fill_value=0)
        LepVars["lep_pt"] = (candidatelep_p4.pt).to_numpy().filled(fill_value=0)
        LepVars["lep_pt_ratio"] = (candidatelep_p4.pt / candidatefj.pt).to_numpy().filled(fill_value=0)
        LepVars["lep_reliso"] = lep_reliso.to_numpy().filled(fill_value=0)
        LepVars["lep_miso"] = lep_miso.to_numpy().filled(fill_value=0)

        Others = {}
        Others["n_bjets_L"] = (
            ak.sum(ak4_outside_ak8.btagDeepFlavB > btagWPs["deepJet"][self._year]["L"], axis=1)
            .to_numpy()
            .filled(fill_value=0)
        )
        Others["n_bjets_M"] = (
            ak.sum(ak4_outside_ak8.btagDeepFlavB > btagWPs["deepJet"][self._year]["M"], axis=1)
            .to_numpy()
            .filled(fill_value=0)
        )
        Others["n_bjets_T"] = (
            ak.sum(ak4_outside_ak8.btagDeepFlavB > btagWPs["deepJet"][self._year]["T"], axis=1)
            .to_numpy()
            .filled(fill_value=0)
        )

        Others["rec_W_lnu_pt"] = rec_W_lnu.pt.to_numpy().filled(fill_value=0)
        Others["rec_W_lnu_m"] = rec_W_lnu.mass.to_numpy().filled(fill_value=0)
        Others["rec_W_qq_pt"] = rec_W_qq.pt.to_numpy().filled(fill_value=0)
        Others["rec_W_qq_m"] = rec_W_qq.mass.to_numpy().filled(fill_value=0)
        Others["rec_higgs_pt"] = rec_higgs.pt.to_numpy().filled(fill_value=0)
        Others["rec_higgs_m"] = rec_higgs.mass.to_numpy().filled(fill_value=0)

        Others["mjj"] = mjj.to_numpy().filled(fill_value=0)
        Others["deta"] = deta.to_numpy().filled(fill_value=0)

        METVars = {}
        METVars["met_pt"] = met.pt
        METVars["met_relpt"] = met.pt / candidatefj.pt
        METVars["met_fj_dphi"] = met.delta_phi(candidatefj)

        skimmed_vars = {**FatJetVars, **{"matched_mask": matched_mask}, **genVars, **METVars, **LepVars, **Others}

        # apply selections
        skimmed_vars = {
            key: np.squeeze(np.array(value[selection.all(*selection.names)])) for (key, value) in skimmed_vars.items()
        }

        # fill inference
        assert self.inference is True, "enable --inference to run skimmer"

        for model_name in ["ak8_MD_vminclv2ParT_manual_fixwrap_all_nodes"]:
            pnet_vars = runInferenceTriton(
                self.tagger_resources_path,
                events[selection.all(*selection.names)],
                fj_idx_lep[selection.all(*selection.names)],
                model_name=model_name,
            )

            # pnet_df = self.ak_to_pandas(pnet_vars)
            pnet_df = pd.DataFrame(pnet_vars)

            scores = {"fj_ParT_score": (pnet_df[sigs].sum(axis=1)).values}
            reg_mass = {"fj_ParT_mass": pnet_vars["fj_ParT_mass"]}

            hidNeurons = {}
            for key in pnet_vars:
                if "hidNeuron" in key:
                    hidNeurons[key] = pnet_vars[key]

            skimmed_vars = {**skimmed_vars, **scores, **reg_mass, **hidNeurons}

        for key in skimmed_vars:
            skimmed_vars[key] = skimmed_vars[key].squeeze()

        # convert output to pandas
        df = pd.DataFrame(skimmed_vars)

        # for key in self.skim_vars:
        #     for keykey in self.skim_vars[key]:
        #         assert (
        #             keykey in df.keys()
        #         ), f"make sure you are computing and storing {keykey} in the skimmed_vars dictionnary"

        df = df.dropna()  # very few events would have genjetmass NaN for some reason

        print(f"convert: {time.time() - start:.1f}s")

        print(df)

        # save the output
        fname = events.behavior["__events_factory__"]._partition_key.replace("/", "_")
        fname = "condor_" + fname

        self.save_dfs_parquet(df, fname)

        print(f"dump parquet: {time.time() - start:.1f}s")

        # # TODO: drop NaNs from rootfiles
        # self.dump_root(skimmed_vars, fname)
        # print(f"dump rootfile: {time.time() - start:.1f}s")

        # for now do something like this to dump the parquets in root
        # OUTPATH = "../datafiles/ntuples/"

        # for sample in samples:
        #     print(sample)

        #     for file in os.listdir(f"{OUTPATH}/{sample}/train/"):
        #         if "parquet" not in file:
        #             continue

        #         d = pd.read_parquet(f"{OUTPATH}/{sample}/train/{file}")

        #         with uproot.recreate(f"{OUTPATH}/{sample}/train/out.root", compression=uproot.LZ4(4)) as rfile:
        #             rfile["Events"] = ak.Array(d.to_dict(orient="list", index=True))
        #             rfile["Events"].show()

        #     for file in os.listdir(f"{OUTPATH}/{sample}/test/"):
        #         if "parquet" not in file:
        #             continue

        #         d = pd.read_parquet(f"{OUTPATH}/{sample}/test/{file}")

        #         with uproot.recreate(f"{OUTPATH}/{sample}/test/out.root", compression=uproot.LZ4(4)) as rfile:
        #             rfile["Events"] = ak.Array(d.to_dict(orient="list", index=True))
        #             rfile["Events"].show()
        #     print("--------------------------")

        return {}

    def postprocess(self, accumulator):
        pass
