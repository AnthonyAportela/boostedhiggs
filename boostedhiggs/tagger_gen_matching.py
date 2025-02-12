from typing import Dict, Tuple, Union

import awkward as ak
import numpy as np
from coffea.nanoevents.methods.base import NanoEventsArray
from coffea.nanoevents.methods.nanoaod import FatJetArray, GenParticleArray

d_PDGID = 1
c_PDGID = 4
b_PDGID = 5
g_PDGID = 21
TOP_PDGID = 6

ELE_PDGID = 11
vELE_PDGID = 12
MU_PDGID = 13
vMU_PDGID = 14
TAU_PDGID = 15
vTAU_PDGID = 16

GAMMA_PDGID = 22
Z_PDGID = 23
W_PDGID = 24
HIGGS_PDGID = 25

PI_PDGID = 211
PO_PDGID = 221
PP_PDGID = 111

GEN_FLAGS = ["fromHardProcess", "isLastCopy"]

FILL_NONE_VALUE = -99999

JET_DR = 0.8


def get_pid_mask(genparts: GenParticleArray, pdgids: Union[int, list], ax: int = 2, byall: bool = True) -> ak.Array:
    """
    Get selection mask for gen particles matching any of the pdgIds in ``pdgids``.
    If ``byall``, checks all particles along axis ``ax`` match.
    """
    gen_pdgids = abs(genparts.pdgId)

    if type(pdgids) is list:
        mask = gen_pdgids == pdgids[0]
        for pdgid in pdgids[1:]:
            mask = mask | (gen_pdgids == pdgid)
    else:
        mask = gen_pdgids == pdgids

    return ak.all(mask, axis=ax) if byall else mask


def to_label(array: ak.Array) -> ak.Array:
    return ak.values_astype(array, np.int32)


def match_H(genparts: GenParticleArray, fatjet: FatJetArray):
    """Gen matching for Higgs samples"""
    higgs = genparts[get_pid_mask(genparts, HIGGS_PDGID, byall=False) * genparts.hasFlags(GEN_FLAGS)]

    # only select events that match an specific decay
    # matched_higgs = higgs[ak.argmin(fatjet.delta_r(higgs), axis=1, keepdims=True)][:, 0]
    matched_higgs = higgs[ak.argmin(fatjet.delta_r(higgs), axis=1, keepdims=True)]
    matched_higgs_mask = ak.any(fatjet.delta_r(matched_higgs) < 0.8, axis=1)

    matched_higgs = ak.firsts(matched_higgs)

    matched_higgs_children = matched_higgs.children
    higgs_children = higgs.children

    children_mask = get_pid_mask(matched_higgs_children, [W_PDGID], byall=False)
    is_hww = ak.any(children_mask, axis=1)

    # order by mass, select lower mass child as V* and higher as V
    matched_higgs_children = matched_higgs_children[children_mask]
    children_mass = matched_higgs_children.mass
    v_star = ak.firsts(matched_higgs_children[ak.argmin(children_mass, axis=1, keepdims=True)])
    v = ak.firsts(matched_higgs_children[ak.argmax(children_mass, axis=1, keepdims=True)])

    # VV daughters
    # requires coffea-0.7.21
    all_daus = higgs_children.distinctChildrenDeep
    all_daus = ak.flatten(all_daus, axis=2)
    all_daus_flat = ak.flatten(all_daus, axis=2)
    all_daus_flat_pdgId = abs(all_daus_flat.pdgId)

    # the following tells you about the decay
    num_quarks = ak.sum(all_daus_flat_pdgId <= b_PDGID, axis=1)
    num_leptons = ak.sum(
        (all_daus_flat_pdgId == ELE_PDGID) | (all_daus_flat_pdgId == MU_PDGID) | (all_daus_flat_pdgId == TAU_PDGID),
        axis=1,
    )
    num_electrons = ak.sum(all_daus_flat_pdgId == ELE_PDGID, axis=1)
    num_muons = ak.sum(all_daus_flat_pdgId == MU_PDGID, axis=1)
    num_taus = ak.sum(all_daus_flat_pdgId == TAU_PDGID, axis=1)

    # the following tells you about the matching
    # prongs except neutrino
    neutrinos = (
        (all_daus_flat_pdgId == vELE_PDGID) | (all_daus_flat_pdgId == vMU_PDGID) | (all_daus_flat_pdgId == vTAU_PDGID)
    )
    leptons = (all_daus_flat_pdgId == ELE_PDGID) | (all_daus_flat_pdgId == MU_PDGID) | (all_daus_flat_pdgId == TAU_PDGID)

    # num_m: number of matched leptons
    # number of quarks excludes neutrino and leptons
    num_m_quarks = ak.sum(fatjet.delta_r(all_daus_flat[~neutrinos & ~leptons]) < JET_DR, axis=1)
    num_m_leptons = ak.sum(fatjet.delta_r(all_daus_flat[leptons]) < JET_DR, axis=1)
    num_m_cquarks = ak.sum(fatjet.delta_r(all_daus_flat[all_daus_flat.pdgId == b_PDGID]) < JET_DR, axis=1)

    lep_daughters = all_daus_flat[leptons]
    # parent = ak.firsts(lep_daughters[fatjet.delta_r(lep_daughters) < JET_DR].distinctParent)
    parent = ak.firsts(lep_daughters.distinctParent)
    iswlepton = parent.mass == v.mass
    iswstarlepton = parent.mass == v_star.mass

    genVars = {"fj_genH_pt": ak.fill_none(higgs.pt, FILL_NONE_VALUE)}

    genVVars = {
        "fj_genH_jet": fatjet.delta_r(higgs[:, 0]),
        "fj_genV_dR": fatjet.delta_r(v),
        "fj_genVstar": fatjet.delta_r(v_star),
        "genV_genVstar_dR": v.delta_r(v_star),
    }

    genHVVVars = {
        "fj_isHVV": is_hww,
        "fj_isHVV_Matched": matched_higgs_mask,
        "fj_isHVV_4q": to_label((num_quarks == 4) & (num_leptons == 0)),
        "fj_isHVV_elenuqq": to_label((num_electrons == 1) & (num_quarks == 2) & (num_leptons == 1)),
        "fj_isHVV_munuqq": to_label((num_muons == 1) & (num_quarks == 2) & (num_leptons == 1)),
        "fj_isHVV_taunuqq": to_label((num_taus == 1) & (num_quarks == 2) & (num_leptons == 1)),
        "fj_isHVV_Vlepton": iswlepton,
        "fj_isHVV_Vstarlepton": iswstarlepton,
        "fj_genRes_mass": matched_higgs.mass,
        "fj_nquarks": num_m_quarks,
        "fj_ncquarks": num_m_cquarks,
        "fj_lepinprongs": num_m_leptons,
    }

    genVars = {**genVars, **genVVars, **genHVVVars}

    return genVars, is_hww


def match_V(genparts: GenParticleArray, fatjet: FatJetArray):
    vs = genparts[get_pid_mask(genparts, [W_PDGID, Z_PDGID], byall=False) * genparts.hasFlags(GEN_FLAGS)]
    matched_vs = vs[ak.argmin(fatjet.delta_r(vs), axis=1, keepdims=True)]
    matched_vs_mask = ak.any(fatjet.delta_r(matched_vs) < JET_DR, axis=1)

    daughters = ak.flatten(matched_vs.distinctChildren, axis=2)
    daughters = daughters[daughters.hasFlags(["fromHardProcess", "isLastCopy"])]
    daughters_pdgId = abs(daughters.pdgId)
    decay = (
        # 2 quarks * 1
        (ak.sum(daughters_pdgId < b_PDGID, axis=1) == 2) * 1
        # >=1 electron * 3
        + (ak.sum(daughters_pdgId == ELE_PDGID, axis=1) >= 1) * 3
        # >=1 muon * 5
        + (ak.sum(daughters_pdgId == MU_PDGID, axis=1) >= 1) * 5
        # >=1 tau * 7
        + (ak.sum(daughters_pdgId == TAU_PDGID, axis=1) >= 1) * 7
    )

    daughters_nov = daughters[
        ((daughters_pdgId != vELE_PDGID) & (daughters_pdgId != vMU_PDGID) & (daughters_pdgId != vTAU_PDGID))
    ]
    nprongs = ak.sum(fatjet.delta_r(daughters_nov) < JET_DR, axis=1)

    lepdaughters = daughters[
        ((daughters_pdgId == ELE_PDGID) | (daughters_pdgId == MU_PDGID) | (daughters_pdgId == TAU_PDGID))
    ]
    lepinprongs = 0
    if len(lepdaughters) > 0:
        lepinprongs = ak.sum(fatjet.delta_r(lepdaughters) < JET_DR, axis=1)  # should be 0 or 1

    # number of c quarks
    cquarks = daughters_nov[abs(daughters_nov.pdgId) == c_PDGID]
    ncquarks = ak.sum(fatjet.delta_r(cquarks) < JET_DR, axis=1)

    matched_vdaus_mask = ak.any(fatjet.delta_r(daughters) < 0.8, axis=1)
    matched_mask = matched_vs_mask & matched_vdaus_mask
    genVars = {
        "fj_isV": np.ones(len(genparts), dtype="bool"),
        "fj_isV_Matched": matched_mask,
        "fj_isV_2q": to_label(decay == 1),
        "fj_isV_elenu": to_label(decay == 3),
        "fj_isV_munu": to_label(decay == 5),
        "fj_isV_taunu": to_label(decay == 7),
        "fj_nprongs": nprongs,
        "fj_lepinprongs": lepinprongs,
        "fj_ncquarks": ncquarks,
    }

    genVars["fj_isV_lep"] = (genVars["fj_isV_elenu"] == 1) | (genVars["fj_isV_munu"] == 1) | (genVars["fj_isV_taunu"] == 1)

    return genVars, matched_mask


def match_Top(genparts: GenParticleArray, fatjet: FatJetArray):
    tops = genparts[get_pid_mask(genparts, TOP_PDGID, byall=False) * genparts.hasFlags(GEN_FLAGS)]
    # tops = tops[ak.argsort(tops.pt, ascending=False)]
    matched_tops = tops[fatjet.delta_r(tops) < JET_DR]
    num_matched_tops = ak.sum(fatjet.delta_r(matched_tops) < JET_DR, axis=1)

    # take all possible daughters!
    daughters = ak.flatten(tops.distinctChildren, axis=2)
    daughters = daughters[daughters.hasFlags(["fromHardProcess", "isLastCopy"])]
    daughters_pdgId = abs(daughters.pdgId)

    wboson_daughters = ak.flatten(daughters[(daughters_pdgId == W_PDGID)].distinctChildren, axis=2)
    wboson_daughters = wboson_daughters[wboson_daughters.hasFlags(["fromHardProcess", "isLastCopy"])]
    wboson_daughters_pdgId = abs(wboson_daughters.pdgId)

    # take all possible granddaughters!
    granddaughters = ak.flatten(tops.distinctChildren.distinctChildren, axis=3)
    granddaughters = granddaughters[granddaughters.hasFlags(["fromHardProcess", "isLastCopy"])]
    granddaughters_pdgId = abs(granddaughters.pdgId) 

    # print('\n',granddaughters_pdgId,'\n')
    
    bquark = daughters[(daughters_pdgId == b_PDGID)]
    neutrinos = (
        (wboson_daughters_pdgId == vELE_PDGID)
        | (wboson_daughters_pdgId == vMU_PDGID)
        | (wboson_daughters_pdgId == vTAU_PDGID)
    )
    leptons = (
        (wboson_daughters_pdgId == ELE_PDGID) | (wboson_daughters_pdgId == MU_PDGID) | (wboson_daughters_pdgId == TAU_PDGID)
    )    
    
    grandleptons = (
        (granddaughters_pdgId == ELE_PDGID) | (granddaughters_pdgId == MU_PDGID)
    )

    # print('\n',ak.flatten(granddaughters[grandleptons], axis=2).pdgId,'\n')

    # get leptons that come from tops
    top_leps = ak.flatten(granddaughters[grandleptons], axis=2)
    top_leps = top_leps[ak.argsort(top_leps.pt, ascending=False)]

    # get b's that come from the leptonic top
    W_lep = wboson_daughters[leptons]
    # print('\n',W_lep.pdgId,'\n')

    lep_W = W_lep.distinctParent
    lep_W = lep_W[lep_W.hasFlags(["fromHardProcess", "isLastCopy"])]
    # print('\n',lep_W.pdgId,'\n')

    lep_t = lep_W.distinctParent
    lep_t = lep_t[lep_t.hasFlags(["fromHardProcess", "isLastCopy"])]
    # print('\n',lep_t.pdgId,'\n')

    lep_t_children = lep_t.distinctChildren
    lep_t_children = lep_t_children[lep_t_children.hasFlags(["fromHardProcess", "isLastCopy"])]
    # print('\n',lep_t_children.pdgId,'\n')

    lep_b = ak.flatten(lep_t_children[abs(lep_t_children.pdgId) == b_PDGID],axis=2)
    # print('\n',lep_b.pdgId,'\n')

    
    quarks = ~leptons & ~neutrinos
    cquarks = wboson_daughters_pdgId == c_PDGID
    electrons = wboson_daughters_pdgId == ELE_PDGID
    muons = wboson_daughters_pdgId == MU_PDGID
    taus = wboson_daughters_pdgId == TAU_PDGID

    # get tau decays from V daughters
    taudaughters = wboson_daughters[(wboson_daughters_pdgId == TAU_PDGID)].children
    taudaughters = taudaughters[taudaughters.hasFlags(["isLastCopy"])]
    taudaughters_pdgId = abs(taudaughters.pdgId)
    taudecay = (
        # pions/kaons (hadronic tau) * 1
        (
            ak.sum(
                (taudaughters_pdgId == ELE_PDGID) | (taudaughters_pdgId == MU_PDGID),
                axis=2,
            )
            == 0
        )
        * 1
        # 1 electron * 3
        + (ak.sum(taudaughters_pdgId == ELE_PDGID, axis=2) == 1) * 3
        # 1 muon * 5
        + (ak.sum(taudaughters_pdgId == MU_PDGID, axis=2) == 1) * 5
    )
    # flatten taudecay - so painful
    taudecay = ak.sum(taudecay, axis=-1)

    # get number of matched daughters
    num_m_quarks_nob = ak.sum(fatjet.delta_r(wboson_daughters[quarks]) < JET_DR, axis=1)
    num_m_bquarks = ak.sum(fatjet.delta_r(bquark) < JET_DR, axis=1)
    num_m_cquarks = ak.sum(fatjet.delta_r(wboson_daughters[cquarks]) < JET_DR, axis=1)
    num_m_leptons = ak.sum(fatjet.delta_r(wboson_daughters[leptons]) < JET_DR, axis=1)
    num_m_electrons = ak.sum(fatjet.delta_r(wboson_daughters[electrons]) < JET_DR, axis=1)
    num_m_muons = ak.sum(fatjet.delta_r(wboson_daughters[muons]) < JET_DR, axis=1)
    num_m_taus = ak.sum(fatjet.delta_r(wboson_daughters[taus]) < JET_DR, axis=1)

    matched_tops_mask = ak.any(fatjet.delta_r(tops) < JET_DR, axis=1)
    matched_topdaus_mask = ak.any(fatjet.delta_r(daughters) < JET_DR, axis=1)
    matched_mask = matched_tops_mask & matched_topdaus_mask

    genVars = {
        "fj_isTop": np.ones(len(genparts), dtype="bool"),
        "fj_isTop_Matched": matched_mask,  # at least one top and one daughter matched..
        "fj_Top_numMatched": num_matched_tops,  # number of tops matched
        "fj_isTop_W_lep_b": to_label((num_m_leptons == 1) & (num_m_bquarks == 1)),
        "fj_isTop_W_lep": to_label(num_m_leptons == 1),
        "fj_isTop_W_ele_b": to_label((num_m_electrons == 1) & (num_m_leptons == 1) & (num_m_bquarks == 1)),
        "fj_isTop_W_ele": to_label((num_m_electrons == 1) & (num_m_leptons == 1)),
        "fj_isTop_W_mu_b": to_label((num_m_muons == 1) & (num_m_leptons == 1) & (num_m_bquarks == 1)),
        "fj_isTop_W_mu": to_label((num_m_muons == 1) & (num_m_leptons == 1)),
        "fj_isTop_W_tau_b": to_label((num_m_taus == 1) & (num_m_leptons == 1) & (num_m_bquarks == 1)),
        "fj_isTop_W_tau": to_label((num_m_taus == 1) & (num_m_leptons == 1)),
        "fj_Top_nquarksnob": num_m_quarks_nob,  # number of quarks from W decay (not b) matched in dR
        "fj_Top_nbquarks": num_m_bquarks,  # number of b quarks ..
        "fj_Top_ncquarks": num_m_cquarks,  # number of c quarks ..
        "fj_Top_nleptons": num_m_leptons,  # number of leptons ..
        "fj_Top_nele": num_m_electrons,  # number of electrons...
        "fj_Top_nmu": num_m_muons,  # number of muons...
        "fj_Top_ntau": num_m_taus,  # number of taus...
        "fj_Top_taudecay": taudecay,  # taudecay (1: hadronic, 3: electron, 5: muon)
        "first_gTopLep_pt": ak.firsts(top_leps.pt),
        "first_gTopLep_eta": ak.firsts(top_leps.eta),
        "first_gTopLep_phi": ak.firsts(top_leps.phi),
        "first_gTopLep_mass": ak.firsts(top_leps.mass),
        "gLep_b_pt": ak.firsts(lep_b.pt),
        "gLep_b_eta": ak.firsts(lep_b.eta),
        "gLep_b_phi": ak.firsts(lep_b.phi),
        "gLep_b_mass": ak.firsts(lep_b.mass),
    }

    return genVars, matched_mask


def match_QCD(genparts: GenParticleArray, fatjets: FatJetArray) -> Tuple[np.array, Dict[str, np.array]]:
    """Gen matching for QCD samples, arguments as defined in `tagger_gen_matching`."""

    partons = genparts[get_pid_mask(genparts, [g_PDGID] + list(range(1, b_PDGID + 1)), ax=1, byall=False)]
    matched_mask = ak.any(fatjets.delta_r(partons) < JET_DR, axis=1)

    genVars = {
        "fj_isQCD": np.ones(len(genparts), dtype="bool"),
        "fj_isQCD_Matched": matched_mask,
        "fj_isQCDb": (fatjets.nBHadrons == 1),
        "fj_isQCDbb": (fatjets.nBHadrons > 1),
        "fj_isQCDc": (fatjets.nCHadrons == 1) * (fatjets.nBHadrons == 0),
        "fj_isQCDcc": (fatjets.nCHadrons > 1) * (fatjets.nBHadrons == 0),
        "fj_isQCDothers": (fatjets.nBHadrons == 0) & (fatjets.nCHadrons == 0),
    }

    # genVars["fj_isQCD"] = (
    #     (genVars["fj_isQCDb"] == 1)
    #     | (genVars["fj_isQCDbb"] == 1)
    #     | (genVars["fj_isQCDc"] == 1)
    #     | (genVars["fj_isQCDcc"] == 1)
    #     | (genVars["fj_isQCDothers"] == 1)
    # )

    genVars = {key: to_label(var) for key, var in genVars.items()}

    return genVars, matched_mask


def get_genjet_vars(events: NanoEventsArray, fatjets: FatJetArray):
    """Matched fat jet to gen-level jet and gets gen jet vars"""
    GenJetVars = {}

    # NanoAOD automatically matched ak8 fat jets
    # No soft dropped gen jets however
    GenJetVars["fj_genjetmass"] = fatjets.matched_gen.mass
    matched_gen_jet_mask = np.ones(len(events), dtype="bool")

    return GenJetVars, matched_gen_jet_mask