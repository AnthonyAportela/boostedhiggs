from typing import Dict, List, Tuple, Union

import awkward as ak
import numpy as np
from coffea.analysis_tools import PackedSelection
from coffea.nanoevents.methods.base import NanoEventsArray
from coffea.nanoevents.methods.nanoaod import FatJetArray, GenParticleArray
from coffea.nanoevents.methods.candidate import PtEtaPhiMCandidateArray

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

P4 = {
    "eta": "Eta",
    "phi": "Phi",
    "mass": "Mass",
    "pt": "Pt",
}


def get_pid_mask(
    genparts: GenParticleArray,
    pdgids: Union[int, list],
    ax: int = 2,
    byall: bool = True,
) -> ak.Array:
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

def match_H(genparts: GenParticleArray, fatjet: FatJetArray, dau_pdgid=W_PDGID):
    """Gen matching for Higgs samples"""
    higgs = genparts[get_pid_mask(genparts, HIGGS_PDGID, byall=False) * genparts.hasFlags(GEN_FLAGS)]

    # only select events that match an specific decay
    # matched_higgs = higgs[ak.argmin(fatjet.delta_r(higgs), axis=1, keepdims=True)][:, 0]
    matched_higgs = higgs[ak.argmin(fatjet.delta_r(higgs), axis=1, keepdims=True)]
    matched_higgs_mask = ak.any(fatjet.delta_r(matched_higgs) < 0.8, axis=1)

    matched_higgs = ak.firsts(matched_higgs)

    matched_higgs_children = matched_higgs.children
    higgs_children = higgs.children

    genVars = {"fj_genH_pt": ak.fill_none(higgs.pt, FILL_NONE_VALUE)}

    if dau_pdgid == W_PDGID:
        children_mask = get_pid_mask(matched_higgs_children, [W_PDGID], byall=False)
        is_hww = ak.any(children_mask, axis=1)

        # order by mass, select lower mass child as V* and higher as V
        matched_higgs_children = matched_higgs_children[children_mask]
        children_mass = matched_higgs_children.mass
        v_star = ak.firsts(matched_higgs_children[ak.argmin(children_mass, axis=1, keepdims=True)])
        v = ak.firsts(matched_higgs_children[ak.argmax(children_mass, axis=1, keepdims=True)])

        genVVars = {
            "fj_genH_jet": fatjet.delta_r(higgs[:, 0]),
            "fj_genV_dR": fatjet.delta_r(v),
            "fj_genVstar": fatjet.delta_r(v_star),
            "genV_genVstar_dR": v.delta_r(v_star),
        }

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

        gen_lepton = ak.firsts(lep_daughters)

        genHVVVars = {
            "fj_nquarks": num_m_quarks,
            "fj_ncquarks": num_m_cquarks,
            "fj_lepinprongs": num_m_leptons,
            "fj_H_VV_4q": to_label((num_quarks == 4) & (num_leptons == 0)),
            "fj_H_VV_elenuqq": to_label((num_electrons == 1) & (num_quarks == 2) & (num_leptons == 1)),
            "fj_H_VV_munuqq": to_label((num_muons == 1) & (num_quarks == 2) & (num_leptons == 1)),
            "fj_H_VV_taunuqq": to_label((num_taus == 1) & (num_quarks == 2) & (num_leptons == 1)),
            "fj_H_VV_isVlepton": iswlepton,
            "fj_H_VV_isVstarlepton": iswstarlepton,
            "fj_H_VV": is_hww,
            "fj_H_VV_isMatched": matched_higgs_mask,
            "gen_Vlep_pt": gen_lepton.pt,
            # "genlep_dR_lep": lepton.delta_r(gen_lepton)
            "fj_genRes_mass": matched_higgs.mass,
        }

        genVars = {**genVars, **genVVars, **genHVVVars}

    elif dau_pdgid == TAU_PDGID:
        children_mask = get_pid_mask(matched_higgs_children, [TAU_PDGID], byall=False)
        daughters = matched_higgs_children[children_mask]

        is_htt_matched = ak.any(children_mask, axis=1)

        # taudaughters = daughters[(abs(daughters.pdgId) == TAU_PDGID)].children
        taudaughters = daughters[(abs(daughters.pdgId) == TAU_PDGID)].distinctChildrenDeep
        taudaughters = taudaughters[taudaughters.hasFlags(["isLastCopy"])]
        taudaughters_pdgId = abs(taudaughters.pdgId)

        taudaughters = taudaughters[
            ((taudaughters_pdgId != vELE_PDGID) & (taudaughters_pdgId != vMU_PDGID) & (taudaughters_pdgId != vTAU_PDGID))
        ]
        taudaughters_pdgId = abs(taudaughters.pdgId)

        flat_taudaughters_pdgId = ak.flatten(taudaughters_pdgId, axis=2)
        taudecay = (
            # pions/kaons (full hadronic tau) * 1
            (
                (
                    ak.sum(
                        (flat_taudaughters_pdgId == PI_PDGID)
                        | (flat_taudaughters_pdgId == PO_PDGID)
                        | (flat_taudaughters_pdgId == PP_PDGID),
                        axis=1,
                    )
                    > 0
                )
            )
            * 1
            # 1 electron * 3
            + (ak.sum(flat_taudaughters_pdgId == ELE_PDGID, axis=1) == 1) * 3
            # 1 muon * 5
            + (ak.sum(flat_taudaughters_pdgId == MU_PDGID, axis=1) == 1) * 5
            # two leptons
            + (
                (ak.sum(flat_taudaughters_pdgId == ELE_PDGID, axis=1) == 2)
                | (ak.sum(flat_taudaughters_pdgId == MU_PDGID, axis=1) == 2)
            )
            * 7
        )

        elehad = taudecay == 4
        muhad = taudecay == 6
        leplep = taudecay == 7
        hadhad = ~elehad & ~muhad & ~leplep

        genHTTVars = {
            "fj_H_tt_hadhad": to_label(hadhad),
            "fj_H_tt_elehad": to_label(elehad),
            "fj_H_tt_muhad": to_label(muhad),
            "fj_H_tt_leplep": to_label(leplep),
            "fj_H_tt_isMatched": is_htt_matched,
        }

        genVars = {**genVars, **genHTTVars}

    return genVars, is_hww


def match_H_lep(genparts: GenParticleArray, fatjet: FatJetArray, candidatelep_p4: PtEtaPhiMCandidateArray, dau_pdgid=W_PDGID):
    """Gen matching for Higgs samples"""
    higgs = genparts[get_pid_mask(genparts, HIGGS_PDGID, byall=False) * genparts.hasFlags(GEN_FLAGS)]

    # only select events that match an specific decay
    # matched_higgs = higgs[ak.argmin(fatjet.delta_r(higgs), axis=1, keepdims=True)][:, 0]
    matched_higgs = higgs[ak.argmin(fatjet.delta_r(higgs), axis=1, keepdims=True)]
    matched_higgs_mask = ak.any(fatjet.delta_r(matched_higgs) < 0.8, axis=1)

    matched_higgs = ak.firsts(matched_higgs)

    matched_higgs_children = matched_higgs.children
    higgs_children = higgs.children

    genVars = {"fj_genH_pt": ak.fill_none(higgs.pt, FILL_NONE_VALUE)}

    if dau_pdgid == W_PDGID:
        children_mask = get_pid_mask(matched_higgs_children, [W_PDGID], byall=False)
        is_hww = ak.any(children_mask, axis=1)

        # order by mass, select lower mass child as V* and higher as V
        matched_higgs_children = matched_higgs_children[children_mask]
        children_mass = matched_higgs_children.mass
        v_star = ak.firsts(matched_higgs_children[ak.argmin(children_mass, axis=1, keepdims=True)])
        v = ak.firsts(matched_higgs_children[ak.argmax(children_mass, axis=1, keepdims=True)])

        genVVars = {
            "fj_genH_jet": fatjet.delta_r(higgs[:, 0]),
            "fj_genV_dR": fatjet.delta_r(v),
            "fj_genVstar": fatjet.delta_r(v_star),
            "genV_genVstar_dR": v.delta_r(v_star),
        }

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

        gen_lepton = ak.firsts(lep_daughters)

            
        # print('===================================')
        # print(len(is_hww))
        # print(len(matched_higgs_mask))
        # print('===================================')
        
        genHVVVars = {
            "fj_nquarks": num_m_quarks,
            "fj_ncquarks": num_m_cquarks,
            "fj_lepinprongs": num_m_leptons,
            "fj_H_VV_4q": to_label((num_quarks == 4) & (num_leptons == 0)),
            "fj_H_VV_elenuqq": to_label((num_electrons == 1) & (num_quarks == 2) & (num_leptons == 1)),
            "fj_H_VV_munuqq": to_label((num_muons == 1) & (num_quarks == 2) & (num_leptons == 1)),
            "fj_H_VV_taunuqq": to_label((num_taus == 1) & (num_quarks == 2) & (num_leptons == 1)),
            "fj_H_VV_isVlepton": iswlepton,
            "fj_H_VV_isVstarlepton": iswstarlepton,
            "fj_H_VV": is_hww,
            "fj_H_VV_isMatched": matched_higgs_mask,
            "gen_Vlep_pt": gen_lepton.pt,
            # "genlep_dR_lep": lepton.delta_r(gen_lepton)
            "fj_genRes_mass": matched_higgs.mass,
            "H_dR_lep": candidatelep_p4.delta_r(higgs[:, 0]),
        }

        genVars = {**genVars, **genVVars, **genHVVVars}

    elif dau_pdgid == TAU_PDGID:
        children_mask = get_pid_mask(matched_higgs_children, [TAU_PDGID], byall=False)
        daughters = matched_higgs_children[children_mask]

        is_htt_matched = ak.any(children_mask, axis=1)

        # taudaughters = daughters[(abs(daughters.pdgId) == TAU_PDGID)].children
        taudaughters = daughters[(abs(daughters.pdgId) == TAU_PDGID)].distinctChildrenDeep
        taudaughters = taudaughters[taudaughters.hasFlags(["isLastCopy"])]
        taudaughters_pdgId = abs(taudaughters.pdgId)

        taudaughters = taudaughters[
            ((taudaughters_pdgId != vELE_PDGID) & (taudaughters_pdgId != vMU_PDGID) & (taudaughters_pdgId != vTAU_PDGID))
        ]
        taudaughters_pdgId = abs(taudaughters.pdgId)

        flat_taudaughters_pdgId = ak.flatten(taudaughters_pdgId, axis=2)
        taudecay = (
            # pions/kaons (full hadronic tau) * 1
            (
                (
                    ak.sum(
                        (flat_taudaughters_pdgId == PI_PDGID)
                        | (flat_taudaughters_pdgId == PO_PDGID)
                        | (flat_taudaughters_pdgId == PP_PDGID),
                        axis=1,
                    )
                    > 0
                )
            )
            * 1
            # 1 electron * 3
            + (ak.sum(flat_taudaughters_pdgId == ELE_PDGID, axis=1) == 1) * 3
            # 1 muon * 5
            + (ak.sum(flat_taudaughters_pdgId == MU_PDGID, axis=1) == 1) * 5
            # two leptons
            + (
                (ak.sum(flat_taudaughters_pdgId == ELE_PDGID, axis=1) == 2)
                | (ak.sum(flat_taudaughters_pdgId == MU_PDGID, axis=1) == 2)
            )
            * 7
        )

        elehad = taudecay == 4
        muhad = taudecay == 6
        leplep = taudecay == 7
        hadhad = ~elehad & ~muhad & ~leplep

        genHTTVars = {
            "fj_H_tt_hadhad": to_label(hadhad),
            "fj_H_tt_elehad": to_label(elehad),
            "fj_H_tt_muhad": to_label(muhad),
            "fj_H_tt_leplep": to_label(leplep),
            "fj_H_tt_isMatched": is_htt_matched,
        }

        genVars = {**genVars, **genHTTVars}

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
        "fj_nprongs": nprongs,
        "fj_lepinprongs": lepinprongs,
        "fj_ncquarks": ncquarks,
        "fj_V_isMatched": matched_mask,
        "fj_V_2q": to_label(decay == 1),
        "fj_V_elenu": to_label(decay == 3),
        "fj_V_munu": to_label(decay == 5),
        "fj_V_taunu": to_label(decay == 7),
    }
    return genVars, matched_mask


def match_Top(genparts: GenParticleArray, fatjet: FatJetArray):
    tops = genparts[get_pid_mask(genparts, TOP_PDGID, byall=False) * genparts.hasFlags(GEN_FLAGS)]
    matched_tops = tops[fatjet.delta_r(tops) < JET_DR]
    num_matched_tops = ak.sum(fatjet.delta_r(matched_tops) < JET_DR, axis=1)

    # take all possible daughters!
    daughters = ak.flatten(tops.distinctChildren, axis=2)
    daughters = daughters[daughters.hasFlags(["fromHardProcess", "isLastCopy"])]
    daughters_pdgId = abs(daughters.pdgId)

    wboson_daughters = ak.flatten(daughters[(daughters_pdgId == W_PDGID)].distinctChildren, axis=2)
    wboson_daughters = wboson_daughters[wboson_daughters.hasFlags(["fromHardProcess", "isLastCopy"])]
    wboson_daughters_pdgId = abs(wboson_daughters.pdgId)

    bquark = daughters[(daughters_pdgId == 5)]
    neutrinos = (
        (wboson_daughters_pdgId == vELE_PDGID)
        | (wboson_daughters_pdgId == vMU_PDGID)
        | (wboson_daughters_pdgId == vTAU_PDGID)
    )
    leptons = (
        (wboson_daughters_pdgId == ELE_PDGID) | (wboson_daughters_pdgId == MU_PDGID) | (wboson_daughters_pdgId == TAU_PDGID)
    )
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
        "fj_Top_isMatched": matched_mask,  # at least one top and one daugther matched..
        "fj_Top_numMatched": num_matched_tops,  # number of tops matched
        "fj_Top_nquarksnob": num_m_quarks_nob,  # number of quarks from W decay (not b) matched in dR
        "fj_Top_nbquarks": num_m_bquarks,  # number of b quarks ..
        "fj_Top_ncquarks": num_m_cquarks,  # number of c quarks ..
        "fj_Top_nleptons": num_m_leptons,  # number of leptons ..
        "fj_Top_nele": num_m_electrons,  # number of electrons...
        "fj_Top_nmu": num_m_muons,  # number of muons...
        "fj_Top_ntau": num_m_taus,  # number of taus...
        "fj_Top_taudecay": taudecay,  # taudecay (1: hadronic, 3: electron, 5: muon)
    }

    return genVars, matched_mask


def match_QCD(
    genparts: GenParticleArray,
    fatjets: FatJetArray,
    # genlabels: List[str],
) -> Tuple[np.array, Dict[str, np.array]]:
    """Gen matching for QCD samples, arguments as defined in ``tagger_gen_matching``"""

    partons = genparts[get_pid_mask(genparts, [g_PDGID] + list(range(1, b_PDGID + 1)), ax=1, byall=False)]
    matched_mask = ak.any(fatjets.delta_r(partons) < JET_DR, axis=1)

    genVars = {
        "fj_QCDb": (fatjets.nBHadrons == 1),
        "fj_QCDbb": (fatjets.nBHadrons > 1),
        "fj_QCDc": (fatjets.nCHadrons == 1) * (fatjets.nBHadrons == 0),
        "fj_QCDcc": (fatjets.nCHadrons > 1) * (fatjets.nBHadrons == 0),
        "fj_QCDothers": (fatjets.nBHadrons == 0) & (fatjets.nCHadrons == 0),
    }

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


def tagger_gen_matching(
    events: NanoEventsArray,
    genparts: GenParticleArray,
    fatjets: FatJetArray,
    # candidatelep_p4,
    genlabels: List[str],
    label: str,
) -> Tuple[np.array, Dict[str, np.array]]:
    """Does fatjet -> gen-level matching and derives gen-level variables.

    Args:
        events (NanoEventsArray): events.
        genparts (GenParticleArray): event gen particles.
        fatjets (FatJetArray): event fat jets (should be only one fat jet per event!).
        genlabels (List[str]): gen variables to return.
        label (str): dataset label, formatted as
          ``{AK15 or AK8}_{H or QCD}_{(optional) H decay products}``.
        JET_DR (float): max distance between fat jet and gen particle for matching.
          Defaults to 1.0.

    Returns:
        np.array: Boolean selection array of shape ``[len(fatjets)]``.
        Dict[str, np.array]: dict of gen variables.

    """

    if "H" in label:
        print("match_H")
        GenVars, matched_mask = match_H(genparts, fatjets)
        # GenVars["fj_genRes_mass"] = 125 * np.ones(len(events))
    elif "QCD" in label:
        print("match_QCD")
        GenVars, matched_mask = match_QCD(genparts, fatjets)
    elif "VJets" in label:
        print("match_V")
        GenVars, matched_mask = match_V(genparts, fatjets)
    elif "Top" in label:
        print("match_Top")
        GenVars, matched_mask = match_Top(genparts, fatjets)
    else:
        print("no match")
        GenVars = {}
        matched_mask = np.zeros(len(genparts), dtype="bool")

    # genjet_vars, matched_gen_jet_mask = get_genjet_vars(events, fatjets)
    # AllGenVars = {**GenVars, **genjet_vars}

    AllGenVars = {**GenVars, **{"fj_genjetmass": fatjets.matched_gen.mass}}  # add gen jet mass

    # loop to keep only the specified variables in `genlabels`
    # if ``GenVars`` doesn't contain a variable, that variable is not applicable to this sample so fill with 0s
    GenVars = {key: AllGenVars[key] if key in AllGenVars.keys() else np.zeros(len(genparts)) for key in genlabels}
    for key, item in GenVars.items():
        try:
            GenVars[key] = GenVars[key].to_numpy()
        except Exception:
            continue

    return matched_mask, GenVars
    # return matched_mask * matched_gen_jet_mask, GenVars


# def FILL_NONE_VALUE(
#     arr: ak.Array,
#     value: float,
#     target: int = None,
#     axis: int = 0,
#     to_numpy: bool = False,
#     clip: bool = True,
# ):
#     """
#     pads awkward array up to ``target`` index along axis ``axis`` with value ``value``,
#     optionally converts to numpy array
#     """
#     if target:
#         ret = ak.fill_none(ak.pad_none(arr, target, axis=axis, clip=clip), value, axis=None)
#     else:
#         ret = ak.fill_none(arr, value, axis=None)
#     return ret.to_numpy() if to_numpy else ret


def add_selection(
    name: str,
    sel: np.ndarray,
    selection: PackedSelection,
    cutflow: dict,
    isData: bool,
    signGenWeights: ak.Array,
):
    """adds selection to PackedSelection object and the cutflow dictionary"""
    selection.add(name, sel)
    cutflow[name] = (
        np.sum(selection.all(*selection.names))
        if isData
        # add up sign of genWeights for MC
        else np.sum(signGenWeights[selection.all(*selection.names)])
    )


def add_selection_no_cutflow(
    name: str,
    sel: np.ndarray,
    selection: PackedSelection,
):
    """adds selection to PackedSelection object"""
    selection.add(name, ak.fill_none(sel, False))


def get_neutrino_z(vis, inv, h_mass=125):
    """
    Reconstruct the mass by taking qq jet, lepton and MET
    Then, solve for the z component of the neutrino momentum
    by requiring that the invariant mass of the group of objects is the Higgs mass = 125
    """
    a = h_mass * h_mass - vis.mass * vis.mass + 2 * vis.x * inv.x + 2 * vis.y * inv.y
    A = 4 * (vis.t * vis.t - vis.z * vis.z)
    B = -4 * a * vis.z
    C = 4 * vis.t * vis.t * (inv.x * inv.x + inv.y * inv.y) - a * a
    delta = B * B - 4 * A * C
    neg = -B / (2 * A)
    neg = ak.nan_to_num(neg)
    pos = np.maximum((-B + np.sqrt(delta)) / (2 * A), (-B - np.sqrt(delta)) / (2 * A))
    pos = ak.nan_to_num(pos)

    invZ = (delta < 0) * neg + (delta > 0) * pos
    neutrino = ak.zip(
        {
            "x": inv.x,
            "y": inv.y,
            "z": invZ,
            "t": np.sqrt(inv.x * inv.x + inv.y * inv.y + invZ * invZ),
        },
        with_name="LorentzVector",
    )
    return neutrino


# TAGGER STUFF
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
qcd = [
    "fj_ParT_probQCDbb",
    "fj_ParT_probQCDcc",
    "fj_ParT_probQCDb",
    "fj_ParT_probQCDc",
    "fj_ParT_probQCDothers",
]
tope = ["fj_ParT_probTopbWev", "fj_ParT_probTopbWtauev"]
topm = ["fj_ParT_probTopbWmv", "fj_ParT_probTopbWtaumv"]
tophad = [
    "fj_ParT_probTopbWqq0c",
    "fj_ParT_probTopbWqq1c",
    "fj_ParT_probTopbWq0c",
    "fj_ParT_probTopbWq1c",
    "fj_ParT_probTopbWtauhv",
]
top = tope + topm + tophad
bkgs = qcd + top
others = [
    "fj_ParT_probHbb",
    "fj_ParT_probHcc",
    "fj_ParT_probHss",
    "fj_ParT_probHqq",
    "fj_ParT_probHtauhtaue",
    "fj_ParT_probHtauhtaum",
    "fj_ParT_probHtauhtauh",
]
