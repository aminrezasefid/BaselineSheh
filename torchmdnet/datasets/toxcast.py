import os
import os.path as osp
import sys

from typing import Callable, List, Optional, Dict

import torch
from torch import Tensor
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.transforms import Compose
from torch_geometric.utils import one_hot, scatter
from tqdm import tqdm

URLS = {
    "precise3d": "https://drive.google.com/uc?export=download&id=1kQDG0QtVW46wlnqcTtoZpLZhWimuw8XX",
    "optimized3d": "https://drive.google.com/uc?export=download&id=1qi1ZDtYGfV0NC1VYTWCkbjjhcc3yluIZ",
    "rdkit3d": "https://drive.google.com/uc?export=download&id=16XBihWqX7GivWSyAho4k3LB0TZDOCgA1",
    "rdkit2d": "https://drive.google.com/uc?export=download&id=19Ft34srAobETlGo6iEbY_AYeglyoWgAI"
}

toxcast_target_dict = {'ACEA_T47D_80hr_Negative': 0, 'ACEA_T47D_80hr_Positive': 1, 'APR_HepG2_CellCycleArrest_24h_dn': 2, 'APR_HepG2_CellCycleArrest_24h_up': 3, 'APR_HepG2_CellCycleArrest_72h_dn': 4, 'APR_HepG2_CellLoss_24h_dn': 5, 'APR_HepG2_CellLoss_72h_dn': 6, 'APR_HepG2_MicrotubuleCSK_24h_dn': 7, 'APR_HepG2_MicrotubuleCSK_24h_up': 8, 'APR_HepG2_MicrotubuleCSK_72h_dn': 9, 'APR_HepG2_MicrotubuleCSK_72h_up': 10, 'APR_HepG2_MitoMass_24h_dn': 11, 'APR_HepG2_MitoMass_24h_up': 12, 'APR_HepG2_MitoMass_72h_dn': 13, 'APR_HepG2_MitoMass_72h_up': 14, 'APR_HepG2_MitoMembPot_1h_dn': 15, 'APR_HepG2_MitoMembPot_24h_dn': 16, 'APR_HepG2_MitoMembPot_72h_dn': 17, 'APR_HepG2_MitoticArrest_24h_up': 18, 'APR_HepG2_MitoticArrest_72h_up': 19, 'APR_HepG2_NuclearSize_24h_dn': 20, 'APR_HepG2_NuclearSize_72h_dn': 21, 'APR_HepG2_NuclearSize_72h_up': 22, 'APR_HepG2_OxidativeStress_24h_up': 23, 'APR_HepG2_OxidativeStress_72h_up': 24, 'APR_HepG2_StressKinase_1h_up': 25, 'APR_HepG2_StressKinase_24h_up': 26, 'APR_HepG2_StressKinase_72h_up': 27, 'APR_HepG2_p53Act_24h_up': 28, 'APR_HepG2_p53Act_72h_up': 29, 'APR_Hepat_Apoptosis_24hr_up': 30, 'APR_Hepat_Apoptosis_48hr_up': 31, 'APR_Hepat_CellLoss_24hr_dn': 32, 'APR_Hepat_CellLoss_48hr_dn': 33, 'APR_Hepat_DNADamage_24hr_up': 34, 'APR_Hepat_DNADamage_48hr_up': 35, 'APR_Hepat_DNATexture_24hr_up': 36, 'APR_Hepat_DNATexture_48hr_up': 37, 'APR_Hepat_MitoFxnI_1hr_dn': 38, 'APR_Hepat_MitoFxnI_24hr_dn': 39, 'APR_Hepat_MitoFxnI_48hr_dn': 40, 'APR_Hepat_NuclearSize_24hr_dn': 41, 'APR_Hepat_NuclearSize_48hr_dn': 42, 'APR_Hepat_Steatosis_24hr_up': 43, 'APR_Hepat_Steatosis_48hr_up': 44, 'ATG_AP_1_CIS_dn': 45, 'ATG_AP_1_CIS_up': 46, 'ATG_AP_2_CIS_dn': 47, 'ATG_AP_2_CIS_up': 48, 'ATG_AR_TRANS_dn': 49, 'ATG_AR_TRANS_up': 50, 'ATG_Ahr_CIS_dn': 51, 'ATG_Ahr_CIS_up': 52, 'ATG_BRE_CIS_dn': 53, 'ATG_BRE_CIS_up': 54, 'ATG_CAR_TRANS_dn': 55, 'ATG_CAR_TRANS_up': 56, 'ATG_CMV_CIS_dn': 57, 'ATG_CMV_CIS_up': 58, 'ATG_CRE_CIS_dn': 59, 'ATG_CRE_CIS_up': 60, 'ATG_C_EBP_CIS_dn': 61, 'ATG_C_EBP_CIS_up': 62, 'ATG_DR4_LXR_CIS_dn': 63, 'ATG_DR4_LXR_CIS_up': 64, 'ATG_DR5_CIS_dn': 65, 'ATG_DR5_CIS_up': 66, 'ATG_E2F_CIS_dn': 67, 'ATG_E2F_CIS_up': 68, 'ATG_EGR_CIS_up': 69, 'ATG_ERE_CIS_dn': 70, 'ATG_ERE_CIS_up': 71, 'ATG_ERRa_TRANS_dn': 72, 'ATG_ERRg_TRANS_dn': 73, 'ATG_ERRg_TRANS_up': 74, 'ATG_ERa_TRANS_up': 75, 'ATG_E_Box_CIS_dn': 76, 'ATG_E_Box_CIS_up': 77, 'ATG_Ets_CIS_dn': 78, 'ATG_Ets_CIS_up': 79, 'ATG_FXR_TRANS_up': 80, 'ATG_FoxA2_CIS_dn': 81, 'ATG_FoxA2_CIS_up': 82, 'ATG_FoxO_CIS_dn': 83, 'ATG_FoxO_CIS_up': 84, 'ATG_GAL4_TRANS_dn': 85, 'ATG_GATA_CIS_dn': 86, 'ATG_GATA_CIS_up': 87, 'ATG_GLI_CIS_dn': 88, 'ATG_GLI_CIS_up': 89, 'ATG_GRE_CIS_dn': 90, 'ATG_GRE_CIS_up': 91, 'ATG_GR_TRANS_dn': 92, 'ATG_GR_TRANS_up': 93, 'ATG_HIF1a_CIS_dn': 94, 'ATG_HIF1a_CIS_up': 95, 'ATG_HNF4a_TRANS_dn': 96, 'ATG_HNF4a_TRANS_up': 97, 'ATG_HNF6_CIS_dn': 98, 'ATG_HNF6_CIS_up': 99, 'ATG_HSE_CIS_dn': 100, 'ATG_HSE_CIS_up': 101, 'ATG_IR1_CIS_dn': 102, 'ATG_IR1_CIS_up': 103, 'ATG_ISRE_CIS_dn': 104, 'ATG_ISRE_CIS_up': 105, 'ATG_LXRa_TRANS_dn': 106, 'ATG_LXRa_TRANS_up': 107, 'ATG_LXRb_TRANS_dn': 108, 'ATG_LXRb_TRANS_up': 109, 'ATG_MRE_CIS_up': 110, 'ATG_M_06_TRANS_up': 111, 'ATG_M_19_CIS_dn': 112, 'ATG_M_19_TRANS_dn': 113, 'ATG_M_19_TRANS_up': 114, 'ATG_M_32_CIS_dn': 115, 'ATG_M_32_CIS_up': 116, 'ATG_M_32_TRANS_dn': 117, 'ATG_M_32_TRANS_up': 118, 'ATG_M_61_TRANS_up': 119, 'ATG_Myb_CIS_dn': 120, 'ATG_Myb_CIS_up': 121, 'ATG_Myc_CIS_dn': 122, 'ATG_Myc_CIS_up': 123, 'ATG_NFI_CIS_dn': 124, 'ATG_NFI_CIS_up': 125, 'ATG_NF_kB_CIS_dn': 126, 'ATG_NF_kB_CIS_up': 127, 'ATG_NRF1_CIS_dn': 128, 'ATG_NRF1_CIS_up': 129, 'ATG_NRF2_ARE_CIS_dn': 130, 'ATG_NRF2_ARE_CIS_up': 131, 'ATG_NURR1_TRANS_dn': 132, 'ATG_NURR1_TRANS_up': 133, 'ATG_Oct_MLP_CIS_dn': 134, 'ATG_Oct_MLP_CIS_up': 135, 'ATG_PBREM_CIS_dn': 136, 'ATG_PBREM_CIS_up': 137, 'ATG_PPARa_TRANS_dn': 138, 'ATG_PPARa_TRANS_up': 139, 'ATG_PPARd_TRANS_up': 140, 'ATG_PPARg_TRANS_up': 141, 'ATG_PPRE_CIS_dn': 142, 'ATG_PPRE_CIS_up': 143, 'ATG_PXRE_CIS_dn': 144, 'ATG_PXRE_CIS_up': 145, 'ATG_PXR_TRANS_dn': 146, 'ATG_PXR_TRANS_up': 147, 'ATG_Pax6_CIS_up': 148, 'ATG_RARa_TRANS_dn': 149, 'ATG_RARa_TRANS_up': 150, 'ATG_RARb_TRANS_dn': 151, 'ATG_RARb_TRANS_up': 152, 'ATG_RARg_TRANS_dn': 153, 'ATG_RARg_TRANS_up': 154, 'ATG_RORE_CIS_dn': 155, 'ATG_RORE_CIS_up': 156, 'ATG_RORb_TRANS_dn': 157, 'ATG_RORg_TRANS_dn': 158, 'ATG_RORg_TRANS_up': 159, 'ATG_RXRa_TRANS_dn': 160, 'ATG_RXRa_TRANS_up': 161, 'ATG_RXRb_TRANS_dn': 162, 'ATG_RXRb_TRANS_up': 163, 'ATG_SREBP_CIS_dn': 164, 'ATG_SREBP_CIS_up': 165, 'ATG_STAT3_CIS_dn': 166, 'ATG_STAT3_CIS_up': 167, 'ATG_Sox_CIS_dn': 168, 'ATG_Sox_CIS_up': 169, 'ATG_Sp1_CIS_dn': 170, 'ATG_Sp1_CIS_up': 171, 'ATG_TAL_CIS_dn': 172, 'ATG_TAL_CIS_up': 173, 'ATG_TA_CIS_dn': 174, 'ATG_TA_CIS_up': 175, 'ATG_TCF_b_cat_CIS_dn': 176, 'ATG_TCF_b_cat_CIS_up': 177, 'ATG_TGFb_CIS_dn': 178, 'ATG_TGFb_CIS_up': 179, 'ATG_THRa1_TRANS_dn': 180, 'ATG_THRa1_TRANS_up': 181, 'ATG_VDRE_CIS_dn': 182, 'ATG_VDRE_CIS_up': 183, 'ATG_VDR_TRANS_dn': 184, 'ATG_VDR_TRANS_up': 185, 'ATG_XTT_Cytotoxicity_up': 186, 'ATG_Xbp1_CIS_dn': 187, 'ATG_Xbp1_CIS_up': 188, 'ATG_p53_CIS_dn': 189, 'ATG_p53_CIS_up': 190, 'BSK_3C_Eselectin_down': 191, 'BSK_3C_HLADR_down': 192, 'BSK_3C_ICAM1_down': 193, 'BSK_3C_IL8_down': 194, 'BSK_3C_MCP1_down': 195, 'BSK_3C_MIG_down': 196, 'BSK_3C_Proliferation_down': 197, 'BSK_3C_SRB_down': 198, 'BSK_3C_Thrombomodulin_down': 199, 'BSK_3C_Thrombomodulin_up': 200, 'BSK_3C_TissueFactor_down': 201, 'BSK_3C_TissueFactor_up': 202, 'BSK_3C_VCAM1_down': 203, 'BSK_3C_Vis_down': 204, 'BSK_3C_uPAR_down': 205, 'BSK_4H_Eotaxin3_down': 206, 'BSK_4H_MCP1_down': 207, 'BSK_4H_Pselectin_down': 208, 'BSK_4H_Pselectin_up': 209, 'BSK_4H_SRB_down': 210, 'BSK_4H_VCAM1_down': 211, 'BSK_4H_VEGFRII_down': 212, 'BSK_4H_uPAR_down': 213, 'BSK_4H_uPAR_up': 214, 'BSK_BE3C_HLADR_down': 215, 'BSK_BE3C_IL1a_down': 216, 'BSK_BE3C_IP10_down': 217, 'BSK_BE3C_MIG_down': 218, 'BSK_BE3C_MMP1_down': 219, 'BSK_BE3C_MMP1_up': 220, 'BSK_BE3C_PAI1_down': 221, 'BSK_BE3C_SRB_down': 222, 'BSK_BE3C_TGFb1_down': 223, 'BSK_BE3C_tPA_down': 224, 'BSK_BE3C_uPAR_down': 225, 'BSK_BE3C_uPAR_up': 226, 'BSK_BE3C_uPA_down': 227, 'BSK_CASM3C_HLADR_down': 228, 'BSK_CASM3C_IL6_down': 229, 'BSK_CASM3C_IL6_up': 230, 'BSK_CASM3C_IL8_down': 231, 'BSK_CASM3C_LDLR_down': 232, 'BSK_CASM3C_LDLR_up': 233, 'BSK_CASM3C_MCP1_down': 234, 'BSK_CASM3C_MCP1_up': 235, 'BSK_CASM3C_MCSF_down': 236, 'BSK_CASM3C_MCSF_up': 237, 'BSK_CASM3C_MIG_down': 238, 'BSK_CASM3C_Proliferation_down': 239, 'BSK_CASM3C_Proliferation_up': 240, 'BSK_CASM3C_SAA_down': 241, 'BSK_CASM3C_SAA_up': 242, 'BSK_CASM3C_SRB_down': 243, 'BSK_CASM3C_Thrombomodulin_down': 244, 'BSK_CASM3C_Thrombomodulin_up': 245, 'BSK_CASM3C_TissueFactor_down': 246, 'BSK_CASM3C_VCAM1_down': 247, 'BSK_CASM3C_VCAM1_up': 248, 'BSK_CASM3C_uPAR_down': 249, 'BSK_CASM3C_uPAR_up': 250, 'BSK_KF3CT_ICAM1_down': 251, 'BSK_KF3CT_IL1a_down': 252, 'BSK_KF3CT_IP10_down': 253, 'BSK_KF3CT_IP10_up': 254, 'BSK_KF3CT_MCP1_down': 255, 'BSK_KF3CT_MCP1_up': 256, 'BSK_KF3CT_MMP9_down': 257, 'BSK_KF3CT_SRB_down': 258, 'BSK_KF3CT_TGFb1_down': 259, 'BSK_KF3CT_TIMP2_down': 260, 'BSK_KF3CT_uPA_down': 261, 'BSK_LPS_CD40_down': 262, 'BSK_LPS_Eselectin_down': 263, 'BSK_LPS_Eselectin_up': 264, 'BSK_LPS_IL1a_down': 265, 'BSK_LPS_IL1a_up': 266, 'BSK_LPS_IL8_down': 267, 'BSK_LPS_IL8_up': 268, 'BSK_LPS_MCP1_down': 269, 'BSK_LPS_MCSF_down': 270, 'BSK_LPS_PGE2_down': 271, 'BSK_LPS_PGE2_up': 272, 'BSK_LPS_SRB_down': 273, 'BSK_LPS_TNFa_down': 274, 'BSK_LPS_TNFa_up': 275, 'BSK_LPS_TissueFactor_down': 276, 'BSK_LPS_TissueFactor_up': 277, 'BSK_LPS_VCAM1_down': 278, 'BSK_SAg_CD38_down': 279, 'BSK_SAg_CD40_down': 280, 'BSK_SAg_CD69_down': 281, 'BSK_SAg_Eselectin_down': 282, 'BSK_SAg_Eselectin_up': 283, 'BSK_SAg_IL8_down': 284, 'BSK_SAg_IL8_up': 285, 'BSK_SAg_MCP1_down': 286, 'BSK_SAg_MIG_down': 287, 'BSK_SAg_PBMCCytotoxicity_down': 288, 'BSK_SAg_PBMCCytotoxicity_up': 289, 'BSK_SAg_Proliferation_down': 290, 'BSK_SAg_SRB_down': 291, 'BSK_hDFCGF_CollagenIII_down': 292, 'BSK_hDFCGF_EGFR_down': 293, 'BSK_hDFCGF_EGFR_up': 294, 'BSK_hDFCGF_IL8_down': 295, 'BSK_hDFCGF_IP10_down': 296, 'BSK_hDFCGF_MCSF_down': 297, 'BSK_hDFCGF_MIG_down': 298, 'BSK_hDFCGF_MMP1_down': 299, 'BSK_hDFCGF_MMP1_up': 300, 'BSK_hDFCGF_PAI1_down': 301, 'BSK_hDFCGF_Proliferation_down': 302, 'BSK_hDFCGF_SRB_down': 303, 'BSK_hDFCGF_TIMP1_down': 304, 'BSK_hDFCGF_VCAM1_down': 305, 'CEETOX_H295R_11DCORT_dn': 306, 'CEETOX_H295R_ANDR_dn': 307, 'CEETOX_H295R_CORTISOL_dn': 308, 'CEETOX_H295R_DOC_dn': 309, 'CEETOX_H295R_DOC_up': 310, 'CEETOX_H295R_ESTRADIOL_dn': 311, 'CEETOX_H295R_ESTRADIOL_up': 312, 'CEETOX_H295R_ESTRONE_dn': 313, 'CEETOX_H295R_ESTRONE_up': 314, 'CEETOX_H295R_OHPREG_up': 315, 'CEETOX_H295R_OHPROG_dn': 316, 'CEETOX_H295R_OHPROG_up': 317, 'CEETOX_H295R_PROG_up': 318, 'CEETOX_H295R_TESTO_dn': 319, 'CLD_ABCB1_48hr': 320, 'CLD_ABCG2_48hr': 321, 'CLD_CYP1A1_24hr': 322, 'CLD_CYP1A1_48hr': 323, 'CLD_CYP1A1_6hr': 324, 'CLD_CYP1A2_24hr': 325, 'CLD_CYP1A2_48hr': 326, 'CLD_CYP1A2_6hr': 327, 'CLD_CYP2B6_24hr': 328, 'CLD_CYP2B6_48hr': 329, 'CLD_CYP2B6_6hr': 330, 'CLD_CYP3A4_24hr': 331, 'CLD_CYP3A4_48hr': 332, 'CLD_CYP3A4_6hr': 333, 'CLD_GSTA2_48hr': 334, 'CLD_SULT2A_24hr': 335, 'CLD_SULT2A_48hr': 336, 'CLD_UGT1A1_24hr': 337, 'CLD_UGT1A1_48hr': 338, 'NCCT_HEK293T_CellTiterGLO': 339, 'NCCT_QuantiLum_inhib_2_dn': 340, 'NCCT_QuantiLum_inhib_dn': 341, 'NCCT_TPO_AUR_dn': 342, 'NCCT_TPO_GUA_dn': 343, 'NHEERL_ZF_144hpf_TERATOSCORE_up': 344, 'NVS_ADME_hCYP19A1': 345, 'NVS_ADME_hCYP1A1': 346, 'NVS_ADME_hCYP1A2': 347, 'NVS_ADME_hCYP2A6': 348, 'NVS_ADME_hCYP2B6': 349, 'NVS_ADME_hCYP2C19': 350, 'NVS_ADME_hCYP2C9': 351, 'NVS_ADME_hCYP2D6': 352, 'NVS_ADME_hCYP3A4': 353, 'NVS_ADME_hCYP4F12': 354, 'NVS_ADME_rCYP2C12': 355, 'NVS_ENZ_hAChE': 356, 'NVS_ENZ_hAMPKa1': 357, 'NVS_ENZ_hAurA': 358, 'NVS_ENZ_hBACE': 359, 'NVS_ENZ_hCASP5': 360, 'NVS_ENZ_hCK1D': 361, 'NVS_ENZ_hDUSP3': 362, 'NVS_ENZ_hES': 363, 'NVS_ENZ_hElastase': 364, 'NVS_ENZ_hFGFR1': 365, 'NVS_ENZ_hGSK3b': 366, 'NVS_ENZ_hMMP1': 367, 'NVS_ENZ_hMMP13': 368, 'NVS_ENZ_hMMP2': 369, 'NVS_ENZ_hMMP3': 370, 'NVS_ENZ_hMMP7': 371, 'NVS_ENZ_hMMP9': 372, 'NVS_ENZ_hPDE10': 373, 'NVS_ENZ_hPDE4A1': 374, 'NVS_ENZ_hPDE5': 375, 'NVS_ENZ_hPI3Ka': 376, 'NVS_ENZ_hPTEN': 377, 'NVS_ENZ_hPTPN11': 378, 'NVS_ENZ_hPTPN12': 379, 'NVS_ENZ_hPTPN13': 380, 'NVS_ENZ_hPTPN9': 381, 'NVS_ENZ_hPTPRC': 382, 'NVS_ENZ_hSIRT1': 383, 'NVS_ENZ_hSIRT2': 384, 'NVS_ENZ_hTrkA': 385, 'NVS_ENZ_hVEGFR2': 386, 'NVS_ENZ_oCOX1': 387, 'NVS_ENZ_oCOX2': 388, 'NVS_ENZ_rAChE': 389, 'NVS_ENZ_rCNOS': 390, 'NVS_ENZ_rMAOAC': 391, 'NVS_ENZ_rMAOAP': 392, 'NVS_ENZ_rMAOBC': 393, 'NVS_ENZ_rMAOBP': 394, 'NVS_ENZ_rabI2C': 395, 'NVS_GPCR_bAdoR_NonSelective': 396, 'NVS_GPCR_bDR_NonSelective': 397, 'NVS_GPCR_g5HT4': 398, 'NVS_GPCR_gH2': 399, 'NVS_GPCR_gLTB4': 400, 'NVS_GPCR_gLTD4': 401, 'NVS_GPCR_gMPeripheral_NonSelective': 402, 'NVS_GPCR_gOpiateK': 403, 'NVS_GPCR_h5HT2A': 404, 'NVS_GPCR_h5HT5A': 405, 'NVS_GPCR_h5HT6': 406, 'NVS_GPCR_h5HT7': 407, 'NVS_GPCR_hAT1': 408, 'NVS_GPCR_hAdoRA1': 409, 'NVS_GPCR_hAdoRA2a': 410, 'NVS_GPCR_hAdra2A': 411, 'NVS_GPCR_hAdra2C': 412, 'NVS_GPCR_hAdrb1': 413, 'NVS_GPCR_hAdrb2': 414, 'NVS_GPCR_hAdrb3': 415, 'NVS_GPCR_hDRD1': 416, 'NVS_GPCR_hDRD2s': 417, 'NVS_GPCR_hDRD4.4': 418, 'NVS_GPCR_hH1': 419, 'NVS_GPCR_hLTB4_BLT1': 420, 'NVS_GPCR_hM1': 421, 'NVS_GPCR_hM2': 422, 'NVS_GPCR_hM3': 423, 'NVS_GPCR_hM4': 424, 'NVS_GPCR_hNK2': 425, 'NVS_GPCR_hOpiate_D1': 426, 'NVS_GPCR_hOpiate_mu': 427, 'NVS_GPCR_hTXA2': 428, 'NVS_GPCR_p5HT2C': 429, 'NVS_GPCR_r5HT1_NonSelective': 430, 'NVS_GPCR_r5HT_NonSelective': 431, 'NVS_GPCR_rAdra1B': 432, 'NVS_GPCR_rAdra1_NonSelective': 433, 'NVS_GPCR_rAdra2_NonSelective': 434, 'NVS_GPCR_rAdrb_NonSelective': 435, 'NVS_GPCR_rNK1': 436, 'NVS_GPCR_rNK3': 437, 'NVS_GPCR_rOpiate_NonSelective': 438, 'NVS_GPCR_rOpiate_NonSelectiveNa': 439, 'NVS_GPCR_rSST': 440, 'NVS_GPCR_rTRH': 441, 'NVS_GPCR_rV1': 442, 'NVS_GPCR_rabPAF': 443, 'NVS_GPCR_rmAdra2B': 444, 'NVS_IC_hKhERGCh': 445, 'NVS_IC_rCaBTZCHL': 446, 'NVS_IC_rCaDHPRCh_L': 447, 'NVS_IC_rNaCh_site2': 448, 'NVS_LGIC_bGABARa1': 449, 'NVS_LGIC_h5HT3': 450, 'NVS_LGIC_hNNR_NBungSens': 451, 'NVS_LGIC_rGABAR_NonSelective': 452, 'NVS_LGIC_rNNR_BungSens': 453, 'NVS_MP_hPBR': 454, 'NVS_MP_rPBR': 455, 'NVS_NR_bER': 456, 'NVS_NR_bPR': 457, 'NVS_NR_cAR': 458, 'NVS_NR_hAR': 459, 'NVS_NR_hCAR_Antagonist': 460, 'NVS_NR_hER': 461, 'NVS_NR_hFXR_Agonist': 462, 'NVS_NR_hFXR_Antagonist': 463, 'NVS_NR_hGR': 464, 'NVS_NR_hPPARa': 465, 'NVS_NR_hPPARg': 466, 'NVS_NR_hPR': 467, 'NVS_NR_hPXR': 468, 'NVS_NR_hRAR_Antagonist': 469, 'NVS_NR_hRARa_Agonist': 470, 'NVS_NR_hTRa_Antagonist': 471, 'NVS_NR_mERa': 472, 'NVS_NR_rAR': 473, 'NVS_NR_rMR': 474, 'NVS_OR_gSIGMA_NonSelective': 475, 'NVS_TR_gDAT': 476, 'NVS_TR_hAdoT': 477, 'NVS_TR_hDAT': 478, 'NVS_TR_hNET': 479, 'NVS_TR_hSERT': 480, 'NVS_TR_rNET': 481, 'NVS_TR_rSERT': 482, 'NVS_TR_rVMAT2': 483, 'OT_AR_ARELUC_AG_1440': 484, 'OT_AR_ARSRC1_0480': 485, 'OT_AR_ARSRC1_0960': 486, 'OT_ER_ERaERa_0480': 487, 'OT_ER_ERaERa_1440': 488, 'OT_ER_ERaERb_0480': 489, 'OT_ER_ERaERb_1440': 490, 'OT_ER_ERbERb_0480': 491, 'OT_ER_ERbERb_1440': 492, 'OT_ERa_EREGFP_0120': 493, 'OT_ERa_EREGFP_0480': 494, 'OT_FXR_FXRSRC1_0480': 495, 'OT_FXR_FXRSRC1_1440': 496, 'OT_NURR1_NURR1RXRa_0480': 497, 'OT_NURR1_NURR1RXRa_1440': 498, 'TOX21_ARE_BLA_Agonist_ch1': 499, 'TOX21_ARE_BLA_Agonist_ch2': 500, 'TOX21_ARE_BLA_agonist_ratio': 501, 'TOX21_ARE_BLA_agonist_viability': 502, 'TOX21_AR_BLA_Agonist_ch1': 503, 'TOX21_AR_BLA_Agonist_ch2': 504, 'TOX21_AR_BLA_Agonist_ratio': 505, 'TOX21_AR_BLA_Antagonist_ch1': 506, 'TOX21_AR_BLA_Antagonist_ch2': 507, 'TOX21_AR_BLA_Antagonist_ratio': 508, 'TOX21_AR_BLA_Antagonist_viability': 509, 'TOX21_AR_LUC_MDAKB2_Agonist': 510, 'TOX21_AR_LUC_MDAKB2_Antagonist': 511, 'TOX21_AR_LUC_MDAKB2_Antagonist2': 512, 'TOX21_AhR_LUC_Agonist': 513, 'TOX21_Aromatase_Inhibition': 514, 'TOX21_AutoFluor_HEK293_Cell_blue': 515, 'TOX21_AutoFluor_HEK293_Media_blue': 516, 'TOX21_AutoFluor_HEPG2_Cell_blue': 517, 'TOX21_AutoFluor_HEPG2_Cell_green': 518, 'TOX21_AutoFluor_HEPG2_Media_blue': 519, 'TOX21_AutoFluor_HEPG2_Media_green': 520, 'TOX21_ELG1_LUC_Agonist': 521, 'TOX21_ERa_BLA_Agonist_ch1': 522, 'TOX21_ERa_BLA_Agonist_ch2': 523, 'TOX21_ERa_BLA_Agonist_ratio': 524, 'TOX21_ERa_BLA_Antagonist_ch1': 525, 'TOX21_ERa_BLA_Antagonist_ch2': 526, 'TOX21_ERa_BLA_Antagonist_ratio': 527, 'TOX21_ERa_BLA_Antagonist_viability': 528, 'TOX21_ERa_LUC_BG1_Agonist': 529, 'TOX21_ERa_LUC_BG1_Antagonist': 530, 'TOX21_ESRE_BLA_ch1': 531, 'TOX21_ESRE_BLA_ch2': 532, 'TOX21_ESRE_BLA_ratio': 533, 'TOX21_ESRE_BLA_viability': 534, 'TOX21_FXR_BLA_Antagonist_ch1': 535, 'TOX21_FXR_BLA_Antagonist_ch2': 536, 'TOX21_FXR_BLA_agonist_ch2': 537, 'TOX21_FXR_BLA_agonist_ratio': 538, 'TOX21_FXR_BLA_antagonist_ratio': 539, 'TOX21_FXR_BLA_antagonist_viability': 540, 'TOX21_GR_BLA_Agonist_ch1': 541, 'TOX21_GR_BLA_Agonist_ch2': 542, 'TOX21_GR_BLA_Agonist_ratio': 543, 'TOX21_GR_BLA_Antagonist_ch2': 544, 'TOX21_GR_BLA_Antagonist_ratio': 545, 'TOX21_GR_BLA_Antagonist_viability': 546, 'TOX21_HSE_BLA_agonist_ch1': 547, 'TOX21_HSE_BLA_agonist_ch2': 548, 'TOX21_HSE_BLA_agonist_ratio': 549, 'TOX21_HSE_BLA_agonist_viability': 550, 'TOX21_MMP_ratio_down': 551, 'TOX21_MMP_ratio_up': 552, 'TOX21_MMP_viability': 553, 'TOX21_NFkB_BLA_agonist_ch1': 554, 'TOX21_NFkB_BLA_agonist_ch2': 555, 'TOX21_NFkB_BLA_agonist_ratio': 556, 'TOX21_NFkB_BLA_agonist_viability': 557, 'TOX21_PPARd_BLA_Agonist_viability': 558, 'TOX21_PPARd_BLA_Antagonist_ch1': 559, 'TOX21_PPARd_BLA_agonist_ch1': 560, 'TOX21_PPARd_BLA_agonist_ch2': 561, 'TOX21_PPARd_BLA_agonist_ratio': 562, 'TOX21_PPARd_BLA_antagonist_ratio': 563, 'TOX21_PPARd_BLA_antagonist_viability': 564, 'TOX21_PPARg_BLA_Agonist_ch1': 565, 'TOX21_PPARg_BLA_Agonist_ch2': 566, 'TOX21_PPARg_BLA_Agonist_ratio': 567, 'TOX21_PPARg_BLA_Antagonist_ch1': 568, 'TOX21_PPARg_BLA_antagonist_ratio': 569, 'TOX21_PPARg_BLA_antagonist_viability': 570, 'TOX21_TR_LUC_GH3_Agonist': 571, 'TOX21_TR_LUC_GH3_Antagonist': 572, 'TOX21_VDR_BLA_Agonist_viability': 573, 'TOX21_VDR_BLA_Antagonist_ch1': 574, 'TOX21_VDR_BLA_agonist_ch2': 575, 'TOX21_VDR_BLA_agonist_ratio': 576, 'TOX21_VDR_BLA_antagonist_ratio': 577, 'TOX21_VDR_BLA_antagonist_viability': 578, 'TOX21_p53_BLA_p1_ch1': 579, 'TOX21_p53_BLA_p1_ch2': 580, 'TOX21_p53_BLA_p1_ratio': 581, 'TOX21_p53_BLA_p1_viability': 582, 'TOX21_p53_BLA_p2_ch1': 583, 'TOX21_p53_BLA_p2_ch2': 584, 'TOX21_p53_BLA_p2_ratio': 585, 'TOX21_p53_BLA_p2_viability': 586, 'TOX21_p53_BLA_p3_ch1': 587, 'TOX21_p53_BLA_p3_ch2': 588, 'TOX21_p53_BLA_p3_ratio': 589, 'TOX21_p53_BLA_p3_viability': 590, 'TOX21_p53_BLA_p4_ch1': 591, 'TOX21_p53_BLA_p4_ch2': 592, 'TOX21_p53_BLA_p4_ratio': 593, 'TOX21_p53_BLA_p4_viability': 594, 'TOX21_p53_BLA_p5_ch1': 595, 'TOX21_p53_BLA_p5_ch2': 596, 'TOX21_p53_BLA_p5_ratio': 597, 'TOX21_p53_BLA_p5_viability': 598, 'Tanguay_ZF_120hpf_AXIS_up': 599, 'Tanguay_ZF_120hpf_ActivityScore': 600, 'Tanguay_ZF_120hpf_BRAI_up': 601, 'Tanguay_ZF_120hpf_CFIN_up': 602, 'Tanguay_ZF_120hpf_CIRC_up': 603, 'Tanguay_ZF_120hpf_EYE_up': 604, 'Tanguay_ZF_120hpf_JAW_up': 605, 'Tanguay_ZF_120hpf_MORT_up': 606, 'Tanguay_ZF_120hpf_OTIC_up': 607, 'Tanguay_ZF_120hpf_PE_up': 608, 'Tanguay_ZF_120hpf_PFIN_up': 609, 'Tanguay_ZF_120hpf_PIG_up': 610, 'Tanguay_ZF_120hpf_SNOU_up': 611, 'Tanguay_ZF_120hpf_SOMI_up': 612, 'Tanguay_ZF_120hpf_SWIM_up': 613, 'Tanguay_ZF_120hpf_TRUN_up': 614, 'Tanguay_ZF_120hpf_TR_up': 615, 'Tanguay_ZF_120hpf_YSE_up': 616}

class TOXCAST(InMemoryDataset):
    def __init__(self, 
                 root: str, 
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 force_reload: bool = False,
                 structure: str = "rdkit3d",
                 dataset_args: List[str] = None):
        self.structure = structure
        self.raw_url = URLS[structure]
        self.labels = [toxcast_target_dict[label] for label in dataset_args] if dataset_args is not None else list(toxcast_target_dict.values())

        if transform is None:
            transform = self._filter_label
        else:
            transform = Compose([transform, self._filter_label])

        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)
        self.load(self.processed_paths[0])

    def mean(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].mean())

    def std(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].std())
    
    @property
    def raw_file_names(self) -> List[str]:
        try:
            import rdkit  # noqa
            file_names = {
                "precise3d": ['toxcast_exp.sdf', 'toxcast_exp.sdf.csv'],
                "optimized3d": ['toxcast_opt.sdf', 'toxcast_opt.sdf.csv'],
                "rdkit3d": ['toxcast.sdf', 'toxcast.sdf.csv'],
                "rdkit2d": ['toxcast_graph.sdf', 'toxcast_graph.sdf.csv']
            }
            return file_names[self.structure]
        except ImportError:
            return ImportError("Please install 'rdkit' to download the dataset.")

    @property
    def processed_file_names(self) -> str:
        return 'data_v3.pt'
    
    def download(self):
        try:
            import rdkit  # noqa
            #import gdown
            file_path = download_url(self.raw_url, self.raw_dir)
            #gdown.download(self.raw_url, output=file_path, quiet=False)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)

        except ImportError:
            print("Please install 'rdkit' to download the dataset.", file=sys.stderr)

    def process(self) -> None:
        try:
            import rdkit
            from rdkit import Chem, RDLogger
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit.Chem.rdchem import HybridizationType
            RDLogger.DisableLog('rdApp.*')

        except ImportError:
            rdkit = None

        if rdkit is None:
            print(("Using a pre-processed version of the dataset. Please "
                   "install 'rdkit' to alternatively process the raw data."),
                  file=sys.stderr)

            data_list = torch.load(self.raw_paths[0])
            data_list = [Data(**data_dict) for data_dict in data_list]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            self.save(data_list, self.processed_paths[0])
            return

      
        types = {'O': 0, 'N': 1, 'C': 2, 'Cl': 3, 'H': 4, 'Si': 5, 'Br': 6, 'Ba': 7, 'Nd': 8, 'Dy': 9, 'In': 10, 'P': 11, 'Sb': 12, 'Co': 13, 'S': 14, 'K': 15, 'Na': 16, 'B': 17, 'Ca': 18, 'Hg': 19, 'Ni': 20, 'Se': 21, 'Tl': 22, 'Cd': 23, 'F': 24, 'Fe': 25, 'Li': 26, 'Yb': 27, 'I': 28, 'Cr': 29, 'Sn': 30, 'Zn': 31, 'Cu': 32, 'Pb': 33, 'As': 34, 'Bi': 35, 'Gd': 36, 'V': 37, 'Mn': 38, 'Au': 39, 'Ti': 40, 'Zr': 41, 'Mo': 42, 'Mg': 43, 'Eu': 44, 'Al': 45, 'Pt': 46, 'Sr': 47, 'Sc': 48, 'Ag': 49, 'Pd': 50, 'Be': 51, 'Ge': 52}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3, BT.DATIVE: 4}


        with open(self.raw_paths[1], 'r') as f:
            target = [[float(x) if x != '-100' and x != '' else -1
                       for x in line.split(',')[1:-1]]
                      for line in f.read().split('\n')[1:-1]]
            y = torch.tensor(target, dtype=torch.float)

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False,
                                   sanitize=False)

        data_list = []
        for i, mol in enumerate(tqdm(suppl)):

            N = mol.GetNumAtoms()

            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)

            ## Create a mask for the diagonal
            # mask = torch.eye(N, dtype=bool)
            
            # # Compute the pairwise distances
            # distances = torch.cdist(pos, pos)
            
            # # Apply the mask to the distances (this will set the diagonal elements to infinity)
            # distances.masked_fill_(mask, float('inf'))
            
            # # Now check for overlapping atoms
            # if not torch.all(distances > 0):
            #     #print(f"Skipping molecule {i} due to overlapping atoms.")
            #     continue
              
            # check if any two atoms are overlapping
            if torch.unique(pos, dim=0).size(0) != N:
                # print(f"Skipping molecule {mol.GetProp('_Name')} as it contains overlapping atoms.")
                continue

            type_idx = []
            atomic_number = []
            aromatic = []
            sp = []
            sp2 = []
            sp3 = []
            num_hs = []
            for atom in mol.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])
                atomic_number.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization = atom.GetHybridization()
                sp.append(1 if hybridization == HybridizationType.SP else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

            z = torch.tensor(atomic_number, dtype=torch.long)

            rows, cols, edge_types = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                rows += [start, end]
                cols += [end, start]
                edge_types += 2 * [bonds[bond.GetBondType()]]

            edge_index = torch.tensor([rows, cols], dtype=torch.long)
            edge_type = torch.tensor(edge_types, dtype=torch.long)
            edge_attr = one_hot(edge_type, num_classes=len(bonds))

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]
            edge_attr = edge_attr[perm]

            row, col = edge_index
            hs = (z == 1).to(torch.float)
            num_hs = scatter(hs[row], col, dim_size=N, reduce='sum').tolist()

            x1 = one_hot(torch.tensor(type_idx), num_classes=len(types))
            x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],
                              dtype=torch.float).t().contiguous()
            x = torch.cat([x1, x2], dim=-1)

            name = mol.GetProp('_Name')
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

            data = Data(
                x=x,
                z=z,
                pos=pos,
                edge_index=edge_index,
                smiles=smiles,
                edge_attr=edge_attr,
                y=y[i].unsqueeze(0),
                name=name,
                idx=i,
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])

    def _filter_label(self, batch):
        if self.labels:
            batch.y = batch.y[:, self.labels]
        return batch
