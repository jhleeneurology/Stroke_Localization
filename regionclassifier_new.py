#!/usr/bin/env python3
from __future__ import annotations

import json, argparse, sys
import pathlib
from collections import Counter, defaultdict
from typing import Dict, Any, Optional, List, Tuple, Literal

# --- Defaults (override with CLI flags) ---
DEFAULT_INPUT_FILE  = "1_results_merged.jsonl"
DEFAULT_OUTPUT_FILE = "1_results_cleaned_final_final.jsonl"

# Canonical region list (ordering preserved in outputs)
REGIONS = [
    "Frontal_cortex","Parietal_cortex","Temporal_cortex","Occipital_cortex",
    "Subcortical","Midbrain","Pons","Medulla","Cerebellum",
]
HEMI_REGIONS = {"Frontal_cortex","Parietal_cortex","Temporal_cortex","Occipital_cortex","Subcortical"}
PF_REGIONS   = {"Midbrain","Pons","Medulla","Cerebellum"}

# (kept for reference but not used)
DAMP_PF_IF_HEMI_HINT  = 0.25
DAMP_HEMI_IF_PF_HINT  = 0.40

# ==================== Normalization helpers ====================
def _truthy(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("null", "none", ""):
            return None
        if s in ("true", "t", "yes", "y", "1"):
            return True
        if s in ("false", "f", "no", "n", "0"):
            return False
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, bool):
        return x
    return None

def _any_truthy(*vals: Any) -> Optional[bool]:
    has_true = any(_truthy(v) is True for v in vals)
    has_val  = any(_truthy(v) is not None for v in vals)
    if has_true:
        return True
    if has_val:
        return False
    return None

def _side_from_keys(d: Optional[Dict[str, Any]], right_key: str, left_key: str) -> Optional[str]:
    """
    Return 'R' or 'L' if exactly one of the two keys is truthy.
    If both/none truthy → None.
    """
    if not isinstance(d, dict):
        return None
    r = _truthy(d.get(right_key))
    l = _truthy(d.get(left_key))
    if r is True and l is not True:
        return "R"
    if l is True and r is not True:
        return "L"
    return None

def _get_bool(d: Dict[str, Any], *path) -> Optional[bool]:
    """Safely walk a dict path; coerce to tri-state bool (True/False/None)."""
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict): return None
        cur = cur.get(k)
    return _truthy(cur)

def _any_side_true(d: Dict[str, Any], *path) -> Optional[bool]:
    """Return True if either .R or .L at a path is true, False if either present and both false, else None."""
    node = d
    for k in path:
        if not isinstance(node, dict): return None
        node = node.get(k)
    if not isinstance(node, dict): return None
    r = _truthy(node.get("R"))
    l = _truthy(node.get("L"))
    if r is True or l is True: return True
    if r is not None or l is not None: return False
    return None

def _empty_normalized() -> Dict[str, Any]:
    return {
        "posterior_fossa_signs": {
            "eom_limitation": None, "skew": None, "nystagmus": None, "vertigo": None,
            "nystagmus_direction": None,
        },
        "pupils": {
            "anisocoria": None,
            "R_pupil_light_reflex_absent_or_sluggish": None,
            "L_pupil_light_reflex_absent_or_sluggish": None,
        },
        "cortical_signs": {
            "aphasia": None, "neglect": None, "apraxia": None,
            "R_visual_field_cut": None, "L_visual_field_cut": None,
            "R_gaze_preference": None, "L_gaze_preference": None,
        },
        "nonspecific_signs": {
            "dysarthria": None, "dysmetria_ataxia": None,
            "R_face_weakness": None, "L_face_weakness": None,
            "R_arm_weakness": None, "L_arm_weakness": None,
            "R_leg_weakness": None, "L_leg_weakness": None,
            "weakness_involvement": {"face_arm_gt_leg": None, "leg_gt_face_arm": None, "unknown": None},
            "weakness_side_only": {}, "weakness_lr_comparison": {},
        },
        "sensory": {
            "R_face_numb": None, "L_face_numb": None,
            "R_arm_numb": None, "L_arm_numb": None,
            "R_leg_numb": None, "L_leg_numb": None,
            "cortical_sensory_loss": None,
            "sensory_involvement": {"face_arm_gt_leg": None, "leg_gt_face_arm": None, "unknown": None},
            "sensory_side_only": {},
        },
        "_detail": {"EOM": {}, "PUP_raw": {}, "CTX_raw": {}, "NS_raw": {}, "SENS_raw": {}},
    }

def normalize_survey(survey_or_details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dual-ingest:
      - If input looks like the NEW tagger output (under survey_details.details),
        map it to the localizer's internal normalized structure.
      - Else fall back to legacy mapping (old schema).
    Always returns a normalized dict.
    """
    if not isinstance(survey_or_details, dict):
        return _empty_normalized()

    # ---------- Detect NEW structure ----------
    details = None
    if isinstance(survey_or_details.get("survey_details") if isinstance(survey_or_details, dict) else None, dict):
        details = (survey_or_details.get("survey_details") or {}).get("details") or {}
    else:
        # Possibly already the details blob itself
        if isinstance(survey_or_details.get("eom") if isinstance(survey_or_details, dict) else None, dict):
            details = survey_or_details

    if isinstance(details, dict) and {"eom","pupils","cortical_signs"}.issubset(details.keys()):
        D = details  # shorthand

        # ----- Posterior fossa signs (PF) -----
        PF = {
            "eom_limitation": _any_side_true(D, "eom", "unspecified_limit") or
                              _any_side_true(D, "eom", "cn3_palsy") or
                              _any_side_true(D, "eom", "cn4_palsy") or
                              _any_side_true(D, "eom", "cn6_palsy"),
            "skew":  (_any_side_true(D, "skew", "skew_side") or _get_bool(D, "skew", "skew_legacy")),
            "nystagmus": _get_bool(D, "nystagmus", "pathologic"),
            "vertigo":    _get_bool(D, "vertigo", "present"),
            "nystagmus_direction": (D.get("nystagmus") or {}).get("direction"),
        }

        # Pupils
        def either_true_false(r_fixed, r_slugg):
            if r_fixed is True or r_slugg is True: return True
            if r_fixed is False or r_slugg is False: return False
            return None

        PUP = {
            "anisocoria": _get_bool(D, "pupils", "anisocoria"),
            "R_pupil_light_reflex_absent_or_sluggish": either_true_false(
                _get_bool(D, "pupils", "reactivity", "fixed", "R"),
                _get_bool(D, "pupils", "reactivity", "sluggish", "R")
            ),
            "L_pupil_light_reflex_absent_or_sluggish": either_true_false(
                _get_bool(D, "pupils", "reactivity", "fixed", "L"),
                _get_bool(D, "pupils", "reactivity", "sluggish", "L")
            ),
        }

        # Cortical signs
        CTX_block = D.get("cortical_signs") or {}
        aph = CTX_block.get("aphasia")
        aph_present = _truthy((aph or {}).get("present"))
        aph_type    = (aph or {}).get("type")

        CTX = {
            "aphasia": aph_present,
            "neglect": True if (_any_side_true(D, "cortical_signs", "neglect_side") is True) else (
                        False if (_any_side_true(D, "cortical_signs", "neglect_side") is False) else None),
            "apraxia": True if (_any_side_true(D, "cortical_signs", "apraxia_side") is True) else (
                        False if (_any_side_true(D, "cortical_signs", "apraxia_side") is False) else None),
            "R_visual_field_cut": _get_bool(D, "cortical_signs", "visual_field_cut", "R"),
            "L_visual_field_cut": _get_bool(D, "cortical_signs", "visual_field_cut", "L"),
            "R_gaze_preference":  _get_bool(D, "cortical_signs", "gaze_preference", "R"),
            "L_gaze_preference":  _get_bool(D, "cortical_signs", "gaze_preference", "L"),
        }

        # ---------- Non-specific / motor ----------
        weak = D.get("weakness") or {}
        face_node = (weak.get("face") or {})
        arm_node  = (weak.get("arm")  or {})
        leg_node  = (weak.get("leg")  or {})

        # Explicit sided booleans if present
        face_R_exp = _truthy(face_node.get("R"))
        face_L_exp = _truthy(face_node.get("L"))
        arm_R_exp  = _truthy(arm_node.get("R"))
        arm_L_exp  = _truthy(arm_node.get("L"))
        leg_R_exp  = _truthy(leg_node.get("R"))
        leg_L_exp  = _truthy(leg_node.get("L"))

        # Fallback proxies from lr_comparison (kept)
        face_R_cmp = _get_bool(D, "weakness", "lr_comparison", "face", "R_gt_L")
        face_L_cmp = _get_bool(D, "weakness", "lr_comparison", "face", "L_gt_R")

        # Choose explicit if available; else proxy
        R_face_final = face_R_exp if face_R_exp is not None else face_R_cmp
        L_face_final = face_L_exp if face_L_exp is not None else face_L_cmp

        NS = {
            "dysarthria": _get_bool(D, "dysarthria", "present"),

            # Face weakness: prefer explicit booleans; else proxy
            "R_face_weakness": R_face_final,
            "L_face_weakness": L_face_final,

            # Arm/leg weakness: explicit booleans
            "R_arm_weakness": arm_R_exp,
            "L_arm_weakness": arm_L_exp,
            "R_leg_weakness": leg_R_exp,
            "L_leg_weakness": leg_L_exp,

            "weakness_involvement": {
                "face_arm_gt_leg": _get_bool(D, "weakness", "involvement_pattern", "face_arm_gt_leg"),
                "leg_gt_face_arm": _get_bool(D, "weakness", "involvement_pattern", "leg_gt_face_arm"),
                "unknown": None,
            },
            "weakness_side_only": {
                "R": _get_bool(D, "weakness", "side_only", "R"),
                "L": _get_bool(D, "weakness", "side_only", "L"),
            },
            "weakness_lr_comparison": {
                "overall": {
                    "R_gt_L": _get_bool(D, "weakness", "lr_comparison", "overall", "R_gt_L"),
                    "L_gt_R": _get_bool(D, "weakness", "lr_comparison", "overall", "L_gt_R"),
                },
                "face": {
                    "R_gt_L": _get_bool(D, "weakness", "lr_comparison", "face", "R_gt_L"),
                    "L_gt_R": _get_bool(D, "weakness", "lr_comparison", "face", "L_gt_R"),
                },
                "arm": {
                    "R_gt_L": _get_bool(D, "weakness", "lr_comparison", "arm", "R_gt_L"),
                    "L_gt_R": _get_bool(D, "weakness", "lr_comparison", "arm", "L_gt_R"),
                },
                "leg": {
                    "R_gt_L": _get_bool(D, "weakness", "lr_comparison", "leg", "R_gt_L"),
                    "L_gt_R": _get_bool(D, "weakness", "lr_comparison", "leg", "L_gt_R"),
                },
            },
        }

        # Sensory
        SENS = {
            "R_face_numb": _get_bool(D, "sensory_loss", "face", "R"),
            "L_face_numb": _get_bool(D, "sensory_loss", "face", "L"),
            "R_arm_numb":  _get_bool(D, "sensory_loss", "arm", "R"),
            "L_arm_numb":  _get_bool(D, "sensory_loss", "arm", "L"),
            "R_leg_numb":  _get_bool(D, "sensory_loss", "leg", "R"),
            "L_leg_numb":  _get_bool(D, "sensory_loss", "leg", "L"),
            "cortical_sensory_loss": _get_bool(D, "sensory_loss", "cortical_sensory_loss"),
            "sensory_involvement": {
                "face_arm_gt_leg": _get_bool(D, "sensory_loss", "involvement_pattern", "face_arm_gt_leg"),
                "leg_gt_face_arm": _get_bool(D, "sensory_loss", "involvement_pattern", "leg_gt_face_arm"),
                "unknown":         _get_bool(D, "sensory_loss", "involvement_pattern", "unknown"),
            },
            "sensory_side_only": {
                "R": _get_bool(D, "sensory_loss", "side_only", "R"),
                "L": _get_bool(D, "sensory_loss", "side_only", "L"),
            },
        }

        # Detail passthrough
        detail = {
            "EOM": {
                "R_CNIII_palsy": _get_bool(D, "eom", "cn3_palsy", "R"),
                "L_CNIII_palsy": _get_bool(D, "eom", "cn3_palsy", "L"),
                "R_CNIV_palsy":  _get_bool(D, "eom", "cn4_palsy", "R"),
                "L_CNIV_palsy":  _get_bool(D, "eom", "cn4_palsy", "L"),
                "R_CNVI_palsy":  _get_bool(D, "eom", "cn6_palsy", "R"),
                "L_CNVI_palsy":  _get_bool(D, "eom", "cn6_palsy", "L"),
                "R_unspecified_limitation": _get_bool(D, "eom", "unspecified_limit", "R"),
                "L_unspecified_limitation": _get_bool(D, "eom", "unspecified_limit", "L"),
            },
            "PUP_raw": {
                "fixed_R":    _get_bool(D, "pupils", "reactivity", "fixed", "R"),
                "fixed_L":    _get_bool(D, "pupils", "reactivity", "fixed", "L"),
                "sluggish_R": _get_bool(D, "pupils", "reactivity", "sluggish", "R"),
                "sluggish_L": _get_bool(D, "pupils", "reactivity", "sluggish", "L"),
                "brisk_R": None, "brisk_L": None,
                "anisocoria": _get_bool(D, "pupils", "anisocoria"),
            },
            "CTX_raw": {
                "aphasia": {"present": aph_present, "type": aph_type},
                "neglect": {
                    "L": _get_bool(D, "cortical_signs", "neglect_side", "L"),
                    "R": _get_bool(D, "cortical_signs", "neglect_side", "R"),
                },
                "apraxia": {
                    "L": _get_bool(D, "cortical_signs", "apraxia_side", "L"),
                    "R": _get_bool(D, "cortical_signs", "apraxia_side", "R"),
                    "type": CTX_block.get("apraxia_type")
                },
                "R_visual_field_cut": _get_bool(D, "cortical_signs", "visual_field_cut", "R"),
                "L_visual_field_cut": _get_bool(D, "cortical_signs", "visual_field_cut", "L"),
                "R_gaze_preference":  _get_bool(D, "cortical_signs", "gaze_preference", "R"),
                "L_gaze_preference":  _get_bool(D, "cortical_signs", "gaze_preference", "L"),
            },
            "NS_raw":  weak,
            "SENS_raw": D.get("sensory_loss") or {},
        }

        return {
            "posterior_fossa_signs": PF,
            "pupils": PUP,
            "cortical_signs": CTX,
            "nonspecific_signs": NS,
            "sensory": SENS,
            "_detail": detail,
        }

    # ---------- Legacy path (old schema) ----------
    PF_raw   = survey_or_details.get("posterior_fossa_signs", {}) or {}
    CTX_raw  = survey_or_details.get("cortical_signs", {}) or {}
    PUP_raw  = survey_or_details.get("pupils", {}) or {}
    NS_raw   = survey_or_details.get("nonspecific_signs", {}) or {}
    SENS_raw = survey_or_details.get("sensory", {}) or {}

    eom = PF_raw.get("EOM") or PF_raw.get("eom") or {}
    eom_limitation = _any_truthy(
        eom.get("R_CNIII_palsy"), eom.get("L_CNIII_palsy"),
        eom.get("R_CNIV_palsy"),  eom.get("L_CNIV_palsy"),
        eom.get("R_CNVI_palsy"),  eom.get("L_CNVI_palsy"),
        eom.get("R_unspecified_limitation"), eom.get("L_unspecified_limitation"),
        PF_raw.get("eom_limitation")
    )
    pf_skew      = _truthy(PF_raw.get("skew"))
    pf_nystagmus = _any_truthy(PF_raw.get("nystagmus_pathologic"), PF_raw.get("nystagmus"))
    pf_vertigo   = _truthy(PF_raw.get("vertigo"))

    PF = {
        "eom_limitation": eom_limitation,
        "skew": pf_skew,
        "nystagmus": pf_nystagmus,
        "vertigo": pf_vertigo,
        "nystagmus_direction": PF_raw.get("nystagmus_direction")
    }

    fixed_R    = _truthy(PUP_raw.get("fixed_R"))
    fixed_L    = _truthy(PUP_raw.get("fixed_L"))
    sluggish_R = _truthy(PUP_raw.get("sluggish_R"))
    sluggish_L = _truthy(PUP_raw.get("sluggish_L"))
    brisk_R    = _truthy(PUP_raw.get("brisk_R"))
    brisk_L    = _truthy(PUP_raw.get("brisk_L"))

    def plr_flag(side_fixed, side_sluggish, side_brisk, legacy):
        if _truthy(side_brisk) is True:
            return False
        return _any_truthy(side_fixed, side_sluggish, legacy)

    PUP = {
        "anisocoria": _truthy(PUP_raw.get("anisocoria")),
        "R_pupil_light_reflex_absent_or_sluggish": plr_flag(fixed_R, sluggish_R, brisk_R, PUP_raw.get("R_pupil_light_reflex_absent_or_sluggish")),
        "L_pupil_light_reflex_absent_or_sluggish": plr_flag(fixed_L, sluggish_L, brisk_L, PUP_raw.get("L_pupil_light_reflex_absent_or_sluggish")),
    }

    aph = CTX_raw.get("aphasia")
    aph_present = _truthy(aph.get("present")) if isinstance(aph, dict) else _truthy(aph)

    CTX = {
        "aphasia": aph_present,
        "neglect": _truthy(CTX_raw.get("neglect")),
        "apraxia": _truthy(CTX_raw.get("apraxia")),
        "R_visual_field_cut": _truthy(CTX_raw.get("R_visual_field_cut")),
        "L_visual_field_cut": _truthy(CTX_raw.get("L_visual_field_cut")),
        "R_gaze_preference": _truthy(CTX_raw.get("R_gaze_preference")),
        "L_gaze_preference": _truthy(CTX_raw.get("L_gaze_preference")),
    }

    dysarthria    = _any_truthy(NS_raw.get("dysarthria"), CTX_raw.get("dysarthria"))
    dysmetria_any = _any_truthy(
        NS_raw.get("dysmetria_ataxia"),
        NS_raw.get("R_dysmetria_ataxia"),
        NS_raw.get("L_dysmetria_ataxia")
    )

    NS = {
        "dysarthria": dysarthria,
        "dysmetria_ataxia": dysmetria_any,
        "R_face_weakness": _truthy(NS_raw.get("R_face_weakness")),
        "L_face_weakness": _truthy(NS_raw.get("L_face_weakness")),
        "R_arm_weakness":  _truthy(NS_raw.get("R_arm_weakness")),
        "L_arm_weakness":  _truthy(NS_raw.get("L_arm_weakness")),
        "R_leg_weakness":  _truthy(NS_raw.get("R_leg_weakness")),
        "L_leg_weakness":  _truthy(NS_raw.get("L_leg_weakness")),
        "weakness_involvement": {
            "face_arm_gt_leg": _truthy((NS_raw.get("weakness_involvement") or {}).get("face_arm_gt_leg")),
            "leg_gt_face_arm": _truthy((NS_raw.get("weakness_involvement") or {}).get("leg_gt_face_arm")),
            "unknown":         _truthy((NS_raw.get("weakness_involvement") or {}).get("unknown")),
        },
        "weakness_side_only": (NS_raw.get("weakness_side_only") or {}),
        "weakness_lr_comparison": (NS_raw.get("weakness_lr_comparison") or {}),
    }

    sinv_raw = SENS_raw.get("sensory_involvement") or {}
    SENS = {
        "R_face_numb": _truthy(SENS_raw.get("R_face_numb")),
        "L_face_numb": _truthy(SENS_raw.get("L_face_numb")),
        "R_arm_numb":  _truthy(SENS_raw.get("R_arm_numb")),
        "L_arm_numb":  _truthy(SENS_raw.get("L_arm_numb")),
        "R_leg_numb":  _truthy(SENS_raw.get("R_leg_numb")),
        "L_leg_numb":  _truthy(SENS_raw.get("L_leg_numb")),
        "cortical_sensory_loss": _truthy(SENS_raw.get("cortical_sensory_loss")),
        "sensory_involvement": {
            "face_arm_gt_leg": _truthy(sinv_raw.get("face_arm_gt_leg")),
            "leg_gt_face_arm": _truthy(sinv_raw.get("leg_gt_face_arm")),
            "unknown":         _truthy(sinv_raw.get("unknown")),
        },
        "sensory_side_only": (SENS_raw.get("sensory_side_only") or {}),
    }

    detail = {
        "EOM": {
            "R_CNIII_palsy": _truthy(eom.get("R_CNIII_palsy")),
            "L_CNIII_palsy": _truthy(eom.get("L_CNIII_palsy")),
            "R_CNIV_palsy":  _truthy(eom.get("R_CNIV_palsy")),
            "L_CNIV_palsy":  _truthy(eom.get("L_CNIV_palsy")),
            "R_CNVI_palsy":  _truthy(eom.get("R_CNVI_palsy")),
            "L_CNVI_palsy":  _truthy(eom.get("L_CNVI_palsy")),
            "R_unspecified_limitation": _truthy(eom.get("R_unspecified_limitation")),
            "L_unspecified_limitation": _truthy(eom.get("L_unspecified_limitation")),
        },
        "PUP_raw": {
            "fixed_R": fixed_R, "fixed_L": fixed_L,
            "sluggish_R": sluggish_R, "sluggish_L": sluggish_L,
            "brisk_R": brisk_R, "brisk_L": brisk_L,
            "anisocoria": _truthy(PUP_raw.get("anisocoria")),
        },
        "CTX_raw": CTX_raw,
        "NS_raw":  NS_raw,
        "SENS_raw": SENS_raw,
    }

    return {
        "posterior_fossa_signs": PF,
        "pupils": PUP,
        "cortical_signs": CTX,
        "nonspecific_signs": NS,
        "sensory": SENS,
        "_detail": detail,
    }

# --- Uniform side inference helpers ---
def _vote_count(votes: List[Optional[str]]) -> Optional[str]:
    vs = [v for v in votes if v in ("Left","Right")]
    return Counter(vs).most_common(1)[0][0] if vs else None

def _vfcut_to_lesion_side(r_vfc: Optional[bool], l_vfc: Optional[bool]) -> Optional[str]:
    if r_vfc is True:
        return "Left"
    if l_vfc is True:
        return "Right"
    return None

def _gaze_pref_to_lesion_side(r_gp: Optional[bool], l_gp: Optional[bool]) -> Optional[str]:
    if r_gp is True:
        return "Right"
    if l_gp is True:
        return "Left"
    return None

# ============ Small helpers ============
def t(x) -> bool:
    return x is True

def _any_true_in_obj(o: Any) -> bool:
    """Return True if any boolean True is found anywhere in a nested structure."""
    if isinstance(o, bool):
        return o is True
    if isinstance(o, dict):
        return any(_any_true_in_obj(v) for v in o.values())
    if isinstance(o, (list, tuple)):
        return any(_any_true_in_obj(v) for v in o)
    return False


def majority_side(flags: List[Tuple[str, Any]]) -> Optional[str]:
    votes = [s for s, v in flags if t(v)]
    return Counter(votes).most_common(1)[0][0] if votes else None

def contra(side: Optional[str]) -> Optional[str]:
    return {"R":"Left", "L":"Right"}.get(side)

def ipsi(side: Optional[str]) -> Optional[str]:
    return {"R":"Right", "L":"Left"}.get(side)



def _is_contradictory(node: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(node, dict):
        return False
    return _truthy(node.get("L_gt_R")) is True and _truthy(node.get("R_gt_L")) is True


def limb_motor_side(ns: Dict[str, Any]) -> Optional[str]:
    """
    Returns 'R'/'L' indicating the side of WEAKNESS using a weighted voting system.
    This is more robust to conflicting data than a rigid hierarchical check.
    """
    if not isinstance(ns, dict):
        return None

    score_R, score_L = 0.0, 0.0
    votes = []

    # --- Define weights for different pieces of evidence ---
    W_LIMB_EXPLICIT = 1.0  # Strong evidence
    W_SIDE_ONLY     = 1.5  # Very strong evidence
    W_OVERALL_COMP  = 0.75 # Moderate evidence (can be unreliable)
    W_FACE_EXPLICIT = 0.5  # Weaker evidence for limb weakness

    # --- Tally scores from all available data ---

    # 1. Explicit limb weakness
    if t(ns.get("R_arm_weakness")):
        score_R += W_LIMB_EXPLICIT
        votes.append(f"R_arm(+{W_LIMB_EXPLICIT})")
    if t(ns.get("L_arm_weakness")):
        score_L += W_LIMB_EXPLICIT
        votes.append(f"L_arm(+{W_LIMB_EXPLICIT})")
    if t(ns.get("R_leg_weakness")):
        score_R += W_LIMB_EXPLICIT
        votes.append(f"R_leg(+{W_LIMB_EXPLICIT})")
    if t(ns.get("L_leg_weakness")):
        score_L += W_LIMB_EXPLICIT
        votes.append(f"L_leg(+{W_LIMB_EXPLICIT})")

    # 2. "Side only" flag
    side_only = ns.get("weakness_side_only") or {}
    if t(side_only.get("R")):
        score_R += W_SIDE_ONLY
        votes.append(f"side_only_R(+{W_SIDE_ONLY})")
    if t(side_only.get("L")):
        score_L += W_SIDE_ONLY
        votes.append(f"side_only_L(+{W_SIDE_ONLY})")

    # 3. Overall L/R comparison
    comp = (ns.get("weakness_lr_comparison") or {}).get("overall", {})
    if not _is_contradictory(comp):
        if t(comp.get("R_gt_L")):
            score_R += W_OVERALL_COMP
            votes.append(f"overall_R>L(+{W_OVERALL_COMP})")
        if t(comp.get("L_gt_R")):
            score_L += W_OVERALL_COMP
            votes.append(f"overall_L>R(+{W_OVERALL_COMP})")

    # 4. Face weakness (as a weaker, contributing factor)
    if t(ns.get("R_face_weakness")):
        score_R += W_FACE_EXPLICIT
        votes.append(f"R_face(+{W_FACE_EXPLICIT})")
    if t(ns.get("L_face_weakness")):
        score_L += W_FACE_EXPLICIT
        votes.append(f"L_face(+{W_FACE_EXPLICIT})")

    # --- Decide winner ---
    # Use a small tolerance band; if scores are too close, it's ambiguous.
    TIE_THRESHOLD = 0.1

    if abs(score_R - score_L) < TIE_THRESHOLD:
        return None # Scores are tied or too close to call

    if score_R > score_L:
        return "R"

    if score_L > score_R:
        return "L"

    return None

def limb_sensory_side(sens: Dict[str, Any]) -> Optional[str]:
    """
    Try explicit limb numbness first; if absent, fall back to sensory_side_only {R,L}.
    Returns 'R'/'L' (side of NUMBNESS), not lesion side.
    """
    side = majority_side([
        ("R", sens.get("R_arm_numb")), ("R", sens.get("R_leg_numb")),
        ("L", sens.get("L_arm_numb")), ("L", sens.get("L_leg_numb")),
    ])
    if side in ("R","L"):
        return side

    side_only = sens.get("sensory_side_only") or {}
    side_fallback = _side_from_keys(side_only, "R", "L")
    return side_fallback

def _region_from_label(label: str) -> str:
    return label.split(" ",1)[1] if label.startswith(("Left ","Right ")) else label

def _side_from_best_sided_label(label_scores: Dict[str, float]) -> Optional[str]:
    best_side, best_score = None, float("-inf")
    for lab, sc in label_scores.items():
        if lab.startswith("Left ") or lab.startswith("Right "):
            side = "Left" if lab.startswith("Left ") else "Right"
            if sc > best_score:
                best_side, best_score = side, sc
    return best_side

def _coerce_uniform_side(preferred: Optional[str],
                         label_scores: Dict[str, float],
                         loc: Dict[str, Any]) -> Optional[str]:
    if preferred in ("Left","Right"):
        return preferred
    best_from_scores = _side_from_best_sided_label(label_scores)
    if best_from_scores:
        return best_from_scores
    ihl = loc.get("inferred_hemisphere_label")
    if ihl in ("Left","Right"):
        return ihl
    return None

# ============ Scoring utilities ============
class Scorer:
    def __init__(self):
        self.region_scores: Dict[str, float] = defaultdict(float)
        self.sided_scores: Dict[str, float] = defaultdict(float)  # "Left Frontal_cortex"
        self.reasons: Dict[str, List[str]] = defaultdict(list)

    def add(self, region: str, pts: float, rationale: str, side: Optional[str]=None):
        self.region_scores[region] += pts
        self.reasons[region].append(f"+{pts:g}: {rationale}")
        if side is not None:
            label = f"{side} {region}"
            self.sided_scores[label] += pts
            self.reasons[label].append(f"+{pts:g}: {rationale}")

    def scale_region(self, region: str, factor: float, note: str):
        old = self.region_scores.get(region, 0.0)
        self.region_scores[region] = old * factor
        if note:
            self.reasons[region].append(f"×{factor:g}: {note}")
        suffix = f" {region}"
        for lab, score in list(self.sided_scores.items()):
            if lab.endswith(suffix):
                self.sided_scores[lab] = score * factor
                if note:
                    self.reasons[lab].append(f"×{factor:g}: {note}")

    def top5(self) -> Tuple[List[str], List[str], Dict[str, float], Dict[str, List[str]]]:
        label_scores: Dict[str, float] = dict(self.sided_scores)
        for r, s in self.region_scores.items():
            label_scores[r] = label_scores.get(r, 0.0) + s

        region_rank = {r:i for i,r in enumerate(REGIONS)}
        def keyfun(item):
            label, score = item
            region = _region_from_label(label)
            side_rank = 0 if label.startswith("Left ") else (1 if label.startswith("Right ") else 2)
            return (-score, region_rank.get(region, 999), side_rank, label)

        top = sorted(label_scores.items(), key=keyfun)[:4]
        labels = [lab for lab,_ in top]
        regions_in_top: List[str] = []
        seen = set()
        for lab,_ in top:
            region = _region_from_label(lab)
            if region not in seen:
                regions_in_top.append(region); seen.add(region)
        return labels, regions_in_top, dict(label_scores), self.reasons

# Rule weights (tunable)
W = {
    # Posterior fossa
    "EOM_CNIII": 3.0, "EOM_CNIV": 2.5,
    "EOM_CNVI": 2.5,
    "EOM_unspec": 1.5,
    "skew": 1.2, "vertigo": 1.0, "nystagmus": 1.2,
    "pupil_fixed": 3.0, "pupil_sluggish": 1.5, "anisocoria": 0.8,

    # Cortical
    "aphasia_base": 2.5, "aphasia_fluent": 1.2, "aphasia_nonfluent": 1.2,
    "aphasia_global": 1.0, "aphasia_anomic_subcort": 1.0, "aphasia_conduction": 1.2,
    "neglect_R_parietal": 2.0, "apraxia": 0.8,
    "gaze_pref": 0.8, "visual_cut": 1.2,

    # Non-specific
    "dysarthria": 0.6, "ataxia_generic": 1.0, "ataxia_sided_bundle": 0.8,

    # Motor/sensory patterns
    "motor_no_cortical_subcort": 1.5,
    "weak_face_arm_gt_leg": 1.0, "weak_leg_gt_face_arm": 1.0,
    "sens_face_arm_gt_leg": 1.0, "sens_leg_gt_face_arm": 1.0,
    "hemisensory_pure_subcort": 1.2,
    "crossed_sens_face_body": 1.4,

    # Pons patterns
    "pons_crossed_face_limb": 2.0, "pons_same_side_all": 1.6,

    # PF gating/bundle
    "pf_bundle_boost": 1.2,
    "pf_aphasia_dampen": 0.5,
}

def _apply_pf_bundle_and_aphasia_gating(scorer: Scorer, norm: Dict[str, Any], aphasia_present: bool):
    PF = norm.get("posterior_fossa_signs", {}) or {}
    pf_flags = [PF.get("vertigo"), PF.get("nystagmus"), PF.get("eom_limitation"), PF.get("skew")]
    bundle_ct = sum(1 for f in pf_flags if f is True)
    pf_bundle = bundle_ct >= 2

    pf_regions = ["Midbrain","Pons","Medulla","Cerebellum"]

    if pf_bundle:
        for r in pf_regions:
            scorer.region_scores[r] += W["pf_bundle_boost"]
            scorer.reasons[r].append(f"+{W['pf_bundle_boost']}: PF bundle present (≥2 PF signs)")

    if aphasia_present and not pf_bundle:
        for r in pf_regions:
            old = scorer.region_scores[r]
            new = old * W["pf_aphasia_dampen"]
            if new != old:
                scorer.region_scores[r] = new
                scorer.reasons[r].append(f"×{W['pf_aphasia_dampen']}: Aphasia present without PF bundle (PF dampened)")

# --- Original single-return ladder (kept for backward compat if other code calls it) ---
# --- Hybrid (cortical-precedence) uniform side inference ---
def infer_uniform_side(norm: Dict[str, Any]) -> Tuple[Optional[str], str]:
    """
    Hybrid voter with cortical-sign precedence and a tie band.
    Weights (tunable):
      - Cortical (aphasia; neglect+VF; gaze): high
      - Motor / Sensory (contralateral): moderate
    Behavior:
      - Aphasia ⇒ strong LEFT vote (typical dominance) unless outweighed by strong right cortical evidence.
      - Neglect ⇒ RIGHT vote.
      - Visual field cut: R cut ⇒ LEFT; L cut ⇒ RIGHT.
      - Gaze preference: ipsilateral vote (small).
      - Motor (limb) and Sensory (limb) give contralateral votes ONLY if side is explicit / determinable.
      - Face-only motor contributes, but with a smaller weight than limb motor.
      - Tie band τ prevents a weak motor cue from overruling strong cortical evidence.
    Returns:
      (uniform_side: 'Left'/'Right'/None, rationale: str)
    """

    CTX  = norm.get("cortical_signs", {}) or {}
    NS   = norm.get("nonspecific_signs", {}) or {}
    SENS = norm.get("sensory", {}) or {}
    DTL  = norm.get("_detail", {}) or {}

    # ---- tunables (feel free to tweak) ----
    W_CORTICAL = 3.0
    W_MOTOR    = 1.3   # limb motor
    W_FACE     = 0.8   # face-only motor
    W_SENS     = 1.0
    W_GAZE     = 0.9
    W_NEGLECT  = 3.0
    W_VF       = 1.6
    TAU        = 1.5   # tie band: |S_L - S_R| < TAU → undecided

    votes_L: List[str] = []
    votes_R: List[str] = []
    reasons: List[str] = []

    def add_left(w: float, why: str):
        votes_L.append(f"{w:g}:{why}")
    def add_right(w: float, why: str):
        votes_R.append(f"{w:g}:{why}")

    # ===== Cortical-first =====
    # Aphasia → Left (strong, typical dominance; keep as high weight)
    if CTX.get("aphasia") is True:
        add_left(W_CORTICAL, "aphasia")

    # Neglect → Right
    ngl_raw = (DTL.get("CTX_raw") or {}).get("neglect") or {}
    if _truthy(ngl_raw.get("L")) or CTX.get("neglect") is True:
        add_right(W_NEGLECT, "neglect")

    # Visual field cut: R cut → Left; L cut → Right
    if CTX.get("R_visual_field_cut") is True:
        add_left(W_VF, "R VF cut")
    if CTX.get("L_visual_field_cut") is True:
        add_right(W_VF, "L VF cut")

    # Gaze preference: ipsilateral (small)
    if CTX.get("R_gaze_preference") is True:
        add_right(W_GAZE, "R gaze pref")
    if CTX.get("L_gaze_preference") is True:
        add_left(W_GAZE, "L gaze pref")

    # ===== Motor (contralateral) =====
    motor_side_RL = limb_motor_side(NS)       # 'R'/'L'/None (patient deficit side)
    if motor_side_RL == "R":
        add_left(W_MOTOR, "motor R→contra Left")
    elif motor_side_RL == "L":
        add_right(W_MOTOR, "motor L→contra Right")

    # Face-only motor (smaller weight, only if limb motor indeterminate)
    if motor_side_RL is None:
        face_R = t(NS.get("R_face_weakness"))
        face_L = t(NS.get("L_face_weakness"))
        if face_R and not face_L:
            add_left(W_FACE, "face motor R→contra Left")
        elif face_L and not face_R:
            add_right(W_FACE, "face motor L→contra Right")

    # ===== Sensory (contralateral) =====
    sens_side_RL = limb_sensory_side(SENS)    # 'R'/'L'/None
    if sens_side_RL == "R":
        add_left(W_SENS, "sensory R→contra Left")
    elif sens_side_RL == "L":
        add_right(W_SENS, "sensory L→contra Right")

    # ===== Tally =====
    def sum_weights(vs: List[str]) -> float:
        s = 0.0
        for item in vs:
            try:
                w_str, _ = item.split(":", 1)
                s += float(w_str)
            except Exception:
                pass
        return s

    S_left  = sum_weights(votes_L)
    S_right = sum_weights(votes_R)

    # Build readable rationale
    if votes_L:  reasons.append("Left votes:  "  + ", ".join(votes_L))
    if votes_R:  reasons.append("Right votes: " + ", ".join(votes_R))
    reasons.append(f"S_left={S_left:.2f}  S_right={S_right:.2f}  τ={TAU}")

    # Decision with tie band
    if abs(S_left - S_right) < TAU:
        reasons.append("Δ below τ → Undetermined")
        return None, " | ".join(reasons)

    side = "Left" if S_left > S_right else "Right"
    reasons.append(f"winner ⇒ {side}")
    return side, " | ".join(reasons)


# --- New: explicit deficit vs lesion side using the same ladder ---
def infer_sides_from_norm(norm: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], List[str], str]:
    """
    Returns:
      deficit_side_RL: 'R'/'L'/None  (patient deficit side, limbs preferred)
      lesion_side_LR:  'Left'/'Right'/None
      steps: list[str] (breadcrumb rationale)
      summary: str     (one-liner)
    Ladder:
      1) Motor (contra) → deficit R/L → lesion contra
      2) Neglect + Hemianopsia (consensus) → lesion
      3) Gaze preference (ipsi) → lesion
      4) Sensory (contra) → deficit R/L → lesion contra
      5) Aphasia → Left (low-confidence)
      6) Posterior fossa ipsi (CN III/IV/VI, pupil, dysmetria)
    """
    CTX  = norm.get("cortical_signs", {}) or {}
    NS   = norm.get("nonspecific_signs", {}) or {}
    SENS = norm.get("sensory", {}) or {}
    DTL  = norm.get("_detail", {}) or {}

    steps: List[str] = []
    # 1) Motor first (deficit → contra lesion)
    motor_side_RL = limb_motor_side(NS)
    if motor_side_RL in ("R","L"):
        lesion = "Left" if motor_side_RL == "R" else "Right"
        steps.append(f"1) motor→contra({motor_side_RL}) ⇒ {lesion} (locked)")
        return motor_side_RL, lesion, steps, f"motor→contra({motor_side_RL}) ⇒ {lesion}"

    # Face-only proxy (still “motor”)
    face_side_RL = "R" if t(NS.get("R_face_weakness")) else ("L" if t(NS.get("L_face_weakness")) else None)
    if face_side_RL and not (t(NS.get("R_face_weakness")) and t(NS.get("L_face_weakness"))):
        lesion = "Left" if face_side_RL == "R" else "Right"
        steps.append(f"1a) face motor→contra({face_side_RL}) ⇒ {lesion} (locked)")
        return face_side_RL, lesion, steps, f"face motor→contra({face_side_RL}) ⇒ {lesion}"

    # 2) Neglect + Hemianopsia (lesion consensus)
    ngl_raw = (DTL.get("CTX_raw") or {}).get("neglect") or {}
    neglect_lesion = "Right" if _truthy(ngl_raw.get("L")) else ("Left" if _truthy(ngl_raw.get("R")) else None)
    vf_lesion = _vfcut_to_lesion_side(CTX.get("R_visual_field_cut"), CTX.get("L_visual_field_cut"))
    votes = [v for v in [neglect_lesion, vf_lesion] if v]
    if votes:
        if len(votes) == 1 or (len(votes) == 2 and votes[0] == votes[1]):
            lesion = votes[0]
            steps.append(f"2) neglect/hemianopsia ⇒ {lesion}")
            return None, lesion, steps, f"neglect/hemianopsia ⇒ {lesion}"
        steps.append(f"2) neglect/hemianopsia disagree: {votes}")

    # 3) Gaze preference (ipsi)
    if t(CTX.get("R_gaze_preference")):
        steps.append("3) gaze preference ⇒ Right")
        return None, "Right", steps, "gaze preference ⇒ Right"
    if t(CTX.get("L_gaze_preference")):
        steps.append("3) gaze preference ⇒ Left")
        return None, "Left", steps, "gaze preference ⇒ Left"

    # 4) Sensory (deficit → contra lesion)
    sens_side_RL = limb_sensory_side(SENS)
    if sens_side_RL in ("R","L"):
        lesion = "Left" if sens_side_RL == "R" else "Right"
        steps.append(f"4) sensory→contra({sens_side_RL}) ⇒ {lesion}")
        return sens_side_RL, lesion, steps, f"sensory→contra({sens_side_RL}) ⇒ {lesion}"

    # 5) Aphasia (Left, low-confidence)
    if t(CTX.get("aphasia")):
        steps.append("5) aphasia ⇒ Left (low-confidence)")
        return None, "Left", steps, "aphasia⇒Left (low-confidence)"

    # 6) PF ipsilateral
    E  = (DTL.get("EOM") or {})
    PR = (DTL.get("PUP_raw") or {})
    NSR = (DTL.get("NS_raw") or {})           # NEW
    ipsi_votes: List[str] = []

    if t(E.get("R_CNIII_palsy")): ipsi_votes.append("Right")
    if t(E.get("L_CNIII_palsy")): ipsi_votes.append("Left")
    if t(E.get("R_CNIV_palsy")):  ipsi_votes.append("Right")
    if t(E.get("L_CNIV_palsy")):  ipsi_votes.append("Left")
    if t(E.get("R_CNVI_palsy")):  ipsi_votes.append("Right")
    if t(E.get("L_CNVI_palsy")):  ipsi_votes.append("Left")

    if t(PR.get("fixed_R")) or t(PR.get("sluggish_R")): ipsi_votes.append("Right")
    if t(PR.get("fixed_L")) or t(PR.get("sluggish_L")): ipsi_votes.append("Left")

    # Ipsilateral cerebellar signs (dysmetria/ataxia) — posterior fossa
    if t(NSR.get("R_dysmetria_ataxia")): ipsi_votes.append("Right")   # NEW
    if t(NSR.get("L_dysmetria_ataxia")): ipsi_votes.append("Left")    # NEW

    if ipsi_votes:
        les = Counter(ipsi_votes).most_common(1)[0][0]
        steps.append(f"6) PF ipsi votes={ipsi_votes} ⇒ {les}")
        return None, les, steps, f"PF ipsi ⇒ {les}"


    steps.append("insufficient evidence")
    return None, None, steps, "insufficient evidence"

# ============ Core localization logic (scored) ============
def localize_with_side(survey_or_case: Dict[str, Any]) -> Dict[str, Any]:
    norm = normalize_survey(survey_or_case or {})
    if not isinstance(norm, dict):
        norm = _empty_normalized()

    PF   = norm.get("posterior_fossa_signs", {}) or {}
    CTX  = norm.get("cortical_signs", {}) or {}
    PUP  = norm.get("pupils", {}) or {}
    NS   = norm.get("nonspecific_signs", {}) or {}
    SENS = norm.get("sensory", {}) or {}
    DTL  = norm.get("_detail", {}) or {}

    scorer = Scorer()

    # ===== 1) Explicit mappings (posterior fossa details) =====
    E = (DTL.get("EOM") or {})

    # CN III → Midbrain
    if t(E.get("R_CNIII_palsy")):
        scorer.add("Midbrain", W["EOM_CNIII"], "R CN III palsy → R midbrain", side="Right")
    if t(E.get("L_CNIII_palsy")):
        scorer.add("Midbrain", W["EOM_CNIII"], "L CN III palsy → L midbrain", side="Left")

    # CN IV → Midbrain
    if t(E.get("R_CNIV_palsy")):
        scorer.add("Midbrain", W["EOM_CNIV"], "R CN IV palsy → R midbrain", side="Right")
    if t(E.get("L_CNIV_palsy")):
        scorer.add("Midbrain", W["EOM_CNIV"], "L CN IV palsy → L midbrain", side="Left")

    # CN VI → Pons
    if t(E.get("R_CNVI_palsy")):
        scorer.add("Pons", W["EOM_CNVI"], "R CN VI palsy → R pons", side="Right")
    if t(E.get("L_CNVI_palsy")):
        scorer.add("Pons", W["EOM_CNVI"], "L CN VI palsy → L pons", side="Left")

    # Unspecified EOM limitation → PF but weak
    if t(E.get("R_unspecified_limitation")):
        scorer.add("Midbrain", W["EOM_unspec"], "R EOM limitation (unspecified) → R midbrain", side="Right")
        scorer.add("Pons",     W["EOM_unspec"], "R EOM limitation (unspecified) → R pons",     side="Right")
    if t(E.get("L_unspecified_limitation")):
        scorer.add("Midbrain", W["EOM_unspec"], "L EOM limitation (unspecified) → L midbrain", side="Left")
        scorer.add("Pons",     W["EOM_unspec"], "L EOM limitation (unspecified) → L pons",     side="Left")

    if t(PF.get("skew")):
        for r in ["Midbrain","Pons","Medulla","Cerebellum"]:
            scorer.add(r, W["skew"], "Skew → posterior fossa")
    if t(PF.get("vertigo")):
        for r in ["Midbrain","Pons","Medulla","Cerebellum"]:
            scorer.add(r, W["vertigo"], "Vertigo → posterior fossa")
    if t(PF.get("nystagmus")):
        for r in ["Pons","Medulla","Cerebellum"]:
            scorer.add(r, W["nystagmus"], "Nystagmus (pathologic) → pons/medulla/cerebellum")

    PR = (DTL.get("PUP_raw") or {})
    if t(PR.get("fixed_R")):
        scorer.add("Midbrain", W["pupil_fixed"], "Fixed R pupil → R midbrain", side="Right")
    if t(PR.get("fixed_L")):
        scorer.add("Midbrain", W["pupil_fixed"], "Fixed L pupil → L midbrain", side="Left")
    if t(PR.get("sluggish_R")):
        for r in ["Midbrain","Pons","Medulla"]:
            scorer.add(r, W["pupil_sluggish"], "Sluggish R pupil → brainstem (R)", side="Right")
    if t(PR.get("sluggish_L")):
        for r in ["Midbrain","Pons","Medulla"]:
            scorer.add(r, W["pupil_sluggish"], "Sluggish L pupil → brainstem (L)", side="Left")
    if t(PR.get("anisocoria")):
        scorer.add("Midbrain", W["anisocoria"], "Anisocoria → possible midbrain")

    # ===== 2) Cortical signs (incl. subtyping) =====
    CTX_RAW = DTL.get("CTX_raw") or {}
    aph_block = CTX_RAW.get("aphasia") if isinstance(CTX_RAW.get("aphasia"), dict) else None
    aph_type  = (aph_block or {}).get("type") if aph_block else None

    if t(CTX.get("aphasia")):
        scorer.add("Frontal_cortex",  W["aphasia_base"], "Aphasia → L frontal/temporal", side="Left")
        scorer.add("Temporal_cortex", W["aphasia_base"], "Aphasia → L frontal/temporal", side="Left")
        if isinstance(aph_type, str):
            low = aph_type.strip().lower()
            if low == "fluent":
                scorer.add("Temporal_cortex", W["aphasia_fluent"], "Fluent aphasia → L temporal", side="Left")
            elif low == "nonfluent":
                scorer.add("Frontal_cortex",  W["aphasia_nonfluent"], "Nonfluent aphasia → L frontal", side="Left")
            elif low == "global":
                scorer.add("Frontal_cortex",  W["aphasia_global"], "Global aphasia → L frontal + L temporal", side="Left")
                scorer.add("Temporal_cortex", W["aphasia_global"], "Global aphasia → L frontal + L temporal", side="Left")
            elif low == "anomic":
                scorer.add("Subcortical", W["aphasia_anomic_subcort"], "Anomic aphasia → subcortical")
            elif low == "conduction":
                scorer.add("Temporal_cortex", W["aphasia_conduction"], "Conduction aphasia → L temporal+parietal", side="Left")
                scorer.add("Parietal_cortex", W["aphasia_conduction"], "Conduction aphasia → L temporal+parietal", side="Left")

    neglect_present = t(CTX.get("neglect")) or (isinstance(CTX_RAW.get("neglect"), str) and CTX_RAW.get("neglect").strip() != "")
    if neglect_present:
        scorer.add("Parietal_cortex", W["neglect_R_parietal"], "Neglect → R parietal", side="Right")

    if t(CTX.get("R_visual_field_cut")):
        for r in ["Occipital_cortex","Parietal_cortex","Temporal_cortex"]:
            scorer.add(r, W["visual_cut"], "R HH → L retro-chiasmal", side="Left")
    if t(CTX.get("L_visual_field_cut")):
        for r in ["Occipital_cortex","Parietal_cortex","Temporal_cortex"]:
            scorer.add(r, W["visual_cut"], "L HH → R retro-chiasmal", side="Right")

    if t(CTX.get("R_gaze_preference")):
        scorer.add("Frontal_cortex", W["gaze_pref"], "R gaze preference → frontal eye fields", side="Right")
    if t(CTX.get("L_gaze_preference")):
        scorer.add("Frontal_cortex", W["gaze_pref"], "L gaze preference → frontal eye fields", side="Left")

    if t(CTX.get("apraxia")):
        scorer.add("Parietal_cortex", W["apraxia"], "Apraxia → parietal ± premotor frontal")
        scorer.add("Frontal_cortex",  W["apraxia"], "Apraxia → parietal ± premotor frontal")

    # ===== 3) Non-specific =====
    if t(NS.get("dysarthria")):
        for r in ["Frontal_cortex","Subcortical","Pons","Cerebellum"]:
            scorer.add(r, W["dysarthria"], "Dysarthria (nonspecific)")

    if t(NS.get("dysmetria_ataxia")):
        for r in ["Cerebellum","Pons","Subcortical"]:
            scorer.add(r, W["ataxia_generic"], "Ataxia → cerebellar/pontine/internal capsule")

    NS_RAW = DTL.get("NS_raw") or {}
    if t(NS_RAW.get("R_dysmetria_ataxia")):
        scorer.add("Cerebellum", W["ataxia_sided_bundle"], "R dysmetria/ataxia → R cerebellum", side="Right")
        scorer.add("Pons",       W["ataxia_sided_bundle"], "R dysmetria/ataxia → R pons",       side="Right")
        scorer.add("Midbrain",   W["ataxia_sided_bundle"], "R dysmetria/ataxia → L midbrain (SCP)",   side="Left")
        scorer.add("Subcortical",W["ataxia_sided_bundle"], "R dysmetria/ataxia → L subcortical",side="Left")
    if t(NS_RAW.get("L_dysmetria_ataxia")):
        scorer.add("Cerebellum", W["ataxia_sided_bundle"], "L dysmetria/ataxia → L cerebellum", side="Left")
        scorer.add("Pons",       W["ataxia_sided_bundle"], "L dysmetria/ataxia → L pons",       side="Left")
        scorer.add("Midbrain",   W["ataxia_sided_bundle"], "L dysmetria/ataxia → R midbrain (SCP)",   side="Right")
        scorer.add("Subcortical",W["ataxia_sided_bundle"], "L dysmetria/ataxia → R subcortical",side="Right")

    # ===== 4) Motor / sensory patterns =====
    motor_side = limb_motor_side(NS)
    face_R, face_L = t(NS.get("R_face_weakness")), t(NS.get("L_face_weakness"))
    face_side = "R" if face_R else ("L" if face_L else None)

    cortical_any = any(t(CTX.get(k)) for k in ["aphasia","neglect","apraxia","R_visual_field_cut","L_visual_field_cut","R_gaze_preference","L_gaze_preference"])
    motor_any    = any(t(NS.get(k))  for k in ["R_arm_weakness","L_arm_weakness","R_leg_weakness","L_leg_weakness"])
    if (motor_any or motor_side in ("R","L")) and not cortical_any:
        scorer.add("Subcortical", W["motor_no_cortical_subcort"], "Motor w/o cortical signs → subcortical")

    wsum = (NS.get("weakness_involvement") or {})
    if t(wsum.get("face_arm_gt_leg")):
        for r in ["Frontal_cortex","Parietal_cortex","Temporal_cortex","Subcortical"]:
            scorer.add(r, W["weak_face_arm_gt_leg"], "Weakness face/arm ≫ leg → MCA cortical ± deep")
    if t(wsum.get("leg_gt_face_arm")):
        scorer.add("Frontal_cortex", W["weak_leg_gt_face_arm"], "Weakness leg ≫ face/arm → ACA medial frontal")

    # Sensory
    SENSn = norm.get("sensory", {}) or {}
    s_face_R, s_face_L = t(SENSn.get("R_face_numb")), t(SENSn.get("L_face_numb"))
    s_face_side = "R" if s_face_R else ("L" if s_face_L else None)
    s_limb_side = limb_sensory_side(SENSn)

    # Unilateral limb sensory → contra parietal/subcortical (prevents empty scores on sensory-only cases)
    if s_limb_side in ("R","L"):
        les = contra(s_limb_side)
        if les:
            scorer.add("Parietal_cortex", 0.8, "Unilateral limb sensory → contra parietal", side=les)
            scorer.add("Subcortical",     0.6, "Unilateral limb sensory → possible thalamic/deep", side=les)


    if s_face_side and s_limb_side and s_face_side != s_limb_side:
        for r in ["Medulla","Pons"]:
            scorer.add(r, W["crossed_sens_face_body"], "Crossed face/body sensory → lateral medulla/pons")

    if t(SENSn.get("cortical_sensory_loss")):
        scorer.add("Parietal_cortex", 1.2, "Cortical sensory loss → parietal cortex")

    sinv = (SENSn.get("sensory_involvement") or {})
    if t(sinv.get("face_arm_gt_leg")):
        for r in ["Parietal_cortex","Subcortical"]:
            scorer.add(r, W["sens_face_arm_gt_leg"], "Sensory face/arm ≫ leg → MCA parietal ± deep")
    if t(sinv.get("leg_gt_face_arm")):
        scorer.add("Parietal_cortex", W["sens_leg_gt_face_arm"], "Sensory leg ≫ face/arm → ACA medial parietal")

    hemisensory = all([
        any(t(SENSn.get(k)) for k in ["R_face_numb","L_face_numb"]),
        any(t(SENSn.get(k)) for k in ["R_arm_numb","L_arm_numb"]),
        any(t(SENSn.get(k)) for k in ["R_leg_numb","L_leg_numb"]),
    ])
    if hemisensory and not cortical_any and not t(SENSn.get("cortical_sensory_loss")):
        scorer.add("Subcortical", W["hemisensory_pure_subcort"], "Pure hemisensory w/o cortical signs → thalamic/deep")

    # ===== 5) Brainstem side logic (pons patterns) =====
    face_contra_label = contra(face_side) if face_side else None
    face_ipsi_label   = ipsi(face_side)   if face_side else None

    if face_side:
        if face_contra_label:
            scorer.add("Frontal_cortex", 0.8, "Face weakness → contra supratentorial", side=face_contra_label)
            scorer.add("Subcortical",    0.8, "Face weakness → contra deep",          side=face_contra_label)
            scorer.add("Midbrain",       0.4, "Face weakness → possible contra midbrain", side=face_contra_label)

    crossed_face_limb = bool(face_side and motor_side and face_side != motor_side)
    same_side_face_limb = bool(face_side and motor_side and face_side == motor_side)
    no_cortical_bias = not cortical_any

    if crossed_face_limb and face_ipsi_label and no_cortical_bias:
        scorer.add("Pons", 0.9, "Crossed face (ipsi) + limb (contra) → ipsi pons", side=face_ipsi_label)
        scorer.add("Pons", W["pons_crossed_face_limb"], "Crossed motor signs (face vs limbs) → lateral pontine", side=face_ipsi_label)
    elif same_side_face_limb and no_cortical_bias:
        les_side = contra(face_side)
        if les_side:
            scorer.add("Pons", W["pons_same_side_all"], "Same-side face+limb weakness → pontine (supranuclear facial)", side=les_side)

    # ===== PF bundle & aphasia gating =====
    _apply_pf_bundle_and_aphasia_gating(scorer, norm, aphasia_present=t(CTX.get("aphasia")))

    # New, explicit side inference (deficit + lesion)
    deficit_side_RL, lesion_side_LR, side_steps, side_summary = infer_sides_from_norm(norm)

    # --- Safety net: if no scores at all, seed minimal side-based priors so predictions aren't empty
    if not scorer.region_scores and not scorer.sided_scores and lesion_side_LR in ("Left","Right"):
        for r in ["Subcortical","Parietal_cortex","Frontal_cortex"]:
            scorer.add(r, 0.01, "Seed from side inference", side=lesion_side_LR)

    # ===== Build Top-5 from scores =====

    top_labels, union_regions, all_label_scores, reasons = scorer.top5()

    # ---- Force a side unless the survey is empty / all-false ----
    has_any_positive_signal = _any_true_in_obj(norm)  # any True anywhere in normalized structure?
    if lesion_side_LR is None and has_any_positive_signal:
        # Prefer best *sided* label from scores; else inferred hemisphere from deficits; else deterministic default
        best_from_scores = _side_from_best_sided_label(all_label_scores)  # "Left"/"Right" or None
        inferred_hemi = contra(limb_motor_side(NS) or limb_sensory_side(SENS))  # "Left"/"Right"/None

        forced = best_from_scores or inferred_hemi or "Left"
        lesion_side_LR = forced
        side_steps.append(f"F) forced side due to non-empty survey ⇒ {forced}")
        side_summary = (side_summary + " | forced=" + forced) if side_summary else f"forced={forced}"

    return {
        "top_regions_union": union_regions,
        "top_regions_with_side": top_labels,
        "label_scores": {k: round(v, 3) for k,v in all_label_scores.items()},
        "rationales": {k: reasons[k] for k in top_labels},

        "inferred_motor_side": limb_motor_side(norm.get("nonspecific_signs", {})),
        "inferred_sensory_side": limb_sensory_side(norm.get("sensory", {})),
        "inferred_hemisphere_label": contra(limb_motor_side(norm.get("nonspecific_signs", {})) or limb_sensory_side(norm.get("sensory", {}))),

        "deficit_side": deficit_side_RL,
        "lesion_side": lesion_side_LR,
        "side_rationale_steps": side_steps,
        "side_rationale_summary": side_summary,

        "uniform_side": lesion_side_LR,
        "uniform_side_rationale": side_summary,
    }


# ---------- Build compact predictions from scores (prefer sided over unsided) ----------
def to_prediction_correlation_from_scores(label_scores: Dict[str, float], k:int=5, preferred_side: Optional[str]=None) -> List[Dict[str, Any]]:
    """
    Build a de-duplicated, sided-preferred prediction list from label_scores.
    If preferred_side is provided ('Left'/'Right'):
      - Keep sided labels only if they match preferred_side.
      - Convert unsided labels to preferred_side in the output.
    """
    buckets: Dict[str, Dict[str, Tuple[float, str]]] = {}
    for lab, sc in label_scores.items():
        if lab.startswith("Left ") or lab.startswith("Right "):
            side_long, region = lab.split(" ", 1)
            side = "L" if side_long == "Left" else "R"
            b = buckets.setdefault(region, {})
            if side not in b or sc > b[side][0]:
                b[side] = (sc, lab)
        else:
            region = lab
            b = buckets.setdefault(region, {})
            if "U" not in b or sc > b["U"][0]:
                b["U"] = (sc, lab)

    chosen: List[Tuple[str, float]] = []

    for region, b in buckets.items():
        if preferred_side in ("Left","Right"):
            want = "L" if preferred_side == "Left" else "R"
            if want in b:
                chosen.append((b[want][1], b[want][0]))
            elif "U" in b:
                chosen.append((b["U"][1], b["U"][0]))
        else:
            has_sided = ("L" in b) or ("R" in b)
            if has_sided:
                if "L" in b: chosen.append((b["L"][1], b["L"][0]))
                if "R" in b: chosen.append((b["R"][1], b["R"][0]))
            elif "U" in b:
                chosen.append((b["U"][1], b["U"][0]))

    def keyfun(item: Tuple[str, float]):
        lab, sc = item
        if lab.startswith("Left ") or lab.startswith("Right "):
            region = lab.split(" ", 1)[1]
            side_rank = 0 if lab.startswith("Left ") else 1
        else:
            region = lab
            side_rank = 2
        return (-sc, region, side_rank, lab)

    chosen_sorted = sorted(chosen, key=keyfun)[:k]

    out: List[Dict[str, Any]] = []
    for lab, _sc in chosen_sorted:
        if lab.startswith("Left ") or lab.startswith("Right "):
            side_long, loc = lab.split(" ", 1)
            side = "L" if side_long == "Left" else "R"
        else:
            loc = lab
            if preferred_side in ("Left","Right"):
                side = "L" if preferred_side == "Left" else "R"
            else:
                side = None
        out.append({"side": side, "location": loc, "justification": None})
    return out

# ---------- Generic deduper for arrays of {side, location} ----------
def dedup_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    sided_present = {}
    for e in entries:
        loc = e.get("location")
        side = e.get("side")
        if side is not None:
            sided_present[loc] = True

    out = []
    for e in entries:
        loc = e.get("location")
        side = e.get("side")
        key = (side, loc)
        if side is None and sided_present.get(loc):
            continue
        if key not in seen:
            out.append(e)
            seen.add(key)
    return out

# ---------- Compartment helpers ----------
def region_to_compartment(region: str) -> Optional[str]:
    if region in PF_REGIONS:
        return "posterior_fossa"
    if region in HEMI_REGIONS:
        return "hemisphere"
    return None

def locations_to_true_compartment(locs: List[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(locs, list):
        return None
    has_pf = any((e.get("location") in PF_REGIONS) for e in locs if isinstance(e, dict))
    has_hemi = any((e.get("location") in HEMI_REGIONS) for e in locs if isinstance(e, dict))
    if has_pf:
        return "posterior_fossa"
    if has_hemi:
        return "hemisphere"
    return None

def scores_to_pred_compartment(label_scores: Dict[str, float]) -> Optional[str]:
    if not label_scores:
        return None
    top_lab, _ = max(label_scores.items(), key=lambda kv: kv[1])
    top_region = _region_from_label(top_lab)
    return region_to_compartment(top_region)

# ---------- Metrics ----------
def confusion_and_metrics(y_true: List[str], y_pred: List[str]) -> None:
    labels = ["hemisphere", "posterior_fossa"]
    idx = {lab:i for i,lab in enumerate(labels)}
    cm = [[0,0],[0,0]]
    for t,p in zip(y_true, y_pred):
        if t not in idx or p not in idx:
            continue
        cm[idx[t]][idx[p]] += 1

    total = sum(sum(r) for r in cm)
    correct = cm[0][0] + cm[1][1]
    acc = correct / total if total else 0.0

    def class_metrics(pos_idx: int):
        TP = cm[pos_idx][pos_idx]
        FN = sum(cm[pos_idx]) - TP
        FP = sum(cm[r][pos_idx] for r in range(2)) - TP
        TN = total - TP - FN - FP
        sens = TP / (TP + FN) if (TP+FN) else 0.0
        spec = TN / (TN + FP) if (TN+FP) else 0.0
        prec = TP / (TP + FP) if (TP+FP) else 0.0
        f1 = 2*prec*sens / (prec+sens) if (prec+sens) else 0.0
        return TP, FN, FP, TN, sens, spec, f1

    print("\n=== Confusion Matrix (Compartment) (rows=TRUE, cols=PRED) ===")
    print(f"{'':16s}{labels[0]:>14s}{labels[1]:>18s}")
    for i, lab in enumerate(labels):
        print(f"{lab:16s}{cm[i][0]:14d}{cm[i][1]:18d}")

    print(f"\nOverall Accuracy: {acc:.4f}  (N={total})")

    for i, lab in enumerate(labels):
        TP,FN,FP,TN,sens,spec,f1 = class_metrics(i)
        support = sum(cm[i])
        print(f" - {lab}: support={support}, sensitivity={sens:.3f}, specificity={spec:.3f}, f1={f1:.3f}")

# ============ Robust loader/writer that PRESERVES input shape ============
InputKind = Literal["jsonl", "json_list", "json_cases"]

def load_cases_anyshape(path: str) -> tuple[List[Dict[str, Any]], InputKind, Optional[Dict[str, Any]]]:
    txt = pathlib.Path(path).read_text(encoding="utf-8").strip()
    if not txt:
        return [], "json_list", None

    try:
        data = json.loads(txt)
        if isinstance(data, dict):
            if "cases" in data and isinstance(data["cases"], list):
                return data["cases"], "json_cases", data
            raise ValueError("JSON object input must contain a 'cases' array.")
        if isinstance(data, list):
            return data, "json_list", None
        raise ValueError("Unsupported JSON root type. Expected array or object with 'cases'.")
    except json.JSONDecodeError:
        cases: List[Dict[str, Any]] = []
        for i, line in enumerate(txt.splitlines(), 1):
            ln = line.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSONL at line {i}: {e}")
            if not isinstance(obj, dict):
                raise ValueError(f"Line {i} is not a JSON object.")
            cases.append(obj)
        return cases, "jsonl", None

def write_same_shape(kind: InputKind, root: Optional[Dict[str, Any]], out_path: str, cases: List[Dict[str, Any]]) -> None:
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if kind == "jsonl":
        with open(out_path, "w", encoding="utf-8") as f:
            for c in cases:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        return
    if kind == "json_list":
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(cases, f, ensure_ascii=False, indent=2)
        return
    if kind == "json_cases":
        wrapper = dict(root or {})
        wrapper["cases"] = cases
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(wrapper, f, ensure_ascii=False, indent=2)
        return
    raise ValueError(f"Unknown input kind: {kind}")

# ============ Main ============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT_FILE, help="Input (JSONL / JSON array / JSON object with 'cases').")
    ap.add_argument("--output", default=DEFAULT_OUTPUT_FILE, help="Output path; preserves input container shape.")
    ap.add_argument("--inplace", action="store_true", help="Overwrite the input file in place.")
    ap.add_argument("--debug", action="store_true", help="Print top-5 labels and rationales for each case to stderr.")
    args = ap.parse_args()

    in_path = args.input
    out_path = in_path if args.inplace else args.output

    cases, kind, root = load_cases_anyshape(in_path)

    augmented: List[Dict[str, Any]] = []
    y_true: List[str] = []
    y_pred: List[str] = []

    for idx, case in enumerate(cases, 1):
        c = dict(case)  # shallow copy
        case_id = str(c.get("case_id") or c.get("hadm_id") or f"case_{idx}")

        # IMPORTANT: pass the whole case; normalizer will detect schema
        survey_input = c

        # RUN LOCALIZER
        loc = localize_with_side(survey_input)

        # Build compact predictions from scores
        label_scores = loc.get("label_scores", {})
        preferred = _coerce_uniform_side(loc.get("uniform_side"), label_scores, loc)

        corr = to_prediction_correlation_from_scores(
            label_scores,
            k=4,
            preferred_side=preferred
        )

        corr = dedup_entries(corr)

        c["localization_outcome"] = loc
        c["prediction"] = {"correlation": corr}

        # Compact any stroke_location in input
        if isinstance(c.get("stroke_location"), list):
            c["stroke_location"] = dedup_entries(c["stroke_location"])

        augmented.append(c)

        # ===== Confusion matrix collection (compartment) =====
        true_comp = locations_to_true_compartment(c.get("stroke_location") or [])
        pred_comp = scores_to_pred_compartment(loc.get("label_scores", {}))

        if true_comp is not None and pred_comp is not None:
            y_true.append(true_comp)
            y_pred.append(pred_comp)

        if args.debug:
            print(f"\n=== {case_id} ===", file=sys.stderr)
            print("Predictions:", corr, file=sys.stderr)
            for lab in loc.get("top_regions_with_side", []):
                rs = loc["rationales"].get(lab, [])
                if rs:
                    print(f"  {lab}:", file=sys.stderr)
                    for r in rs:
                        print(f"   - {r}", file=sys.stderr)

    write_same_shape(kind, root, out_path, augmented)
    print(f"Wrote {len(augmented)} augmented cases → {out_path}")

    # Print confusion matrix & metrics
    if y_true and y_pred:
        confusion_and_metrics(y_true, y_pred)
    else:
        print("Insufficient data to compute compartment confusion matrix (missing true or predicted compartments).")

if __name__ == "__main__":
    main()
