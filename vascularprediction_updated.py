#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# ===============================
# Hard-coded input / output paths
# ===============================
INPUT_FILE  = "total.jsonl"        # accepts .json or .jsonl
OUTPUT_FILE = "1_results_cleaned_vascular_final.jsonl"     # writes JSONL

# CSV outputs (global)
CONFUSION_CSV = "out/confusion_matrix_territory.csv"                    # territory-only (all)
CONFUSION_SIDE_CSV = "out/confusion_matrix_side_territory.csv"          # side-aware (all)

# CSV outputs (stratified)
CONFUSION_CSV_HEMI = "out/confusion_matrix_territory_hemisphere.csv"    # territory-only (hemisphere)
CONFUSION_CSV_PF   = "out/confusion_matrix_territory_posterior_fossa.csv"  # territory-only (posterior fossa)
CONFUSION_SIDE_CSV_HEMI = "out/confusion_matrix_side_territory_hemisphere.csv"  # side-aware (hemisphere)
CONFUSION_SIDE_CSV_PF   = "out/confusion_matrix_side_territory_posterior_fossa.csv"  # side-aware (posterior fossa)

# JSONL outputs (misclassification summaries)
MIS_TERRITORY_JSONL = "out/misclassified_territory.jsonl"
MIS_SIDE_JSONL      = "out/misclassified_side_territory.jsonl"
MIS_CATEGORY_JSONL  = "out/misclassified_category.jsonl"

# JSONL outputs (FULL original case objects + error annotations)
MIS_TERRITORY_FULL_JSONL = "out/misclassified_territory_full.jsonl"
MIS_SIDE_FULL_JSONL      = "out/misclassified_side_territory_full.jsonl"
MIS_CATEGORY_FULL_JSONL  = "out/misclassified_category_full.jsonl"

# -------- Labels & index (territory-only) --------
ALLOWED_TERRITORIES = ("MCA", "ACA", "PCA", "BA", "Vertebral")
IDX = {lab: i for i, lab in enumerate(ALLOWED_TERRITORIES)}

# -------- Side-aware labels --------
SIDES = ("Left","Right")
SIDE_TERRITORY_LABELS = tuple(f"{s} {t}" for t in ALLOWED_TERRITORIES for s in SIDES)
SIDE_IDX = {lab: i for i, lab in enumerate(SIDE_TERRITORY_LABELS)}

# -------- Stratum label-sets --------
HEMI_TERRS = ("MCA","ACA","PCA")
PF_TERRS   = ("BA","Vertebral")

# -------- Category labels --------
CATEGORY_LABELS = ("hemisphere", "posterior_fossa")
CAT_IDX = {lab: i for i, lab in enumerate(CATEGORY_LABELS)}

# -------- Region → Territory mapping (list-based) --------
REGION_TO_TERRITORY = {
    "Frontal_cortex":  ["MCA", "ACA"],
    "Parietal_cortex": ["MCA"],
    "Temporal_cortex": ["MCA"],             # extend later if you add posterior temporal → PCA
    "Occipital_cortex":["PCA"],
    "Subcortical":     ["MCA"],             # basal ganglia (MCA) vs thalamus (PCA) — refine upstream if needed
    "Midbrain":        ["BA"],
    "Pons":            ["BA"],
    "Medulla":         ["Vertebral"],
    "Cerebellum":      ["Vertebral", "BA"], # PICA vs SCA
}

# ---------- I/O helpers ----------
def load_json_or_jsonl(path: Path):
    raw = path.read_text(encoding="utf-8", errors="replace").lstrip("\ufeff").strip()
    if not raw:
        return []
    # JSON (whole-file) or JSONL
    if raw[0] in "[{":
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and "cases" in data:
                return data["cases"]
            if isinstance(data, list):
                return data
            return [data]
        except json.JSONDecodeError:
            pass
    out = []
    for i, line in enumerate(raw.splitlines(), 1):
        ln = line.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError as e:
            print(f"Skipping line {i}: {e}")
    return out

def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _save_jsonl(path: Path, rows: List[Dict[str, Any]]):
    # kept for backward-compat with existing calls below
    write_jsonl(path, rows)

# ---------- small utils ----------
def _truthy(x: Any, *, key_if_dict: Optional[str] = "present") -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, dict):
        if key_if_dict is not None and key_if_dict in x:
            return bool(x[key_if_dict])
        for v in x.values():
            if isinstance(v, bool) and v:
                return True
        return False
    return False

def _any_true(d: Dict[str, Any], keys: List[str]) -> bool:
    if not isinstance(d, dict):
        return False
    return any(bool(d.get(k)) for k in keys)

def _norm_side(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = str(s).strip().lower()
    if s in ("l", "left"):
        return "Left"
    if s in ("r", "right"):
        return "Right"
    return None

def _get_category(case: Dict[str, Any]) -> Optional[str]:
    lab = case.get("localization_label")
    cat: Optional[str] = None
    if isinstance(lab, dict):
        cat = lab.get("category") or lab.get("type") or lab.get("group")
    elif isinstance(lab, str):
        cat = lab
    if not cat:
        return None
    cat = str(cat).strip().lower()
    return cat if cat in ("hemisphere","posterior_fossa") else None

# ---------- NEW: helpers for the new survey schema ----------
def _details(case: Dict[str, Any]) -> Dict[str, Any]:
    """Return the canonical details dict for the new survey schema."""
    sd = case.get("survey_details") or {}
    return sd.get("details") or {}

# ---------- Feature readers / detectors (updated to new schema) ----------
def _get_weakness_hint(case: Dict[str, Any]) -> Optional[str]:
    d = _details(case)
    w = d.get("weakness") or {}
    pat = w.get("involvement_pattern") or {}
    if pat.get("face_arm_gt_leg"):
        return "MCA"
    if pat.get("leg_gt_face_arm"):
        return "ACA"
    return None

def _has_cortical_language(case: Dict[str, Any]) -> bool:
    d = _details(case)
    cort = d.get("cortical_signs") or {}
    aph = cort.get("aphasia")
    return _truthy(aph, key_if_dict="present")

def _has_visual_field_cut(case: Dict[str, Any]) -> bool:
    d = _details(case)
    vf = (d.get("cortical_signs") or {}).get("visual_field_cut") or {}
    return bool(vf.get("R")) or bool(vf.get("L"))

def _infer_side_from_weakness(case: Dict[str, Any]) -> Optional[str]:
    """
    Return 'L' or 'R' lesion side inferred from weakness (contralateral rule).
    Prefers explicit side_only; falls back to lr_comparison majorities.
    """
    d = _details(case)
    w = d.get("weakness") or {}
    side_only = w.get("side_only") or {}
    if side_only.get("R") and not side_only.get("L"):
        return "L"
    if side_only.get("L") and not side_only.get("R"):
        return "R"

    comp = (w.get("lr_comparison") or {})
    scores = {"R": 0, "L": 0}
    for part in ("face", "arm", "leg"):
        p = comp.get(part) or {}
        if p.get("R_gt_L"): scores["R"] += 1
        if p.get("L_gt_R"): scores["L"] += 1
    if scores["R"] > scores["L"]:
        return "L"
    if scores["L"] > scores["R"]:
        return "R"
    return None

def _iter_correlations(case: Dict[str, Any]):
    """
    Yield correlation dicts, preferring prediction.correlation, else clinical_correlation.correlation.
    """
    pred = (case.get("prediction") or {}).get("correlation")
    if isinstance(pred, list) and pred:
        for it in pred:
            yield it or {}
        return
    clin = (case.get("clinical_correlation") or {}).get("correlation")
    if isinstance(clin, list) and clin:
        for it in clin:
            yield it or {}

def _territories_from_prediction(case: Dict[str, Any]) -> List[str]:
    out, seen = [], set()
    for item in _iter_correlations(case):
        loc = (item or {}).get("location")
        terrs = REGION_TO_TERRITORY.get(loc, [])
        if isinstance(terrs, str):
            terrs = [terrs]
        for terr in terrs:
            if terr in ALLOWED_TERRITORIES and terr not in seen:
                seen.add(terr); out.append(terr)
    return out

def _saw_region(case: Dict[str, Any], region_name: str) -> bool:
    for it in _iter_correlations(case):
        if (it or {}).get("location") == region_name:
            return True
    return False

def _has_posterior_fossa_signs(case: Dict[str, Any]) -> bool:
    d = _details(case)

    eom = d.get("eom") or {}
    cn_palsy = any([
        any((eom.get("cn3_palsy") or {}).values()),
        any((eom.get("cn4_palsy") or {}).values()),
        any((eom.get("cn6_palsy") or {}).values()),
    ])
    eom_lim = any((eom.get("unspecified_limit") or {}).values())

    skew = any(((d.get("skew") or {}).get("skew_side") or {}).values()) or bool((d.get("skew") or {}).get("skew_legacy"))
    vertigo = bool((d.get("vertigo") or {}).get("present"))

    nyst = d.get("nystagmus") or {}
    nyst_path = bool(nyst.get("pathologic"))

    saw_pf_region = any([
        _saw_region(case, "Pons"),
        _saw_region(case, "Midbrain"),
        _saw_region(case, "Medulla"),
        _saw_region(case, "Cerebellum"),
    ])

    return nyst_path or skew or cn_palsy or eom_lim or (vertigo and (nyst_path or skew)) or saw_pf_region

def _has_posterior_fossa_strong(case: Dict[str, Any]) -> bool:
    d = _details(case)

    eom = d.get("eom") or {}
    cn_palsy = any([
        any((eom.get("cn3_palsy") or {}).values()),
        any((eom.get("cn4_palsy") or {}).values()),
        any((eom.get("cn6_palsy") or {}).values()),
    ])
    eom_lim = any((eom.get("unspecified_limit") or {}).values())

    skew = any(((d.get("skew") or {}).get("skew_side") or {}).values()) or bool((d.get("skew") or {}).get("skew_legacy"))
    vertigo = bool((d.get("vertigo") or {}).get("present"))

    nyst = d.get("nystagmus") or {}
    nyst_path = bool(nyst.get("pathologic"))

    saw_midbrain = _saw_region(case, "Midbrain")
    saw_pons    = _saw_region(case, "Pons")
    saw_medulla = _saw_region(case, "Medulla")
    saw_cereb   = _saw_region(case, "Cerebellum")

    indicators = [nyst_path, skew, cn_palsy, eom_lim, saw_midbrain, saw_pons, saw_medulla]
    strong_count = sum(1 for x in indicators if x)

    if cn_palsy:
        return True
    if strong_count >= 2:
        return True
    if (saw_midbrain or saw_pons or saw_medulla) and (nyst_path or skew or eom_lim or vertigo):
        return True
    if saw_cereb and (nyst_path or skew or eom_lim):
        return True
    return False

# ---------- NEW: infer predicted side from predictions (no GT leakage) ----------
def _infer_side_from_prediction(case: Dict[str, Any]) -> Optional[str]:
    """
    Decide side from model/clinical correlations ONLY:
      1) Majority vote over correlation[*].side (prediction preferred, else clinical)
      2) If tie/absent, fallback to localization_outcome.uniform_side / inferred_hemisphere_label
      3) If still absent, fallback to weakness-based contralateral inference from survey
    Returns normalized 'Left'/'Right' or None.
    """
    counts = {"Left": 0, "Right": 0}
    for it in _iter_correlations(case):
        s = _norm_side((it or {}).get("side"))
        if s:
            counts[s] += 1

    if counts["Left"] > counts["Right"]:
        return "Left"
    if counts["Right"] > counts["Left"]:
        return "Right"

    lo = (case.get("localization_outcome") or {})
    s2 = _norm_side(lo.get("uniform_side") or lo.get("inferred_hemisphere_label"))
    if s2:
        return s2

    s3 = _infer_side_from_weakness(case)
    return _norm_side(s3) if s3 else None

# ---------- Clinical pruning & prioritization ----------
def _apply_clinical_pruning(territories: List[str],
                            weakness_hint: Optional[str],
                            case: Dict[str, Any]) -> List[str]:
    terrs = list(territories)

    if weakness_hint == "MCA":
        terrs = [t for t in terrs if t != "ACA"] or terrs
    elif weakness_hint == "ACA":
        terrs = [t for t in terrs if t != "MCA"] or terrs

    if _has_cortical_language(case) and "MCA" in terrs:
        terrs = ["MCA"] + [t for t in terrs if t != "MCA"]
        terrs = [t for t in terrs if not (t == "PCA" and "MCA" in terrs)] or terrs

    if _has_visual_field_cut(case) and "PCA" in terrs and not _has_posterior_fossa_strong(case):
        terrs = ["PCA"] + [t for t in terrs if t != "PCA"]

    pf_strong = _has_posterior_fossa_strong(case)
    anterior_hint = _has_cortical_language(case) or (weakness_hint in ("MCA","ACA"))
    if anterior_hint and not pf_strong:
        anterior = [t for t in terrs if t in ("MCA","ACA","PCA")]
        posterior = [t for t in terrs if t in ("BA","Vertebral")]
        if anterior:
            terrs = anterior + posterior

    return terrs

def _prioritize(weakness_hint: Optional[str], terrs_from_pred: List[str], case: Dict[str, Any]) -> List[str]:
    pf_strong = _has_posterior_fossa_strong(case)

    posterior = [t for t in terrs_from_pred if t in ("PCA","BA","Vertebral")]
    cortex    = [t for t in terrs_from_pred if t in ("MCA","ACA")]

    if pf_strong:
        ordered = posterior + cortex
    else:
        ordered = cortex + posterior

    if weakness_hint:
        ordered = [weakness_hint] + [t for t in ordered if t != weakness_hint]

    if _saw_region(case, "Subcortical") and not ordered:
        ordered = [weakness_hint or "MCA"]
    if not ordered:
        ordered = [weakness_hint or "MCA"]

    ordered = _apply_clinical_pruning(ordered, weakness_hint, case)

    # Return TOP-1 vascular territory only.
    # This keeps vascular_estimate["vascular_territory"] as a one-item list
    # so downstream code remains compatible with the original schema.
    for t in ordered:
        if t in ALLOWED_TERRITORIES:
            return [t]

    return [weakness_hint or "MCA"]

# ---------- Final estimator ----------
def estimate_vascular_territory(case: Dict[str, Any]) -> Dict[str, Any]:
    weakness_hint = _get_weakness_hint(case)
    terrs_from_pred = _territories_from_prediction(case)
    territories = _prioritize(weakness_hint, terrs_from_pred, case)

    # infer side strictly from correlations/predictions (no GT)
    side = _infer_side_from_prediction(case)

    conf = 0.5
    if weakness_hint: conf += 0.25
    if terrs_from_pred: conf += 0.2
    if _has_posterior_fossa_signs(case): conf += 0.05
    # No multi-territory penalty because this version returns TOP-1 only.
    conf = max(0.0, min(1.0, conf))

    rationale_bits = []
    if weakness_hint: rationale_bits.append(f"weakness pattern → {weakness_hint}")
    if _has_cortical_language(case): rationale_bits.append("aphasia → favor MCA")
    if _has_visual_field_cut(case): rationale_bits.append("visual field cut → favor PCA")
    if _has_posterior_fossa_signs(case): rationale_bits.append("posterior fossa signs present")
    if terrs_from_pred: rationale_bits.append(f"predicted regions → {', '.join(terrs_from_pred)}")

    # side source explanation (optional)
    used_corr = False
    for it in _iter_correlations(case):
        if (it or {}).get("side"):
            used_corr = True
            break
    if used_corr:
        rationale_bits.append("side from correlation (majority)")
    else:
        lo = (case.get("localization_outcome") or {})
        if lo.get("uniform_side") or lo.get("inferred_hemisphere_label"):
            rationale_bits.append("side from localization_outcome (fallback)")
        elif (_details(case).get("weakness")):
            rationale_bits.append("side from weakness contralateral inference (fallback)")

    if not rationale_bits:
        rationale_bits.append("defaulted to MCA due to limited cues")

    return {
        "vascular_territory": territories,
        "side": side,  # 'Left'/'Right' or None
        "confidence": round(conf, 2),
        "rationale": "; ".join(rationale_bits),
    }

# ---------- Metrics (global territory-only) ----------
def _first_allowed(gt_list: Any) -> Optional[str]:
    if not gt_list:
        return None
    if isinstance(gt_list, str):
        return gt_list if gt_list in ALLOWED_TERRITORIES else None
    if isinstance(gt_list, list):
        for lab in gt_list:
            if lab in ALLOWED_TERRITORIES:
                return lab
    return None

def _list_allowed(gt_list: Any) -> List[str]:
    if not gt_list:
        return []
    if isinstance(gt_list, str):
        return [gt_list] if gt_list in ALLOWED_TERRITORIES else []
    if isinstance(gt_list, list):
        return [lab for lab in gt_list if lab in ALLOWED_TERRITORIES]
    return []

def confusion_and_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    n = len(ALLOWED_TERRITORIES)
    cm = [[0]*n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        cm[IDX[t]][IDX[p]] += 1

    total_samples = sum(sum(row) for row in cm)
    correct = sum(cm[i][i] for i in range(n))
    overall_acc = correct / total_samples if total_samples else 0.0

    per_class = {}
    accs = []
    for i, lab in enumerate(ALLOWED_TERRITORIES):
        TP = cm[i][i]
        FN = sum(cm[i][j] for j in range(n) if j != i)
        FP = sum(cm[j][i] for j in range(n) if j != i)
        TN = total_samples - TP - FN - FP

        denom = TP + FP + FN + TN
        acc = (TP + TN) / denom if denom else 0.0
        accs.append(acc)

        recall      = TP / (TP + FN) if (TP + FN) else 0.0
        specificity = TN / (TN + FP) if (TN + FP) else 0.0
        f1          = (2*TP)/(2*TP + FP + FN) if (2*TP + FP + FN) else 0.0

        per_class[lab] = {
            "support": TP + FN,
            "accuracy": round(acc, 4),
            "recall_sensitivity": round(recall, 4),
            "specificity": round(specificity, 4),
            "f1": round(f1, 4),
        }

    macro_acc = sum(accs)/len(accs) if accs else 0.0
    return {
        "confusion_matrix": cm,
        "labels": list(ALLOWED_TERRITORIES),
        "overall_accuracy": round(overall_acc, 4),
        "macro_accuracy": round(macro_acc, 4),
        "per_class": per_class,
        "total": total_samples,
    }

def microaverage_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    # NOTE: This returns micro-accuracy (TP/N) under multiple names for continuity.
    # If you prefer true micro P/R/F1, replace this with aggregated TP/FP/FN logic.
    K = len(ALLOWED_TERRITORIES)
    n = K
    cm = [[0]*n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        cm[IDX[t]][IDX[p]] += 1

    N = sum(sum(row) for row in cm)
    TP = sum(cm[i][i] for i in range(n))
    if N == 0:
        return {"micro_precision": 0.0, "micro_sensitivity": 0.0, "micro_specificity": 0.0, "micro_f1": 0.0}

    precision = TP / N
    recall    = TP / N
    f1        = TP / N
    micro_specificity = ((K - 2) * N + TP) / ((K - 1) * N) if K > 1 else 0.0

    return {
        "micro_precision": round(precision, 4),
        "micro_sensitivity": round(recall, 4),
        "micro_specificity": round(micro_specificity, 4),
        "micro_f1": round(f1, 4),
    }

# ---------- Metrics (side-aware global) ----------
def confusion_and_metrics_side(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    n = len(SIDE_TERRITORY_LABELS)
    cm = [[0]*n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        cm[SIDE_IDX[t]][SIDE_IDX[p]] += 1

    total_samples = sum(sum(row) for row in cm)
    correct = sum(cm[i][i] for i in range(n))
    overall_acc = correct / total_samples if total_samples else 0.0

    per_class = {}
    accs = []
    for i, lab in enumerate(SIDE_TERRITORY_LABELS):
        TP = cm[i][i]
        FN = sum(cm[i][j] for j in range(n) if j != i)
        FP = sum(cm[j][i] for j in range(n) if j != i)
        TN = total_samples - TP - FN - FP

        denom = TP + FP + FN + TN
        acc = (TP + TN) / denom if denom else 0.0
        accs.append(acc)

        recall      = TP / (TP + FN) if (TP + FN) else 0.0
        specificity = TN / (TN + FP) if (TN + FP) else 0.0
        f1          = (2*TP)/(2*TP + FP + FN) if (2*TP + FP + FN) else 0.0

        per_class[lab] = {
            "support": TP + FN,
            "accuracy": round(acc, 4),
            "recall_sensitivity": round(recall, 4),
            "specificity": round(specificity, 4),
            "f1": round(f1, 4),
        }

    macro_acc = sum(accs)/len(accs) if accs else 0.0
    return {
        "confusion_matrix": cm,
        "labels": list(SIDE_TERRITORY_LABELS),
        "overall_accuracy": round(overall_acc, 4),
        "macro_accuracy": round(macro_acc, 4),
        "per_class": per_class,
        "total": total_samples,
    }

def microaverage_metrics_side(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    # NOTE: returns micro-accuracy (TP/N) under multiple names for continuity.
    K = len(SIDE_TERRITORY_LABELS)
    n = K
    cm = [[0]*n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        cm[SIDE_IDX[t]][SIDE_IDX[p]] += 1

    N = sum(sum(row) for row in cm)
    TP = sum(cm[i][i] for i in range(n))
    if N == 0:
        return {"micro_precision": 0.0, "micro_sensitivity": 0.0, "micro_specificity": 0.0, "micro_f1": 0.0}

    precision = TP / N
    recall    = TP / N
    f1        = TP / N
    micro_specificity = ((K - 2) * N + TP) / ((K - 1) * N) if K > 1 else 0.0

    return {
        "micro_precision": round(precision, 4),
        "micro_sensitivity": round(recall, 4),
        "micro_specificity": round(micro_specificity, 4),
        "micro_f1": round(f1, 4),
    }

# ---------- Stratified metric builders ----------
def _metrics_with_labelset(labels: Tuple[str, ...],
                           y_true: List[str],
                           y_pred: List[str],
                           *,
                           add_other: bool = True,
                           other_label: str = "Other") -> Dict[str, Any]:
    base_labels = list(labels)
    use_labels = base_labels + ([other_label] if add_other else [])
    idx = {lab: i for i, lab in enumerate(use_labels)}
    n = len(use_labels)

    cm = [[0]*n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        if t not in base_labels:
            continue
        p_lab = p if p in base_labels else (other_label if add_other else None)
        if p_lab is None:
            continue
        cm[idx[t]][idx[p_lab]] += 1

    total = sum(sum(r) for r in cm)
    correct = sum(cm[i][i] for i in range(len(base_labels)))
    overall_acc = correct/total if total else 0.0

    per_class, accs = {}, []
    for i, lab in enumerate(base_labels):
        TP = cm[i][i]
        FN = sum(cm[i][j] for j in range(n) if j != i)
        FP = sum(cm[j][i] for j in range(n) if j != i)
        TN = total - TP - FN - FP
        denom = TP + FP + FN + TN
        acc = (TP + TN) / denom if denom else 0.0
        accs.append(acc)
        recall      = TP / (TP + FN) if (TP + FN) else 0.0
        specificity = TN / (TN + FP) if (TN + FP) else 0.0
        f1          = (2*TP)/(2*TP + FP + FN) if (2*TP + FP + FN) else 0.0
        per_class[lab] = {
            "support": TP + FN,
            "accuracy": round(acc, 4),
            "recall_sensitivity": round(recall, 4),
            "specificity": round(specificity, 4),
            "f1": round(f1, 4),
        }

    macro_acc = sum(accs)/len(accs) if accs else 0.0
    return {
        "confusion_matrix": cm,
        "labels": use_labels,
        "overall_accuracy": round(overall_acc, 4),
        "macro_accuracy": round(macro_acc, 4),
        "per_class": per_class,
        "total": total,
    }

def _metrics_with_labelset_side(labels_terr: Tuple[str, ...],
                                y_true: List[str],
                                y_pred: List[str],
                                *,
                                add_other: bool = True,
                                other_label: str = "Other") -> Dict[str, Any]:
    base_labels = tuple(f"{s} {t}" for t in labels_terr for s in SIDES)
    use_labels = list(base_labels) + ([other_label] if add_other else [])
    idx = {lab: i for i, lab in enumerate(use_labels)}
    n = len(use_labels)

    cm = [[0]*n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        if t not in base_labels:
            continue
        p_lab = p if p in base_labels else (other_label if add_other else None)
        if p_lab is None:
            continue
        cm[idx[t]][idx[p_lab]] += 1

    total = sum(sum(r) for r in cm)
    correct = sum(cm[i][i] for i in range(len(base_labels)))
    overall_acc = correct/total if total else 0.0

    per_class, accs = {}, []
    for i, lab in enumerate(base_labels):
        TP = cm[i][i]
        FN = sum(cm[i][j] for j in range(n) if j != i)
        FP = sum(cm[j][i] for j in range(n) if j != i)
        TN = total - TP - FN - FP
        denom = TP + FP + FN + TN
        acc = (TP + TN)/denom if denom else 0.0
        accs.append(acc)
        recall      = TP / (TP + FN) if (TP + FN) else 0.0
        specificity = TN / (TN + FP) if (TN + FP) else 0.0
        f1          = (2*TP)/(2*TP + FP + FN) if (2*TP + FP + FN) else 0.0
        per_class[lab] = {
            "support": TP + FN,
            "accuracy": round(acc, 4),
            "recall_sensitivity": round(recall, 4),
            "specificity": round(specificity, 4),
            "f1": round(f1, 4),
        }

    macro_acc = sum(accs)/len(accs) if accs else 0.0
    return {
        "confusion_matrix": cm,
        "labels": use_labels,
        "overall_accuracy": round(overall_acc, 4),
        "macro_accuracy": round(macro_acc, 4),
        "per_class": per_class,
        "total": total,
    }

# ---------- Category metrics ----------
def _predicted_category_from_territory(pred_terr: Optional[str]) -> Optional[str]:
    if pred_terr in HEMI_TERRS:
        return "hemisphere"
    if pred_terr in PF_TERRS:
        return "posterior_fossa"
    return None

def category_confusion_and_metrics(y_true_cat: List[str], y_pred_cat: List[str]) -> Dict[str, Any]:
    n = len(CATEGORY_LABELS)
    cm = [[0]*n for _ in range(n)]
    for t, p in zip(y_true_cat, y_pred_cat):
        cm[CAT_IDX[t]][CAT_IDX[p]] += 1

    total = sum(sum(r) for r in cm)
    correct = sum(cm[i][i] for i in range(n))
    overall_acc = correct/total if total else 0.0

    per_class_acc = {}
    for i, lab in enumerate(CATEGORY_LABELS):
        TP = cm[i][i]
        FN = sum(cm[i][j] for j in range(n) if j != i)
        FP = sum(cm[j][i] for j in range(n) if j != i)
        TN = total - TP - FN - FP
        denom = TP + FP + FN + TN
        acc = (TP + TN)/denom if denom else 0.0
        per_class_acc[lab] = round(acc, 4)

    return {
        "confusion_matrix": cm,
        "labels": list(CATEGORY_LABELS),
        "overall_accuracy": round(overall_acc, 4),
        "per_class_accuracy": per_class_acc,
        "total": total,
    }

def _save_confusion_csv(cm: List[List[int]], labels: List[str], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("TRUE\\PRED," + ",".join(labels) + "\n")
        for i, row in enumerate(cm):
            f.write(labels[i] + "," + ",".join(str(x) for x in row) + "\n")

# ---------- Main ----------
def main():
    inp, outp = Path(INPUT_FILE), Path(OUTPUT_FILE)
    cases = load_json_or_jsonl(inp)
    print(f"Loaded {len(cases)} cases from {INPUT_FILE}")

    # Global pools
    eval_true_terr, eval_pred_terr = [], []
    eval_true_side, eval_pred_side = [], []

    # Stratified by localization_label.category
    terr_true_hemi, terr_pred_hemi = [], []
    terr_true_pf,   terr_pred_pf   = [], []
    side_true_hemi, side_pred_hemi = [], []
    side_true_pf,   side_pred_pf   = [], []

    # Category pools (binary)
    cat_true, cat_pred = [], []

    augmented: List[Dict[str, Any]] = []
    n_cat_missing = 0

    # mismatch collectors (summaries + FULL originals)
    mis_territory: List[Dict[str, Any]] = []
    mis_side: List[Dict[str, Any]] = []
    mis_category: List[Dict[str, Any]] = []

    mis_territory_full: List[Dict[str, Any]] = []
    mis_side_full: List[Dict[str, Any]] = []
    mis_category_full: List[Dict[str, Any]] = []

    for idx, case in enumerate(cases):
        est = estimate_vascular_territory(case)
        out_case = dict(case); out_case["vascular_estimate"] = est
        augmented.append(out_case)

        hadm_id = case.get("hadm_id") or case.get("id") or case.get("_id") or idx

        pred_terrs_all = [t for t in (est.get("vascular_territory") or []) if t in ALLOWED_TERRITORIES]
        yhat_side_raw = est.get("side")
        yhat_side = _norm_side(yhat_side_raw)

        # ground truth territory
        gt_container = (case.get("vascular_territory_result") or {})
        gt_list_raw = gt_container.get("vascular_territory")
        if gt_list_raw is None:
            gt_list_raw = case.get("vascular_territory")
        gt_terrs_all = _list_allowed(gt_list_raw)
        ygt_terr_primary = _first_allowed(gt_list_raw)

        # ground truth side (for eval only)
        ygt_side = _norm_side(
            gt_container.get("side") or
            gt_container.get("stroke_location_side") or
            case.get("ground_truth_side") or
            case.get("stroke_location_side") or
            case.get("side")
        )

        # ----- Align prediction/GT for evaluation: treat lists as unordered sets -----
        overlap = [t for t in pred_terrs_all if t in gt_terrs_all]
        if overlap:
            ygt_terr_eval = overlap[0]
            yhat_terr_eval = overlap[0]
        else:
            ygt_terr_eval = ygt_terr_primary
            yhat_terr_eval = pred_terrs_all[0] if pred_terrs_all else None

        # category (GT) and category (pred) — use eval prediction
        cat = _get_category(case)
        pred_cat = _predicted_category_from_territory(yhat_terr_eval)
        if cat is None:
            n_cat_missing += 1

        # ----- Global (all), plus collect territory mistakes -----
        if ygt_terr_eval is not None and yhat_terr_eval is not None:
            eval_true_terr.append(ygt_terr_eval)
            eval_pred_terr.append(yhat_terr_eval)
            if ygt_terr_eval != yhat_terr_eval:
                mis_territory.append({
                    "idx": idx,
                    "hadm_id": hadm_id,
                    "GT_territory": ygt_terr_eval,
                    "PRED_territory": yhat_terr_eval,
                    "GT_side": ygt_side or "",
                    "PRED_side": yhat_side or "",
                    "confidence": est.get("confidence"),
                    "weakness_hint": _get_weakness_hint(case) or "",
                    "has_posterior_fossa_signs": _has_posterior_fossa_signs(case),
                    "has_aphasia": _has_cortical_language(case),
                    "has_visual_field_cut": _has_visual_field_cut(case),
                    "predicted_regions": "|".join(_territories_from_prediction(case)),
                    "rationale": est.get("rationale", "")
                })
                full_entry = dict(case)
                full_entry["vascular_estimate"] = est
                full_entry["__error__"] = {
                    "type": "territory_mismatch",
                    "GT_territory": ygt_terr_eval,
                    "PRED_territory": yhat_terr_eval,
                    "GT_side": ygt_side,
                    "PRED_side": yhat_side,
                    "confidence": est.get("confidence"),
                    "rationale": est.get("rationale"),
                    "GT_territories_all": gt_terrs_all,
                    "PRED_territories_all": pred_terrs_all,
                }
                mis_territory_full.append(full_entry)

        # ----- Side+Territory (ALL) + collect side+territory mistakes -----
        if ygt_terr_eval is not None and ygt_side is not None and yhat_terr_eval is not None and yhat_side is not None:
            t_true = f"{ygt_side} {ygt_terr_eval}"
            t_pred = f"{yhat_side} {yhat_terr_eval}"
            eval_true_side.append(t_true)
            eval_pred_side.append(t_pred)
            if t_true != t_pred:
                mis_side.append({
                    "idx": idx,
                    "hadm_id": hadm_id,
                    "GT_side_territory": t_true,
                    "PRED_side_territory": t_pred,
                    "GT_side": ygt_side, "GT_territory": ygt_terr_eval,
                    "PRED_side": yhat_side, "PRED_territory": yhat_terr_eval,
                    "confidence": est.get("confidence"),
                    "rationale": est.get("rationale", "")
                })
                full_entry = dict(case)
                full_entry["vascular_estimate"] = est
                full_entry["__error__"] = {
                    "type": "side_territory_mismatch",
                    "GT_side_territory": t_true,
                    "PRED_side_territory": t_pred,
                    "GT_side": ygt_side,
                    "GT_territory": ygt_terr_eval,
                    "PRED_side": yhat_side,
                    "PRED_territory": yhat_terr_eval,
                    "confidence": est.get("confidence"),
                    "rationale": est.get("rationale"),
                    "GT_territories_all": gt_terrs_all,
                    "PRED_territories_all": pred_terrs_all,
                }
                mis_side_full.append(full_entry)

        # Category confusion (+ collect mistakes)
        if cat in CATEGORY_LABELS and pred_cat in CATEGORY_LABELS:
            cat_true.append(cat); cat_pred.append(pred_cat)
            if cat != pred_cat:
                mis_category.append({
                    "idx": idx,
                    "hadm_id": hadm_id,
                    "GT_category": cat,
                    "PRED_category": pred_cat,
                    "GT_territory": ygt_terr_eval or "",
                    "PRED_territory": yhat_terr_eval or "",
                    "confidence": est.get("confidence"),
                    "rationale": est.get("rationale", "")
                })
                full_entry = dict(case)
                full_entry["vascular_estimate"] = est
                full_entry["__error__"] = {
                    "type": "category_mismatch",
                    "GT_category": cat,
                    "PRED_category": pred_cat,
                    "GT_territory": ygt_terr_eval,
                    "PRED_territory": yhat_terr_eval,
                    "confidence": est.get("confidence"),
                    "rationale": est.get("rationale"),
                }
                mis_category_full.append(full_entry)

        # ----- Stratify by provided category -----
        if (cat and ygt_terr_eval is not None and yhat_terr_eval is not None):
            if cat == "hemisphere":
                terr_true_hemi.append(ygt_terr_eval); terr_pred_hemi.append(yhat_terr_eval)
                if ygt_side is not None and yhat_side is not None:
                    side_true_hemi.append(f"{ygt_side} {ygt_terr_eval}")
                    side_pred_hemi.append(f"{yhat_side} {yhat_terr_eval}")
            elif cat == "posterior_fossa":
                terr_true_pf.append(ygt_terr_eval); terr_pred_pf.append(yhat_terr_eval)
                if ygt_side is not None and yhat_side is not None:
                    side_true_pf.append(f"{ygt_side} {ygt_terr_eval}")
                    side_pred_pf.append(f"{yhat_side} {yhat_terr_eval}")

    write_jsonl(outp, augmented)
    print(f"Wrote {len(augmented)} cases → {OUTPUT_FILE}")
    if n_cat_missing:
        print(f"(Note) {n_cat_missing} case(s) missing localization_label.category; excluded from stratified metrics.")

    # =========================
    # Global (all territories)
    # =========================
    if eval_true_terr:
        res_all = confusion_and_metrics(eval_true_terr, eval_pred_terr)
        micro = microaverage_metrics(eval_true_terr, eval_pred_terr)
        cm, labels = res_all["confusion_matrix"], res_all["labels"]

        print("\n=== Confusion Matrix (Territory only, ALL) (rows=TRUE, cols=PRED) ===")
        header = " " * 12 + "  ".join(f"{lab:>10}" for lab in labels); print(header)
        for i, lab in enumerate(labels):
            print(f"{lab:>10}  " + "  ".join(f"{cm[i][j]:>10d}" for j in range(len(labels))))

        print(f"\nOverall Accuracy: {res_all['overall_accuracy']:.4f}  (N={res_all['total']})")
        print(f"Macro Accuracy:   {res_all['macro_accuracy']:.4f}\n")
        print("Per-class metrics (accuracy):")
        for lab in labels:
            m = res_all["per_class"][lab]
            print(f" - {lab}: support={m['support']}, accuracy={m['accuracy']}, "
                  f"sensitivity={m['recall_sensitivity']}, specificity={m['specificity']}, f1={m['f1']}")

        print("\nMicro-averaged metrics:")
        for k, v in micro.items():
            print(f" {k}: {v}")

        _save_confusion_csv(cm, labels, Path(CONFUSION_CSV))
        print(f"\nSaved CSV → {CONFUSION_CSV}")
    else:
        print("No comparable ground-truth (territory) labels found; metrics skipped.")

    if eval_true_side:
        res_all_s = confusion_and_metrics_side(eval_true_side, eval_pred_side)
        micro_s = microaverage_metrics_side(eval_true_side, eval_pred_side)
        cm_s, labels_s = res_all_s["confusion_matrix"], res_all_s["labels"]

        print("\n=== Confusion Matrix (Side + Territory, ALL) (rows=TRUE, cols=PRED) ===")
        header = " " * 12 + "  ".join(f"{lab:>16}" for lab in labels_s); print(header)
        for i, lab in enumerate(labels_s):
            print(f"{lab:>12}  " + "  ".join(f"{cm_s[i][j]:>16d}" for j in range(len(labels_s))))

        print(f"\nOverall Accuracy: {res_all_s['overall_accuracy']:.4f}  (N={res_all_s['total']})")
        print(f"Macro Accuracy:   {res_all_s['macro_accuracy']:.4f}\n")
        print("Per-class metrics (side+territory, accuracy):")
        for lab in labels_s:
            m = res_all_s["per_class"][lab]
            print(f" - {lab}: support={m['support']}, accuracy={m['accuracy']}, "
                  f"sensitivity={m['recall_sensitivity']}, specificity={m['specificity']}, f1={m['f1']}")

        print("\nMicro-averaged metrics (side-aware):")
        for k, v in micro_s.items():
            print(f" {k}: {v}")

        _save_confusion_csv(cm_s, labels_s, Path(CONFUSION_SIDE_CSV))
        print(f"\nSaved side-aware confusion matrix CSV → {CONFUSION_SIDE_CSV}")
    else:
        print("No cases with BOTH side + territory GT/pred; side-aware metrics (ALL) skipped.")

    # =========================
    # Stratified by localization_label
    # =========================
    if terr_true_hemi:
        hemi_labels = tuple(t for t in HEMI_TERRS
                            if any(x == t for x in terr_true_hemi + terr_pred_hemi))
        res_hemi = _metrics_with_labelset(hemi_labels, terr_true_hemi, terr_pred_hemi)
        cm_h, labs_h = res_hemi["confusion_matrix"], res_hemi["labels"]

        print("\n=== Confusion Matrix (Territory only, stratified: HEMISPHERE) ===")
        header = " " * 12 + "  ".join(f"{lab:>10}" for lab in labs_h); print(header)
        for i, lab in enumerate(labs_h):
            print(f"{lab:>10}  " + "  ".join(f"{cm_h[i][j]:>10d}" for j in range(len(labs_h))))

        print(f"\nOverall Accuracy (hemisphere): {res_hemi['overall_accuracy']:.4f}  (N={res_hemi['total']})")
        print(f"Macro Accuracy (hemisphere):   {res_hemi['macro_accuracy']:.4f}\n")
        print("Per-class (hemisphere, accuracy):")
        for lab in hemi_labels:
            m = res_hemi["per_class"][lab]
            print(f" - {lab}: support={m['support']}, accuracy={m['accuracy']}, "
                  f"sensitivity={m['recall_sensitivity']}, specificity={m['specificity']}, f1={m['f1']}")

        _save_confusion_csv(cm_h, list(labs_h), Path(CONFUSION_CSV_HEMI))
        print(f"\nSaved CSV → {CONFUSION_CSV_HEMI}")
    else:
        print("No hemisphere cases (with GT+pred) found; hemisphere metrics skipped.")

    if terr_true_pf:
        pf_labels = tuple(t for t in PF_TERRS
                          if any(x == t for x in terr_true_pf + terr_pred_pf))
        res_pf = _metrics_with_labelset(pf_labels, terr_true_pf, terr_pred_pf)
        cm_p, labs_p = res_pf["confusion_matrix"], res_pf["labels"]

        print("\n=== Confusion Matrix (Territory only, stratified: POSTERIOR FOSSA) ===")
        header = " " * 12 + "  ".join(f"{lab:>10}" for lab in labs_p); print(header)
        for i, lab in enumerate(labs_p):
            print(f"{lab:>10}  " + "  ".join(f"{cm_p[i][j]:>10d}" for j in range(len(labs_p))))

        print(f"\nOverall Accuracy (posterior fossa): {res_pf['overall_accuracy']:.4f}  (N={res_pf['total']})")
        print(f"Macro Accuracy (posterior fossa):   {res_pf['macro_accuracy']:.4f}\n")
        print("Per-class (posterior fossa, accuracy):")
        for lab in pf_labels:
            m = res_pf["per_class"][lab]
            print(f" - {lab}: support={m['support']}, accuracy={m['accuracy']}, "
                  f"sensitivity={m['recall_sensitivity']}, specificity={m['specificity']}, f1={m['f1']}")

        _save_confusion_csv(cm_p, list(labs_p), Path(CONFUSION_CSV_PF))
        print(f"\nSaved CSV → {CONFUSION_CSV_PF}")
    else:
        print("No posterior fossa cases (with GT+pred) found; posterior fossa metrics skipped.")

    # =========================
    # Binary Category (Hemisphere vs Posterior fossa)
    # =========================
    if cat_true:
        cres = category_confusion_and_metrics(cat_true, cat_pred)
        ccm, clabels = cres["confusion_matrix"], cres["labels"]
        print("\n=== Confusion Matrix (Category: Hemisphere vs Posterior fossa) ===")
        header = " " * 24 + "  ".join(f"{lab:>20}" for lab in clabels)
        print(header)
        for i, lab in enumerate(clabels):
            print(f"{('TRUE: ' + lab):>24}  " + "  ".join(f"{ccm[i][j]:>20d}" for j in range(len(clabels))))
        print(f"\nOverall Accuracy (category): {cres['overall_accuracy']:.4f}  (N={cres['total']})")
        print("Per-class accuracy:")
        for lab in clabels:
            print(f" - {lab}: {cres['per_class_accuracy'][lab]:.4f}")

        _save_confusion_csv(
            ccm,
            [f"PRED: {x.title().replace('_',' ')}" for x in clabels],
            Path("out/confusion_matrix_category.csv"),
        )
        print(f"\nSaved category confusion CSV → out/confusion_matrix_category.csv")
    else:
        print("No cases with usable category GT+prediction; category matrix skipped.")

    # =========================
    # Write misclassifications as JSONL (summaries + FULL originals)
    # =========================
    if mis_territory:
        _save_jsonl(Path(MIS_TERRITORY_JSONL), mis_territory)
        _save_jsonl(Path(MIS_TERRITORY_FULL_JSONL), mis_territory_full)
        print(f"\nSaved misclassified territory (summary) → {MIS_TERRITORY_JSONL}")
        print(f"Saved misclassified territory (FULL)    → {MIS_TERRITORY_FULL_JSONL}")
    else:
        print("\nNo territory misclassifications 🎉")

    if mis_side:
        _save_jsonl(Path(MIS_SIDE_JSONL), mis_side)
        _save_jsonl(Path(MIS_SIDE_FULL_JSONL), mis_side_full)
        print(f"\nSaved misclassified side+territory (summary) → {MIS_SIDE_JSONL}")
        print(f"Saved misclassified side+territory (FULL)    → {MIS_SIDE_FULL_JSONL}")
    else:
        print("\nNo side+territory misclassifications (with both sides present)")

    if mis_category:
        _save_jsonl(Path(MIS_CATEGORY_JSONL), mis_category)
        _save_jsonl(Path(MIS_CATEGORY_FULL_JSONL), mis_category_full)
        print(f"\nSaved misclassified category (summary) → {MIS_CATEGORY_JSONL}")
        print(f"Saved misclassified category (FULL)    → {MIS_CATEGORY_FULL_JSONL}")
    else:
        print("\nNo category (hemisphere vs posterior fossa) misclassifications")

if __name__ == "__main__":
    main()
