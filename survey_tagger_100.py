#!/usr/bin/env python3
"""
Symptom Survey Tagger (Google Gemini)

- Strategy: Uses a robust category-by-category extraction on raw clinical text.

ENV:
  GOOGLE_API_KEY
  LLM_CONCURRENCY (optional, default 8)

Install:
  pip install google-generativeai>=0.5 python-dotenv tqdm
"""
from __future__ import annotations
import os, sys, argparse, json, pathlib, time, random
from typing import Dict, Any, List, Optional, Tuple
from string import Template
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from threading import BoundedSemaphore

load_dotenv()

# ==============================
# Hardcoded Configuration
# ==============================
INPUT_PATH = "1_results_cleaned.only_gold_yes.jsonl"
OUTPUT_PATH = "1_results_cleaned.only_gold_yes_symptomextract.jsonl"
MODEL_NAME = "gemini-2.5-flash-lite"

# Global cap on concurrent LLM calls (can be overridden by CLI/env)
DEFAULT_LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "8"))
_llm_sem: Optional[BoundedSemaphore] = None

# ==============================
# Google Gemini client
# ==============================
try:
    import google.generativeai as genai
except Exception:
    print("Please `pip install google-generativeai>=0.5 python-dotenv tqdm`", file=sys.stderr)
    raise

def configure_gemini_and_get_model():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: Missing required config: GOOGLE_API_KEY", file=sys.stderr)
        sys.exit(1)
    genai.configure(api_key=api_key)
    return MODEL_NAME

# ==============================
# Schema & Definitions
# ==============================
def make_details_block() -> Dict[str, Any]:
    """Defines the full, empty JSON schema for a neurological survey."""
    return {
        "eom": {
            "cn3_palsy": {"R": False, "L": False}, "cn4_palsy": {"R": False, "L": False},
            "cn6_palsy": {"R": False, "L": False}, "unspecified_limit": {"R": False, "L": False},
        },
        "skew": {"skew_side": {"R": False, "L": False}, "skew_legacy": False},
        "vertigo": {"present": False},
        "nystagmus": {"pathologic": False, "direction": None},
        "pupils": { "anisocoria": False, "reactivity": {"fixed": {"R": False, "L": False}, "sluggish": {"R": False, "L": False}}},
        "cortical_signs": {
            "aphasia": {"present": False, "type": None}, "neglect_side": {"R": False, "L": False},
            "apraxia_side": {"R": False, "L": False}, "apraxia_type": None,
            "visual_field_cut": {"R": False, "L": False}, "gaze_preference": {"R": False, "L": False}
        },
        "dysarthria": {"present": False},
        "weakness": {
            "side_only": {"R": False, "L": False}, "face": {"R": False, "L": False},
            "arm":  {"R": False, "L": False}, "leg":  {"R": False, "L": False},
            "involvement_pattern": {"face_arm_gt_leg": False, "leg_gt_face_arm": False},
            "lr_comparison": {
                "overall": {"R_gt_L": False, "L_gt_R": False, "symmetric": False},
                "face":    {"R_gt_L": False, "L_gt_R": False, "symmetric": False},
                "arm":     {"R_gt_L": False, "L_gt_R": False, "symmetric": False},
                "leg":     {"R_gt_L": False, "L_gt_R": False, "symmetric": False}
            }
        },
        "sensory_loss": {
            "face": {"R": False, "L": False}, "arm":  {"R": False, "L": False},
            "leg":  {"R": False, "L": False}, "cortical_sensory_loss": False,
            "cortical_sensory_examples": [],
            "involvement_pattern": {"face_arm_gt_leg": False, "leg_gt_face_arm": False, "unknown": False},
            "side_only": {"R": False, "L": False}
        }
    }

CATEGORIES = list(make_details_block().keys())

CATEGORY_DEFINITIONS = {
    "eom": """- **eom (Extraocular Movements)**:
  - `cn3_palsy`: True for "(R/L) CN III palsy" or "ptosis + eye 'down and out'".
  - `cn4_palsy`: True for "CN IV palsy" or "vertical diplopia worse on downgaze".
  - `cn6_palsy`: True for "(R/L) CN VI palsy" or "limited abduction".
  - `unspecified_limit`: True for restricted EOMs if no specific nerve is named.
  - Negations like "EOMI" or "full EOM" mean all `eom` booleans are `false`.""",
    "skew": """- **skew**:
  - `skew_side`: Set R/L to `true` if the side of vertical misalignment is stated (e.g., "left eye lower" -> L=true).
  - `skew_legacy`: Set to `true` if skew is present but side is unclear.""",
    "vertigo": """- **vertigo**:
  - `present`: True only for true vertigo (spinning/room-spinning). "Lightheadedness" alone is `false`.""",
    "nystagmus": """- **nystagmus**:
  - `pathologic`: True if any nystagmus is stated.
  - `direction`: Can be 'R', 'L', 'U', 'D', 'torsional', 'bidirectional', or 'other'.""",
    "pupils": """- **pupils**:
  - `anisocoria`: True for "unequal pupils".
  - `reactivity.fixed`: True for "(R/L) pupil fixed/nonreactive".
  - `reactivity.sluggish`: True for "(R/L) sluggish reaction".
  - "PERRL/PERRLA" implies `anisocoria` is `false` and all `reactivity` booleans are `false`.""",
    "cortical_signs": """- **cortical_signs**:
  - `aphasia.present`: True for any language impairment.
  - `aphasia.type`: Can be 'expressive', 'receptive', 'global', 'anomic', or 'other'.
  - `neglect_side`: True for hemispatial neglect/extinction on that side.
  - `apraxia_side`: True for sided limb apraxia.
  - `visual_field_cut`: True for homonymous hemianopia/quadrantanopia to that side.
  - `gaze_preference`: True if gaze preference/deviation to that side is stated.""",
    "dysarthria": """- **dysarthria**:
  - `present`: True for dysarthria or slurred speech. Do not infer from aphasia.""",
    "weakness": """- **weakness**:
  - `side_only`: True only if side is stated without body part (e.g., "right-sided weakness").
  - `face`/`arm`/`leg`: True for explicit weakness in that compartment.
  - `involvement_pattern`: `face_arm_gt_leg` or `leg_gt_face_arm` if distribution is described.
  - `lr_comparison`: Only set if weakness is bilateral. For `overall`, `face`, `arm`, or `leg`, set exactly one of `R_gt_L`, `L_gt_R`, or `symmetric` to true based on text like "R>L" or "symmetric".""",
    "sensory_loss": """- **sensory_loss**:
  - Fields are analogous to weakness.
  - `cortical_sensory_loss`: True for deficits in graphesthesia, stereognosis, etc. Add examples to `cortical_sensory_examples`."""
}

def category_schema_slice(category: str) -> Dict[str, Any]:
    """Returns the schema structure for a single category."""
    base = make_details_block()
    if category not in base:
        raise ValueError(f"Unknown category '{category}'. Allowed={list(base.keys())}")
    return {category: base[category]}

# ==============================
# Prompt for Optimized Category-by-Category Workflow
# ==============================
CATEGORY_EXTRACT_PROMPT_TEMPLATE = Template(
    """You are a careful, literal clinical NLP annotator.
Your task is to parse the provided clinical text and fill out the JSON schema for the **$category_name** category.

CRITICAL RULES:
- **BI-STATE LOGIC**: `true` if the finding is present, `false` if it is normal, absent, or not mentioned.
- **LITERAL INTERPRETATION**: Do not infer findings beyond the text provided.
- **STRICT JSON OUTPUT**: Output ONLY a single, valid JSON object matching the schema. No commentary or extra keys.
- If a field is not explicitly stated in the text, set it to `false` or `null` exactly as the schema requires.

---
## DETAILED FIELD DEFINITIONS for "$category_name"
---
$field_definitions
---

## Clinical Text
<<<
Chief Complaint: $chief_complaint_text

Neurological Exam: $neuro_exam_text
>>>

## JSON Schema to Fill (for the "$category_name" category ONLY)
<<<
$schema_json
>>>
"""
)

# ==============================
# Helpers: retries, JSON extraction, type coercion
# ==============================
def with_retry(call_fn, *, retries=4, base_delay=0.6, jitter=0.25):
    attempt = 0
    while True:
        try:
            return call_fn()
        except Exception as e:
            msg = (str(e) or "").lower()
            retryable_hits = ("rate limit", "429", "timeout", "temporarily unavailable", "unavailable", "503", "502", "500")
            retryable = any(tok in msg for tok in retryable_hits)
            if not retryable or attempt >= retries:
                raise
            delay = base_delay * (2 ** attempt) + random.random() * jitter
            time.sleep(delay)
            attempt += 1

def _extract_json_blob(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    # Remove markdown code fences if present
    if t.startswith("```"):
        lines = [ln for ln in t.splitlines() if not ln.strip().startswith("```")]
        t = "\n".join(lines).strip()
    # Find first '{' and last '}' span
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    return t[start:end+1]

def _coerce_bools(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _coerce_bools(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_coerce_bools(v) for v in x]
    if isinstance(x, str):
        low = x.strip().lower()
        if low == "true": return True
        if low == "false": return False
    return x

# ==============================
# LLM Call
# ==============================
def call_llm(model, prompt: str) -> Tuple[Any, Dict[str, int]]:
    """Calls the LLM, returning the parsed JSON and a dict of token counts."""
    global _llm_sem
    if _llm_sem is None:
        # Fallback if not initialized; should be set in cli()
        _llm_sem = BoundedSemaphore(DEFAULT_LLM_CONCURRENCY)

    with _llm_sem:
        content = ""
        try:
            config = genai.GenerationConfig(response_mime_type="application/json")
            response = model.generate_content(prompt, generation_config=config)
            content = getattr(response, "text", "") or ""
            
            token_counts = { "input": 0, "output": 0, "total": 0 }
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                token_counts = {
                    "input": usage.prompt_token_count,
                    "output": usage.candidates_token_count,
                    "total": usage.total_token_count,
                }

            blob = _extract_json_blob(content)
            if not blob:
                raise RuntimeError("No JSON found in model response")
            
            parsed = json.loads(blob)
            parsed = _coerce_bools(parsed)
            return parsed, token_counts
        except json.JSONDecodeError as je:
            raise RuntimeError(f"Malformed JSON from model: {je}: {content!r}")
        except Exception:
            # Let with_retry handle transient failures
            raise

# ==============================
# Core Processing Logic
# ==============================
def extract_for_category(model, category: str, cc_text: str, ne_text: str) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Makes a single, focused API call to extract data for one category."""
    schema_slice = category_schema_slice(category)
    field_definitions = CATEGORY_DEFINITIONS.get(category, "")

    prompt = CATEGORY_EXTRACT_PROMPT_TEMPLATE.substitute(
        category_name=category,
        chief_complaint_text=cc_text,
        neuro_exam_text=ne_text,
        schema_json=json.dumps(schema_slice, indent=2),
        field_definitions=field_definitions
    )

    def _call():
        return call_llm(model, prompt)

    details_partial, token_counts = with_retry(_call)
    return details_partial, token_counts

def process_one_case(model, row: Dict[str, Any]) -> Tuple[Dict[str, Any], int, Dict[str, int]]:
    """Processes a single case, returning the result, call count, and token counts."""
    cc_text = (row.get("chief_complaint_text") or "").strip()
    ne_text = (row.get("neuro_exam_text") or "").strip()

    final_details: Dict[str, Any] = {}
    case_tokens = {"input": 0, "output": 0, "total": 0}

    # Fan-out across categories, but actual API concurrency is gated by _llm_sem
    with ThreadPoolExecutor(max_workers=len(CATEGORIES)) as ex:
        future_to_category = {
            ex.submit(extract_for_category, model, cat, cc_text, ne_text): cat
            for cat in CATEGORIES
        }
        for future in as_completed(future_to_category):
            category = future_to_category[future]
            try:
                result, tokens = future.result()
                if category in result:
                    final_details[category] = result[category]
                else:
                    final_details[category] = category_schema_slice(category)[category]
                
                # Aggregate tokens for the case
                case_tokens["input"] += tokens.get("input", 0)
                case_tokens["output"] += tokens.get("output", 0)
                case_tokens["total"] += tokens.get("total", 0)
            except Exception as e:
                hadm_id = row.get('hadm_id', 'UNKNOWN')
                print(f"ERROR processing category '{category}' for case {hadm_id}: {e}", file=sys.stderr)
                final_details[category] = category_schema_slice(category)[category]

    # Create a new row to avoid modifying the original
    out_row = {
        "hadm_id": row.get("hadm_id"),
        "chief_complaint_text": row.get("chief_complaint_text"),
        "neuro_exam_text": row.get("neuro_exam_text"),
        "survey_details": {"details": final_details}
    }

    return out_row, len(CATEGORIES), case_tokens

# ==============================
# I/O and Main CLI
# ==============================
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    full_obj = json.loads(line)
                    data.append({
                        "hadm_id": full_obj.get("hadm_id"),
                        "chief_complaint_text": full_obj.get("chief_complaint_text"),
                        "neuro_exam_text": full_obj.get("neuro_exam_text")
                    })
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON line in input: {line.strip()}", file=sys.stderr)
    return data

def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _fmt_dur(sec: float) -> str:
    sec = int(round(sec))
    h, rem = divmod(sec, 3600); m, s = divmod(rem, 60)
    return f"{h}h {m}m {s}s" if h else (f"{m}m {s}s" if m else f"{s}s")

def cli():
    global _llm_sem

    ap = argparse.ArgumentParser(description="Gemini Clinical Survey Tagger (Hardcoded)")
    ap.add_argument("--workers", type=int, default=4, help="Number of parallel cases to process.")
    ap.add_argument("--limit", type=int, default=None, help="Limit processing to the first N cases.")
    ap.add_argument("--concurrency", type=int, default=DEFAULT_LLM_CONCURRENCY,
                    help="Global max concurrent LLM calls (default from LLM_CONCURRENCY env or 8).")
    args = ap.parse_args()

    # Initialize global semaphore for LLM concurrency
    _llm_sem = BoundedSemaphore(args.concurrency)

    model_name = configure_gemini_and_get_model()
    print(f"Using hardcoded model: {model_name}")
    model = genai.GenerativeModel(model_name)

    print(f"Reading from hardcoded path: {INPUT_PATH}...")
    rows = read_jsonl(INPUT_PATH)
    if args.limit and args.limit > 0:
        print(f"Limiting to the first {args.limit} of {len(rows)} cases.")
        rows = rows[:args.limit]

    total = len(rows)
    out_rows: List[Dict[str, Any]] = []
    global_calls = 0
    global_tokens = {"input": 0, "output": 0, "total": 0}
    start_all = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        with tqdm(total=total, desc="Processing cases") as pbar:
            futures = {ex.submit(process_one_case, model, row): row for row in rows}
            for future in as_completed(futures):
                row_info = futures[future]
                try:
                    out_row, case_calls, case_tokens = future.result()
                    out_rows.append(out_row)
                    global_calls += case_calls
                    global_tokens["input"] += case_tokens.get("input", 0)
                    global_tokens["output"] += case_tokens.get("output", 0)
                    global_tokens["total"] += case_tokens.get("total", 0)
                except Exception as e:
                    hadm_id = row_info.get('hadm_id', 'UNKNOWN')
                    pbar.write(f"FATAL ERROR on case {hadm_id}: {e}")
                pbar.update(1)

    # Sort results to match input order for consistency
    original_order_map = {row['hadm_id']: i for i, row in enumerate(rows)}
    out_rows.sort(key=lambda r: original_order_map.get(r.get('hadm_id'), float('inf')))

    write_jsonl(OUTPUT_PATH, out_rows)

    total_dur = time.time() - start_all
    print(f"\nFinished. Wrote {len(out_rows)} records to {OUTPUT_PATH} in {_fmt_dur(total_dur)}.")
    print(f"Total API calls made (intended): {global_calls}")
    print(f"Total tokens: "
          f"Input={global_tokens['input']:,} | "
          f"Output={global_tokens['output']:,} | "
          f"Total={global_tokens['total']:,}")

if __name__ == "__main__":
    cli()