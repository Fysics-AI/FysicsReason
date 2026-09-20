#!/usr/bin/env python3
"""Evaluate answer-only task1-task5 predictions.

Input JSONL format, one prediction per line:
  {
    "index": 0,
    "task_source": "task1",
    "wsv_answer": "...",
    "final_answer": "..."
  }

For task2 relative bbox predictions, include image_width and image_height in
the corresponding JSONL row.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import difflib
import json
import warnings
from math import isfinite
from pathlib import Path
import re
from typing import Any

import pandas as pd

try:
    from math_verify import parse as math_verify_parse, verify as math_verify_verify
except ImportError:
    math_verify_parse = None
    math_verify_verify = None

TASKS = {"task1", "task2", "task3", "task4", "task5"}

SUMMARY_FIELDS = ("S_wsv", "S_r", "S")

TASK5_SKIPPED_STRUCTURED_RESULT_KEYS = {
    "temperature",
    "temperature_c",
    "time_s",
    "contact_area_m2",
    "conductance_factor_w_per_k",
}

TASK4_MATERIAL_SYNONYMS = {
    "clay": ["clay", "mud", "earthen", "pottery"],
    "plastic": ["plastic", "polymer", "resin"],
    "styrofoam": ["styrofoam", "foam", "polystyrene", "packing foam"],
    "wax": ["wax", "paraffin", "beeswax", "waxy"],
    "wood": ["wood", "wooden", "timber", "lumber"],
    "rubber": ["rubber", "rubbery", "latex", "elastomer", "gel", "slime", "goo", "squishy"],
    "adhesive": ["adhesive", "glue", "sticky", "sealant"],
    "paper": ["paper", "paperboard", "cardboard", "cardstock"],
    "ceramic": ["ceramic", "pottery", "porcelain", "earthenware"],
    "plant": ["plant", "leaf", "vegetation", "botanical"],
    "gelatin": ["gelatin", "gelatine", "collagen gel"],
    "gel": ["gel", "jelly", "hydrogel", "gooey gel"],
    "fur": ["fur", "furry", "animal hair", "pelt"],
    "dirt": ["dirt", "soil", "earth", "mud"],
    "concrete": ["concrete", "cement", "cement mix"],
    "minerals": ["mineral", "minerals", "rock", "crystal"],
    "ice": ["ice", "frozen water", "icy", "ice block"],
    "goo": ["goo", "slime", "sticky sludge", "gunk"],
    "powder": ["powder", "dust", "powdered", "granular powder"],
    "metal": ["metal", "metallic", "alloy", "steel", "iron", "aluminum"],
    "textiles": ["textile", "textiles", "fabric", "cloth", "woven", "fiber"],
    "stone": ["stone", "rock", "rocky", "granite"],
    "glass": ["glass", "glassy", "crystal", "transparent glass"],
    "foam": ["foam", "foamy", "sponge", "spongy"],
}

TASK1_TARGET_ATTRIBUTE_ALIASES = {
    "number": ["number", "count", "many"],
    "distance": ["distance", "far", "length"],
    "speed": ["speed", "velocity"],
    "acceleration": ["acceleration"],
    "time": ["time", "duration"],
    "angle": ["angle"],
    "area": ["area"],
    "volume": ["volume"],
    "mass": ["mass", "weight"],
    "force": ["force"],
    "energy": ["energy"],
    "power": ["power"],
    "pressure": ["pressure"],
    "temperature": ["temperature"],
    "probability": ["probability", "chance"],
    "ratio": ["ratio"],
    "value": ["value"],
    "height": ["height"],
    "width": ["width"],
    "radius": ["radius"],
    "diameter": ["diameter"],
    "perimeter": ["perimeter", "circumference"],
    "sum": ["sum", "total"],
}

TASK1_TARGET_STOPWORDS = {
    "a", "an", "the", "of", "for", "to", "in", "on", "at", "by", "with", "from",
    "is", "are", "was", "were", "be", "been", "being", "how", "what", "which",
    "find", "determine", "calculate", "compute", "give", "needed", "need",
    "required", "require", "current",
}


def simple_singularize(token: str) -> str:
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ves") and len(token) > 4:
        return token[:-3] + "f"
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def task1_try_math_verify_match(pred_answer: str, gt_answer: str) -> bool | None:
    if math_verify_parse is None or math_verify_verify is None:
        return None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SyntaxWarning)
            gold_parsed = math_verify_parse(gt_answer)
            pred_parsed = math_verify_parse(pred_answer)
        if not gold_parsed or not pred_parsed:
            return None
        return bool(math_verify_verify(gold_parsed, pred_parsed))
    except Exception:
        return None


def task1_canonical_target_attribute(text: Any) -> str | None:
    tokens = set(task1_tokenize_target(text))
    for attr, aliases in TASK1_TARGET_ATTRIBUTE_ALIASES.items():
        if any(alias in tokens for alias in aliases):
            return attr
    return None


def normalize_task1_target_text(text: Any) -> str:
    text = str(text or "").lower().strip()
    text = re.sub(r"<image\d+>", " ", text)
    text = text.replace("'s", "")
    text = re.sub(r"how many\s+", "number of ", text)
    text = re.sub(r"what is the\s+", "", text)
    text = re.sub(r"what is\s+", "", text)
    text = re.sub(r"find (an?|the)?\s*", "", text)
    text = re.sub(r"determine (an?|the)?\s*", "", text)
    text = re.sub(r"calculate (an?|the)?\s*", "", text)
    text = re.sub(r"compute (an?|the)?\s*", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def task1_tokenize_target(text: Any) -> list[str]:
    normalized = normalize_task1_target_text(text)
    tokens = []
    for token in normalized.split():
        if token in TASK1_TARGET_STOPWORDS:
            continue
        tokens.append(simple_singularize(token))
    return tokens


def task1_target_exact_match(gold_text: Any, pred_text: Any) -> bool:
    return normalize_task1_target_text(gold_text) == normalize_task1_target_text(pred_text)


def task1_target_token_f1(gold_tokens: list[str], pred_tokens: list[str]) -> float:
    if not gold_tokens or not pred_tokens:
        return 0.0
    gold_counter = Counter(gold_tokens)
    pred_counter = Counter(pred_tokens)
    overlap = sum((gold_counter & pred_counter).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(pred_counter.values())
    recall = overlap / sum(gold_counter.values())
    return 2 * precision * recall / (precision + recall)


def task1_target_score(gold_synonyms: list[str], pred_text: str) -> dict[str, Any]:
    pred_tokens = task1_tokenize_target(pred_text)
    pred_attr = task1_canonical_target_attribute(pred_text)
    best_f1 = 0.0
    best_gold = None
    any_exact_match = False
    any_attribute_match = False

    for gold_text in gold_synonyms:
        gold_tokens = task1_tokenize_target(gold_text)
        gold_attr = task1_canonical_target_attribute(gold_text)
        current_exact = task1_target_exact_match(gold_text, pred_text)
        current_attr = gold_attr is not None and pred_attr == gold_attr
        current_f1 = task1_target_token_f1(gold_tokens, pred_tokens)
        if current_exact:
            any_exact_match = True
        if current_attr:
            any_attribute_match = True
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_gold = gold_text

    target_token_f1 = round(best_f1, 4)
    return {
        "pred_target": pred_text,
        "target_best_gold": best_gold,
        "target_exact_match": any_exact_match,
        "target_attribute_match": any_attribute_match,
        "target_token_f1": target_token_f1,
        "target_correct": any_exact_match or any_attribute_match,
        "target_correct_max": max(float(any_exact_match or any_attribute_match), target_token_f1),
    }


def task1_judge_extracted_answer(
    pred_answer: Any,
    gt_answer: Any,
    gt_answer_option: Any,
) -> bool:
    """Compare an answer-only task1 value directly against its ground truth."""
    gt_option = str(gt_answer_option or "").strip().upper()
    if re.fullmatch(r"[A-Z]", gt_option):
        return str(pred_answer or "").strip().upper() == gt_option
    return task1_try_math_verify_match(str(pred_answer), str(gt_answer)) is True


def task4_property_match(pred_property: str, new_properties: Any) -> bool:
    if hasattr(new_properties, "tolist"):
        new_properties = new_properties.tolist()
    if not isinstance(new_properties, list):
        return False
    pred_norm = normalize_text(pred_property)
    if not pred_norm:
        return False
    pred_parts = [part for part in re.split(r"[^a-z0-9]+", pred_norm) if part]
    for candidate in new_properties:
        candidate_norm = normalize_text(candidate)
        if not candidate_norm:
            continue
        if pred_norm == candidate_norm or pred_norm in candidate_norm or candidate_norm in pred_norm:
            return True
        candidate_parts = [part for part in re.split(r"[^a-z0-9]+", candidate_norm) if part]
        if any(part in candidate_parts for part in pred_parts):
            return True
        if any(part in pred_parts for part in candidate_parts):
            return True
        if any(tokens_match(pred_part, candidate_part) for pred_part in pred_parts for candidate_part in candidate_parts):
            return True
    return False


def task4_material_match(pred_material: str, gt_object: Any) -> bool:
    if not isinstance(gt_object, dict):
        return False
    gt_materials = gt_object.get("materials")
    if hasattr(gt_materials, "tolist"):
        gt_materials = gt_materials.tolist()
    if not isinstance(gt_materials, list):
        return False
    return any(text_contains_any_synonym(pred_material, gt_material, TASK4_MATERIAL_SYNONYMS) for gt_material in gt_materials)


def normalize_text(text: Any) -> str:
    text = str(text or "").strip().lower()
    return re.sub(r"\s+", " ", text)


def json_loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, float) and pd.isna(value):
        return default
    if isinstance(value, (list, dict)):
        return value
    if not str(value).strip():
        return default
    try:
        return json.loads(str(value))
    except Exception:
        return default


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "item") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        return to_jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    return value


def extract_number_list(text: Any) -> list[float]:
    if isinstance(text, (int, float)) and isfinite(float(text)):
        return [float(text)]
    if isinstance(text, (list, tuple)):
        out = []
        for item in text:
            out.extend(extract_number_list(item))
        return out
    value = str(text or "")
    number_pattern = r"(?<![A-Za-z_^])[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?(?![A-Za-z_])"
    return [float(x) for x in re.findall(number_pattern, value)]


def parse_bbox(value: Any) -> list[float] | None:
    nums = extract_number_list(value)
    if len(nums) < 4:
        return None
    return nums[:4]


def bbox_iou(pred: list[float] | None, gt: list[float] | None) -> float:
    if pred is None or gt is None or len(pred) != 4 or len(gt) != 4:
        return 0.0
    px1, py1, px2, py2 = pred
    gx1, gy1, gx2, gy2 = gt
    if px1 > px2 or py1 > py2 or gx1 > gx2 or gy1 > gy2:
        return 0.0
    ix1, iy1 = max(px1, gx1), max(py1, gy1)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    pred_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    gt_area = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
    union = pred_area + gt_area - inter
    return inter / union if union > 0 else 0.0


def valid_bbox(pred: list[float] | None, *, low: float | None = None, high: float | None = None) -> bool:
    if pred is None or len(pred) != 4:
        return False
    for value in pred:
        if not isinstance(value, (int, float)) or not isfinite(float(value)):
            return False
        if low is not None and float(value) < low:
            return False
        if high is not None and float(value) > high:
            return False
    x1, y1, x2, y2 = pred
    return x1 <= x2 and y1 <= y2


def map_bbox_relative_1000(pred: list[float] | None, image_width: Any, image_height: Any) -> list[float] | None:
    if not valid_bbox(pred, low=0.0, high=1000.0):
        return None
    width = first_positive_number(image_width)
    height = first_positive_number(image_height)
    if width is None or height is None:
        return None
    return [
        pred[0] * width / 1000.0,
        pred[1] * height / 1000.0,
        pred[2] * width / 1000.0,
        pred[3] * height / 1000.0,
    ]


def map_bbox_relative_1(pred: list[float] | None, image_width: Any, image_height: Any) -> list[float] | None:
    if not valid_bbox(pred, low=0.0, high=1.0):
        return None
    width = first_positive_number(image_width)
    height = first_positive_number(image_height)
    if width is None or height is None:
        return None
    return [
        pred[0] * width,
        pred[1] * height,
        pred[2] * width,
        pred[3] * height,
    ]


def task2_bbox_to_absolute(
    pred_bbox: list[float] | None,
    image_width: float,
    image_height: float,
    coordinate_mode: Any = None,
) -> tuple[list[float] | None, str | None]:
    mode = str(coordinate_mode or "").strip().lower()
    if mode == "relative_1000":
        mapped = map_bbox_relative_1000(pred_bbox, image_width, image_height)
        return mapped, "relative_1000" if mapped is not None else None
    if mode == "relative_1":
        mapped = map_bbox_relative_1(pred_bbox, image_width, image_height)
        return mapped, "unit_relative" if mapped is not None else None
    if mode == "absolute":
        return (pred_bbox, "absolute") if valid_bbox(pred_bbox) else (None, None)

    if valid_bbox(pred_bbox, low=0.0, high=1.0):
        mapped = map_bbox_relative_1(pred_bbox, image_width, image_height)
        return mapped, "unit_relative"
    if (
        valid_bbox(pred_bbox, low=0.0, high=1000.0)
        and pred_bbox is not None
        and max(abs(float(value)) for value in pred_bbox) > max(float(image_width), float(image_height), 1.0)
    ):
        mapped = map_bbox_relative_1000(pred_bbox, image_width, image_height)
        return mapped, "relative_1000"
    return (pred_bbox, "absolute") if valid_bbox(pred_bbox) else (None, None)


def parse_interval(value: Any) -> list[float] | None:
    nums = extract_number_list(value)
    if len(nums) < 2:
        return None
    return [nums[0], nums[1]]


def interval_iou(pred: list[float] | None, gt: list[float] | None) -> float:
    if pred is None or gt is None or len(pred) != 2 or len(gt) != 2:
        return 0.0
    ps, pe = pred
    gs, ge = gt
    if ps > pe or gs > ge:
        return 0.0
    inter = max(0.0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return inter / union if union > 0 else 0.0


def extract_range_numbers(value: Any) -> tuple[float, float] | None:
    nums = extract_number_list(value)
    if not nums:
        return None
    if len(nums) == 1:
        return nums[0], nums[0]
    return min(nums[0], nums[1]), max(nums[0], nums[1])


def extract_single_value(value: Any) -> float | None:
    interval = extract_range_numbers(value)
    if interval is None:
        return None
    low, high = interval
    return low if low == high else (low + high) / 2.0


def relative_error(value: float, target: float) -> float:
    if target == 0:
        return 0.0 if value == 0 else float("inf")
    return abs(value - target) / abs(target)


def score_against_range(pred_text: Any, gt_range_text: Any) -> dict[str, Any]:
    pred = extract_single_value(pred_text)
    gt_range = extract_range_numbers(gt_range_text)
    if pred is None or gt_range is None:
        return {"pred_value": pred, "in_range": False, "score": None}
    low, high = gt_range
    tolerance = 1e-9
    if low - tolerance <= pred <= high + tolerance:
        return {"pred_value": pred, "gt_min": low, "gt_max": high, "in_range": True, "relative_error": 0.0, "score": 1.0}
    bound = low if pred < low else high
    err = relative_error(pred, bound)
    return {"pred_value": pred, "gt_min": low, "gt_max": high, "in_range": False, "relative_error": err, "score": 1.0 / (1.0 + err)}


def normalize_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")


def parse_structured_range(value: Any) -> Any:
    try:
        return ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return None


def is_structured_range(value: Any) -> bool:
    text = str(value or "").strip()
    if not text or "{" not in text:
        return False
    parsed = parse_structured_range(text)
    return parsed is None or isinstance(parsed, (dict, list, tuple))


def leaf_numbers(value: Any) -> list[float]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    text = re.sub(r"(?<=\d)-(?=\d)", " to ", str(value))
    nums = extract_number_list(text)
    if not nums:
        return []
    if "integrated over" in text.lower():
        return nums[:1]
    return nums[:2]


def collect_structured_leaves(value: Any, path: tuple[str, ...] = ()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from collect_structured_leaves(child, path + (str(key),))
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from collect_structured_leaves(child, path)
    else:
        yield path, value


def extract_structured_derived_range(value: Any) -> tuple[float, float] | None:
    parsed = parse_structured_range(value)
    if parsed is None:
        return None
    values_by_key: dict[str, list[float]] = {}
    for path, leaf in collect_structured_leaves(parsed):
        if not path:
            continue
        key = normalize_key(path[-1])
        if key in TASK5_SKIPPED_STRUCTURED_RESULT_KEYS:
            continue
        nums = leaf_numbers(leaf)
        if nums:
            values_by_key.setdefault(key, []).extend(nums)
    if len(values_by_key) != 1:
        return None
    nums = next(iter(values_by_key.values()))
    return min(nums), max(nums)


def with_scores(item: dict[str, Any], s_wsv: float | None, s_r: float | None) -> dict[str, Any]:
    s_value = None if s_wsv is None or s_r is None else s_wsv * s_r
    output = dict(item)
    output.update({"S_wsv": s_wsv, "S_r": s_r, "S": s_value})
    return output


def build_token_variants(token: str) -> set[str]:
    token = normalize_text(token)
    variants = {token}
    if not token:
        return variants
    rewrite_rules = [
        ("ibility", "ible"), ("ability", "able"), ("ility", "le"), ("encies", "ent"), ("ency", ""),
        ("ances", ""), ("ance", ""), ("ences", ""), ("ence", ""), ("ization", "ize"),
        ("isation", "ise"), ("ation", "ate"), ("ition", "ite"), ("ption", "b"), ("sion", ""),
        ("tion", ""), ("ments", ""), ("ment", ""), ("ness", ""), ("ities", "y"),
        ("ity", ""), ("ships", ""), ("ship", ""), ("ings", ""), ("ing", ""),
        ("ers", ""), ("er", ""), ("eds", ""), ("ed", ""), ("ables", "able"),
        ("able", ""), ("ibles", "ible"), ("ible", ""), ("ives", "ive"), ("ive", ""),
        ("ials", "ial"), ("ial", ""), ("als", "al"), ("al", ""), ("ous", ""),
        ("ful", ""), ("less", ""), ("ic", ""), ("ly", ""), ("ies", "y"),
        ("es", ""), ("s", ""),
    ]
    changed = True
    while changed:
        changed = False
        current = list(variants)
        for value in current:
            for suffix, repl in rewrite_rules:
                if len(value) <= len(suffix) + 2:
                    continue
                if value.endswith(suffix):
                    candidate = value[: -len(suffix)] + repl
                    candidate = candidate.strip()
                    if candidate and candidate not in variants:
                        variants.add(candidate)
                        changed = True
    return {variant for variant in variants if len(variant) >= 3}


def tokens_match(left: str, right: str) -> bool:
    left_variants = build_token_variants(left)
    right_variants = build_token_variants(right)
    if left_variants & right_variants:
        return True
    for left_variant in left_variants:
        for right_variant in right_variants:
            if left_variant == right_variant:
                return True
            if len(left_variant) >= 4 and len(right_variant) >= 4:
                if left_variant in right_variant or right_variant in left_variant:
                    return True
            if len(left_variant) >= 5 and len(right_variant) >= 5:
                if difflib.SequenceMatcher(None, left_variant, right_variant).ratio() >= 0.84:
                    return True
    return False


def text_contains_any_synonym(pred_text: str, canonical: str, synonym_dict: dict[str, list[str]]) -> bool:
    pred_norm = normalize_text(pred_text)
    canonical_norm = normalize_text(canonical)
    if not pred_norm or not canonical_norm:
        return False
    candidates = [canonical_norm]
    if canonical_norm in synonym_dict:
        candidates.extend(synonym_dict[canonical_norm])
    pred_parts = [part for part in re.split(r"[^a-z0-9]+", pred_norm) if part]
    for candidate in candidates:
        if not candidate:
            continue
        if candidate in pred_norm:
            return True
        if any(part and (part in candidate or candidate in part) for part in pred_parts):
            return True
    return False


def load_task_tables(submit_root: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    for task in sorted(TASKS):
        path = submit_root / "data" / task / f"{task}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing task parquet: {path}")
        tables[task] = pd.read_parquet(path)
    return tables


def get_task_row(tables: dict[str, pd.DataFrame], task: str, index: Any) -> pd.Series:
    if task not in tables:
        raise ValueError(f"Unknown task_source={task!r}; expected one of {sorted(TASKS)}")
    try:
        row_index = int(index)
    except Exception as exc:
        raise ValueError(f"index must be an integer row index, got {index!r}") from exc
    df = tables[task]
    if row_index < 0 or row_index >= len(df):
        raise IndexError(f"index={row_index} out of range for {task} with {len(df)} rows")
    return df.iloc[row_index]


def first_positive_number(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if isfinite(numeric) and numeric > 0:
            return numeric
    return None


def task2_image_dimensions(item: dict[str, Any], row: pd.Series) -> tuple[float, float]:
    width = first_positive_number(item.get("image_width"), row.get("image_width"), 1280.0)
    height = first_positive_number(item.get("image_height"), row.get("image_height"), 720.0)
    return float(width), float(height)


def evaluate_prediction(
    item: dict[str, Any],
    row: pd.Series,
    task2_bbox_coordinate_mode: str = "auto",
) -> dict[str, Any]:
    task = str(row["task_source"])
    intermediate = item.get("wsv_answer")
    final = item["final_answer"].strip()
    if task == "task1":
        synonyms = json_loads(row.get("target_quantity_synonyms_json"), [])
        target_metrics = task1_target_score(synonyms, intermediate)
        answer_correct = task1_judge_extracted_answer(
            pred_answer=final,
            gt_answer=row.get("gt_answer"),
            gt_answer_option=row.get("answer_option"),
        )
        return with_scores(item, float(target_metrics["target_correct_max"]), float(answer_correct))

    if task == "task2":
        pred_bbox_raw = parse_bbox(intermediate)
        image_width, image_height = task2_image_dimensions(item, row)
        coordinate_mode = task2_bbox_coordinate_mode
        pred_bbox, _ = task2_bbox_to_absolute(
            pred_bbox_raw,
            image_width,
            image_height,
            coordinate_mode,
        )
        gt_bbox = json_loads(row.get("gt_bbox_json"), [])
        iou = bbox_iou(pred_bbox, gt_bbox)
        pred_answer = final.upper()
        gt_letters = [str(value).strip().upper() for value in json_loads(row.get("gt_answer_letters_json"), [])]
        answer_correct = pred_answer in gt_letters
        return with_scores(item, iou, float(answer_correct))

    if task == "task3":
        pred_interval = parse_interval(intermediate)
        gt_interval = json_loads(row.get("gt_interval_json"), [])
        iou = interval_iou(pred_interval, gt_interval)
        pred_answer = final.upper()
        answer_correct = pred_answer == str(row.get("gt_answer_option")).strip().upper()
        return with_scores(item, iou, float(answer_correct))

    if task == "task4":
        if not isinstance(intermediate, dict):
            raise ValueError("task4 wsv_answer must be an object in the clear-data format")
        pred_property = str(intermediate.get("property") or "")
        pred_object0_material = str(intermediate.get("object0_material") or "")
        pred_object1_material = str(intermediate.get("object1_material") or "")
        # ``new_property`` is the historical synonym set used by scoring.
        # Parquet may expose it as a numpy array rather than JSON text.
        property_value = row.get("new_property")
        if hasattr(property_value, "tolist"):
            property_value = property_value.tolist()
        gt_properties = json_loads(property_value, json_loads(row.get("gt_properties"), []))
        gt_obj0 = json_loads(row.get("gt_object0_materials"), json_loads(row.get("gt_object0_materials_json"), []))
        gt_obj1 = json_loads(row.get("gt_object1_materials"), json_loads(row.get("gt_object1_materials_json"), []))
        gt_obj0_materials = []
        gt_obj1_materials = []
        for obj in gt_obj0 if isinstance(gt_obj0, list) else [gt_obj0]:
            if isinstance(obj, dict):
                mats = obj.get("materials")
                if hasattr(mats, "tolist"):
                    mats = mats.tolist()
                if isinstance(mats, list):
                    gt_obj0_materials.extend(str(v) for v in mats)
                elif mats is not None:
                    gt_obj0_materials.append(str(mats))
            elif obj is not None:
                gt_obj0_materials.append(str(obj))
        for obj in gt_obj1 if isinstance(gt_obj1, list) else [gt_obj1]:
            if isinstance(obj, dict):
                mats = obj.get("materials")
                if hasattr(mats, "tolist"):
                    mats = mats.tolist()
                if isinstance(mats, list):
                    gt_obj1_materials.extend(str(v) for v in mats)
                elif mats is not None:
                    gt_obj1_materials.append(str(mats))
            elif obj is not None:
                gt_obj1_materials.append(str(obj))
        # ``gt_label`` was the historical name; submitted task4 data now
        # keeps the equivalent value under the canonical ``gt_answer`` name.
        gt_answer = str(row.get("gt_label", row.get("gt_answer"))).strip()
        answer_correct = final == gt_answer
        prop_correct = task4_property_match(pred_property, gt_properties)
        obj0_correct = task4_material_match(pred_object0_material, {"materials": gt_obj0_materials})
        obj1_correct = task4_material_match(pred_object1_material, {"materials": gt_obj1_materials})
        wsv = float(prop_correct) * 0.4 + float(obj0_correct) * 0.3 + float(obj1_correct) * 0.3
        return with_scores(item, wsv, float(answer_correct))

    if task == "task5":
        property_score = score_against_range(intermediate, row.get("gt_property_range"))
        raw_answer_range = row.get("gt_answer_range")
        answer_range = raw_answer_range
        is_exp = str(row.get("exp") or "").strip().lower() in {"1", "true", "yes"}
        if is_exp and is_structured_range(raw_answer_range):
            structured_range = extract_structured_derived_range(raw_answer_range)
            if structured_range is not None:
                answer_range = f"[{structured_range[0]}, {structured_range[1]}]"
        answer_score = score_against_range(final, answer_range)
        return with_scores(item, property_score["score"], answer_score["score"])

    raise ValueError(f"Unsupported task_source={task!r}")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_judged.jsonl")


def valid_metric_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(field)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def mean_or_none(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"count": len(rows)}
    for field in SUMMARY_FIELDS:
        values = valid_metric_values(rows, field)
        out[field] = mean_or_none(values)
    return out


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"by_task": {}, "Avg": {}}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        groups[str(item.get("task_source"))].append(item)
    for task in sorted(groups):
        summary["by_task"][task] = metric_summary(groups[task])

    task_stats = [summary["by_task"][task] for task in sorted(groups)]
    summary["Avg"] = {
        "count": len(task_stats),
        **{
            field: mean_or_none([stats[field] for stats in task_stats if stats[field] is not None])
            for field in SUMMARY_FIELDS
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate answer-only task1-task5 prediction JSONL.")
    parser.add_argument("input_jsonl", help="Prediction JSONL in the clear-data format.")
    parser.add_argument("--submit-root", default=str(Path(__file__).resolve().parents[1]), help="Path to submit root.")
    parser.add_argument("--output", help="Output judged JSONL path. Defaults to <input>_judged.jsonl.")
    parser.add_argument("--summary-output", help="Optional summary JSON path.")
    parser.add_argument(
        "--task2-bbox-coordinate-mode",
        choices=("auto", "absolute", "relative_1", "relative_1000"),
        default="auto",
        help=(
            "How to interpret task2 predicted bboxes. 'auto' selects the format "
            "with the strongest overlap after converting candidates."
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output) if args.output else default_output_path(input_path)
    submit_root = Path(args.submit_root)
    tables = load_task_tables(submit_root)

    results = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for line_no, item in iter_jsonl(input_path):
            task = item.get("task_source")
            if task not in TASKS:
                raise ValueError(f"{input_path}:{line_no} invalid task_source={task!r}")
            if not isinstance(item.get("final_answer"), str):
                raise ValueError(f"{input_path}:{line_no} final_answer must be a string")
            row = get_task_row(tables, task, item.get("index"))
            judged = evaluate_prediction(item, row, args.task2_bbox_coordinate_mode)
            results.append(judged)
            out.write(json.dumps(to_jsonable(judged), ensure_ascii=False) + "\n")

    summary = summarize(results)
    summary_text = json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2)
    print(summary_text)
    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(summary_text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
