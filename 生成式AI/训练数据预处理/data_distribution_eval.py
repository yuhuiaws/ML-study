#!/usr/bin/env python3
"""
Data distribution evaluation for Bedrock-format SFT conversation JSONL.

Pipeline:
  1. Metadata annotation (optional, via AWS Bedrock Claude):
       Annotate each sample with language, task_type, difficulty if not present.
  2. Multi-dimensional distribution statistics:
       - Language distribution (multilingual)
       - Task type distribution (multi-task)
       - Character/token length distribution (input & output)
       - Difficulty distribution (low / medium / high)
  3. Cross-dimensional analysis (e.g. language × task, language × difficulty).
  4. Alert system with configurable thresholds.
  5. Validation set splitting (stratified by distribution).
  6. Automated report generation (JSON + optional matplotlib visualizations).

All parameters are exposed via CLI flags (see --help).
"""

import argparse
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as _fm
    _fm._load_fontmanager(try_read_cache=False)
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"] + plt.rcParams["font.sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ============================================================
# Debug log collector
# ============================================================

_debug_lines: List[str] = []


def _debug(msg: str) -> None:
    """Append a debug message to the collector and print it."""
    _debug_lines.append(msg)
    print(msg)


# ============================================================
# Configuration
# ============================================================

@dataclass
class EvalConfig:
    """All tuneable parameters for the data distribution evaluation."""

    # -- I/O --
    input: str = "./zh_mixed_filtered.jsonl"
    output_annotated: str = "./zh_mixed_annotated.jsonl"
    report_json: str = "./distribution_report.json"
    report_dir: str = "./distribution_plots"

    # -- Metadata annotation --
    enable_annotation: bool = True
    annotation_model_id: str = "global.anthropic.claude-opus-4-6-v1"
    bedrock_region: str = "us-east-1"

    # -- Validation split --
    enable_val_split: bool = False
    val_ratio: float = 0.1
    val_output: str = "./val_split.jsonl"
    train_output: str = "./train_split.jsonl"
    random_seed: int = 42

    # -- Enable visualization --
    enable_plots: bool = True

    # -- Language alert thresholds --
    lang_min_pct: float = 1.0          # target language < this % → alert
    lang_min_abs: int = 5000           # and absolute count < this → alert

    # -- Task alert thresholds --
    task_max_single_pct: float = 50.0  # single task > this % → alert
    task_min_pct: float = 3.0          # important task < this % → alert
    task_min_abs: int = 2000           # task count < this → alert
    multi_turn_min_pct: float = 20.0   # multi-turn ratio < this % → alert

    # -- Difficulty alert thresholds --
    difficulty_target: Dict[str, float] = field(
        default_factory=lambda: {"低": 30.0, "中": 50.0, "高": 20.0}
    )
    difficulty_tolerance: float = 15.0  # deviation > this % → alert

    # -- Length category boundaries (based on total_char_len) --
    length_short_max: int = 200         # total_char_len <= this → "短"
    length_long_min: int = 1000         # total_char_len >= this → "长"
    # everything in between → "中"

    # -- Length category alert thresholds --
    length_cat_target: Dict[str, float] = field(
        default_factory=lambda: {"短": 25.0, "中": 50.0, "长": 25.0}
    )
    length_cat_tolerance: float = 15.0  # deviation > this % → alert

    # -- Cross-dimension alert threshold --
    cross_min_count: int = 100          # cross cell < this → alert

    # -- Length percentiles to report --
    length_percentiles: List[int] = field(
        default_factory=lambda: [10, 25, 50, 75, 90, 99]
    )


# ============================================================
# Text extraction helpers
# ============================================================

def _get_msg_text(msg: dict) -> str:
    """Extract text from a Bedrock-format message content block."""
    content = msg.get("content", [])
    if content and isinstance(content, list):
        return content[0].get("text", "")
    return ""


def extract_input_text(conv: dict) -> str:
    """Concatenate all user turn texts."""
    parts = []
    for msg in conv.get("messages", []):
        if msg.get("role") == "user":
            text = _get_msg_text(msg)
            if text:
                parts.append(text)
    return "\n".join(parts)


def extract_output_text(conv: dict) -> str:
    """Concatenate all assistant turn texts."""
    parts = []
    for msg in conv.get("messages", []):
        if msg.get("role") == "assistant":
            text = _get_msg_text(msg)
            if text:
                parts.append(text)
    return "\n".join(parts)


def extract_full_text(conv: dict) -> str:
    """Concatenate all message texts."""
    parts = []
    for msg in conv.get("messages", []):
        text = _get_msg_text(msg)
        if text:
            parts.append(text)
    return "\n".join(parts)


def count_turns(conv: dict) -> int:
    """Count the number of messages in a conversation."""
    return len(conv.get("messages", []))


def is_multi_turn(conv: dict) -> bool:
    """Return True if conversation has more than 2 messages (1 user + 1 assistant)."""
    return count_turns(conv) > 2


# ============================================================
# Language detection (heuristic, no LLM needed)
# ============================================================

def detect_language_heuristic(text: str) -> str:
    """Detect language heuristically based on script analysis.

    Returns one of: "中文", "英文", "中英混合", "其他"
    """
    if not text.strip():
        return "其他"

    cjk_count = 0
    latin_count = 0
    total_alpha = 0

    for ch in text:
        cp = ord(ch)
        # CJK Unified Ideographs
        if (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF
                or 0x20000 <= cp <= 0x2A6DF or 0x2A700 <= cp <= 0x2B73F):
            cjk_count += 1
            total_alpha += 1
        # Basic Latin letters
        elif (0x0041 <= cp <= 0x005A or 0x0061 <= cp <= 0x007A):
            latin_count += 1
            total_alpha += 1

    if total_alpha == 0:
        return "其他"

    cjk_ratio = cjk_count / total_alpha
    latin_ratio = latin_count / total_alpha

    if cjk_ratio > 0.7:
        return "中文"
    elif latin_ratio > 0.7:
        return "英文"
    elif cjk_count > 0 and latin_count > 0:
        return "中英混合"
    else:
        return "其他"


# ============================================================
# Character-length statistics
# ============================================================

def compute_char_length(text: str) -> int:
    """Compute character length (Unicode)."""
    return len(text)


def classify_length_category(total_char_len: int,
                              short_max: int = 200,
                              long_min: int = 1000) -> str:
    """Classify a sample's total character length into a category.

    Returns one of: "短", "中", "长".
    - total_char_len <= short_max → "短"
    - total_char_len >= long_min  → "长"
    - otherwise                   → "中"
    """
    if total_char_len <= short_max:
        return "短"
    elif total_char_len >= long_min:
        return "长"
    else:
        return "中"


def compute_percentiles(values: List[float],
                        percentiles: List[int]) -> Dict[str, float]:
    """Compute specified percentiles for a list of values.

    Returns dict like {"P10": val, "P25": val, ...}.
    """
    if not values:
        return {f"P{p}": 0.0 for p in percentiles}

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    result = {}
    for p in percentiles:
        # Linear interpolation
        idx = (p / 100.0) * (n - 1)
        lower = int(math.floor(idx))
        upper = int(math.ceil(idx))
        if lower == upper:
            result[f"P{p}"] = sorted_vals[lower]
        else:
            frac = idx - lower
            result[f"P{p}"] = sorted_vals[lower] * (1 - frac) + sorted_vals[upper] * frac
    return result


# ============================================================
# Bedrock LLM annotation
# ============================================================

def _call_bedrock(client, model_id: str, prompt: str,
                  max_tokens: int = 1024) -> str:
    """Call Bedrock Claude API and return response text.

    Uses the Anthropic Messages API format via invoke_model.
    Raises on any API or parsing error (caller decides whether to retry/fallback).
    """
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
    })
    response = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"].strip()


def validate_bedrock_connection(client, model_id: str) -> None:
    """Send a trivial request to validate that the Bedrock client and model work.

    Raises RuntimeError with a diagnostic message on failure.
    """
    try:
        reply = _call_bedrock(client, model_id, "请回复OK", max_tokens=16)
    except Exception as e:
        err_type = type(e).__name__
        raise RuntimeError(
            f"Bedrock 连通性测试失败。\n"
            f"  model_id : {model_id}\n"
            f"  error    : {err_type}: {e}\n"
            f"请检查:\n"
            f"  1. AWS 凭证是否已配置 (aws configure / 环境变量 / IAM role)\n"
            f"  2. model_id 是否正确，当前区域是否有权限访问该模型\n"
            f"  3. bedrock-runtime endpoint 在当前区域是否可用"
        ) from e
    _debug(f"Bedrock 连通性测试通过 (model={model_id}, reply={reply!r})")


def parse_json_response(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()

    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass

    return {}


def _format_conv_for_annotation(conv: dict) -> str:
    """Build a compact text representation of a conversation for annotation prompt."""
    lines = []
    sys_texts = conv.get("system", [])
    if sys_texts:
        sys_str = sys_texts[0].get("text", "") if sys_texts else ""
        if sys_str:
            lines.append(f"[系统] {sys_str}")
    for msg in conv.get("messages", []):
        role = msg.get("role", "")
        text = _get_msg_text(msg)
        if role == "user":
            lines.append(f"用户: {text[:500]}")
        elif role == "assistant":
            lines.append(f"助手: {text[:500]}")
    return "\n".join(lines)


_ANNOTATION_PROMPT_TEMPLATE = (
    "你是一个SFT训练数据分析专家。请分析以下对话样本，给出三个维度的标注：\n\n"
    "1. language（语言）：从以下选项中选择一个：中文、英文、中英混合、其他\n"
    "2. task_type（任务类型）：从以下选项中选择一个：\n"
    "   QA问答、多轮闲聊、情感分类、文本摘要、代码生成、数学推理、"
    "翻译、创意写作、信息提取、逻辑推理、角色扮演、知识问答、其他\n"
    "3. difficulty（难度）：从以下选项中选择一个：低、中、高\n"
    "   - 低：简单事实性回答、简短闲聊\n"
    "   - 中：需要一定推理或组织的回答\n"
    "   - 高：复杂推理、多步骤任务、专业知识\n\n"
    "对话内容:\n{conv_text}\n\n"
    "请以JSON格式返回，格式如下:\n"
    '{{"language": "中文", "task_type": "QA问答", "difficulty": "中"}}\n\n'
    "只返回JSON,不要其他文字。"
)

_REQUIRED_ANNOTATION_KEYS = ("language", "task_type", "difficulty")


def annotate_sample(conv: dict, client, model_id: str) -> Dict[str, str]:
    """Annotate a single sample with language, task_type, difficulty via LLM.

    Returns dict with keys: language, task_type, difficulty.
    On API or parse failure, falls back to heuristic and logs the error.
    """
    conv_text = _format_conv_for_annotation(conv)
    prompt = _ANNOTATION_PROMPT_TEMPLATE.format(conv_text=conv_text)

    # --- Step 1: call Bedrock API ---
    try:
        reply = _call_bedrock(client, model_id, prompt, max_tokens=256)
    except Exception as e:
        _debug(f"  [WARN] Bedrock API 调用失败: {type(e).__name__}: {e}")
        return _fallback_annotation(conv)

    # --- Step 2: parse JSON response ---
    parsed = parse_json_response(reply)
    if not parsed:
        _debug(f"  [WARN] LLM 返回无法解析为JSON: {reply[:200]!r}")
        return _fallback_annotation(conv)

    missing = [k for k in _REQUIRED_ANNOTATION_KEYS if k not in parsed]
    if missing:
        _debug(f"  [WARN] LLM 返回缺少字段 {missing}: {parsed}")
        return _fallback_annotation(conv)

    return {
        "language": parsed["language"],
        "task_type": parsed["task_type"],
        "difficulty": parsed["difficulty"],
    }


def _fallback_annotation(conv: dict) -> Dict[str, str]:
    """Fallback: heuristic language detection, default task/difficulty."""
    full_text = extract_full_text(conv)
    return {
        "language": detect_language_heuristic(full_text),
        "task_type": "其他",
        "difficulty": "中",
    }


def annotate_batch(conversations: List[dict], client,
                   model_id: str) -> Tuple[List[Dict[str, str]], int]:
    """Annotate a batch of conversations.

    Returns (list of metadata dicts, count of successful LLM annotations).
    """
    results = []
    llm_ok = 0
    for conv in conversations:
        meta = annotate_sample(conv, client, model_id)
        results.append(meta)
        # If task_type != "其他" or difficulty != "中", it likely came from LLM
        if meta["task_type"] != "其他":
            llm_ok += 1
    return results, llm_ok


# ============================================================
# Metadata extraction (from existing or newly annotated)
# ============================================================

def extract_metadata(conv: dict,
                     length_short_max: int = 200,
                     length_long_min: int = 1000) -> Dict[str, Any]:
    """Extract metadata from a conversation.

    If metadata fields exist in conv["metadata"], use them.
    Otherwise compute what we can (language heuristic, length).
    length_short_max / length_long_min control length_category boundaries.
    """
    existing = conv.get("metadata", {})
    full_text = extract_full_text(conv)
    input_text = extract_input_text(conv)
    output_text = extract_output_text(conv)
    total_len = compute_char_length(full_text)

    return {
        "language": existing.get("language", detect_language_heuristic(full_text)),
        "task_type": existing.get("task_type", "未标注"),
        "difficulty": existing.get("difficulty", "未标注"),
        "input_char_len": compute_char_length(input_text),
        "output_char_len": compute_char_length(output_text),
        "total_char_len": total_len,
        "num_turns": count_turns(conv),
        "is_multi_turn": is_multi_turn(conv),
        "length_category": classify_length_category(
            total_len, length_short_max, length_long_min),
    }


# ============================================================
# Distribution statistics
# ============================================================

def compute_language_distribution(metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute language distribution statistics.

    Returns dict with counts, percentages, and total chars per language.
    """
    lang_counter = Counter()
    lang_chars = defaultdict(int)

    for m in metadata_list:
        lang = m["language"]
        lang_counter[lang] += 1
        lang_chars[lang] += m["total_char_len"]

    total = len(metadata_list)
    distribution = {}
    for lang, count in lang_counter.most_common():
        distribution[lang] = {
            "count": count,
            "percentage": round(count / total * 100, 2) if total > 0 else 0.0,
            "total_chars": lang_chars[lang],
        }

    return {"total_samples": total, "distribution": distribution}


def compute_task_distribution(metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute task type distribution statistics."""
    task_counter = Counter()
    for m in metadata_list:
        task_counter[m["task_type"]] += 1

    total = len(metadata_list)
    multi_turn_count = sum(1 for m in metadata_list if m["is_multi_turn"])
    multi_turn_pct = round(multi_turn_count / total * 100, 2) if total > 0 else 0.0

    distribution = {}
    for task, count in task_counter.most_common():
        distribution[task] = {
            "count": count,
            "percentage": round(count / total * 100, 2) if total > 0 else 0.0,
        }

    return {
        "total_samples": total,
        "distribution": distribution,
        "multi_turn_count": multi_turn_count,
        "multi_turn_percentage": multi_turn_pct,
    }


def compute_length_distribution(metadata_list: List[Dict[str, Any]],
                                percentiles: List[int]) -> Dict[str, Any]:
    """Compute character length distribution for input, output, total."""
    input_lens = [m["input_char_len"] for m in metadata_list]
    output_lens = [m["output_char_len"] for m in metadata_list]
    total_lens = [m["total_char_len"] for m in metadata_list]

    return {
        "input": {
            "mean": round(sum(input_lens) / len(input_lens), 2) if input_lens else 0.0,
            "min": min(input_lens) if input_lens else 0,
            "max": max(input_lens) if input_lens else 0,
            "percentiles": compute_percentiles(input_lens, percentiles),
        },
        "output": {
            "mean": round(sum(output_lens) / len(output_lens), 2) if output_lens else 0.0,
            "min": min(output_lens) if output_lens else 0,
            "max": max(output_lens) if output_lens else 0,
            "percentiles": compute_percentiles(output_lens, percentiles),
        },
        "total": {
            "mean": round(sum(total_lens) / len(total_lens), 2) if total_lens else 0.0,
            "min": min(total_lens) if total_lens else 0,
            "max": max(total_lens) if total_lens else 0,
            "percentiles": compute_percentiles(total_lens, percentiles),
        },
    }


def compute_difficulty_distribution(metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute difficulty distribution statistics."""
    diff_counter = Counter()
    for m in metadata_list:
        diff_counter[m["difficulty"]] += 1

    total = len(metadata_list)
    distribution = {}
    for diff, count in diff_counter.most_common():
        distribution[diff] = {
            "count": count,
            "percentage": round(count / total * 100, 2) if total > 0 else 0.0,
        }

    return {"total_samples": total, "distribution": distribution}


def compute_length_category_distribution(
        metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute length category (短/中/长) distribution statistics."""
    cat_counter = Counter()
    for m in metadata_list:
        cat_counter[m["length_category"]] += 1

    total = len(metadata_list)
    distribution = {}
    for cat, count in cat_counter.most_common():
        distribution[cat] = {
            "count": count,
            "percentage": round(count / total * 100, 2) if total > 0 else 0.0,
        }

    return {"total_samples": total, "distribution": distribution}


# ============================================================
# Cross-dimensional analysis
# ============================================================

def compute_cross_distribution(metadata_list: List[Dict[str, Any]],
                               dim_a: str, dim_b: str) -> Dict[str, Any]:
    """Compute cross-distribution for two dimensions.

    Returns a nested dict: {dim_a_value: {dim_b_value: count}}.
    """
    cross = defaultdict(lambda: defaultdict(int))
    for m in metadata_list:
        val_a = m.get(dim_a, "未知")
        val_b = m.get(dim_b, "未知")
        cross[val_a][val_b] += 1

    # Convert to regular dict
    result = {}
    for a_val, b_counts in cross.items():
        result[a_val] = dict(b_counts)

    return {"dim_a": dim_a, "dim_b": dim_b, "cross": result}


def compute_4d_cross_distribution(
        metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute the 4-dimensional cross-distribution.

    Groups samples by (language, task_type, length_category, difficulty).
    Returns dict with dimensions, cells (sorted by count desc), total_samples,
    and num_cells (number of non-zero cells).
    """
    _DIMS = ("language", "task_type", "length_category", "difficulty")
    counter: Counter = Counter()
    for m in metadata_list:
        key = tuple(m.get(d, "未知") for d in _DIMS)
        counter[key] += 1

    cells = []
    for key, count in counter.most_common():
        cell = {d: v for d, v in zip(_DIMS, key)}
        cell["count"] = count
        cells.append(cell)

    return {
        "dimensions": list(_DIMS),
        "cells": cells,
        "total_samples": len(metadata_list),
        "num_cells": len(cells),
    }


# ============================================================
# Alert generation
# ============================================================

def check_language_alerts(lang_dist: Dict[str, Any],
                          cfg: EvalConfig) -> List[Dict[str, str]]:
    """Check language distribution for alerts."""
    alerts = []
    total = lang_dist["total_samples"]
    for lang, info in lang_dist["distribution"].items():
        if lang == "其他":
            continue
        pct = info["percentage"]
        count = info["count"]
        if pct < cfg.lang_min_pct and count < cfg.lang_min_abs:
            alerts.append({
                "level": "warning",
                "dimension": "language",
                "message": (
                    f"语言 '{lang}' 样本不足: "
                    f"数量={count} (<{cfg.lang_min_abs}), "
                    f"占比={pct}% (<{cfg.lang_min_pct}%). "
                    f"建议补充数据。"
                ),
            })
    return alerts


def check_task_alerts(task_dist: Dict[str, Any],
                      cfg: EvalConfig) -> List[Dict[str, str]]:
    """Check task distribution for alerts."""
    alerts = []
    total = task_dist["total_samples"]

    # Check multi-turn ratio
    if task_dist["multi_turn_percentage"] < cfg.multi_turn_min_pct:
        alerts.append({
            "level": "warning",
            "dimension": "task",
            "message": (
                f"多轮对话占比过低: {task_dist['multi_turn_percentage']}% "
                f"(<{cfg.multi_turn_min_pct}%). "
                f"建议补充多轮对话数据。"
            ),
        })

    # Check each task type
    for task, info in task_dist["distribution"].items():
        if task in ("未标注", "其他"):
            continue
        pct = info["percentage"]
        count = info["count"]

        # Single task dominance
        if pct > cfg.task_max_single_pct:
            alerts.append({
                "level": "warning",
                "dimension": "task",
                "message": (
                    f"任务类型 '{task}' 占比过高: {pct}% "
                    f"(>{cfg.task_max_single_pct}%). "
                    f"分布偏斜，建议下采样或补充其他类型。"
                ),
            })

        # Under-represented task
        if count < cfg.task_min_abs:
            alerts.append({
                "level": "warning",
                "dimension": "task",
                "message": (
                    f"任务类型 '{task}' 样本不足: "
                    f"数量={count} (<{cfg.task_min_abs}). "
                    f"建议补充数据。"
                ),
            })

    return alerts


def check_difficulty_alerts(diff_dist: Dict[str, Any],
                            cfg: EvalConfig) -> List[Dict[str, str]]:
    """Check difficulty distribution for alerts.

    Target ratio: 低:中:高 ≈ 3:5:2 (configurable).
    """
    alerts = []
    dist = diff_dist["distribution"]

    for level, target_pct in cfg.difficulty_target.items():
        actual_pct = dist.get(level, {}).get("percentage", 0.0)
        deviation = abs(actual_pct - target_pct)
        if deviation > cfg.difficulty_tolerance:
            direction = "过高" if actual_pct > target_pct else "过低"
            alerts.append({
                "level": "warning",
                "dimension": "difficulty",
                "message": (
                    f"难度 '{level}' 占比{direction}: "
                    f"实际={actual_pct}%, 目标={target_pct}%, "
                    f"偏差={deviation:.1f}% (>{cfg.difficulty_tolerance}%). "
                ),
            })

    return alerts


def check_length_category_alerts(
        len_cat_dist: Dict[str, Any],
        cfg: EvalConfig) -> List[Dict[str, str]]:
    """Check length category distribution for alerts.

    Target ratio: 短:中:长 ≈ 25:50:25 (configurable).
    """
    alerts = []
    dist = len_cat_dist["distribution"]

    for cat, target_pct in cfg.length_cat_target.items():
        actual_pct = dist.get(cat, {}).get("percentage", 0.0)
        deviation = abs(actual_pct - target_pct)
        if deviation > cfg.length_cat_tolerance:
            direction = "过高" if actual_pct > target_pct else "过低"
            alerts.append({
                "level": "warning",
                "dimension": "length_category",
                "message": (
                    f"长度类别 '{cat}' 占比{direction}: "
                    f"实际={actual_pct}%, 目标={target_pct}%, "
                    f"偏差={deviation:.1f}% (>{cfg.length_cat_tolerance}%). "
                ),
            })

    return alerts


def check_cross_alerts(cross_4d: Dict[str, Any],
                       min_count: int = 100) -> List[Dict[str, str]]:
    """Check 4D cross-dimensional distribution for sparse cells.

    Accepts the structure returned by compute_4d_cross_distribution:
    {"dimensions": [...], "cells": [{"language": ..., ..., "count": N}, ...], ...}
    """
    alerts = []
    dims = cross_4d["dimensions"]
    dim_label = "×".join(dims)

    for cell in cross_4d["cells"]:
        count = cell["count"]
        if count < min_count:
            parts = " × ".join(f"{d}='{cell[d]}'" for d in dims)
            alerts.append({
                "level": "info",
                "dimension": dim_label,
                "message": (
                    f"{parts} "
                    f"样本稀疏: 数量={count} (<{min_count}). "
                    f"交叉维度可能需要补充。"
                ),
            })

    return alerts


# ============================================================
# Validation set splitting
# ============================================================

def stratified_split(metadata_list: List[Dict[str, Any]],
                     val_ratio: float = 0.1,
                     seed: int = 42) -> Tuple[List[int], List[int]]:
    """Split indices into train and val sets, stratified by all cross dimensions.

    Stratification key: language × task_type × difficulty × length_category.
    Returns (train_indices, val_indices).
    """
    rng = random.Random(seed)

    # Group indices by stratification key (all 4 dimensions)
    strata: Dict[str, List[int]] = defaultdict(list)
    for i, m in enumerate(metadata_list):
        key = (f"{m.get('language', '未知')}"
               f"_{m.get('task_type', '未知')}"
               f"_{m.get('difficulty', '未知')}"
               f"_{m.get('length_category', '未知')}")
        strata[key].append(i)

    train_indices = []
    val_indices = []

    for key, indices in strata.items():
        rng.shuffle(indices)
        n_val = max(1, int(len(indices) * val_ratio))
        if len(indices) <= 1:
            # Too few samples: put all in train
            train_indices.extend(indices)
        else:
            val_indices.extend(indices[:n_val])
            train_indices.extend(indices[n_val:])

    return sorted(train_indices), sorted(val_indices)


# ============================================================
# Visualization
# ============================================================

def plot_bar_chart(data: Dict[str, int], title: str, xlabel: str,
                   ylabel: str, filepath: str) -> None:
    """Plot a bar chart and save to file."""

    print("matplotlib is ", HAS_MATPLOTLIB)
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    labels = list(data.keys())
    values = list(data.values())

    bars = ax.bar(labels, values, color="#4a90d9", edgecolor="white")

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                str(val), ha="center", va="bottom", fontsize=9)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


def plot_histogram(values: List[float], title: str, xlabel: str,
                   filepath: str, bins: int = 50) -> None:
    """Plot a histogram and save to file."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(values, bins=bins, color="#4a90d9", edgecolor="white", alpha=0.8)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("数量", fontsize=11)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


def plot_cross_heatmap(cross_data: Dict[str, Dict[str, int]],
                       dim_a_label: str, dim_b_label: str,
                       title: str, filepath: str) -> None:
    """Plot a cross-distribution heatmap and save to file."""
    if not HAS_MATPLOTLIB:
        return

    a_keys = sorted(cross_data.keys())
    b_keys_set = set()
    for b_counts in cross_data.values():
        b_keys_set.update(b_counts.keys())
    b_keys = sorted(b_keys_set)

    if not a_keys or not b_keys:
        return

    matrix = []
    for a in a_keys:
        row = []
        for b in b_keys:
            row.append(cross_data.get(a, {}).get(b, 0))
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(max(8, len(b_keys) * 1.2),
                                     max(4, len(a_keys) * 0.8)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(b_keys)))
    ax.set_xticklabels(b_keys, rotation=45, ha="right")
    ax.set_yticks(range(len(a_keys)))
    ax.set_yticklabels(a_keys)

    # Add text annotations
    for i in range(len(a_keys)):
        for j in range(len(b_keys)):
            ax.text(j, i, str(matrix[i][j]),
                    ha="center", va="center", fontsize=8)

    ax.set_xlabel(dim_b_label, fontsize=11)
    ax.set_ylabel(dim_a_label, fontsize=11)
    ax.set_title(title, fontsize=14)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


def generate_plots(metadata_list: List[Dict[str, Any]],
                   lang_dist: Dict[str, Any],
                   task_dist: Dict[str, Any],
                   diff_dist: Dict[str, Any],
                   len_cat_dist: Dict[str, Any],
                   report_dir: str) -> List[str]:
    """Generate all visualization plots. Returns list of file paths.

    Computes 2D pairwise cross-distributions internally for heatmap rendering.
    """
    if not HAS_MATPLOTLIB:
        return []

    os.makedirs(report_dir, exist_ok=True)
    paths = []

    print("in the generate plot!!")
    # 1. Language distribution bar chart
    lang_counts = {k: v["count"] for k, v in lang_dist["distribution"].items()}
    path = os.path.join(report_dir, "language_distribution.png")
    plot_bar_chart(lang_counts, "语言分布", "语言", "样本数", path)
    paths.append(path)

    # 2. Task distribution bar chart
    task_counts = {k: v["count"] for k, v in task_dist["distribution"].items()}
    path = os.path.join(report_dir, "task_distribution.png")
    plot_bar_chart(task_counts, "任务类型分布", "任务类型", "样本数", path)
    paths.append(path)

    # 3. Difficulty distribution bar chart
    diff_counts = {k: v["count"] for k, v in diff_dist["distribution"].items()}
    path = os.path.join(report_dir, "difficulty_distribution.png")
    plot_bar_chart(diff_counts, "难度分布", "难度", "样本数", path)
    paths.append(path)

    # 4. Length category distribution bar chart
    len_cat_counts = {k: v["count"] for k, v in len_cat_dist["distribution"].items()}
    path = os.path.join(report_dir, "length_category_distribution.png")
    plot_bar_chart(len_cat_counts, "长度类别分布", "长度类别", "样本数", path)
    paths.append(path)

    # 5. Input length histogram
    input_lens = [m["input_char_len"] for m in metadata_list]
    path = os.path.join(report_dir, "input_length_histogram.png")
    plot_histogram(input_lens, "输入长度分布（字符）", "字符数", path)
    paths.append(path)

    # 6. Output length histogram
    output_lens = [m["output_char_len"] for m in metadata_list]
    path = os.path.join(report_dir, "output_length_histogram.png")
    plot_histogram(output_lens, "输出长度分布（字符）", "字符数", path)
    paths.append(path)

    # 7. All pairwise 2D cross-distribution heatmaps (computed on the fly)
    _dims = ["language", "task_type", "difficulty", "length_category"]
    _dim_labels = {
        "language": "语言", "task_type": "任务类型",
        "difficulty": "难度", "length_category": "长度类别",
    }
    for i in range(len(_dims)):
        for j in range(i + 1, len(_dims)):
            cross_dist = compute_cross_distribution(
                metadata_list, _dims[i], _dims[j])
            dim_a = cross_dist["dim_a"]
            dim_b = cross_dist["dim_b"]
            label_a = _dim_labels.get(dim_a, dim_a)
            label_b = _dim_labels.get(dim_b, dim_b)
            filename = f"cross_{dim_a}_{dim_b}.png"
            path = os.path.join(report_dir, filename)
            plot_cross_heatmap(cross_dist["cross"], label_a, label_b,
                               f"{label_a} × {label_b} 交叉分布", path)
            paths.append(path)

    return paths


# ============================================================
# Pipeline orchestration
# ============================================================

def run(cfg: EvalConfig, bedrock_client=None) -> Dict[str, Any]:
    """Execute the full distribution evaluation pipeline.

    Returns the complete report dict.
    """

    # 0. Setup debug/report directories and timestamp
    global _debug_lines
    _debug_lines = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    debug_dir = os.path.join(script_dir, "debug-log")
    report_out_dir = os.path.join(script_dir, "report")
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(report_out_dir, exist_ok=True)

    _debug(f"=== Data Distribution Eval Debug Log ===")
    _debug(f"Timestamp: {timestamp}")
    _debug(f"Input:  {cfg.input}")
    _debug(f"{'=' * 60}")

    # 1. Load conversations
    conversations: List[dict] = []
    with open(cfg.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                conversations.append(json.loads(line))

    total = len(conversations)
    _debug(f"Loaded {total} conversations from {cfg.input}")

    # 2. Annotate metadata if enabled
    if cfg.enable_annotation:
        # Check if any sample lacks metadata
        needs_annotation = []
        for i, conv in enumerate(conversations):
            meta = conv.get("metadata", {})
            if not all(k in meta for k in ("language", "task_type", "difficulty")):
                needs_annotation.append(i)

        if needs_annotation:
            _debug(f"Annotating {len(needs_annotation)} samples via LLM...")
            _debug(f"  model_id : {cfg.annotation_model_id}")
            _debug(f"  region   : {cfg.bedrock_region}")

            if bedrock_client is None:
                if not HAS_BOTO3:
                    raise ImportError(
                        "boto3 is required for annotation. "
                        "Install it with: pip install boto3"
                    )
                bedrock_client = boto3.client(
                    "bedrock-runtime", region_name=cfg.bedrock_region
                )

            # Validate connection before processing all samples
            validate_bedrock_connection(bedrock_client, cfg.annotation_model_id)

            llm_ok = 0
            for idx_num, i in enumerate(needs_annotation):
                meta = annotate_sample(conversations[i], bedrock_client,
                                       cfg.annotation_model_id)
                if "metadata" not in conversations[i]:
                    conversations[i]["metadata"] = {}
                conversations[i]["metadata"].update(meta)
                # Track LLM success (fallback sets task_type="其他")
                if meta["task_type"] != "其他":
                    llm_ok += 1
                if (idx_num + 1) % 50 == 0:
                    _debug(f"  progress: {idx_num + 1}/{len(needs_annotation)} "
                           f"(LLM OK: {llm_ok})")

            _debug(f"Annotation done: {llm_ok}/{len(needs_annotation)} "
                   f"via LLM, {len(needs_annotation) - llm_ok} via heuristic fallback")

            # Save annotated data
            with open(cfg.output_annotated, "w", encoding="utf-8") as fout:
                for conv in conversations:
                    fout.write(json.dumps(conv, ensure_ascii=False) + "\n")
            _debug(f"Annotated data saved to {cfg.output_annotated}")
        else:
            _debug("All samples already have metadata, skipping annotation.")

    # 3. Extract metadata for all samples
    metadata_list = [extract_metadata(conv, cfg.length_short_max, cfg.length_long_min)
                     for conv in conversations]

    # 4. Compute distributions
    lang_dist = compute_language_distribution(metadata_list)
    task_dist = compute_task_distribution(metadata_list)
    length_dist = compute_length_distribution(metadata_list, cfg.length_percentiles)
    diff_dist = compute_difficulty_distribution(metadata_list)
    len_cat_dist = compute_length_category_distribution(metadata_list)

    # 5. 4D cross-dimensional analysis
    cross_4d = compute_4d_cross_distribution(metadata_list)

    # 6. Generate alerts
    alerts = []
    alerts.extend(check_language_alerts(lang_dist, cfg))
    alerts.extend(check_task_alerts(task_dist, cfg))
    alerts.extend(check_difficulty_alerts(diff_dist, cfg))
    alerts.extend(check_length_category_alerts(len_cat_dist, cfg))
    alerts.extend(check_cross_alerts(cross_4d, cfg.cross_min_count))

    # 7. Validation split
    split_info = {}
    if cfg.enable_val_split:
        train_idx, val_idx = stratified_split(
            metadata_list, cfg.val_ratio, cfg.random_seed)

        # Write train and val JSONL
        with open(cfg.train_output, "w", encoding="utf-8") as f:
            for i in train_idx:
                f.write(json.dumps(conversations[i], ensure_ascii=False) + "\n")
        with open(cfg.val_output, "w", encoding="utf-8") as f:
            for i in val_idx:
                f.write(json.dumps(conversations[i], ensure_ascii=False) + "\n")

        split_info = {
            "train_count": len(train_idx),
            "val_count": len(val_idx),
            "val_ratio_actual": round(len(val_idx) / total, 4) if total > 0 else 0.0,
        }
        _debug(f"Split: train={len(train_idx)}, val={len(val_idx)}")

    # 8. Generate visualization plots
    plot_paths = []
    if cfg.enable_plots:
        print("enable plots")
        plot_paths = generate_plots(
            metadata_list, lang_dist, task_dist, diff_dist,
            len_cat_dist, cfg.report_dir)

    # 9. Assemble report
    report = {
        "total_samples": total,
        "language_distribution": lang_dist,
        "task_distribution": task_dist,
        "length_distribution": length_dist,
        "difficulty_distribution": diff_dist,
        "length_category_distribution": len_cat_dist,
        "cross_distribution_4d": cross_4d,
        "alerts": alerts,
        "split_info": split_info,
        "plot_paths": plot_paths,
        "config": {
            "length_short_max": cfg.length_short_max,
            "length_long_min": cfg.length_long_min,
            "cross_min_count": cfg.cross_min_count,
        },
    }

    # 10. Save JSON report
    with open(cfg.report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    _debug(f"Report saved to {cfg.report_json}")

    # 11. Print summary
    _print_summary(report)

    # 12. Save debug log to ./debug-log/
    _debug(f"\n{'=' * 60}")
    _debug(f"Debug log complete. Total samples processed: {total}")
    debug_log_path = os.path.join(
        debug_dir, f"data_distribution_eval_debug_{timestamp}.log")
    with open(debug_log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(_debug_lines) + "\n")

    # 13. Save report to ./report/ (JSON + TXT)
    report_json_path = os.path.join(
        report_out_dir, f"data_distribution_eval_report_{timestamp}.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    report_txt_path = os.path.join(
        report_out_dir, f"data_distribution_eval_report_{timestamp}.txt")
    summary_lines = _build_summary_lines(report)
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Timestamp : {timestamp}\n")
        f.write(f"Input     : {cfg.input}\n\n")
        f.write("\n".join(summary_lines) + "\n")

    print(f"\nDebug log saved to: {debug_log_path}")
    print(f"Report saved to:    {report_txt_path}")
    print(f"Report (JSON):      {report_json_path}")

    return report


def _build_summary_lines(report: Dict[str, Any]) -> List[str]:
    """Build a human-readable summary as a list of lines."""
    lines: List[str] = []
    lines.append(f"{'='*70}")
    lines.append(f"  数据分布评估报告")
    lines.append(f"{'='*70}")
    lines.append(f"")
    lines.append(f"总样本数: {report['total_samples']}")

    # Language
    lines.append(f"")
    lines.append(f"--- 语言分布 ---")
    for lang, info in report["language_distribution"]["distribution"].items():
        lines.append(f"  {lang}: {info['count']} ({info['percentage']}%)")

    # Task
    lines.append(f"")
    lines.append(f"--- 任务类型分布 ---")
    td = report["task_distribution"]
    for task, info in td["distribution"].items():
        lines.append(f"  {task}: {info['count']} ({info['percentage']}%)")
    lines.append(f"  多轮对话占比: {td['multi_turn_percentage']}%")

    # Difficulty
    lines.append(f"")
    lines.append(f"--- 难度分布 ---")
    for diff, info in report["difficulty_distribution"]["distribution"].items():
        lines.append(f"  {diff}: {info['count']} ({info['percentage']}%)")

    # Length category
    if "length_category_distribution" in report:
        cfg_info = report.get("config", {})
        short_max = cfg_info.get("length_short_max", "?")
        long_min = cfg_info.get("length_long_min", "?")
        lines.append(f"")
        lines.append(f"--- 长度类别分布 (短≤{short_max}, 中, 长≥{long_min} 字符) ---")
        for cat, info in report["length_category_distribution"]["distribution"].items():
            lines.append(f"  {cat}: {info['count']} ({info['percentage']}%)")

    # Length
    lines.append(f"")
    lines.append(f"--- 长度分布 (字符) ---")
    for part in ("input", "output", "total"):
        ld = report["length_distribution"][part]
        pcts = ld["percentiles"]
        pct_str = ", ".join(f"{k}={v:.0f}" for k, v in pcts.items())
        lines.append(f"  {part}: mean={ld['mean']:.0f}, min={ld['min']}, "
                     f"max={ld['max']}, {pct_str}")

    # 4D cross-distribution
    if "cross_distribution_4d" in report:
        cross_4d = report["cross_distribution_4d"]
        lines.append(f"")
        lines.append(f"--- 4D 交叉分布 (language×task_type×length_category×difficulty) ---")
        lines.append(f"  非零组合数: {cross_4d['num_cells']}")
        top_n = 10
        cells = cross_4d["cells"]
        lines.append(f"  Top-{min(top_n, len(cells))} 组合:")
        for cell in cells[:top_n]:
            lines.append(f"    {cell['language']} × {cell['task_type']} × "
                         f"{cell['length_category']} × {cell['difficulty']}: "
                         f"{cell['count']}")
        sparse_count = sum(1 for c in cells if c["count"] < report.get(
            "config", {}).get("cross_min_count", 100))
        lines.append(f"  稀疏组合 (低于阈值): {sparse_count}")

    # Alerts
    if report["alerts"]:
        warnings = [a for a in report["alerts"] if a["level"] == "warning"]
        infos = [a for a in report["alerts"] if a["level"] == "info"]
        if warnings:
            lines.append(f"")
            lines.append(f"--- 告警 ({len(warnings)} 条) ---")
            for a in warnings:
                lines.append(f"  [WARNING] [{a['dimension']}] {a['message']}")
        if infos:
            lines.append(f"")
            lines.append(f"--- 提示 ({len(infos)} 条) ---")
            for a in infos[:10]:  # Cap at 10
                lines.append(f"  [INFO] [{a['dimension']}] {a['message']}")
            if len(infos) > 10:
                lines.append(f"  ... 还有 {len(infos) - 10} 条提示")
    else:
        lines.append(f"")
        lines.append(f"--- 无告警 ---")

    # Split info
    if report.get("split_info"):
        si = report["split_info"]
        lines.append(f"")
        lines.append(f"--- 验证集划分 ---")
        lines.append(f"  训练集: {si['train_count']}, 验证集: {si['val_count']}, "
                     f"实际比例: {si['val_ratio_actual']}")

    return lines


def _print_summary(report: Dict[str, Any]) -> None:
    """Print a human-readable summary to stdout."""
    for line in _build_summary_lines(report):
        print(line)


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Data distribution evaluation for Bedrock-format "
                    "SFT conversation JSONL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Basic evaluation (heuristic language detection, no LLM annotation):
  python data_distribution_eval.py -i filtered.jsonl --no-enable-annotation

  # Full evaluation with LLM annotation:
  python data_distribution_eval.py -i filtered.jsonl --enable-annotation

  # With validation split:
  python data_distribution_eval.py -i filtered.jsonl --enable-val-split --val-ratio 0.1

  # Custom alert thresholds:
  python data_distribution_eval.py -i filtered.jsonl --lang-min-abs 1000 --task-min-abs 500
""",
    )

    io_grp = p.add_argument_group("I/O")
    io_grp.add_argument("-i", "--input", default="./zh_mixed_filtered.jsonl",
                        help="Input JSONL path (default: %(default)s)")
    io_grp.add_argument("--output-annotated",
                        default="./zh_mixed_annotated.jsonl",
                        help="Output annotated JSONL path (default: %(default)s)")
    io_grp.add_argument("--report-json", default="./distribution_report.json",
                        help="JSON report output path (default: %(default)s)")
    io_grp.add_argument("--report-dir", default="./distribution_plots",
                        help="Directory for plot images (default: %(default)s)")

    annot = p.add_argument_group("Annotation")
    annot.add_argument("--enable-annotation",
                       action=argparse.BooleanOptionalAction, default=True,
                       help="Enable LLM annotation of metadata (default: on)")
    annot.add_argument("--annotation-model-id",
                       default="global.anthropic.claude-opus-4-6-v1",
                       help="Bedrock model ID for annotation (default: %(default)s)")
    annot.add_argument("--bedrock-region", default="us-east-1",
                       help="AWS region for Bedrock (default: %(default)s)")

    split = p.add_argument_group("Validation split")
    split.add_argument("--enable-val-split",
                       action=argparse.BooleanOptionalAction, default=False,
                       help="Enable validation set splitting (default: off)")
    split.add_argument("--val-ratio", type=float, default=0.1,
                       help="Fraction of data for validation (default: %(default)s)")
    split.add_argument("--val-output", default="./val_split.jsonl",
                       help="Validation set output path (default: %(default)s)")
    split.add_argument("--train-output", default="./train_split.jsonl",
                       help="Training set output path (default: %(default)s)")
    split.add_argument("--random-seed", type=int, default=42,
                       help="Random seed for splitting (default: %(default)s)")

    viz = p.add_argument_group("Visualization")
    viz.add_argument("--enable-plots",
                     action=argparse.BooleanOptionalAction, default=True,
                     help="Generate visualization plots (default: on)")

    lang_alert = p.add_argument_group("Language alert thresholds")
    lang_alert.add_argument("--lang-min-pct", type=float, default=1.0,
                            help="Min language percentage before alert (default: %(default)s)")
    lang_alert.add_argument("--lang-min-abs", type=int, default=5000,
                            help="Min language absolute count before alert (default: %(default)s)")

    task_alert = p.add_argument_group("Task alert thresholds")
    task_alert.add_argument("--task-max-single-pct", type=float, default=50.0,
                            help="Max single task percentage before alert (default: %(default)s)")
    task_alert.add_argument("--task-min-pct", type=float, default=3.0,
                            help="Min task percentage before alert (default: %(default)s)")
    task_alert.add_argument("--task-min-abs", type=int, default=2000,
                            help="Min task count before alert (default: %(default)s)")
    task_alert.add_argument("--multi-turn-min-pct", type=float, default=20.0,
                            help="Min multi-turn conversation percentage (default: %(default)s)")

    diff_alert = p.add_argument_group("Difficulty alert thresholds")
    diff_alert.add_argument("--difficulty-tolerance", type=float, default=15.0,
                            help="Max deviation from target ratio (default: %(default)s)")

    len_cat = p.add_argument_group("Length category settings")
    len_cat.add_argument("--length-short-max", type=int, default=200,
                         help="Max total chars for '短' category (default: %(default)s)")
    len_cat.add_argument("--length-long-min", type=int, default=1000,
                         help="Min total chars for '长' category (default: %(default)s)")
    len_cat.add_argument("--length-cat-tolerance", type=float, default=15.0,
                         help="Max deviation from target length category ratio "
                              "(default: %(default)s)")

    cross_alert = p.add_argument_group("Cross-dimension alert thresholds")
    cross_alert.add_argument("--cross-min-count", type=int, default=100,
                             help="Min count per cross-dimension cell "
                                  "before alert (default: %(default)s)")

    return p


def args_to_config(args: argparse.Namespace) -> EvalConfig:
    """Map parsed CLI args -> EvalConfig."""
    return EvalConfig(
        input=args.input,
        output_annotated=args.output_annotated,
        report_json=args.report_json,
        report_dir=args.report_dir,
        enable_annotation=args.enable_annotation,
        annotation_model_id=args.annotation_model_id,
        bedrock_region=args.bedrock_region,
        enable_val_split=args.enable_val_split,
        val_ratio=args.val_ratio,
        val_output=args.val_output,
        train_output=args.train_output,
        random_seed=args.random_seed,
        enable_plots=args.enable_plots,
        lang_min_pct=args.lang_min_pct,
        lang_min_abs=args.lang_min_abs,
        task_max_single_pct=args.task_max_single_pct,
        task_min_pct=args.task_min_pct,
        task_min_abs=args.task_min_abs,
        multi_turn_min_pct=args.multi_turn_min_pct,
        difficulty_tolerance=args.difficulty_tolerance,
        length_short_max=args.length_short_max,
        length_long_min=args.length_long_min,
        length_cat_tolerance=args.length_cat_tolerance,
        cross_min_count=args.cross_min_count,
    )


def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = args_to_config(args)
    run(cfg)


if __name__ == "__main__":
    main()
