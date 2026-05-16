#!/usr/bin/env python3
"""
Advanced quality filtering for Bedrock-format SFT conversation JSONL.

Pipeline (applied in order, fail-fast per conversation):
  1. Turn-level rule filters (cheap):
       empty turns, repeated assistant, role alternation, truncation detection
  2. Dialogue-level rule filters:
       turn count, user low-info ratio, assistant stalling
  3. LLM-based filters (optional, via AWS Bedrock Claude):
       content moderation, PII detection, quality scoring

All parameters are exposed via CLI flags (see --help).
"""

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False


# ============================================================
# Configuration
# ============================================================

DEFAULT_TRUNCATION_MARKERS = ["...", "…", "[截断]", "[truncated]", "——未完"]

DEFAULT_LOW_INFO_PATTERNS = [
    "嗯", "哦", "好的", "是的", "ok", "OK", "Ok",
    "好", "对", "行", "嗯嗯", "哦哦", "是",
    "yes", "Yes", "no", "No", "yeah", "yep",
    "谢谢", "thanks", "Thank you", "好吧",
]


@dataclass
class FilterConfig:
    """All tuneable parameters for the advanced quality filter pipeline."""

    # -- I/O --
    input: str = "./zh_mixed_deduped.jsonl"
    output: str = "./zh_mixed_filtered.jsonl"
    report_json: str = "./report_for_advanced_quality_filter.json"  # optional JSON report path

    # -- Turn-level rule filters --
    filter_empty_turns: bool = True
    filter_repeated_assistant: bool = True
    filter_role_alternation: bool = True
    filter_truncation: bool = True
    truncation_markers: List[str] = field(
        default_factory=lambda: list(DEFAULT_TRUNCATION_MARKERS)
    )

    # -- Dialogue-level rule filters --
    filter_turn_count: bool = True
    min_turns: int = 2
    max_turns: int = 100
    filter_user_low_info: bool = True
    low_info_patterns: List[str] = field(
        default_factory=lambda: list(DEFAULT_LOW_INFO_PATTERNS)
    )
    max_low_info_ratio: float = 0.6
    filter_stalling: bool = True
    max_repeat_ratio: float = 0.5

    # -- LLM-based filters --
    enable_llm_quality: bool = True
    llm_quality_threshold: float = 6.0
    enable_llm_moderation: bool = True
    enable_llm_pii: bool = True

    # -- Bedrock settings --
    bedrock_model_id: str = "us.anthropic.claude-opus-4-6-v1"
    bedrock_region: str = "us-east-1"


# ============================================================
# Helper: extract message text
# ============================================================

def _get_msg_text(msg: dict) -> str:
    """Extract text from a Bedrock-format message content block."""
    content = msg.get("content", [])
    if content and isinstance(content, list):
        return content[0].get("text", "")
    return ""


def _format_conv_full(conv: dict) -> str:
    """Format a full conversation for logging, showing system + all turns."""
    lines = []
    sys_texts = conv.get("system", [])
    if sys_texts:
        sys_str = sys_texts[0].get("text", "")
        if sys_str:
            lines.append(f"    [系统] {sys_str}")
    for msg in conv.get("messages", []):
        role = msg.get("role", "unknown")
        text = _get_msg_text(msg)
        tag = "用户" if role == "user" else "助手"
        lines.append(f"    [{tag}] {text}")
    turns = len(conv.get("messages", []))
    total_len = sum(len(_get_msg_text(m)) for m in conv.get("messages", []))
    lines.append(f"    -- {turns} turns, {total_len} chars --")
    return "\n".join(lines)


# ============================================================
# Turn-level Filters
# ============================================================

def check_empty_turns(conv: dict) -> Tuple[bool, Optional[str]]:
    """Reject if any message has empty or whitespace-only text."""
    for idx, msg in enumerate(conv.get("messages", [])):
        text = _get_msg_text(msg)
        if not text.strip():
            return False, f"empty_turn@msg{idx}"
    return True, None


def check_repeated_assistant(conv: dict) -> Tuple[bool, Optional[str]]:
    """Reject if two adjacent assistant responses (possibly separated by user
    turns) have identical text."""
    messages = conv.get("messages", [])
    prev_assistant_text = None
    for idx, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            text = _get_msg_text(msg).strip()
            if prev_assistant_text is not None and text == prev_assistant_text:
                return False, f"repeated_assistant@msg{idx}"
            prev_assistant_text = text
    return True, None


def check_role_alternation(conv: dict) -> Tuple[bool, Optional[str]]:
    """Reject if messages don't alternate user/assistant or first is not user."""
    messages = conv.get("messages", [])
    if not messages:
        return True, None
    if messages[0].get("role") != "user":
        return False, "role_alternation@msg0"
    for idx in range(1, len(messages)):
        if messages[idx].get("role") == messages[idx - 1].get("role"):
            return False, f"role_alternation@msg{idx}"
    return True, None


def check_truncation(conv: dict,
                     markers: List[str] = None) -> Tuple[bool, Optional[str]]:
    """Reject if last assistant message ends with a truncation marker,
    or any assistant message has text length at a suspicious cutoff."""
    if markers is None:
        markers = DEFAULT_TRUNCATION_MARKERS

    messages = conv.get("messages", [])
    suspicious_lengths = {4096, 8192, 16384, 32768}

    # Check last assistant message for truncation markers
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "assistant":
            text = _get_msg_text(messages[idx]).strip()
            for marker in markers:
                if text.endswith(marker):
                    return False, f"truncation@msg{idx}"
            break

    # Check all assistant messages for suspicious exact-length cutoffs
    for idx, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            text = _get_msg_text(msg)
            if len(text) in suspicious_lengths:
                return False, f"truncation@msg{idx}"

    return True, None


# ============================================================
# Dialogue-level Filters
# ============================================================

def check_turn_count(conv: dict, min_turns: int = 2,
                     max_turns: int = 100) -> Tuple[bool, Optional[str]]:
    """Reject if message count is outside [min_turns, max_turns]."""
    n = len(conv.get("messages", []))
    if n < min_turns or n > max_turns:
        return False, f"turn_count:{n}"
    return True, None


def check_user_low_info(conv: dict,
                        patterns: List[str] = None,
                        max_ratio: float = 0.6) -> Tuple[bool, Optional[str]]:
    """Reject if too many user messages are low-information."""
    if patterns is None:
        patterns = DEFAULT_LOW_INFO_PATTERNS

    pattern_set = set(patterns)
    user_msgs = [msg for msg in conv.get("messages", [])
                 if msg.get("role") == "user"]
    if not user_msgs:
        return True, None

    low_info_count = 0
    for msg in user_msgs:
        text = _get_msg_text(msg).strip()
        if text in pattern_set or len(text) <= 3:
            low_info_count += 1

    ratio = low_info_count / len(user_msgs)
    if ratio > max_ratio:
        return False, f"user_low_info:{ratio:.2f}"
    return True, None


def _char_overlap_ratio(text_a: str, text_b: str) -> float:
    """Character n-gram overlap coefficient between two texts."""
    n = 3
    if len(text_a) < n or len(text_b) < n:
        return 1.0 if text_a == text_b else 0.0
    ngrams_a = set(text_a[i:i + n] for i in range(len(text_a) - n + 1))
    ngrams_b = set(text_b[i:i + n] for i in range(len(text_b) - n + 1))
    if not ngrams_a or not ngrams_b:
        return 0.0
    intersection = len(ngrams_a & ngrams_b)
    return intersection / min(len(ngrams_a), len(ngrams_b))


def check_stalling(conv: dict,
                   max_repeat_ratio: float = 0.5) -> Tuple[bool, Optional[str]]:
    """Reject if too many consecutive assistant turns repeat content."""
    assistant_texts = []
    for msg in conv.get("messages", []):
        if msg.get("role") == "assistant":
            assistant_texts.append(_get_msg_text(msg).strip())

    if len(assistant_texts) < 2:
        return True, None

    stalling_count = 0
    total_pairs = len(assistant_texts) - 1
    for i in range(total_pairs):
        overlap = _char_overlap_ratio(assistant_texts[i], assistant_texts[i + 1])
        if overlap > 0.8:
            stalling_count += 1

    ratio = stalling_count / total_pairs
    if ratio > max_repeat_ratio:
        return False, f"stalling:{ratio:.2f}"
    return True, None


# ============================================================
# LLM Helpers
# ============================================================

def format_conversation_for_llm(conv: dict) -> str:
    """Format a conversation for LLM evaluation.

    Includes system prompt if present.
    Format: [系统] text / 用户: text / 助手: text
    """
    lines = []
    sys_texts = conv.get("system", [])
    if sys_texts:
        sys_str = sys_texts[0].get("text", "") if sys_texts else ""
        if sys_str:
            lines.append(f"[系统] {sys_str}")
    for msg in conv.get("messages", []):
        role = msg.get("role", "")
        if role == "user":
            label = "用户"
        elif role == "assistant":
            label = "助手"
        else:
            continue
        text = _get_msg_text(msg)
        if text:
            lines.append(f"{label}: {text}")
    return "\n".join(lines)


def parse_json_response(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    # Try to find JSON in markdown code block
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()

    # Try direct JSON parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to find a JSON object in the text
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass

    return {}


def _call_bedrock(client, model_id: str, prompt: str,
                  max_tokens: int = 1024) -> str:
    """Call Bedrock API and return response text."""
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


# ============================================================
# LLM-based Filters
# ============================================================

def llm_check_quality(conv: dict, client, model_id: str,
                      threshold: float = 6.0) -> Tuple[bool, dict]:
    """Evaluate conversation quality using LLM.

    Multi-turn (>=4 msgs): 完整性/连贯性/信息密度/自然度/上下文一致性/知识准确性/回答深度/对话推进
    Single-turn (2 msgs): 完整性/信息密度/知识准确性/自然度
    Returns (pass, scores_dict).
    """
    formatted = format_conversation_for_llm(conv)
    n_msgs = len(conv.get("messages", []))

    if n_msgs >= 4:
        dimensions = "完整性、连贯性、信息密度、自然度、上下文一致性、知识准确性、回答深度、对话推进"
        example_scores = (
            '"完整性": 8, "连贯性": 7, "信息密度": 8, '
            '"自然度": 7, "上下文一致性": 8, "知识准确性": 7, '
            '"回答深度": 8, "对话推进": 7'
        )
    else:
        dimensions = "完整性、信息密度、知识准确性、自然度"
        example_scores = '"完整性": 8, "信息密度": 7, "知识准确性": 8, "自然度": 7'

    prompt = (
        "你是一个SFT数据质量评估专家。请对以下对话的质量进行评分。\n\n"
        f"评分维度(每项1-10分): {dimensions}\n\n"
        f"对话内容:\n{formatted}\n\n"
        "请以JSON格式返回评分结果,格式如下:\n"
        f'{{"scores": {{{example_scores}}}, "average": 7.5}}\n\n'
        "只返回JSON,不要其他文字。"
    )

    try:
        reply = _call_bedrock(client, model_id, prompt, max_tokens=512)
        parsed = parse_json_response(reply)
        if not parsed or "average" not in parsed:
            return False, {"error": "parse_failed", "raw": reply}
        avg = float(parsed["average"])
        passed = avg >= threshold
        return passed, parsed
    except Exception as e:
        return False, {"error": str(e)}


def llm_check_moderation(conv: dict, client,
                         model_id: str) -> Tuple[bool, dict]:
    """Check conversation for content safety violations using LLM.

    Categories: 涉黄/涉恐/涉暴/涉政/辱骂/歧视/语言暴力
    Returns (safe, detail_dict).
    """
    formatted = format_conversation_for_llm(conv)

    prompt = (
        "你是一个内容安全审核专家。请检查以下对话是否包含不安全内容。\n\n"
        "检查类别: 涉黄、涉恐、涉暴、涉政、辱骂、歧视、语言暴力\n\n"
        f"对话内容:\n{formatted}\n\n"
        "请以JSON格式返回审核结果:\n"
        '如果安全: {"safe": true, "categories": []}\n'
        '如果不安全: {"safe": false, "categories": ["涉暴"], "evidence": "..."}\n\n'
        "只返回JSON,不要其他文字。"
    )

    try:
        reply = _call_bedrock(client, model_id, prompt, max_tokens=512)
        parsed = parse_json_response(reply)
        if not parsed or "safe" not in parsed:
            return False, {"error": "parse_failed", "raw": reply}
        safe = bool(parsed["safe"])
        return safe, parsed
    except Exception as e:
        return False, {"error": str(e)}


def llm_check_pii(conv: dict, client,
                  model_id: str) -> Tuple[bool, dict]:
    """Detect PII in conversation using LLM.

    PII types: 姓名/手机号/身份证/银行卡/邮箱/地址/IP
    Returns (no_pii, detail_dict).
    """
    formatted = format_conversation_for_llm(conv)

    prompt = (
        "你是一个隐私信息检测专家。请检查以下对话是否包含个人隐私信息(PII)。\n\n"
        "检测类型: 真实姓名、手机号、身份证号、银行卡号、邮箱地址、家庭住址、IP地址\n\n"
        f"对话内容:\n{formatted}\n\n"
        "请以JSON格式返回检测结果:\n"
        '如果无PII: {"has_pii": false, "items": []}\n'
        '如果有PII: {"has_pii": true, "items": [{"type": "手机号", "value": "138****"}]}\n\n'
        "只返回JSON,不要其他文字。"
    )

    try:
        reply = _call_bedrock(client, model_id, prompt, max_tokens=512)
        parsed = parse_json_response(reply)
        if not parsed or "has_pii" not in parsed:
            return False, {"error": "parse_failed", "raw": reply}
        has_pii = bool(parsed["has_pii"])
        return not has_pii, parsed
    except Exception as e:
        return False, {"error": str(e)}


# ============================================================
# Pipeline Orchestration
# ============================================================

def run(cfg: FilterConfig, bedrock_client=None) -> dict:
    """Execute the full quality filter pipeline. Returns stats dict."""

    # 0. Prepare output directories for debug and report
    script_dir = os.path.dirname(os.path.abspath(__file__))
    debug_dir = os.path.join(script_dir, "debug-log")
    report_dir = os.path.join(script_dir, "report")
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_log_path = os.path.join(debug_dir, f"advanced_quality_filter_debug_{timestamp}.log")
    report_json_path = os.path.join(report_dir, f"advanced_quality_filter_report_{timestamp}.json")
    report_txt_path = os.path.join(report_dir, f"advanced_quality_filter_report_{timestamp}.txt")

    # 1. Load conversations
    conversations: List[dict] = []
    with open(cfg.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                conversations.append(json.loads(line))

    total = len(conversations)

    # 2. Create Bedrock client if any LLM filter is enabled
    any_llm = cfg.enable_llm_quality or cfg.enable_llm_moderation or cfg.enable_llm_pii
    if any_llm and bedrock_client is None:
        if not HAS_BOTO3:
            raise ImportError(
                "boto3 is required for LLM filters. "
                "Install it with: pip install boto3"
            )
        bedrock_client = boto3.client(
            "bedrock-runtime", region_name=cfg.bedrock_region
        )

    stats: dict = {
        "total": total,
        "kept": 0,
        "filtered": 0,
        "reasons": {},
        "llm_calls": 0,
    }
    report_entries: List[dict] = []
    # Per-reason sample collector: store up to 3 example conversations per reason
    reason_examples: Dict[str, list] = {}

    # 3. Apply filters per conversation (fail-fast)
    survivors: List[dict] = []

    with open(debug_log_path, "w", encoding="utf-8") as debug_f:
        debug_f.write(f"=== Advanced Quality Filter Debug Log ===\n")
        debug_f.write(f"Timestamp: {timestamp}\n")
        debug_f.write(f"Input:  {cfg.input}\n")
        debug_f.write(f"Output: {cfg.output}\n")
        debug_f.write(f"{'=' * 60}\n\n")

        for conv_idx, conv in enumerate(conversations):
            reason = None
            entry = {"index": conv_idx, "status": "kept", "reason": None,
                     "llm_details": {}}

            # --- Turn-level rule filters ---
            if cfg.filter_empty_turns and reason is None:
                passed, reason = check_empty_turns(conv)

            if cfg.filter_repeated_assistant and reason is None:
                passed, reason = check_repeated_assistant(conv)

            if cfg.filter_role_alternation and reason is None:
                passed, reason = check_role_alternation(conv)

            if cfg.filter_truncation and reason is None:
                passed, reason = check_truncation(conv, cfg.truncation_markers)

            # --- Dialogue-level rule filters ---
            if cfg.filter_turn_count and reason is None:
                passed, reason = check_turn_count(conv, cfg.min_turns, cfg.max_turns)

            if cfg.filter_user_low_info and reason is None:
                passed, reason = check_user_low_info(
                    conv, cfg.low_info_patterns, cfg.max_low_info_ratio
                )

            if cfg.filter_stalling and reason is None:
                passed, reason = check_stalling(conv, cfg.max_repeat_ratio)

            # --- LLM moderation ---
            llm_filter_detail = None  # detail dict of the LLM filter that rejected
            if cfg.enable_llm_moderation and reason is None:
                
                stats["llm_calls"] += 1
                passed, detail = llm_check_moderation(
                    conv, bedrock_client, cfg.bedrock_model_id
                )
                entry["llm_details"]["moderation"] = detail
                if not passed:
                    reason = "llm_moderation"
                    llm_filter_detail = detail

            # --- LLM PII ---
            if cfg.enable_llm_pii and reason is None:
                stats["llm_calls"] += 1
                passed, detail = llm_check_pii(
                    conv, bedrock_client, cfg.bedrock_model_id
                )
                entry["llm_details"]["pii"] = detail

                if not passed:
                    reason = "llm_pii"
                    llm_filter_detail = detail

            # --- LLM quality (most expensive, last) ---
            if cfg.enable_llm_quality and reason is None:
                stats["llm_calls"] += 1
                passed, detail = llm_check_quality(
                    conv, bedrock_client, cfg.bedrock_model_id,
                    cfg.llm_quality_threshold
                )
                entry["llm_details"]["quality"] = detail


                if not passed:
                    reason = "llm_quality"
                    llm_filter_detail = detail

            # --- Record result ---
            if reason is None:
                survivors.append(conv)
                stats["kept"] += 1
            else:
                stats["filtered"] += 1
                base_reason = reason.split("@")[0].split(":")[0]
                stats["reasons"][base_reason] = stats["reasons"].get(base_reason, 0) + 1
                entry["status"] = "filtered"
                entry["reason"] = reason

                # Write debug info for every filtered conversation
                debug_f.write(f"{'=' * 60}\n")
                debug_f.write(f"[FILTERED #{stats['filtered']}] idx={conv_idx}  reason={reason}\n")
                debug_f.write(f"{'=' * 60}\n")
                debug_f.write(_format_conv_full(conv) + "\n")
                # Write LLM filter details if available
                if llm_filter_detail:
                    debug_f.write(f"  ---- LLM 判定详情 ----\n")
                    if reason == "llm_moderation":
                        cats = llm_filter_detail.get("categories", [])
                        evidence = llm_filter_detail.get("evidence", "")
                        debug_f.write(f"  违规类别: {', '.join(cats) if cats else '未知'}\n")
                        if evidence:
                            debug_f.write(f"  证据: {evidence}\n")
                    elif reason == "llm_pii":
                        items = llm_filter_detail.get("items", [])
                        if items:
                            for it in items:
                                debug_f.write(f"  PII类型: {it.get('type', '未知')}  "
                                              f"值: {it.get('value', 'N/A')}\n")
                        else:
                            debug_f.write(f"  检测到PII (详情未返回)\n")
                    elif reason == "llm_quality":
                        scores = llm_filter_detail.get("scores", {})
                        avg = llm_filter_detail.get("average", "N/A")
                        debug_f.write(f"  综合评分: {avg}  (阈值: {cfg.llm_quality_threshold})\n")
                        if scores:
                            parts = [f"{k}={v}" for k, v in scores.items()]
                            debug_f.write(f"  各项评分: {', '.join(parts)}\n")
                    # Fallback: if detail has error field
                    if "error" in llm_filter_detail:
                        debug_f.write(f"  错误: {llm_filter_detail['error']}\n")
                        raw = llm_filter_detail.get("raw", "")
                        if raw:
                            debug_f.write(f"  原始响应: {raw[:200]}\n")
                debug_f.write("\n")

                # Collect up to 3 examples per reason for the report
                if base_reason not in reason_examples:
                    reason_examples[base_reason] = []
                if len(reason_examples[base_reason]) < 3:
                    msgs = conv.get("messages", [])
                    preview = []
                    for m in msgs[:3]:
                        text = _get_msg_text(m)
                        if len(text) > 100:
                            text = text[:100] + "..."
                        preview.append({"role": m.get("role", ""), "text": text})
                    reason_examples[base_reason].append(preview)

            report_entries.append(entry)

        debug_f.write(f"\n{'=' * 60}\n")
        debug_f.write(f"Debug log complete. Total filtered: {stats['filtered']}\n")

    # 4. Write survivors
    with open(cfg.output, "w", encoding="utf-8") as fout:
        for conv in survivors:
            fout.write(json.dumps(conv, ensure_ascii=False) + "\n")

    # 5. Optional JSON report (legacy path from CLI --report-json)
    if cfg.report_json:
        report = {"stats": stats, "entries": report_entries}
        with open(cfg.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    # 6. Build structured report (debug-log & report directories)
    sorted_reasons = sorted(stats["reasons"].items(), key=lambda x: -x[1])

    # Determine which filter methods were active
    active_filters = []
    if cfg.filter_empty_turns:
        active_filters.append("empty_turns (空消息过滤)")
    if cfg.filter_repeated_assistant:
        active_filters.append("repeated_assistant (重复助手回复过滤)")
    if cfg.filter_role_alternation:
        active_filters.append("role_alternation (角色交替检查)")
    if cfg.filter_truncation:
        active_filters.append("truncation (截断检测)")
    if cfg.filter_turn_count:
        active_filters.append(f"turn_count (轮次数量过滤, min={cfg.min_turns}, max={cfg.max_turns})")
    if cfg.filter_user_low_info:
        active_filters.append(f"user_low_info (用户低信息量过滤, max_ratio={cfg.max_low_info_ratio})")
    if cfg.filter_stalling:
        active_filters.append(f"stalling (助手重复内容过滤, max_repeat_ratio={cfg.max_repeat_ratio})")
    if cfg.enable_llm_moderation:
        active_filters.append(f"llm_moderation (LLM内容审核, model={cfg.bedrock_model_id})")
    if cfg.enable_llm_pii:
        active_filters.append(f"llm_pii (LLM隐私信息检测, model={cfg.bedrock_model_id})")
    if cfg.enable_llm_quality:
        active_filters.append(f"llm_quality (LLM质量评分, threshold={cfg.llm_quality_threshold}, model={cfg.bedrock_model_id})")

    # JSON report
    report_data = {
        "timestamp": timestamp,
        "input_file": cfg.input,
        "output_file": cfg.output,
        "summary": {
            "total_samples": stats["total"],
            "kept_samples": stats["kept"],
            "filtered_samples": stats["filtered"],
            "keep_rate": round(stats["kept"] / stats["total"] * 100, 2) if stats["total"] else 0,
            "filter_rate": round(stats["filtered"] / stats["total"] * 100, 2) if stats["total"] else 0,
            "llm_calls": stats["llm_calls"],
        },
        "active_filters": active_filters,
        "filter_reasons": [
            {
                "reason": reason,
                "count": count,
                "percentage": round(count / stats["filtered"] * 100, 2) if stats["filtered"] else 0,
                "examples": reason_examples.get(reason, []),
            }
            for reason, count in sorted_reasons
        ],
        "config": {
            "filter_empty_turns": cfg.filter_empty_turns,
            "filter_repeated_assistant": cfg.filter_repeated_assistant,
            "filter_role_alternation": cfg.filter_role_alternation,
            "filter_truncation": cfg.filter_truncation,
            "truncation_markers": cfg.truncation_markers,
            "filter_turn_count": cfg.filter_turn_count,
            "min_turns": cfg.min_turns,
            "max_turns": cfg.max_turns,
            "filter_user_low_info": cfg.filter_user_low_info,
            "max_low_info_ratio": cfg.max_low_info_ratio,
            "filter_stalling": cfg.filter_stalling,
            "max_repeat_ratio": cfg.max_repeat_ratio,
            "enable_llm_quality": cfg.enable_llm_quality,
            "llm_quality_threshold": cfg.llm_quality_threshold,
            "enable_llm_moderation": cfg.enable_llm_moderation,
            "enable_llm_pii": cfg.enable_llm_pii,
            "bedrock_model_id": cfg.bedrock_model_id,
            "bedrock_region": cfg.bedrock_region,
        },
    }
    with open(report_json_path, "w", encoding="utf-8") as rf:
        json.dump(report_data, rf, ensure_ascii=False, indent=2)

    # Text report
    with open(report_txt_path, "w", encoding="utf-8") as tf:
        tf.write("=" * 60 + "\n")
        tf.write("  Advanced Quality Filter — Processing Report\n")
        tf.write("=" * 60 + "\n\n")
        tf.write(f"Timestamp : {timestamp}\n")
        tf.write(f"Input     : {cfg.input}\n")
        tf.write(f"Output    : {cfg.output}\n\n")

        tf.write("-" * 40 + "\n")
        tf.write("  Active Filter Methods\n")
        tf.write("-" * 40 + "\n")
        for af in active_filters:
            tf.write(f"  * {af}\n")
        tf.write("\n")

        tf.write("-" * 40 + "\n")
        tf.write("  Sample Statistics\n")
        tf.write("-" * 40 + "\n")
        tf.write(f"  Total samples   : {stats['total']}\n")
        tf.write(f"  Kept samples    : {stats['kept']}\n")
        tf.write(f"  Filtered samples: {stats['filtered']}\n")
        keep_pct = round(stats["kept"] / stats["total"] * 100, 2) if stats["total"] else 0
        filt_pct = round(stats["filtered"] / stats["total"] * 100, 2) if stats["total"] else 0
        tf.write(f"  Keep rate       : {keep_pct}%\n")
        tf.write(f"  Filter rate     : {filt_pct}%\n")
        if stats["llm_calls"]:
            tf.write(f"  LLM calls       : {stats['llm_calls']}\n")
        tf.write("\n")

        if sorted_reasons:
            tf.write("-" * 40 + "\n")
            tf.write("  Filter Reasons Breakdown\n")
            tf.write("-" * 40 + "\n")
            for reason, count in sorted_reasons:
                pct = round(count / stats["filtered"] * 100, 2) if stats["filtered"] else 0
                tf.write(f"  {reason:40s} : {count:6d}  ({pct}%)\n")
            tf.write("\n")

            tf.write("-" * 40 + "\n")
            tf.write("  Filtered Sample Examples (up to 3 per reason)\n")
            tf.write("-" * 40 + "\n")
            for reason, count in sorted_reasons:
                tf.write(f"\n  [{reason}] ({count} filtered)\n")
                examples = reason_examples.get(reason, [])
                for i, ex in enumerate(examples, 1):
                    tf.write(f"    Example {i}:\n")
                    for turn in ex:
                        tf.write(f"      {turn['role']}: {turn['text']}\n")

        tf.write("\n" + "=" * 60 + "\n")
        tf.write("  End of Report\n")
        tf.write("=" * 60 + "\n")

    # 7. Print summary to console
    print(f"\nTotal:    {stats['total']}")
    print(f"Kept:     {stats['kept']}")
    print(f"Filtered: {stats['filtered']}")
    if stats["reasons"]:
        print("Filter reasons:")
        for reason, count in sorted(stats["reasons"].items(),
                                    key=lambda x: -x[1]):
            print(f"  {reason}: {count}")
    if stats["llm_calls"]:
        print(f"LLM calls: {stats['llm_calls']}")
    print(f"\nDebug log saved to: {debug_log_path}")
    print(f"Report saved to:    {report_txt_path}")
    print(f"Report (JSON):      {report_json_path}")

    return stats


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Advanced quality filtering for Bedrock-format SFT conversation JSONL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Default (rule-based filters only):
  python advanced_quality_filter.py -i zh_mixed_deduped.jsonl -o zh_mixed_filtered.jsonl

  # With LLM moderation and PII detection:
  python advanced_quality_filter.py -i in.jsonl -o out.jsonl \\
      --enable-llm-moderation --enable-llm-pii

  # Full LLM pipeline (quality + moderation + PII):
  python advanced_quality_filter.py -i in.jsonl -o out.jsonl \\
      --enable-llm-quality --enable-llm-moderation --enable-llm-pii

  # Disable turn-count filter, custom thresholds:
  python advanced_quality_filter.py --no-filter-turn-count --max-low-info-ratio 0.8

  # Generate JSON report:
  python advanced_quality_filter.py -i in.jsonl -o out.jsonl --report-json report.json
""",
    )

    io_grp = p.add_argument_group("I/O")
    io_grp.add_argument("-i", "--input", default="./zh_mixed_deduped.jsonl",
                        help="Input JSONL path (default: %(default)s)")
    io_grp.add_argument("-o", "--output", default="./zh_mixed_filtered.jsonl",
                        help="Output JSONL path (default: %(default)s)")
    io_grp.add_argument("--report-json", default="",
                        help="Optional JSON report output path")

    turn = p.add_argument_group("Turn-level rule filters")
    turn.add_argument("--filter-empty-turns",
                      action=argparse.BooleanOptionalAction, default=True,
                      help="Filter conversations with empty turns (default: on)")
    turn.add_argument("--filter-repeated-assistant",
                      action=argparse.BooleanOptionalAction, default=True,
                      help="Filter consecutive identical assistant responses (default: on)")
    turn.add_argument("--filter-role-alternation",
                      action=argparse.BooleanOptionalAction, default=True,
                      help="Filter non-alternating roles (default: on)")
    turn.add_argument("--filter-truncation",
                      action=argparse.BooleanOptionalAction, default=True,
                      help="Filter truncated responses (default: on)")

    dial = p.add_argument_group("Dialogue-level rule filters")
    dial.add_argument("--filter-turn-count",
                      action=argparse.BooleanOptionalAction, default=True,
                      help="Filter by turn count range (default: on)")
    dial.add_argument("--min-turns", type=int, default=2,
                      help="Minimum message count (default: %(default)s)")
    dial.add_argument("--max-turns", type=int, default=100,
                      help="Maximum message count (default: %(default)s)")
    dial.add_argument("--filter-user-low-info",
                      action=argparse.BooleanOptionalAction, default=True,
                      help="Filter conversations with too many low-info user turns (default: on)")
    dial.add_argument("--max-low-info-ratio", type=float, default=0.6,
                      help="Max ratio of low-info user messages (default: %(default)s)")
    dial.add_argument("--filter-stalling",
                      action=argparse.BooleanOptionalAction, default=True,
                      help="Filter assistant stalling/repetition (default: on)")
    dial.add_argument("--max-repeat-ratio", type=float, default=0.5,
                      help="Max ratio of stalling assistant pairs (default: %(default)s)")

    llm = p.add_argument_group("LLM-based filters")
    llm.add_argument("--enable-llm-quality",
                     action=argparse.BooleanOptionalAction, default=True,
                     help="Enable LLM quality scoring (default: off)")
    llm.add_argument("--llm-quality-threshold", type=float, default=6.0,
                     help="Min average quality score to keep (default: %(default)s)")
    llm.add_argument("--enable-llm-moderation",
                     action=argparse.BooleanOptionalAction, default=True,
                     help="Enable LLM content moderation (default: off)")
    llm.add_argument("--enable-llm-pii",
                     action=argparse.BooleanOptionalAction, default=True,
                     help="Enable LLM PII detection (default: off)")

    bed = p.add_argument_group("Bedrock settings")
    bed.add_argument("--bedrock-model-id",
                     default="us.anthropic.claude-opus-4-6-v1",
                     help="Bedrock model ID (default: %(default)s)")
    bed.add_argument("--bedrock-region", default="us-east-1",
                     help="AWS region for Bedrock (default: %(default)s)")

    return p


def args_to_config(args: argparse.Namespace) -> FilterConfig:
    """Map parsed CLI args -> FilterConfig."""
    return FilterConfig(
        input=args.input,
        output=args.output,
        report_json=args.report_json,
        filter_empty_turns=args.filter_empty_turns,
        filter_repeated_assistant=args.filter_repeated_assistant,
        filter_role_alternation=args.filter_role_alternation,
        filter_truncation=args.filter_truncation,
        filter_turn_count=args.filter_turn_count,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        filter_user_low_info=args.filter_user_low_info,
        max_low_info_ratio=args.max_low_info_ratio,
        filter_stalling=args.filter_stalling,
        max_repeat_ratio=args.max_repeat_ratio,
        enable_llm_quality=args.enable_llm_quality,
        llm_quality_threshold=args.llm_quality_threshold,
        enable_llm_moderation=args.enable_llm_moderation,
        enable_llm_pii=args.enable_llm_pii,
        bedrock_model_id=args.bedrock_model_id,
        bedrock_region=args.bedrock_region,
    )


def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = args_to_config(args)
    run(cfg)


if __name__ == "__main__":
    main()
