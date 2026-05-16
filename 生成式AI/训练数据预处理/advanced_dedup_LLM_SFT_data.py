#!/usr/bin/env python3
"""
Advanced (Layer-2) semantic deduplication for Bedrock-format SFT conversation JSONL.

Pipeline:
  Stage 1 -- Full-sample embedding dedup:
      Embed input+output concatenated text via AWS Bedrock Cohere Embed v3 multilingual.
      Cluster by cosine similarity (threshold ~ 0.85); keep best-scored per cluster.

  Stage 2 -- Input-level dedup (similar question -> keep best answer):
      Embed only user turns (input); cluster by cosine similarity (threshold ~ 0.7).
      For each cluster: evaluate output quality via LLM-as-judge
      (AWS Bedrock Claude Opus 4.6); for multi-turn dialogs score each assistant
      turn and compute weighted average.
      Keep the conversation with the best output per cluster.

Embedding: AWS Bedrock Cohere Embed Multilingual v3.
LLM Judge: AWS Bedrock Claude Opus 4.6.

All parameters are exposed via CLI flags (see --help).
"""

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import os
from datetime import datetime

try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False


# ============================================================
# Configuration
# ============================================================

@dataclass
class AdvancedDedupConfig:
    """All tuneable parameters for the advanced dedup pipeline."""

    # -- I/O --
    input: str = "./zh_mixed_deduped.jsonl"
    output: str = "./zh_mixed_advanced_deduped.jsonl"

    # -- Stage switches --
    enable_full_sample_dedup: bool = True
    enable_input_dedup: bool = True

    # -- Embedding settings --
    embed_model_id: str = "cohere.embed-multilingual-v3"
    embed_batch_size: int = 96   # max texts per Cohere Embed API call
    embed_input_type: str = "search_document"

    # -- Full-sample dedup --
    full_sample_threshold: float = 0.85  # cosine similarity threshold

    # -- Input-level dedup --
    input_threshold: float = 0.7  # cosine similarity threshold

    # -- Output quality scoring --
    scoring_method: str = "llm"  # "heuristic" or "llm"
    judge_model_id: str = "us.anthropic.claude-opus-4-6-v1"

    # -- Heuristic scoring weights --
    weight_completeness: float = 0.4
    weight_info_density: float = 0.6

    # -- Multi-turn weighting --
    multi_turn_weight_mode: str = "linear_increasing"  # "equal" | "linear_increasing"

    # -- Bedrock settings --
    bedrock_region: str = "us-east-1"


# ============================================================
# Text extraction
# ============================================================

def _get_msg_text(msg: dict) -> str:
    """Extract text from a Bedrock-format message content block."""
    content = msg.get("content", [])
    if content and isinstance(content, list):
        return content[0].get("text", "")
    return ""


def extract_full_text(conv: dict, separator: str = "\n") -> str:
    """Concatenate all message content texts (input+output), ignoring role/system."""
    parts = []
    for msg in conv.get("messages", []):
        text = _get_msg_text(msg)
        if text:
            parts.append(text)
    return separator.join(parts)


def extract_input_text(conv: dict, separator: str = "\n") -> str:
    """Concatenate only user turns (input), ignoring assistant and system."""
    parts = []
    for msg in conv.get("messages", []):
        if msg.get("role") == "user":
            text = _get_msg_text(msg)
            if text:
                parts.append(text)
    return separator.join(parts)


def extract_assistant_texts(conv: dict) -> List[str]:
    """Extract all assistant turn texts as a list (preserving order)."""
    texts = []
    for msg in conv.get("messages", []):
        if msg.get("role") == "assistant":
            texts.append(_get_msg_text(msg))
    return texts


# ============================================================
# Vector utilities (no numpy dependency)
# ============================================================

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns 0.0 when either vector has zero norm.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must have the same length")
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ============================================================
# Union-Find
# ============================================================

class UnionFind:
    """Path-compressed union-find with rank."""

    def __init__(self, elements):
        self.parent = {x: x for x in elements}
        self.rank = {x: 0 for x in elements}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def groups(self) -> Dict[int, List[int]]:
        """Return {root: [members]} mapping."""
        g: Dict[int, List[int]] = defaultdict(list)
        for x in self.parent:
            g[self.find(x)].append(x)
        return dict(g)


# ============================================================
# Embedding via AWS Bedrock Cohere Embed v3
# ============================================================

def embed_texts(texts: List[str], client, model_id: str,
                input_type: str = "search_document",
                batch_size: int = 96) -> List[List[float]]:
    """Generate embeddings for a list of texts using Bedrock Cohere Embed v3.

    Batches texts into groups of batch_size (max 96 per API call).
    Returns list of embedding vectors in the same order as input texts.
    """
    all_embeddings: List[List[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        body = json.dumps({
            "texts": batch,
            "input_type": input_type,
            "truncate": "END",
        })
        response = client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        result = json.loads(response["body"].read())
        embeddings = result["embeddings"]
        all_embeddings.extend(embeddings)

    return all_embeddings


# ============================================================
# Cosine similarity clustering
# ============================================================

def cluster_by_cosine(embeddings: List[List[float]], indices: List[int],
                      threshold: float) -> "UnionFind":
    """O(n^2) pairwise cosine similarity; returns UnionFind of clusters."""
    n = len(embeddings)
    uf = UnionFind(indices)
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim >= threshold:
                uf.union(indices[i], indices[j])
    return uf


# ============================================================
# Heuristic scoring
# ============================================================

def compute_heuristic_score(conv: dict, w_completeness: float = 0.4,
                            w_info_density: float = 0.6) -> float:
    """Score a conversation heuristically.

    Higher is better.  score = w_comp * completeness + w_info * info_density
    where completeness = n_turns + total_len/100, info_density = avg_asst_len/100.
    """
    messages = conv.get("messages", [])
    n_turns = len(messages)
    total_len = 0
    assistant_lens = []
    for msg in messages:
        text = _get_msg_text(msg)
        total_len += len(text)
        if msg.get("role") == "assistant":
            assistant_lens.append(len(text))
    avg_assistant_len = (sum(assistant_lens) / len(assistant_lens)
                         if assistant_lens else 0.0)

    completeness = n_turns + total_len / 100.0
    info_density = avg_assistant_len / 100.0
    return w_completeness * completeness + w_info_density * info_density


# ============================================================
# LLM helpers
# ============================================================

def _call_bedrock(client, model_id: str, prompt: str,
                  max_tokens: int = 1024) -> str:
    """Call Bedrock Claude API and return response text."""
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


def parse_json_response(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    # Try markdown code block first
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()

    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to find a JSON object anywhere in the text
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass

    return {}


def format_conversation_for_judge(conv: dict) -> str:
    """Format a conversation for LLM judge evaluation."""
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


# ============================================================
# LLM-as-Judge: output quality scoring
# ============================================================

def score_single_turn_output(conv: dict, client, model_id: str) -> float:
    """Score a single-turn conversation's output quality via LLM judge.

    Returns a float score (1-10).  Returns -1.0 on failure.
    """
    formatted = format_conversation_for_judge(conv)

    prompt = (
        "你是一个SFT训练数据质量评估专家。请评估以下对话中助手回答的质量。\n\n"
        "评分维度(每项1-10分): 完整性、信息密度、知识准确性、自然度\n\n"
        f"对话内容:\n{formatted}\n\n"
        "请以JSON格式返回评分结果:\n"
        '{"scores": {"完整性": 8, "信息密度": 7, "知识准确性": 8, "自然度": 7}, '
        '"average": 7.5}\n\n'
        "只返回JSON,不要其他文字。"
    )

    try:
        reply = _call_bedrock(client, model_id, prompt, max_tokens=512)
        parsed = parse_json_response(reply)
        if parsed and "average" in parsed:
            return float(parsed["average"])
    except Exception:
        pass
    return -1.0


def score_multi_turn_output(conv: dict, client, model_id: str,
                            weight_mode: str = "linear_increasing") -> float:
    """Score a multi-turn conversation's output quality via LLM judge.

    Scores each assistant turn individually, then computes weighted average.
    weight_mode: "equal" or "linear_increasing" (later turns weighted higher).
    Returns a float score (1-10).  Returns -1.0 on failure.
    """
    messages = conv.get("messages", [])
    assistant_indices = [i for i, m in enumerate(messages)
                         if m.get("role") == "assistant"]

    if not assistant_indices:
        return -1.0

    # Score each assistant turn in context
    turn_scores: List[float] = []

    for asst_idx in assistant_indices:
        # Build conversation context up to and including this assistant turn
        context_msgs = messages[:asst_idx + 1]
        context_lines = []
        sys_texts = conv.get("system", [])
        if sys_texts:
            sys_str = sys_texts[0].get("text", "") if sys_texts else ""
            if sys_str:
                context_lines.append(f"[系统] {sys_str}")
        for msg in context_msgs:
            role = msg.get("role", "")
            if role == "user":
                context_lines.append(f"用户: {_get_msg_text(msg)}")
            elif role == "assistant":
                context_lines.append(f"助手: {_get_msg_text(msg)}")
        context_text = "\n".join(context_lines)

        asst_text = _get_msg_text(messages[asst_idx])

        prompt = (
            "你是一个SFT训练数据质量评估专家。"
            "请评估以下多轮对话中,最后一条助手回答的质量。\n\n"
            "评分维度(每项1-10分): "
            "完整性、连贯性、信息密度、自然度、上下文一致性、回答深度\n\n"
            f"对话上下文:\n{context_text}\n\n"
            f"需要评分的助手回答:\n{asst_text}\n\n"
            "请以JSON格式返回评分结果:\n"
            '{"scores": {"完整性": 8, "连贯性": 7, "信息密度": 8, '
            '"自然度": 7, "上下文一致性": 8, "回答深度": 7}, "average": 7.5}\n\n'
            "只返回JSON,不要其他文字。"
        )

        try:
            reply = _call_bedrock(client, model_id, prompt, max_tokens=512)
            parsed = parse_json_response(reply)
            if parsed and "average" in parsed:
                turn_scores.append(float(parsed["average"]))
            else:
                turn_scores.append(-1.0)
        except Exception:
            turn_scores.append(-1.0)

    # Filter out failed scores
    valid_entries = [(i, s) for i, s in enumerate(turn_scores) if s >= 0]
    if not valid_entries:
        return -1.0

    # Compute weighted average
    if weight_mode == "linear_increasing":
        # Weight: 1, 2, 3, ... for successive valid turns
        total_weight = 0.0
        weighted_sum = 0.0
        for rank, (_, score) in enumerate(valid_entries, 1):
            weighted_sum += rank * score
            total_weight += rank
        return weighted_sum / total_weight if total_weight > 0 else -1.0
    else:  # equal
        return sum(s for _, s in valid_entries) / len(valid_entries)


def score_output_quality(conv: dict, client, model_id: str,
                         weight_mode: str = "linear_increasing") -> float:
    """Score output quality using LLM-as-judge.

    Dispatches to single-turn or multi-turn scoring based on message count.
    """
    messages = conv.get("messages", [])
    n_msgs = len(messages)

    if n_msgs <= 2:
        return score_single_turn_output(conv, client, model_id)
    else:
        return score_multi_turn_output(conv, client, model_id, weight_mode)


# ============================================================
# Formatting helpers for logging
# ============================================================

def _format_conv_full(conv: dict) -> str:
    """Format a full conversation for logging."""
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
# Best-per-group selection
# ============================================================

def select_best_per_group(groups: Dict[int, List[int]],
                          conversations: List[dict],
                          cfg: "AdvancedDedupConfig",
                          bedrock_client=None,
                          stage_name: str = "") -> Tuple[Set[int], List[str], List[dict]]:
    """From groups, keep only the best index per group.

    Uses LLM-as-judge when scoring_method=='llm' for both stages;
    falls back to heuristic when scoring_method=='heuristic' or client is None.

    Returns (keepers, debug_lines, group_details):
      - keepers: set of indices to keep
      - debug_lines: list of debug log lines
      - group_details: list of dicts for report JSON
    """
    keepers: Set[int] = set()
    debug_lines: List[str] = []
    group_details: List[dict] = []
    dup_group_num = 0

    for members in groups.values():
        if len(members) == 1:
            keepers.add(members[0])
            continue

        dup_group_num += 1
        use_llm = (cfg.scoring_method == "llm"
                    and bedrock_client is not None)
        scoring_label = "LLM-judge" if use_llm else "heuristic"

        header = (f"\n{'='*70}\n"
                  f"  [{stage_name}] Duplicate group #{dup_group_num} "
                  f"({len(members)} members, scoring: {scoring_label})\n"
                  f"{'='*70}")
        print(header)
        debug_lines.append(header)

        scores: Dict[int, float] = {}
        for rank, idx in enumerate(members, 1):
            if use_llm:
                score = score_output_quality(
                    conversations[idx], bedrock_client,
                    cfg.judge_model_id, cfg.multi_turn_weight_mode)
                # Fallback to heuristic if LLM scoring failed
                if score < 0:
                    score = compute_heuristic_score(
                        conversations[idx],
                        cfg.weight_completeness, cfg.weight_info_density)
                    score_line = (f"\n  [{rank}] index={idx}  llm_score=FAILED  "
                                  f"heuristic_fallback={score:.2f}")
                else:
                    score_line = f"\n  [{rank}] index={idx}  llm_score={score:.2f}"
            else:
                score = compute_heuristic_score(
                    conversations[idx],
                    cfg.weight_completeness, cfg.weight_info_density)
                score_line = f"\n  [{rank}] index={idx}  heuristic_score={score:.2f}"

            scores[idx] = score
            print(score_line)
            debug_lines.append(score_line)

            conv_text = (f"  ---- conversation ----\n"
                         f"{_format_conv_full(conversations[idx])}")
            print(conv_text)
            debug_lines.append(conv_text)

        best_idx = max(members, key=lambda i: scores[i])
        best_rank = members.index(best_idx) + 1
        selected_line = (f"\n  >>> SELECTED: [{best_rank}] index={best_idx}  "
                         f"score={scores[best_idx]:.2f}  (by {scoring_label})")
        print(selected_line)
        debug_lines.append(selected_line)
        debug_lines.append("  " + "- " * 35)
        debug_lines.append(_format_conv_full(conversations[best_idx]))

        keepers.add(best_idx)

        # Build report group detail
        removed_indices = sorted(i for i in members if i != best_idx)
        full_text = extract_full_text(conversations[best_idx])
        text_preview = (full_text[:200] + "...") if len(full_text) > 200 else full_text
        conv_preview = []
        for msg in conversations[best_idx].get("messages", []):
            role = msg.get("role", "")
            if role in ("user", "assistant"):
                text = _get_msg_text(msg)
                conv_preview.append({"role": role, "text": text[:100]})
        group_details.append({
            "method": stage_name,
            "group_id": dup_group_num,
            "member_count": len(members),
            "member_indices": members,
            "scoring_method": scoring_label,
            "scores": {str(k): round(v, 2) for k, v in scores.items()},
            "kept_index": best_idx,
            "removed_indices": removed_indices,
            "text_preview": text_preview,
            "conversation_preview": conv_preview,
        })

    return keepers, debug_lines, group_details


# ============================================================
# Full pipeline
# ============================================================

def run(cfg: AdvancedDedupConfig, bedrock_client=None,
        embed_client=None) -> dict:
    """Execute the advanced dedup pipeline. Returns stats dict."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_dir = "./debug-log"
    report_dir = "./report"
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    all_debug_lines: List[str] = []
    all_group_details: List[dict] = []

    all_debug_lines.append("=== Advanced Dedup Debug Log ===")
    all_debug_lines.append(f"Timestamp: {timestamp}")
    all_debug_lines.append(f"Input:  {cfg.input}")
    all_debug_lines.append(f"Output: {cfg.output}")
    all_debug_lines.append(
        f"Methods: full_sample_dedup={cfg.enable_full_sample_dedup}, "
        f"input_dedup={cfg.enable_input_dedup}")
    all_debug_lines.append(f"Scoring: {cfg.scoring_method}")
    all_debug_lines.append(f"Full-sample threshold: {cfg.full_sample_threshold}")
    all_debug_lines.append(f"Input threshold: {cfg.input_threshold}")
    all_debug_lines.append(f"Embed model: {cfg.embed_model_id}")
    all_debug_lines.append(f"Judge model: {cfg.judge_model_id}")
    all_debug_lines.append("=" * 60)

    # 1. Load
    conversations: List[dict] = []
    with open(cfg.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                conversations.append(json.loads(line))

    total = len(conversations)
    alive: Set[int] = set(range(total))

    # 2. Create clients if needed
    needs_embed = cfg.enable_full_sample_dedup or cfg.enable_input_dedup
    needs_llm = cfg.scoring_method == "llm" and cfg.enable_input_dedup

    if (needs_embed or needs_llm) and bedrock_client is None:
        if not HAS_BOTO3:
            raise ImportError(
                "boto3 is required for embedding and LLM scoring. "
                "Install it with: pip install boto3"
            )
        bedrock_client = boto3.client(
            "bedrock-runtime", region_name=cfg.bedrock_region)

    if needs_embed and embed_client is None:
        embed_client = bedrock_client

    stats: dict = {
        "total": total,
        "stages": [],
        "final": total,
    }

    # ---- Stage 1: Full-sample embedding dedup ----
    if cfg.enable_full_sample_dedup and len(alive) > 1:
        stage_input = len(alive)
        alive_list = sorted(alive)
        full_texts = [extract_full_text(conversations[i]) for i in alive_list]

        # Generate embeddings
        embeddings = embed_texts(full_texts, embed_client,
                                 cfg.embed_model_id,
                                 cfg.embed_input_type,
                                 cfg.embed_batch_size)

        # Cluster by cosine similarity
        uf = cluster_by_cosine(embeddings, alive_list,
                               cfg.full_sample_threshold)
        groups = uf.groups()

        # Select best per group (heuristic for stage 1)
        keepers, stage_debug, stage_groups = select_best_per_group(
            groups, conversations, cfg,
            bedrock_client,
            stage_name="full_sample_dedup")

        dup_groups = sum(1 for m in groups.values() if len(m) > 1)
        removed = stage_input - len(keepers)
        alive = keepers

        all_debug_lines.append(f"\n{'#'*60}")
        all_debug_lines.append(
            f"# Stage: full_sample_dedup  (input: {stage_input} samples)")
        all_debug_lines.append(f"{'#'*60}")
        all_debug_lines.extend(stage_debug)
        all_debug_lines.append(
            f"\n  --- full_sample_dedup summary: input={stage_input}, "
            f"output={len(keepers)}, groups={dup_groups}, removed={removed} ---")
        all_group_details.extend(stage_groups)

        stage_method_desc = (
            f"Cohere Embed Multilingual v3 full-text embedding dedup: "
            f"concatenate all message texts per conversation, embed via "
            f"Bedrock Cohere, cluster by cosine similarity "
            f"(threshold >= {cfg.full_sample_threshold}), keep best-scored "
            f"per cluster. Duplicate reason: semantically similar full "
            f"conversations.")
        stats["stages"].append({
            "method": "full_sample_embedding_dedup",
            "method_description": stage_method_desc,
            "input_count": stage_input,
            "output_count": len(keepers),
            "dup_groups": dup_groups,
            "removed": removed,
        })

    # ---- Stage 2: Input-level dedup ----
    if cfg.enable_input_dedup and len(alive) > 1:
        stage_input = len(alive)
        alive_list = sorted(alive)
        input_texts = [extract_input_text(conversations[i]) for i in alive_list]

        # Generate embeddings for input texts only
        embeddings = embed_texts(input_texts, embed_client,
                                 cfg.embed_model_id,
                                 cfg.embed_input_type,
                                 cfg.embed_batch_size)

        # Cluster by cosine similarity with lower threshold
        uf = cluster_by_cosine(embeddings, alive_list,
                               cfg.input_threshold)
        groups = uf.groups()

        # Select best per group (LLM scoring for output quality)
        keepers, stage_debug, stage_groups = select_best_per_group(
            groups, conversations, cfg,
            bedrock_client,
            stage_name="input_dedup")

        dup_groups = sum(1 for m in groups.values() if len(m) > 1)
        removed = stage_input - len(keepers)
        alive = keepers

        all_debug_lines.append(f"\n{'#'*60}")
        all_debug_lines.append(
            f"# Stage: input_dedup  (input: {stage_input} samples)")
        all_debug_lines.append(f"{'#'*60}")
        all_debug_lines.extend(stage_debug)
        all_debug_lines.append(
            f"\n  --- input_dedup summary: input={stage_input}, "
            f"output={len(keepers)}, groups={dup_groups}, removed={removed} ---")
        all_group_details.extend(stage_groups)

        stage_method_desc = (
            f"Cohere Embed Multilingual v3 input-only embedding dedup: "
            f"embed only user turns, cluster by cosine similarity "
            f"(threshold >= {cfg.input_threshold}), evaluate output quality "
            f"via {cfg.scoring_method}, keep best answer per cluster. "
            f"Duplicate reason: semantically similar user inputs with "
            f"different quality answers.")
        stats["stages"].append({
            "method": "input_embedding_dedup",
            "method_description": stage_method_desc,
            "input_count": stage_input,
            "output_count": len(keepers),
            "dup_groups": dup_groups,
            "removed": removed,
        })

    stats["final"] = len(alive)

    # 3. Write output
    with open(cfg.output, "w", encoding="utf-8") as fout:
        for i in sorted(alive):
            fout.write(json.dumps(conversations[i], ensure_ascii=False) + "\n")

    # Print summary
    print(f"\nTotal:        {stats['total']}")
    for stage in stats["stages"]:
        print(f"After {stage['method']:36s}: {stage['output_count']}  "
              f"(groups={stage['dup_groups']}, removed={stage['removed']})")
    print(f"Final kept:   {stats['final']}")

    # ---- Write debug log ----
    total_dup_groups = sum(
        s.get("dup_groups", 0) for s in stats["stages"])
    all_debug_lines.append(
        f"\n{'='*60}\n"
        f"Debug log complete. Total duplicate groups processed: "
        f"{total_dup_groups}")

    debug_path = os.path.join(
        debug_dir, f"advanced_dedup_debug_{timestamp}.log")
    with open(debug_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_debug_lines) + "\n")

    # ---- Write report (.json) ----
    total_removed = stats["total"] - stats["final"]
    dedup_rate = (total_removed / stats["total"] * 100
                  if stats["total"] > 0 else 0.0)

    report_json = {
        "timestamp": timestamp,
        "input_file": cfg.input,
        "output_file": cfg.output,
        "summary": {
            "total_samples": stats["total"],
            "final_kept": stats["final"],
            "total_removed": total_removed,
            "dedup_rate": round(dedup_rate, 2),
        },
        "stages": stats["stages"],
        "duplicate_groups": all_group_details,
        "config": {
            "enable_full_sample_dedup": cfg.enable_full_sample_dedup,
            "enable_input_dedup": cfg.enable_input_dedup,
            "embed_model_id": cfg.embed_model_id,
            "embed_batch_size": cfg.embed_batch_size,
            "full_sample_threshold": cfg.full_sample_threshold,
            "input_threshold": cfg.input_threshold,
            "scoring_method": cfg.scoring_method,
            "judge_model_id": cfg.judge_model_id,
            "weight_completeness": cfg.weight_completeness,
            "weight_info_density": cfg.weight_info_density,
            "multi_turn_weight_mode": cfg.multi_turn_weight_mode,
            "bedrock_region": cfg.bedrock_region,
        },
    }

    report_json_path = os.path.join(
        report_dir, f"advanced_dedup_report_{timestamp}.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_json, f, ensure_ascii=False, indent=2)

    # ---- Write report (.txt) ----
    report_lines: List[str] = []
    report_lines.append("=" * 60)
    report_lines.append(
        "  Advanced Dedup LLM SFT Data \u2014 Processing Report")
    report_lines.append("=" * 60)
    report_lines.append("")
    report_lines.append(f"Timestamp : {timestamp}")
    report_lines.append(f"Input     : {cfg.input}")
    report_lines.append(f"Output    : {cfg.output}")
    report_lines.append("")
    report_lines.append("-" * 40)
    report_lines.append("  Sample Statistics")
    report_lines.append("-" * 40)
    report_lines.append(f"  Total samples   : {stats['total']}")
    report_lines.append(f"  Final kept      : {stats['final']}")
    report_lines.append(f"  Total removed   : {total_removed}")
    report_lines.append(f"  Dedup rate      : {dedup_rate:.1f}%")
    report_lines.append("")
    report_lines.append("-" * 40)
    report_lines.append("  Configuration")
    report_lines.append("-" * 40)
    report_lines.append(
        f"  Scoring method          : {cfg.scoring_method}")
    report_lines.append(
        f"  Embed model             : {cfg.embed_model_id}")
    report_lines.append(
        f"  Judge model             : {cfg.judge_model_id}")
    report_lines.append(
        f"  Full-sample threshold   : {cfg.full_sample_threshold}")
    report_lines.append(
        f"  Input threshold         : {cfg.input_threshold}")
    report_lines.append(
        f"  Multi-turn weight mode  : {cfg.multi_turn_weight_mode}")
    report_lines.append("")
    report_lines.append("-" * 40)
    report_lines.append("  Per-Stage Breakdown")
    report_lines.append("-" * 40)

    for stage in stats["stages"]:
        report_lines.append("")
        report_lines.append(f"  [{stage['method']}]")
        if "method_description" in stage:
            report_lines.append(
                f"    Method description : {stage['method_description']}")
        report_lines.append(
            f"    Input count        : {stage['input_count']}")
        report_lines.append(
            f"    Output count       : {stage['output_count']}")
        report_lines.append(
            f"    Duplicate groups   : {stage['dup_groups']}")
        report_lines.append(
            f"    Removed            : {stage['removed']}")

    if all_group_details:
        report_lines.append("")
        report_lines.append("-" * 40)
        report_lines.append("  Duplicate Group Details")
        report_lines.append("-" * 40)

        for gd in all_group_details:
            report_lines.append("")
            report_lines.append(
                f"  [{gd['method']}] Group #{gd['group_id']} "
                f"({gd['member_count']} members)")
            report_lines.append(
                f"    Scoring method   : {gd['scoring_method']}")
            report_lines.append(
                f"    Member indices   : {gd['member_indices']}")
            score_strs = ", ".join(
                f"idx{k}={v}" for k, v in gd["scores"].items())
            report_lines.append(f"    Scores           : {score_strs}")
            report_lines.append(
                f"    Kept index       : {gd['kept_index']}")
            report_lines.append(
                f"    Removed indices  : {gd['removed_indices']}")
            report_lines.append(
                f"    Text preview     : {gd['text_preview']}")
            report_lines.append("    Conversation preview:")
            for cp in gd.get("conversation_preview", []):
                report_lines.append(
                    f"      {cp['role']}: {cp['text']}")

    report_lines.append("")
    report_lines.append("=" * 60)
    report_lines.append("  End of Report")
    report_lines.append("=" * 60)

    report_txt_path = os.path.join(
        report_dir, f"advanced_dedup_report_{timestamp}.txt")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"\nDebug log saved to: {debug_path}")
    print(f"Report saved to:    {report_txt_path}")
    print(f"Report saved to:    {report_json_path}")

    return stats


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Advanced semantic deduplication for Bedrock-format "
                    "SFT conversation JSONL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Full pipeline (embedding dedup + input-level dedup with LLM judge):
  python advanced_dedup_LLM_SFT_data.py -i deduped.jsonl -o advanced_deduped.jsonl

  # Stage 1 only (full-sample embedding dedup):
  python advanced_dedup_LLM_SFT_data.py --no-enable-input-dedup

  # Stage 2 only (input-level dedup with LLM judge):
  python advanced_dedup_LLM_SFT_data.py --no-enable-full-sample-dedup

  # Use heuristic scoring instead of LLM judge:
  python advanced_dedup_LLM_SFT_data.py --scoring-method heuristic
""",
    )

    io_grp = p.add_argument_group("I/O")
    io_grp.add_argument("-i", "--input", default="./zh_mixed_deduped.jsonl",
                        help="Input JSONL path (default: %(default)s)")
    io_grp.add_argument("-o", "--output",
                        default="./zh_mixed_advanced_deduped.jsonl",
                        help="Output JSONL path (default: %(default)s)")

    stages = p.add_argument_group("Stage switches")
    stages.add_argument("--enable-full-sample-dedup",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Enable full-sample embedding dedup (default: on)")
    stages.add_argument("--enable-input-dedup",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Enable input-level dedup with LLM judge "
                             "(default: on)")

    emb = p.add_argument_group("Embedding settings")
    emb.add_argument("--embed-model-id",
                     default="cohere.embed-multilingual-v3",
                     help="Bedrock embedding model ID (default: %(default)s)")
    emb.add_argument("--embed-batch-size", type=int, default=96,
                     help="Texts per embedding API call (default: %(default)s)")
    emb.add_argument("--embed-input-type", default="search_document",
                     help="Cohere input_type parameter (default: %(default)s)")

    thresholds = p.add_argument_group("Similarity thresholds")
    thresholds.add_argument("--full-sample-threshold", type=float, default=0.85,
                            help="Cosine sim threshold for full-sample dedup "
                                 "(default: %(default)s)")
    thresholds.add_argument("--input-threshold", type=float, default=0.7,
                            help="Cosine sim threshold for input-level dedup "
                                 "(default: %(default)s)")

    scoring = p.add_argument_group("Scoring")
    scoring.add_argument("--scoring-method", choices=["heuristic", "llm"],
                         default="llm",
                         help="Method for selecting best per group "
                              "(default: %(default)s)")
    scoring.add_argument("--judge-model-id",
                         default="us.anthropic.claude-opus-4-6-v1",
                         help="Bedrock model ID for LLM judge "
                              "(default: %(default)s)")
    scoring.add_argument("--weight-completeness", type=float, default=0.4,
                         help="Heuristic: completeness weight "
                              "(default: %(default)s)")
    scoring.add_argument("--weight-info-density", type=float, default=0.6,
                         help="Heuristic: info-density weight "
                              "(default: %(default)s)")
    scoring.add_argument("--multi-turn-weight-mode",
                         choices=["equal", "linear_increasing"],
                         default="linear_increasing",
                         help="Multi-turn weighting mode "
                              "(default: %(default)s)")

    bed = p.add_argument_group("Bedrock settings")
    bed.add_argument("--bedrock-region", default="us-east-1",
                     help="AWS region for Bedrock (default: %(default)s)")

    return p


def args_to_config(args: argparse.Namespace) -> AdvancedDedupConfig:
    """Map parsed CLI args -> AdvancedDedupConfig."""
    return AdvancedDedupConfig(
        input=args.input,
        output=args.output,
        enable_full_sample_dedup=args.enable_full_sample_dedup,
        enable_input_dedup=args.enable_input_dedup,
        embed_model_id=args.embed_model_id,
        embed_batch_size=args.embed_batch_size,
        embed_input_type=args.embed_input_type,
        full_sample_threshold=args.full_sample_threshold,
        input_threshold=args.input_threshold,
        scoring_method=args.scoring_method,
        judge_model_id=args.judge_model_id,
        weight_completeness=args.weight_completeness,
        weight_info_density=args.weight_info_density,
        multi_turn_weight_mode=args.multi_turn_weight_mode,
        bedrock_region=args.bedrock_region,
    )


def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = args_to_config(args)
    run(cfg)


if __name__ == "__main__":
    main()
