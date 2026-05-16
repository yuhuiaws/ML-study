#!/usr/bin/env python3
"""
Layer-1 deduplication for Bedrock-format SFT conversation JSONL.

Pipeline:
  1. Load conversations; extract dedup text (all message content concatenated,
     ignoring role and system prompt).
  2. Apply enabled dedup methods in order:
       - exact  (enable_exact):   SHA-256 hash; group identical texts;
         keep best-scored per group.
       - overlap (enable_overlap): character n-gram overlap ratio,
         O(n^2) pairwise comparison; duplicate when ratio >= threshold.
       - lsh    (enable_lsh):     char n-gram shingling -> MinHash signatures
         -> LSH banding -> verify Jaccard >= threshold.
     Each stage groups near-duplicates (Union-Find / hash groups)
     and keeps the best-scored sample per group.
  3. Write survivors to output JSONL; print stats.

Scoring: heuristic (default) or LLM-based via AWS Bedrock Claude API.

All parameters are exposed via CLI flags (see --help).
"""

import argparse
import hashlib
import json
import os
import random
import re
import struct
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False


# ============================================================
# Configuration
# ============================================================

@dataclass
class DedupConfig:
    """All tuneable parameters for the deduplication pipeline."""

    # -- I/O --
    input: str = "./zh_mixed_cleaned.jsonl"
    output: str = "./zh_mixed_deduped.jsonl"

    # -- Dedup method switches (applied in order: exact -> overlap -> lsh) --
    enable_exact: bool = True
    enable_overlap: bool = True
    enable_lsh: bool = True

    # -- Shared --
    ngram_size: int = 3

    # -- Overlap ratio --
    overlap_threshold: float = 0.7   # min overlap coefficient for duplicates

    # -- MinHash LSH --
    minhash_num_perm: int = 128
    minhash_bands: int = 16
    minhash_rows: int = 8
    minhash_threshold: float = 0.7   # min Jaccard for verified duplicate

    # -- Scoring --
    weight_completeness: float = 0.4
    weight_info_density: float = 0.6
    scoring_method: str = "heuristic"  # "heuristic" or "llm"
    bedrock_model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"
    bedrock_region: str = "us-east-1"

    # -- Misc --
    seed: int = 42


# ============================================================
# Text extraction & scoring
# ============================================================

def extract_dedup_text(conv: dict, separator: str = "\n") -> str:
    """Concatenate all message content texts, ignoring role and system prompt."""
    parts = []
    for msg in conv.get("messages", []):
        for block in msg.get("content", []):
            text = block.get("text", "")
            if text:
                parts.append(text)
    return separator.join(parts)


def compute_score(conv: dict, w_completeness: float = 0.4,
                  w_info_density: float = 0.6) -> float:
    """Score a conversation by turns, total length, and avg assistant length.

    Higher is better. Among duplicates the highest-scored sample is kept.
    """
    messages = conv.get("messages", [])
    n_turns = len(messages)
    total_len = 0
    assistant_lens = []
    for msg in messages:
        text = msg.get("content", [{}])[0].get("text", "")
        total_len += len(text)
        if msg.get("role") == "assistant":
            assistant_lens.append(len(text))
    avg_assistant_len = (sum(assistant_lens) / len(assistant_lens)
                         if assistant_lens else 0.0)

    completeness = n_turns + total_len / 100.0
    info_density = avg_assistant_len / 100.0
    return w_completeness * completeness + w_info_density * info_density


# ============================================================
# Hashing utilities
# ============================================================

def sha256_hex(text: str) -> str:
    """SHA-256 hex digest of UTF-8 encoded text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fnv1a_64(data: bytes) -> int:
    """FNV-1a 64-bit hash."""
    h = 0xcbf29ce484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


# ============================================================
# Character n-grams
# ============================================================

def char_ngrams(text: str, n: int = 3) -> List[str]:
    """Generate character n-grams from text."""
    if len(text) < n:
        return [text] if text else []
    return [text[i:i + n] for i in range(len(text) - n + 1)]


# ============================================================
# SimHash (utility — not used in the pipeline, kept for reference)
# ============================================================

def compute_simhash(text: str, ngram_size: int = 3) -> int:
    """Compute a 64-bit SimHash fingerprint from character n-grams."""
    grams = char_ngrams(text, ngram_size)
    if not grams:
        return 0

    v = [0] * 64
    for gram in grams:
        h = fnv1a_64(gram.encode("utf-8"))
        for i in range(64):
            if h & (1 << i):
                v[i] += 1
            else:
                v[i] -= 1

    fingerprint = 0
    for i in range(64):
        if v[i] > 0:
            fingerprint |= (1 << i)
    return fingerprint


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two 64-bit integers (XOR + popcount)."""
    return bin(a ^ b).count("1")


# ============================================================
# Overlap ratio dedup
# ============================================================

def compute_overlap_ratio(ngrams_a: Set[str], ngrams_b: Set[str]) -> float:
    """Overlap coefficient: |A ∩ B| / min(|A|, |B|).

    Returns 0.0 when either set is empty.
    """
    if not ngrams_a or not ngrams_b:
        return 0.0
    intersection = len(ngrams_a & ngrams_b)
    return intersection / min(len(ngrams_a), len(ngrams_b))


def overlap_ratio_dedup(texts: List[str], indices: List[int],
                        ngram_size: int = 3,
                        threshold: float = 0.7) -> "UnionFind":
    """O(n^2) pairwise overlap-ratio comparison; returns UnionFind of near-duplicates."""
    n = len(texts)
    ngram_sets = [set(char_ngrams(t, ngram_size)) for t in texts]
    uf = UnionFind(indices)
    for i in range(n):
        for j in range(i + 1, n):
            if compute_overlap_ratio(ngram_sets[i], ngram_sets[j]) >= threshold:
                uf.union(indices[i], indices[j])
    return uf


# ============================================================
# MinHash
# ============================================================

class MinHasher:
    """MinHash signature generator using random hash functions."""

    def __init__(self, num_perm: int = 128, ngram_size: int = 3,
                 seed: int = 42):
        self.num_perm = num_perm
        self.ngram_size = ngram_size
        self._mersenne_prime = (1 << 61) - 1
        rng = random.Random(seed)
        self._a = [rng.randint(1, self._mersenne_prime - 1)
                    for _ in range(num_perm)]
        self._b = [rng.randint(0, self._mersenne_prime - 1)
                    for _ in range(num_perm)]

    def _hash_token(self, token: str, i: int) -> int:
        h = fnv1a_64(token.encode("utf-8"))
        return ((self._a[i] * h + self._b[i]) % self._mersenne_prime)

    def signature(self, text: str) -> List[int]:
        """Compute MinHash signature (list of num_perm minimum hashes)."""
        grams = char_ngrams(text, self.ngram_size)
        if not grams:
            return [self._mersenne_prime] * self.num_perm
        sig = []
        for i in range(self.num_perm):
            min_h = min(self._hash_token(g, i) for g in grams)
            sig.append(min_h)
        return sig

    def estimated_jaccard(self, sig1: List[int], sig2: List[int]) -> float:
        """Estimate Jaccard similarity from two MinHash signatures."""
        if len(sig1) != len(sig2):
            raise ValueError("Signatures must have the same length")
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)


# ============================================================
# MinHash LSH dedup
# ============================================================

def minhash_lsh_dedup(texts: List[str], indices: List[int],
                      num_perm: int = 128, bands: int = 16, rows: int = 8,
                      threshold: float = 0.7, ngram_size: int = 3,
                      seed: int = 42) -> "UnionFind":
    """MinHash LSH dedup: banding -> candidate pairs -> verify Jaccard -> UnionFind."""
    hasher = MinHasher(num_perm=num_perm, ngram_size=ngram_size, seed=seed)
    signatures = [hasher.signature(t) for t in texts]
    uf = UnionFind(indices)

    # LSH banding: for each band, hash the band slice -> bucket
    candidates: Set[Tuple[int, int]] = set()
    for band_idx in range(bands):
        buckets: Dict[bytes, List[int]] = defaultdict(list)
        start = band_idx * rows
        end = start + rows
        for pos, sig in enumerate(signatures):
            band_slice = tuple(sig[start:end])
            key = struct.pack(f">{rows}q", *band_slice)
            buckets[key].append(pos)
        for bucket in buckets.values():
            if len(bucket) > 1:
                for i in range(len(bucket)):
                    for j in range(i + 1, len(bucket)):
                        candidates.add((bucket[i], bucket[j]))

    # Verify candidates with full Jaccard estimate
    for pos_i, pos_j in candidates:
        jaccard = hasher.estimated_jaccard(signatures[pos_i], signatures[pos_j])
        if jaccard >= threshold:
            uf.union(indices[pos_i], indices[pos_j])

    return uf


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
# Exact dedup grouping
# ============================================================

def exact_dedup_groups(texts: List[str], alive: Set[int]) -> Dict[int, List[int]]:
    """Group indices by SHA-256 hash of their dedup text.

    Returns {representative: [members]} where representative is the
    first member encountered (analogous to UnionFind.groups() format).
    """
    hash_buckets: Dict[str, List[int]] = defaultdict(list)
    for i in alive:
        h = sha256_hex(texts[i])
        hash_buckets[h].append(i)
    # Convert to {first_member: members} format
    groups: Dict[int, List[int]] = {}
    for members in hash_buckets.values():
        groups[members[0]] = members
    return groups


# ============================================================
# LLM scoring (AWS Bedrock Claude)
# ============================================================

def format_conversation_for_scoring(conv: dict) -> str:
    """Format a conversation into a readable string for LLM scoring.

    Uses 用户:/助手: labels. System text is excluded.
    """
    lines = []
    for msg in conv.get("messages", []):
        role = msg.get("role", "")
        if role == "user":
            label = "用户"
        elif role == "assistant":
            label = "助手"
        else:
            continue
        for block in msg.get("content", []):
            text = block.get("text", "")
            if text:
                lines.append(f"{label}: {text}")
    return "\n".join(lines)


def score_group_with_llm(members: List[int], conversations: List[dict],
                         client, model_id: str,
                         w_completeness: float = 0.4,
                         w_info_density: float = 0.6) -> int:
    """Select the best conversation index from a duplicate group using LLM.

    Calls AWS Bedrock Claude to evaluate candidates. Falls back to heuristic
    scoring on any error.

    Returns the index (from members) of the best conversation.
    """
    if len(members) == 1:
        return members[0]

    # Build numbered list of candidates
    candidate_texts = []
    for rank, idx in enumerate(members, 1):
        formatted = format_conversation_for_scoring(conversations[idx])
        candidate_texts.append(f"【样本{rank}】\n{formatted}")

    prompt = (
        "你是一个数据质量评估专家。以下是几个重复或近似重复的对话样本。\n"
        "请根据以下标准评估哪个样本质量最高:\n"
        "1. 完整性: 对话是否完整,回答是否充分\n"
        "2. 连贯性: 对话是否流畅自然\n"
        "3. 信息密度: 回答是否包含丰富有用的信息\n"
        "4. 自然度: 语言是否自然地道\n\n"
        + "\n\n".join(candidate_texts)
        + f"\n\n请只回答最佳样本的编号(1-{len(members)}),不要解释。"
    )


    print("-----use bedrock model ------!!")
    try:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 16,
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
        reply = result["content"][0]["text"].strip()
        match = re.search(r"\d+", reply)
        if match:
            choice = int(match.group())
            if 1 <= choice <= len(members):
                return members[choice - 1]
        # Invalid digit — fall through to heuristic
    except Exception:
        pass

    print("--------fallback to heuristic !!!-----")
    # Fallback to heuristic
    return max(members, key=lambda i: compute_score(
        conversations[i], w_completeness, w_info_density))


# ============================================================
# Best-per-group selection
# ============================================================

def _format_conv_full(conv: dict) -> str:
    """Format a full conversation for logging, showing system + all turns."""
    lines = []
    sys_texts = conv.get("system", [])
    if sys_texts:
        sys_str = sys_texts[0].get("text", "")
        if sys_str:
            lines.append(f"    [system] {sys_str}")
    for msg in conv.get("messages", []):
        role = msg.get("role", "unknown")
        text = msg.get("content", [{}])[0].get("text", "")
        tag = "用户" if role == "user" else "助手"
        lines.append(f"    [{tag}] {text}")
    turns = len(conv.get("messages", []))
    total_len = sum(
        len(m.get("content", [{}])[0].get("text", ""))
        for m in conv.get("messages", [])
    )
    lines.append(f"    -- {turns} turns, {total_len} chars --")
    return "\n".join(lines)


def select_best_per_group(groups: Dict[int, List[int]],
                          conversations: List[dict],
                          texts: List[str],
                          cfg: "DedupConfig",
                          bedrock_client=None,
                          method_name: str = "",
                          debug_fh=None) -> Tuple[Set[int], List[dict]]:
    """From groups, keep only the best index per group.

    Uses LLM scoring if cfg.scoring_method == "llm" and bedrock_client
    is provided; otherwise uses heuristic scoring.
    Writes detailed debug info to debug_fh (file handle).
    Returns (keepers, report_groups) where report_groups contains
    per-group summaries for the processing report.
    """
    keepers: Set[int] = set()
    report_groups: List[dict] = []
    dup_group_num = 0

    def _debug(msg: str):
        if debug_fh is not None:
            debug_fh.write(msg + "\n")

    for members in groups.values():
        if len(members) == 1:
            keepers.add(members[0])
            continue

        dup_group_num += 1
        use_llm = cfg.scoring_method == "llm" and bedrock_client is not None
        scoring_label = "LLM" if use_llm else "heuristic"

        _debug(f"\n{'='*70}")
        _debug(f"  [{method_name}] Duplicate group #{dup_group_num} "
               f"({len(members)} members, scoring: {scoring_label})")
        _debug(f"{'='*70}")

        scores = {}
        for rank, idx in enumerate(members, 1):
            score = compute_score(conversations[idx],
                                  cfg.weight_completeness,
                                  cfg.weight_info_density)
            scores[idx] = score
            _debug(f"\n  [{rank}] index={idx}  heuristic_score={score:.2f}")
            _debug(f"  ---- dedup text (len={len(texts[idx])}) ----")
            _debug(f"    {repr(texts[idx])}")
            _debug(f"  ---- conversation ----")
            _debug(_format_conv_full(conversations[idx]))

        if use_llm:
            best_idx = score_group_with_llm(
                members, conversations, bedrock_client,
                cfg.bedrock_model_id,
                cfg.weight_completeness, cfg.weight_info_density)
        else:
            best_idx = max(members, key=lambda i: scores[i])

        best_rank = members.index(best_idx) + 1
        _debug(f"\n  >>> SELECTED: [{best_rank}] index={best_idx}  "
               f"heuristic_score={scores[best_idx]:.2f}  (by {scoring_label})")
        _debug(f"  {'- '*35}")
        _debug(_format_conv_full(conversations[best_idx]))

        keepers.add(best_idx)

        # Build a short conversation preview for the report
        first_conv = conversations[members[0]]
        preview_parts = []
        for msg in first_conv.get("messages", [])[:2]:
            role = msg.get("role", "")
            text = msg.get("content", [{}])[0].get("text", "")
            tag = "user" if role == "user" else "assistant"
            preview_parts.append({"role": tag, "text": text[:120]})

        report_groups.append({
            "method": method_name,
            "group_id": dup_group_num,
            "member_count": len(members),
            "member_indices": members,
            "scoring_method": scoring_label,
            "scores": {str(idx): round(scores[idx], 2) for idx in members},
            "kept_index": best_idx,
            "removed_indices": [idx for idx in members if idx != best_idx],
            "text_preview": texts[members[0]][:200],
            "conversation_preview": preview_parts,
        })

    return keepers, report_groups


# ============================================================
# Report writing
# ============================================================

def _method_description(method_name: str, cfg: "DedupConfig") -> str:
    """Return a human-readable description of how a dedup method works."""
    if method_name == "exact":
        return ("SHA-256 exact hash: concatenate all message texts, compute "
                "SHA-256 digest, group samples with identical hashes. "
                "Duplicate reason: texts are character-for-character identical.")
    elif method_name == "overlap":
        return (f"Character {cfg.ngram_size}-gram overlap ratio: compute pairwise "
                f"|A∩B|/min(|A|,|B|) for all surviving samples. "
                f"Duplicate when ratio >= {cfg.overlap_threshold:.2f}. "
                f"Duplicate reason: high character n-gram overlap between texts.")
    elif method_name == "lsh":
        return (f"MinHash LSH: {cfg.minhash_num_perm} permutations, "
                f"{cfg.minhash_bands} bands × {cfg.minhash_rows} rows. "
                f"Candidate pairs from banding, verified with Jaccard >= "
                f"{cfg.minhash_threshold:.2f}. "
                f"Duplicate reason: high estimated Jaccard similarity "
                f"of character {cfg.ngram_size}-gram sets.")
    return ""


def _write_report_txt(path: str, timestamp: str, cfg: "DedupConfig",
                      stats: dict, all_report_groups: List[dict]):
    """Write a human-readable text report."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  Basic Dedup LLM SFT Data — Processing Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp : {timestamp}\n")
        f.write(f"Input     : {cfg.input}\n")
        f.write(f"Output    : {cfg.output}\n\n")

        # Overall summary
        f.write("-" * 40 + "\n")
        f.write("  Sample Statistics\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Total samples   : {stats['total']}\n")
        f.write(f"  Final kept      : {stats['final']}\n")
        total_removed = stats['total'] - stats['final']
        rate = (total_removed / stats['total'] * 100) if stats['total'] else 0
        f.write(f"  Total removed   : {total_removed}\n")
        f.write(f"  Dedup rate      : {rate:.1f}%\n\n")

        # Per-stage breakdown
        f.write("-" * 40 + "\n")
        f.write("  Per-Stage Breakdown\n")
        f.write("-" * 40 + "\n")
        for stage in stats["stages"]:
            m = stage["method"]
            f.write(f"\n  [{m}]\n")
            f.write(f"    Method description : {_method_description(m, cfg)}\n")
            f.write(f"    Input count        : {stage['input_count']}\n")
            f.write(f"    Output count       : {stage['output_count']}\n")
            f.write(f"    Duplicate groups   : {stage['dup_groups']}\n")
            f.write(f"    Removed            : {stage['removed']}\n")

        # Duplicate group details
        if all_report_groups:
            f.write("\n" + "-" * 40 + "\n")
            f.write("  Duplicate Group Details\n")
            f.write("-" * 40 + "\n")
            for g in all_report_groups:
                f.write(f"\n  [{g['method']}] Group #{g['group_id']} "
                        f"({g['member_count']} members)\n")
                f.write(f"    Duplicate reason : "
                        f"{_method_description(g['method'], cfg)}\n")
                f.write(f"    Scoring method   : {g['scoring_method']}\n")
                f.write(f"    Member indices   : {g['member_indices']}\n")
                f.write(f"    Scores           : "
                        f"{', '.join(f'idx{k}={v}' for k, v in g['scores'].items())}\n")
                f.write(f"    Kept index       : {g['kept_index']}\n")
                f.write(f"    Removed indices  : {g['removed_indices']}\n")
                f.write(f"    Text preview     : {g['text_preview'][:100]}...\n")
                if g.get("conversation_preview"):
                    f.write(f"    Conversation preview:\n")
                    for turn in g["conversation_preview"]:
                        f.write(f"      {turn['role']}: {turn['text']}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("  End of Report\n")
        f.write("=" * 60 + "\n")


def _write_report_json(path: str, timestamp: str, cfg: "DedupConfig",
                       stats: dict, all_report_groups: List[dict]):
    """Write a structured JSON report."""
    report = {
        "timestamp": timestamp,
        "input_file": cfg.input,
        "output_file": cfg.output,
        "summary": {
            "total_samples": stats["total"],
            "final_kept": stats["final"],
            "total_removed": stats["total"] - stats["final"],
            "dedup_rate": round(
                (stats["total"] - stats["final"]) / stats["total"] * 100, 2
            ) if stats["total"] else 0,
        },
        "stages": [],
        "duplicate_groups": [],
        "config": {
            "enable_exact": cfg.enable_exact,
            "enable_overlap": cfg.enable_overlap,
            "enable_lsh": cfg.enable_lsh,
            "ngram_size": cfg.ngram_size,
            "overlap_threshold": cfg.overlap_threshold,
            "minhash_num_perm": cfg.minhash_num_perm,
            "minhash_bands": cfg.minhash_bands,
            "minhash_rows": cfg.minhash_rows,
            "minhash_threshold": cfg.minhash_threshold,
            "scoring_method": cfg.scoring_method,
            "weight_completeness": cfg.weight_completeness,
            "weight_info_density": cfg.weight_info_density,
            "seed": cfg.seed,
        },
    }
    for stage in stats["stages"]:
        report["stages"].append({
            "method": stage["method"],
            "method_description": _method_description(stage["method"], cfg),
            "input_count": stage["input_count"],
            "output_count": stage["output_count"],
            "duplicate_groups": stage["dup_groups"],
            "removed": stage["removed"],
        })
    for g in all_report_groups:
        report["duplicate_groups"].append({
            "method": g["method"],
            "duplicate_reason": _method_description(g["method"], cfg),
            "group_id": g["group_id"],
            "member_count": g["member_count"],
            "member_indices": g["member_indices"],
            "scoring_method": g["scoring_method"],
            "scores": g["scores"],
            "kept_index": g["kept_index"],
            "removed_indices": g["removed_indices"],
            "text_preview": g["text_preview"],
            "conversation_preview": g.get("conversation_preview", []),
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


# ============================================================
# Full pipeline
# ============================================================

def run(cfg: DedupConfig, bedrock_client=None) -> dict:
    """Execute the full dedup pipeline. Returns stats dict."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Resolve paths relative to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    debug_dir = os.path.join(script_dir, "debug-log")
    report_dir = os.path.join(script_dir, "report")
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    debug_path = os.path.join(debug_dir, f"basic_dedup_debug_{timestamp}.log")
    report_txt_path = os.path.join(report_dir, f"basic_dedup_report_{timestamp}.txt")
    report_json_path = os.path.join(report_dir, f"basic_dedup_report_{timestamp}.json")

    # 1. Load
    conversations: List[dict] = []
    with open(cfg.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                conversations.append(json.loads(line))

    total = len(conversations)
    texts = [extract_dedup_text(c) for c in conversations]
    alive = set(range(total))  # indices that survive each stage

    # Create Bedrock client if needed
    if cfg.scoring_method == "llm" and bedrock_client is None:
        if not HAS_BOTO3:
            raise ImportError(
                "boto3 is required for LLM scoring. "
                "Install it with: pip install boto3"
            )
        bedrock_client = boto3.client(
            "bedrock-runtime", region_name=cfg.bedrock_region)

    stats: dict = {
        "total": total,
        "stages": [],
        "final": total,
    }
    all_report_groups: List[dict] = []

    # Open debug log
    debug_fh = open(debug_path, "w", encoding="utf-8")
    debug_fh.write(f"=== Basic Dedup Debug Log ===\n")
    debug_fh.write(f"Timestamp: {timestamp}\n")
    debug_fh.write(f"Input:  {cfg.input}\n")
    debug_fh.write(f"Output: {cfg.output}\n")
    debug_fh.write(f"Methods: exact={cfg.enable_exact}, "
                   f"overlap={cfg.enable_overlap}, lsh={cfg.enable_lsh}\n")
    debug_fh.write(f"{'='*60}\n")

    # 2. Apply enabled dedup methods in order: exact -> overlap -> lsh
    methods_to_run: List[Tuple[str, bool]] = [
        ("exact", cfg.enable_exact),
        ("overlap", cfg.enable_overlap),
        ("lsh", cfg.enable_lsh),
    ]

    for method_name, enabled in methods_to_run:
        if not enabled:
            continue

        input_count = len(alive)
        debug_fh.write(f"\n{'#'*60}\n")
        debug_fh.write(f"# Stage: {method_name}  (input: {input_count} samples)\n")
        debug_fh.write(f"{'#'*60}\n")

        if method_name == "exact":
            groups = exact_dedup_groups(texts, alive)
        elif method_name == "overlap":
            alive_list = sorted(alive)
            alive_texts = [texts[i] for i in alive_list]
            uf = overlap_ratio_dedup(alive_texts, alive_list,
                                     ngram_size=cfg.ngram_size,
                                     threshold=cfg.overlap_threshold)
            groups = uf.groups()
        elif method_name == "lsh":
            alive_list = sorted(alive)
            alive_texts = [texts[i] for i in alive_list]
            uf = minhash_lsh_dedup(alive_texts, alive_list,
                                   num_perm=cfg.minhash_num_perm,
                                   bands=cfg.minhash_bands,
                                   rows=cfg.minhash_rows,
                                   threshold=cfg.minhash_threshold,
                                   ngram_size=cfg.ngram_size,
                                   seed=cfg.seed)
            groups = uf.groups()

        keepers, report_groups = select_best_per_group(
            groups, conversations, texts, cfg,
            bedrock_client, method_name=method_name,
            debug_fh=debug_fh)
        all_report_groups.extend(report_groups)

        dup_groups = sum(1 for m in groups.values() if len(m) > 1)
        removed = input_count - len(keepers)
        alive = keepers

        stage_info = {
            "method": method_name,
            "input_count": input_count,
            "output_count": len(keepers),
            "dup_groups": dup_groups,
            "removed": removed,
        }
        stats["stages"].append(stage_info)

        debug_fh.write(f"\n  --- {method_name} summary: "
                       f"input={input_count}, output={len(keepers)}, "
                       f"groups={dup_groups}, removed={removed} ---\n")

    stats["final"] = len(alive)

    # Close debug log
    total_dup_groups = sum(s["dup_groups"] for s in stats["stages"])
    debug_fh.write(f"\n{'='*60}\n")
    debug_fh.write(f"Debug log complete. "
                   f"Total duplicate groups processed: {total_dup_groups}\n")
    debug_fh.close()

    # 3. Write output
    with open(cfg.output, "w", encoding="utf-8") as fout:
        for i in sorted(alive):
            fout.write(json.dumps(conversations[i], ensure_ascii=False) + "\n")

    # 4. Write reports
    _write_report_txt(report_txt_path, timestamp, cfg, stats, all_report_groups)
    _write_report_json(report_json_path, timestamp, cfg, stats, all_report_groups)

    # Print summary to stdout
    print(f"Total:        {stats['total']}")
    for stage in stats["stages"]:
        print(f"After {stage['method']:8s}: {stage['output_count']}  "
              f"(groups={stage['dup_groups']}, removed={stage['removed']})")
    print(f"Final kept:   {stats['final']}")
    print(f"\nDebug log:  {debug_path}")
    print(f"Report:     {report_txt_path}")
    print(f"Report:     {report_json_path}")

    return stats


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Layer-1 deduplication for Bedrock-format SFT conversation JSONL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Default (exact + overlap ratio dedup):
  python basic_dedup_LLM_SFT_data.py -i zh_mixed_cleaned.jsonl -o zh_mixed_deduped.jsonl

  # Exact dedup only:
  python basic_dedup_LLM_SFT_data.py --no-enable-overlap

  # All three methods:
  python basic_dedup_LLM_SFT_data.py --enable-lsh

  # LSH only:
  python basic_dedup_LLM_SFT_data.py --no-enable-exact --no-enable-overlap --enable-lsh

  # LLM-based scoring with Bedrock:
  python basic_dedup_LLM_SFT_data.py --scoring-method llm --bedrock-region us-west-2
""",
    )

    io_grp = p.add_argument_group("I/O")
    io_grp.add_argument("-i", "--input", default="./zh_mixed_cleaned.jsonl",
                        help="Input JSONL path (default: %(default)s)")
    io_grp.add_argument("-o", "--output", default="./zh_mixed_deduped.jsonl",
                        help="Output JSONL path (default: %(default)s)")

    methods = p.add_argument_group("Dedup method switches")
    methods.add_argument("--enable-exact",
                         action=argparse.BooleanOptionalAction,
                         default=True,
                         help="Enable exact (SHA-256) dedup (default: enabled)")
    methods.add_argument("--enable-overlap",
                         action=argparse.BooleanOptionalAction,
                         default=True,
                         help="Enable overlap-ratio dedup (default: enabled)")
    methods.add_argument("--enable-lsh",
                         action=argparse.BooleanOptionalAction,
                         default=True,
                         help="Enable LSH (MinHash) dedup (default: enabled)")
    methods.add_argument("--ngram-size", type=int, default=3,
                         help="Character n-gram size (default: %(default)s)")

    ov = p.add_argument_group("Overlap ratio options")
    ov.add_argument("--overlap-threshold", type=float, default=0.7,
                    help="Min overlap coefficient for duplicates (default: %(default)s)")

    mh = p.add_argument_group("MinHash LSH options")
    mh.add_argument("--minhash-num-perm", type=int, default=128,
                    help="Number of MinHash permutations (default: %(default)s)")
    mh.add_argument("--minhash-bands", type=int, default=16,
                    help="Number of LSH bands (default: %(default)s)")
    mh.add_argument("--minhash-rows", type=int, default=8,
                    help="Rows per LSH band (default: %(default)s)")
    mh.add_argument("--minhash-threshold", type=float, default=0.7,
                    help="Min Jaccard for MinHash duplicate (default: %(default)s)")

    scoring = p.add_argument_group("Scoring")
    scoring.add_argument("--weight-completeness", type=float, default=0.4,
                         help="Weight for completeness score (default: %(default)s)")
    scoring.add_argument("--weight-info-density", type=float, default=0.6,
                         help="Weight for info-density score (default: %(default)s)")
    scoring.add_argument("--scoring-method", choices=["heuristic", "llm"],
                         default="llm",
                         help="Scoring method for selecting best sample (default: %(default)s)")
    scoring.add_argument("--bedrock-model-id",
                         default="global.anthropic.claude-opus-4-6-v1",
                         help="Bedrock model ID for LLM scoring (default: %(default)s)")
    scoring.add_argument("--bedrock-region", default="us-east-1",
                         help="AWS region for Bedrock (default: %(default)s)")

    misc = p.add_argument_group("Misc")
    misc.add_argument("--seed", type=int, default=42,
                      help="Random seed for MinHash (default: %(default)s)")

    return p


def args_to_config(args: argparse.Namespace) -> DedupConfig:
    """Map parsed CLI args -> DedupConfig."""
    return DedupConfig(
        input=args.input,
        output=args.output,
        enable_exact=args.enable_exact,
        enable_overlap=args.enable_overlap,
        enable_lsh=args.enable_lsh,
        ngram_size=args.ngram_size,
        overlap_threshold=args.overlap_threshold,
        minhash_num_perm=args.minhash_num_perm,
        minhash_bands=args.minhash_bands,
        minhash_rows=args.minhash_rows,
        minhash_threshold=args.minhash_threshold,
        weight_completeness=args.weight_completeness,
        weight_info_density=args.weight_info_density,
        scoring_method=args.scoring_method,
        bedrock_model_id=args.bedrock_model_id,
        bedrock_region=args.bedrock_region,
        seed=args.seed,
    )


def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = args_to_config(args)
    run(cfg)


if __name__ == "__main__":
    main()
