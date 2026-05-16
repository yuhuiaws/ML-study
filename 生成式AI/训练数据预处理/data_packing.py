#!/usr/bin/env python3
"""
Data packing and version management for Bedrock-format SFT conversation JSONL.

This is the final stage of the SFT data preprocessing pipeline.  It takes the
cleaned, deduplicated, quality-filtered, and distribution-balanced JSONL and
produces training-ready artefacts:

Pipeline (each step is optional):
  1. Apply chat template   — convert Bedrock-format conversations into the
                             tokeniser's native chat format (e.g. Qwen3).
  2. Sample packing        — merge multiple short samples into one long sample,
                             separated by a special token, to maximise GPU
                             utilisation during training.
  3. Tokenization          — tokenise the (possibly packed) text and build
                             ``input_ids`` / ``labels`` / ``attention_mask``.
  4. Final format export   — write the result as Arrow or Parquet (optionally
                             with compression) for fast data-loading.

All parameters are exposed via CLI flags (see --help).
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

try:
    from transformers import AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# ============================================================
# Debug log collector
# ============================================================

_debug_lines: List[str] = []


def _debug(msg: str) -> None:
    """Append a debug message to the collector and print it."""
    _debug_lines.append(msg)
    print(msg)


IGNORE_INDEX = -100
"""Label value that tells the loss function to skip a token position."""


# ============================================================
# Configuration
# ============================================================

@dataclass
class PackConfig:
    """All tuneable parameters for data packing & version management."""

    # -- I/O --
    input: str = "./zh_mixed_annotated.jsonl"
    output: str = "./train_packed"
    output_format: str = "parquet"          # "parquet" | "arrow"
    parquet_compression: str = "snappy"     # "snappy" | "gzip" | "zstd" | "none"

    # -- Chat template --
    enable_chat_template: bool = True
    model_id: str = "Qwen/Qwen3-8B"
    # If True, use tokenizer.apply_chat_template; otherwise just concatenate
    # the raw messages as plain text.
    add_generation_prompt: bool = False

    # -- Sample packing --
    enable_packing: bool = True
    max_seq_len: int = 4096
    packing_separator: str = ""             # auto-detect from tokenizer if empty
    # If True, drop the last packed sample if it is less than
    # ``min_packing_ratio`` * max_seq_len tokens long.
    drop_short_tail: bool = False
    min_packing_ratio: float = 0.5

    # -- Tokenization --
    enable_tokenization: bool = True
    # When tokenizing packed samples we do NOT set per-sub-sample attention
    # masks (simplified, as stated in the requirements).

    # -- Version / metadata --
    version_tag: str = ""                   # e.g. "v1.0"


# ============================================================
# Helpers: Bedrock → plain messages
# ============================================================

def extract_messages(conversation: Dict[str, Any]) -> List[Dict[str, str]]:
    """Convert a Bedrock-format conversation to a list of
    ``{"role": ..., "content": ...}`` dicts suitable for
    ``tokenizer.apply_chat_template``.

    The *system* field (if present) is emitted as a single
    ``{"role": "system", "content": ...}`` entry at the beginning.
    """
    msgs: List[Dict[str, str]] = []

    # System prompt
    sys_parts = conversation.get("system") or []
    sys_text = " ".join(p.get("text", "") for p in sys_parts).strip()
    if sys_text:
        msgs.append({"role": "system", "content": sys_text})

    for m in conversation.get("messages", []):
        role = m["role"]
        parts = m.get("content", [])
        text = " ".join(p.get("text", "") for p in parts).strip()
        if text:
            msgs.append({"role": role, "content": text})

    return msgs


def messages_to_plain_text(msgs: List[Dict[str, str]]) -> str:
    """Fallback formatter when no tokenizer is available.

    Each message is rendered as ``<|role|>\\ncontent\\n``.
    """
    parts: List[str] = []
    for m in msgs:
        parts.append(f"<|{m['role']}|>\n{m['content']}")
    return "\n".join(parts)


# ============================================================
# Step 1 — Apply chat template
# ============================================================

def apply_chat_template(
    conversations: List[Dict[str, Any]],
    tokenizer: Any,
    add_generation_prompt: bool = False,
) -> List[str]:
    """Apply the tokenizer's chat template to every conversation.

    Returns a list of formatted strings (one per conversation).
    """
    results: List[str] = []
    for conv in conversations:
        msgs = extract_messages(conv)
        try:
            text = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            # Fallback: plain-text rendering
            text = messages_to_plain_text(msgs)
        results.append(text)
    return results


def apply_plain_text_template(
    conversations: List[Dict[str, Any]],
) -> List[str]:
    """Convert conversations to plain text without a tokenizer."""
    return [messages_to_plain_text(extract_messages(c)) for c in conversations]


# ============================================================
# Step 2 — Sample packing
# ============================================================

def resolve_packing_separator(cfg: PackConfig, tokenizer: Any = None) -> str:
    """Determine the separator token used between packed samples.

    Priority:
      1. Explicit ``cfg.packing_separator`` if non-empty.
      2. The tokenizer's EOS token (if a tokenizer is provided).
      3. A generic fallback ``"<|endoftext|>"``.
    """
    if cfg.packing_separator:
        return cfg.packing_separator
    if tokenizer is not None:
        eos = getattr(tokenizer, "eos_token", None)
        if eos:
            return eos
    return "<|endoftext|>"


def pack_samples(
    texts: List[str],
    tokenizer: Any,
    max_seq_len: int,
    separator: str,
    drop_short_tail: bool = False,
    min_packing_ratio: float = 0.5,
) -> List[str]:
    """Greedy bin-packing: concatenate short samples (separated by
    *separator*) until adding the next sample would exceed *max_seq_len*
    tokens.

    Returns a list of packed text strings.
    """
    packed: List[str] = []
    current_parts: List[str] = []
    current_token_count = 0

    sep_token_count = len(tokenizer.encode(separator, add_special_tokens=False))

    for text in texts:
        text_tokens = len(tokenizer.encode(text, add_special_tokens=False))

        if text_tokens > max_seq_len:
            # Single sample already exceeds limit — emit as-is.
            if current_parts:
                packed.append(separator.join(current_parts))
                current_parts = []
                current_token_count = 0
            packed.append(text)
            continue

        # Cost of adding this sample to the current bin.
        cost = text_tokens + (sep_token_count if current_parts else 0)

        if current_token_count + cost > max_seq_len:
            # Flush current bin.
            packed.append(separator.join(current_parts))
            current_parts = [text]
            current_token_count = text_tokens
        else:
            current_parts.append(text)
            current_token_count += cost

    # Flush remainder.
    if current_parts:
        if drop_short_tail and current_token_count < min_packing_ratio * max_seq_len:
            pass  # discard short tail
        else:
            packed.append(separator.join(current_parts))

    return packed


def pack_samples_no_tokenizer(
    texts: List[str],
    max_seq_len: int,
    separator: str,
    drop_short_tail: bool = False,
    min_packing_ratio: float = 0.5,
    chars_per_token: float = 3.5,
) -> List[str]:
    """Character-based packing fallback when no tokenizer is available.

    ``chars_per_token`` is a rough estimate used to convert character
    counts to approximate token counts.
    """
    packed: List[str] = []
    current_parts: List[str] = []
    current_est = 0

    sep_est = len(separator) / chars_per_token

    for text in texts:
        est = len(text) / chars_per_token

        if est > max_seq_len:
            if current_parts:
                packed.append(separator.join(current_parts))
                current_parts = []
                current_est = 0
            packed.append(text)
            continue

        cost = est + (sep_est if current_parts else 0)

        if current_est + cost > max_seq_len:
            packed.append(separator.join(current_parts))
            current_parts = [text]
            current_est = est
        else:
            current_parts.append(text)
            current_est += cost

    if current_parts:
        if drop_short_tail and current_est < min_packing_ratio * max_seq_len:
            pass
        else:
            packed.append(separator.join(current_parts))

    return packed


# ============================================================
# Step 3 — Tokenization
# ============================================================

def tokenize_texts(
    texts: List[str],
    tokenizer: Any,
    max_seq_len: int,
) -> List[Dict[str, List[int]]]:
    """Tokenize a list of texts and return dicts with
    ``input_ids``, ``attention_mask``, and ``labels``.

    For simplicity (as stated in requirements), the attention_mask is a
    simple binary mask (1 for real tokens, 0 for padding) — no
    per-sub-sample custom mask for packed sequences.

    Labels are a copy of input_ids (causal LM objective) with padding
    positions set to -100.
    """
    results: List[Dict[str, List[int]]] = []
    for text in texts:
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
            return_attention_mask=True,
        )
        input_ids = enc["input_ids"]
        attn_mask = enc["attention_mask"]
        labels = [tid if m == 1 else -100 for tid, m in zip(input_ids, attn_mask)]
        results.append({
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels,
        })
    return results


# ============================================================
# Step 3b — Role-aware tokenization (assistant-only loss)
# ============================================================

def _probe_chat_markers(tokenizer: Any) -> Tuple[Optional[str], Optional[str]]:
    """Discover the end-of-turn marker and the assistant role header from the
    tokenizer's chat template by applying it to a probe conversation.

    Returns ``(assistant_header_suffix, eot_marker)``.

    *   ``assistant_header_suffix`` — the text right before the assistant
        content (e.g. ``"assistant\\n"``).
    *   ``eot_marker`` — the end-of-turn marker after the content
        (e.g. ``"<|im_end|>"``).
    """
    probe_msgs = [
        {"role": "user", "content": "XYZZY_PROBE_USER"},
        {"role": "assistant", "content": "XYZZY_PROBE_ASST"},
    ]
    try:
        probe_text = tokenizer.apply_chat_template(
            probe_msgs, tokenize=False, add_generation_prompt=False,
        )
    except Exception:
        return None, None

    asst_idx = probe_text.find("XYZZY_PROBE_ASST")
    if asst_idx < 0:
        return None, None

    # --- assistant header suffix ---
    # Walk backward from asst_idx to find the newline after the role name.
    # E.g. in "…<|im_start|>assistant\nXYZZY…" the header suffix is
    # "assistant\n".  We intentionally stop at the first newline after
    # "assistant" so that model-specific tags like Qwen3's ``<think>``
    # are treated as part of the assistant *output* (and included in the
    # loss), not as part of the header.
    header_region = probe_text[:asst_idx]
    asst_name_pos = header_region.rfind("assistant")
    if asst_name_pos >= 0:
        newline_after = header_region.find("\n", asst_name_pos)
        if newline_after >= 0:
            assistant_header_suffix = header_region[asst_name_pos:newline_after + 1]
        else:
            assistant_header_suffix = header_region[asst_name_pos:]
    else:
        assistant_header_suffix = None

    # --- end-of-turn marker ---
    footer = probe_text[asst_idx + len("XYZZY_PROBE_ASST"):]
    eot = footer.rstrip()
    if not eot:
        eot = None

    return assistant_header_suffix, eot


def tokenize_conversation_with_labels(
    conversation: Dict[str, Any],
    tokenizer: Any,
    add_generation_prompt: bool = False,
) -> Tuple[List[int], List[int]]:
    """Tokenize a single conversation with role-aware label masking.

    System and user tokens receive ``IGNORE_INDEX`` (-100) as labels;
    only assistant tokens retain the real token IDs so that the loss is
    computed exclusively on the model's responses.

    **Strategy** — two methods are tried in order:

    1. **Offset-mapping method** (preferred): tokenize with
       ``return_offsets_mapping=True`` and locate each assistant turn's
       character span in the formatted text via string matching.  This
       is robust to template variations such as Qwen3's ``<think>`` tags
       which are only added to the last assistant turn.

    2. **Progressive prefix comparison** (fallback): apply the chat
       template to progressively longer message prefixes and compare
       token counts.  Used when offset mapping is not available (e.g.
       slow tokenizers or mock tokenizers in tests).

    Returns ``(input_ids, labels)`` — both **unpadded**.
    """
    msgs = extract_messages(conversation)
    if not msgs:
        return [], []

    # Full formatted text
    try:
        full_text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_generation_prompt,
        )
    except Exception:
        full_text = messages_to_plain_text(msgs)

    # ------------------------------------------------------------------
    # Try the offset-mapping method first
    # ------------------------------------------------------------------
    use_offsets = False
    offsets = None
    full_ids = None
    try:
        enc = tokenizer(
            full_text, add_special_tokens=False, return_offsets_mapping=True,
        )
        if enc.get("offset_mapping") is not None:
            full_ids = enc["input_ids"]
            offsets = enc["offset_mapping"]
            use_offsets = True
    except Exception:
        pass

    if full_ids is None:
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    total_len = len(full_ids)
    labels = [IGNORE_INDEX] * total_len

    if use_offsets:
        # ==============================================================
        # Method 1 — offset mapping  (handles Qwen3 <think> tags etc.)
        # ==============================================================
        asst_header_suffix, eot = _probe_chat_markers(tokenizer)

        # Locate every message's content in the full text (in order).
        search_from = 0
        for msg in msgs:
            content = msg["content"]
            if not content:
                continue
            content_start = full_text.find(content, search_from)
            if content_start < 0:
                continue
            content_end = content_start + len(content)

            if msg["role"] == "assistant":
                # ---- span start: right after the assistant header ----
                region_before = full_text[search_from:content_start]
                if asst_header_suffix:
                    hdr_pos = region_before.find(asst_header_suffix)
                    if hdr_pos >= 0:
                        span_start = (
                            search_from + hdr_pos + len(asst_header_suffix)
                        )
                    else:
                        span_start = content_start
                else:
                    span_start = content_start

                # ---- span end: after the end-of-turn marker ----------
                if eot:
                    eot_pos = full_text.find(eot, content_end)
                    if eot_pos >= 0:
                        span_end = eot_pos + len(eot)
                        # Include the trailing newline if present
                        if (span_end < len(full_text)
                                and full_text[span_end] == "\n"):
                            span_end += 1
                    else:
                        span_end = content_end
                else:
                    span_end = content_end

                # Map [span_start, span_end) to token indices via offsets
                for tok_idx, (tok_cs, tok_ce) in enumerate(offsets):
                    if tok_ce > span_start and tok_cs < span_end:
                        labels[tok_idx] = full_ids[tok_idx]

            search_from = content_end

    else:
        # ==============================================================
        # Method 2 — progressive prefix comparison  (fallback)
        # ==============================================================
        for i, msg in enumerate(msgs):
            if msg["role"] != "assistant":
                continue

            prefix_msgs = msgs[:i]
            including_msgs = msgs[: i + 1]

            if prefix_msgs:
                try:
                    prefix_text = tokenizer.apply_chat_template(
                        prefix_msgs, tokenize=False, add_generation_prompt=True,
                    )
                except Exception:
                    prefix_text = messages_to_plain_text(prefix_msgs)
            else:
                prefix_text = ""

            try:
                including_text = tokenizer.apply_chat_template(
                    including_msgs, tokenize=False, add_generation_prompt=False,
                )
            except Exception:
                including_text = messages_to_plain_text(including_msgs)

            prefix_ids = (
                tokenizer.encode(prefix_text, add_special_tokens=False)
                if prefix_text
                else []
            )
            including_ids = tokenizer.encode(
                including_text, add_special_tokens=False,
            )

            start_idx = len(prefix_ids)
            end_idx = len(including_ids)

            for j in range(start_idx, min(end_idx, total_len)):
                labels[j] = full_ids[j]

    return full_ids, labels


def pack_token_samples(
    token_samples: List[Tuple[List[int], List[int]]],
    max_seq_len: int,
    separator_ids: List[int],
    drop_short_tail: bool = False,
    min_packing_ratio: float = 0.5,
) -> List[Tuple[List[int], List[int]]]:
    """Greedy bin-packing at the **token** level.

    Each element of *token_samples* is an ``(input_ids, labels)`` pair
    produced by :func:`tokenize_conversation_with_labels`.  Samples are
    concatenated with *separator_ids* between them; separator tokens
    always receive ``IGNORE_INDEX`` as labels.
    """
    packed: List[Tuple[List[int], List[int]]] = []
    cur_ids: List[int] = []
    cur_labels: List[int] = []

    sep_len = len(separator_ids)

    for ids, lbls in token_samples:
        sample_len = len(ids)

        if sample_len > max_seq_len:
            # Flush current bin
            if cur_ids:
                packed.append((cur_ids, cur_labels))
                cur_ids, cur_labels = [], []
            # Oversized sample — truncate and emit
            packed.append((ids[:max_seq_len], lbls[:max_seq_len]))
            continue

        cost = sample_len + (sep_len if cur_ids else 0)

        if len(cur_ids) + cost > max_seq_len:
            # Flush current bin
            packed.append((cur_ids, cur_labels))
            cur_ids = list(ids)
            cur_labels = list(lbls)
        else:
            if cur_ids:
                cur_ids.extend(separator_ids)
                cur_labels.extend([IGNORE_INDEX] * sep_len)
            cur_ids.extend(ids)
            cur_labels.extend(lbls)

    if cur_ids:
        if drop_short_tail and len(cur_ids) < min_packing_ratio * max_seq_len:
            pass  # discard short tail
        else:
            packed.append((cur_ids, cur_labels))

    return packed


def pad_token_samples(
    packed_samples: List[Tuple[List[int], List[int]]],
    max_seq_len: int,
    pad_token_id: int,
) -> List[Dict[str, List[int]]]:
    """Pad / truncate packed token samples to *max_seq_len*.

    Returns records with ``input_ids``, ``attention_mask``, and ``labels``
    ready for Arrow / Parquet export.
    """
    results: List[Dict[str, List[int]]] = []
    for ids, lbls in packed_samples:
        ids = ids[:max_seq_len]
        lbls = lbls[:max_seq_len]

        real_len = len(ids)
        pad_len = max_seq_len - real_len

        results.append({
            "input_ids": ids + [pad_token_id] * pad_len,
            "attention_mask": [1] * real_len + [0] * pad_len,
            "labels": lbls + [IGNORE_INDEX] * pad_len,
        })
    return results


# ============================================================
# Step 4 — Export to Arrow / Parquet
# ============================================================

def _tokenized_to_arrow_table(records: List[Dict[str, List[int]]]) -> "pa.Table":
    """Convert tokenized records to a PyArrow Table."""
    if not records:
        schema = pa.schema([
            pa.field("input_ids", pa.list_(pa.int32())),
            pa.field("attention_mask", pa.list_(pa.int32())),
            pa.field("labels", pa.list_(pa.int32())),
        ])
        return pa.table({"input_ids": [], "attention_mask": [], "labels": []},
                        schema=schema)

    return pa.table({
        "input_ids": [r["input_ids"] for r in records],
        "attention_mask": [r["attention_mask"] for r in records],
        "labels": [r["labels"] for r in records],
    })


def _texts_to_arrow_table(texts: List[str]) -> "pa.Table":
    """Convert raw texts to a single-column Arrow Table."""
    return pa.table({"text": texts})


def export_arrow(table: "pa.Table", path: str) -> str:
    """Write an Arrow IPC file and return the path."""
    if not path.endswith(".arrow"):
        path = path + ".arrow"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with pa.OSFile(path, "wb") as f:
        writer = pa.ipc.new_file(f, table.schema)
        writer.write_table(table)
        writer.close()
    return path


def export_parquet(
    table: "pa.Table",
    path: str,
    compression: str = "snappy",
) -> str:
    """Write a Parquet file and return the path."""
    if not path.endswith(".parquet"):
        path = path + ".parquet"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    comp = None if compression.lower() == "none" else compression
    pq.write_table(table, path, compression=comp)
    return path


# ============================================================
# Report generation
# ============================================================

def _write_report_json(path: str, stats: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def _write_report_txt(path: str, stats: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("Data Packing & Version Management Report")
    lines.append("=" * 60)
    lines.append(f"Timestamp         : {stats.get('timestamp', 'N/A')}")
    lines.append(f"Input file        : {stats.get('input_file', 'N/A')}")
    lines.append(f"Output path       : {stats.get('output_path', 'N/A')}")
    lines.append(f"Output format     : {stats.get('output_format', 'N/A')}")
    lines.append(f"Version tag       : {stats.get('version_tag', 'N/A')}")
    lines.append("")

    summary = stats.get("summary", {})
    lines.append("--- Summary ---")
    lines.append(f"Total input samples       : {summary.get('total_input_samples', 0)}")
    lines.append(f"Chat template applied     : {summary.get('chat_template_applied', False)}")
    lines.append(f"Model ID                  : {summary.get('model_id', 'N/A')}")
    lines.append(f"Packing enabled           : {summary.get('packing_enabled', False)}")
    lines.append(f"Samples after packing     : {summary.get('samples_after_packing', 0)}")
    lines.append(f"Packing ratio             : {summary.get('packing_ratio', 'N/A')}")
    lines.append(f"Tokenization enabled      : {summary.get('tokenization_enabled', False)}")
    lines.append(f"Max sequence length       : {summary.get('max_seq_len', 0)}")
    lines.append(f"Final sample count        : {summary.get('final_sample_count', 0)}")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# Main pipeline
# ============================================================

def run(cfg: PackConfig) -> Dict[str, Any]:
    """Execute the full data-packing pipeline.

    Returns a stats dictionary summarising the run.
    """
    global _debug_lines
    _debug_lines = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    debug_dir = os.path.join(script_dir, "debug-log")
    report_dir = os.path.join(script_dir, "report")
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    _debug(f"[data_packing] start  timestamp={timestamp}")
    _debug(f"[data_packing] input={cfg.input}")
    _debug(f"[data_packing] output={cfg.output}  format={cfg.output_format}")

    # ------------------------------------------------------------------
    # Load conversations from JSONL
    # ------------------------------------------------------------------
    conversations: List[Dict[str, Any]] = []
    with open(cfg.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            conversations.append(json.loads(line))
    total_input = len(conversations)
    _debug(f"[data_packing] loaded {total_input} conversations")

    # ------------------------------------------------------------------
    # Load tokenizer (if needed)
    # ------------------------------------------------------------------
    tokenizer = None
    if (cfg.enable_chat_template or cfg.enable_tokenization or cfg.enable_packing) and HAS_TRANSFORMERS:
        _debug(f"[data_packing] loading tokenizer for {cfg.model_id}")
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_id,
            trust_remote_code=True,
        )
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        _debug(f"[data_packing] tokenizer loaded  vocab_size={tokenizer.vocab_size}")

    # ------------------------------------------------------------------
    # Pipeline: two paths depending on whether role-aware tokenization
    # is active (chat_template + tokenization both enabled).
    # ------------------------------------------------------------------
    tokenized_records: Optional[List[Dict[str, List[int]]]] = None
    texts: Optional[List[str]] = None

    if cfg.enable_tokenization and tokenizer is not None and cfg.enable_chat_template:
        # ==============================================================
        # NEW PATH — role-aware tokenization (assistant-only loss)
        # ==============================================================
        # 1. Per-conversation: apply chat template + tokenize with label
        #    masking (system/user → IGNORE_INDEX, assistant → real IDs).
        # 2. Pack at the *token* level (preserves label masks exactly).
        # 3. Pad to max_seq_len.
        # ==============================================================
        _debug("[data_packing] role-aware tokenization (assistant-only loss) …")

        token_samples: List[Tuple[List[int], List[int]]] = []
        for conv in conversations:
            ids, lbls = tokenize_conversation_with_labels(
                conv, tokenizer, cfg.add_generation_prompt,
            )
            token_samples.append((ids, lbls))
        _debug(f"[data_packing] tokenized {len(token_samples)} conversations with label masking")

        if cfg.enable_packing:
            separator = resolve_packing_separator(cfg, tokenizer)
            sep_ids = tokenizer.encode(separator, add_special_tokens=False)
            _debug(f"[data_packing] packing token samples  max_seq_len={cfg.max_seq_len}")
            token_samples = pack_token_samples(
                token_samples, cfg.max_seq_len, sep_ids,
                drop_short_tail=cfg.drop_short_tail,
                min_packing_ratio=cfg.min_packing_ratio,
            )
            _debug(f"[data_packing] packed → {len(token_samples)} token-level samples")

        samples_after_packing = len(token_samples)

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        tokenized_records = pad_token_samples(
            token_samples, cfg.max_seq_len, pad_id,
        )
        _debug(f"[data_packing] padded → {len(tokenized_records)} records  max_seq_len={cfg.max_seq_len}")

    else:
        # ==============================================================
        # ORIGINAL PATH — text-based pipeline
        # ==============================================================
        # Step 1: Apply chat template
        if cfg.enable_chat_template and tokenizer is not None:
            _debug("[data_packing] applying chat template …")
            texts = apply_chat_template(
                conversations, tokenizer,
                add_generation_prompt=cfg.add_generation_prompt,
            )
        else:
            _debug("[data_packing] chat template disabled — using plain text")
            texts = apply_plain_text_template(conversations)

        _debug(f"[data_packing] templated {len(texts)} samples")

        # Step 2: Sample packing
        if cfg.enable_packing:
            separator = resolve_packing_separator(cfg, tokenizer)
            _debug(f"[data_packing] packing samples  max_seq_len={cfg.max_seq_len}  sep={repr(separator)}")
            if tokenizer is not None:
                texts = pack_samples(
                    texts, tokenizer, cfg.max_seq_len, separator,
                    drop_short_tail=cfg.drop_short_tail,
                    min_packing_ratio=cfg.min_packing_ratio,
                )
            else:
                texts = pack_samples_no_tokenizer(
                    texts, cfg.max_seq_len, separator,
                    drop_short_tail=cfg.drop_short_tail,
                    min_packing_ratio=cfg.min_packing_ratio,
                )
            _debug(f"[data_packing] packed → {len(texts)} samples")

        samples_after_packing = len(texts)

        # Step 3: Tokenization (without role-aware masking)
        if cfg.enable_tokenization and tokenizer is not None:
            _debug(f"[data_packing] tokenizing {len(texts)} samples  max_seq_len={cfg.max_seq_len}")
            tokenized_records = tokenize_texts(texts, tokenizer, cfg.max_seq_len)
            _debug(f"[data_packing] tokenized {len(tokenized_records)} records")

    # ------------------------------------------------------------------
    # Step 4: Export
    # ------------------------------------------------------------------
    if not HAS_PYARROW:
        _debug("[data_packing] WARNING: pyarrow not installed — skipping export, writing JSONL")
        out_path = cfg.output if cfg.output.endswith(".jsonl") else cfg.output + ".jsonl"
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            if tokenized_records is not None:
                for rec in tokenized_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            else:
                for t in texts:
                    f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
        _debug(f"[data_packing] wrote JSONL fallback → {out_path}")
    else:
        if tokenized_records is not None:
            table = _tokenized_to_arrow_table(tokenized_records)
        else:
            table = _texts_to_arrow_table(texts)

        fmt = cfg.output_format.lower()
        if fmt == "arrow":
            out_path = export_arrow(table, cfg.output)
        else:
            out_path = export_parquet(table, cfg.output, compression=cfg.parquet_compression)
        _debug(f"[data_packing] exported {table.num_rows} rows → {out_path}  format={fmt}")

    # ------------------------------------------------------------------
    # Stats & reports
    # ------------------------------------------------------------------
    packing_ratio = (
        f"{total_input}:{samples_after_packing} "
        f"({total_input / max(samples_after_packing, 1):.2f}x)"
    )
    final_count = len(tokenized_records) if tokenized_records is not None else len(texts)

    stats: Dict[str, Any] = {
        "timestamp": timestamp,
        "input_file": cfg.input,
        "output_path": out_path,
        "output_format": cfg.output_format,
        "version_tag": cfg.version_tag,
        "summary": {
            "total_input_samples": total_input,
            "chat_template_applied": cfg.enable_chat_template and tokenizer is not None,
            "model_id": cfg.model_id if tokenizer is not None else "N/A",
            "packing_enabled": cfg.enable_packing,
            "samples_after_packing": samples_after_packing,
            "packing_ratio": packing_ratio,
            "tokenization_enabled": cfg.enable_tokenization and tokenizer is not None,
            "max_seq_len": cfg.max_seq_len,
            "final_sample_count": final_count,
        },
        "config": {
            "enable_chat_template": cfg.enable_chat_template,
            "enable_packing": cfg.enable_packing,
            "enable_tokenization": cfg.enable_tokenization,
            "max_seq_len": cfg.max_seq_len,
            "packing_separator": cfg.packing_separator,
            "drop_short_tail": cfg.drop_short_tail,
            "min_packing_ratio": cfg.min_packing_ratio,
            "parquet_compression": cfg.parquet_compression,
            "add_generation_prompt": cfg.add_generation_prompt,
            "version_tag": cfg.version_tag,
        },
    }

    report_json_path = os.path.join(report_dir, f"data_packing_report_{timestamp}.json")
    report_txt_path = os.path.join(report_dir, f"data_packing_report_{timestamp}.txt")
    _write_report_json(report_json_path, stats)
    _write_report_txt(report_txt_path, stats)
    _debug(f"[data_packing] reports → {report_json_path}")

    # Debug log
    debug_log_path = os.path.join(debug_dir, f"data_packing_debug_{timestamp}.log")
    with open(debug_log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(_debug_lines))
    _debug(f"[data_packing] debug log → {debug_log_path}")

    return stats


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Data packing & version management for SFT training data.",
    )

    g_io = p.add_argument_group("I/O")
    g_io.add_argument("--input", default=PackConfig.input,
                       help="Input JSONL file (Bedrock-format conversations).")
    g_io.add_argument("--output", default=PackConfig.output,
                       help="Output path (without extension).")
    g_io.add_argument("--output-format", default=PackConfig.output_format,
                       choices=["parquet", "arrow"],
                       help="Output format: parquet or arrow.")
    g_io.add_argument("--parquet-compression", default=PackConfig.parquet_compression,
                       choices=["snappy", "gzip", "zstd", "none"],
                       help="Parquet compression codec.")

    g_tpl = p.add_argument_group("Chat template")
    g_tpl.add_argument("--enable-chat-template", action="store_true", default=True,
                        help="Apply the model's chat template (default: True).")
    g_tpl.add_argument("--no-chat-template", dest="enable_chat_template",
                        action="store_false",
                        help="Disable chat template application.")
    g_tpl.add_argument("--model-id", default=PackConfig.model_id,
                        help="HuggingFace model ID for tokenizer / chat template.")
    g_tpl.add_argument("--add-generation-prompt", action="store_true", default=False,
                        help="Append a generation prompt in chat template.")

    g_pack = p.add_argument_group("Sample packing")
    g_pack.add_argument("--enable-packing", action="store_true", default=True,
                         help="Enable sample packing (default: True).")
    g_pack.add_argument("--no-packing", dest="enable_packing",
                         action="store_false",
                         help="Disable sample packing.")
    g_pack.add_argument("--max-seq-len", type=int, default=PackConfig.max_seq_len,
                         help="Maximum sequence length for packing / tokenization.")
    g_pack.add_argument("--packing-separator", default=PackConfig.packing_separator,
                         help="Separator token between packed samples (auto if empty).")
    g_pack.add_argument("--drop-short-tail", action="store_true", default=False,
                         help="Drop the last packed sample if too short.")
    g_pack.add_argument("--min-packing-ratio", type=float,
                         default=PackConfig.min_packing_ratio,
                         help="Minimum fill ratio for the tail packed sample.")

    g_tok = p.add_argument_group("Tokenization")
    g_tok.add_argument("--enable-tokenization", action="store_true", default=True,
                        help="Enable tokenization (default: True).")
    g_tok.add_argument("--no-tokenization", dest="enable_tokenization",
                        action="store_false",
                        help="Disable tokenization (export text only).")

    g_ver = p.add_argument_group("Version")
    g_ver.add_argument("--version-tag", default=PackConfig.version_tag,
                        help="Optional version tag for the dataset.")

    return p


def args_to_config(args: argparse.Namespace) -> PackConfig:
    return PackConfig(
        input=args.input,
        output=args.output,
        output_format=args.output_format,
        parquet_compression=args.parquet_compression,
        enable_chat_template=args.enable_chat_template,
        model_id=args.model_id,
        add_generation_prompt=args.add_generation_prompt,
        enable_packing=args.enable_packing,
        max_seq_len=args.max_seq_len,
        packing_separator=args.packing_separator,
        drop_short_tail=args.drop_short_tail,
        min_packing_ratio=args.min_packing_ratio,
        enable_tokenization=args.enable_tokenization,
        version_tag=args.version_tag,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = args_to_config(args)
    run(cfg)


if __name__ == "__main__":
    main()
