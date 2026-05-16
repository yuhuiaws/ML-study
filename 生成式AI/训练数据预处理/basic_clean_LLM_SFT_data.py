#!/usr/bin/env python3
"""
Data cleaning and filtering for Bedrock-format conversation JSONL.

Pipeline (per conversation):
  Phase 1 — Text normalization (before filtering):
      full-width→half-width, whitespace, NFD+accents, traditional→simplified
  Phase 2 — Rule-based filtering (drop entire conversation if ANY message fails):
      language, punctuation ratio, length, uppercase, purely numeric,
      URLs, ad keywords, short login/register
  Phase 3 — Optional final normalization (after filtering):
      lowercase, remove punctuation

All parameters are exposed via CLI flags (see --help).
"""

import argparse
import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

# ---------------------------------------------------------------------------
# Optional dependencies (graceful degradation)
# ---------------------------------------------------------------------------
try:
    import opencc
    HAS_OPENCC = True
except ImportError:
    HAS_OPENCC = False


# ============================================================
# Configuration
# ============================================================

DEFAULT_AD_KEYWORDS = [
    # Chinese
    "关注", "转发", "点赞", "订阅", "收藏",
    # English
    "subscribe", "follow us", "like and share", "click here",
    "retweet", "share this",
]

DEFAULT_LOGIN_PREFIXES = [
    "登录", "注册", "login", "sign up", "register", "log in",
]


@dataclass
class CleanConfig:
    """All tuneable parameters for the cleaning pipeline."""

    # -- I/O --
    input: str = "./daily-dialog-bedrock-all.jsonl"
    output: str = "./daily-dialog-bedrock-cleaned.jsonl"

    # -- Phase 1: normalization --
    fullwidth_to_halfwidth: bool = True
    normalize_whitespace: bool = True
    nfd_normalize: bool = True
    traditional_to_simplified: bool = False

    # -- Phase 3: final normalization (applied AFTER filtering) --
    to_lowercase: bool = False
    remove_punctuation: bool = False

    # -- Phase 2: filtering toggles --
    filter_language: bool = True
    allowed_scripts: List[str] = field(default_factory=lambda: ["CJK", "LATIN"])

    filter_punctuation: bool = True
    max_punct_ratio: float = 0.3

    filter_length: bool = True
    min_msg_len: int = 2
    max_msg_len: int = 2000

    filter_uppercase: bool = True
    max_upper_ratio: float = 0.7
    min_alpha_for_upper: int = 10

    filter_numeric: bool = True
    filter_urls: bool = True

    filter_keywords: bool = True
    ad_keywords: List[str] = field(default_factory=lambda: list(DEFAULT_AD_KEYWORDS))
    extra_ad_keywords: List[str] = field(default_factory=list)

    filter_short_login: bool = True
    login_prefixes: List[str] = field(default_factory=lambda: list(DEFAULT_LOGIN_PREFIXES))


# ============================================================
# Phase 1: Text Normalization Functions
# ============================================================

def fullwidth_to_halfwidth(text: str) -> str:
    """Full-width ASCII (U+FF01‥U+FF5E) → half-width; ideographic space → space."""
    out = []
    for ch in text:
        cp = ord(ch)
        if 0xFF01 <= cp <= 0xFF5E:
            out.append(chr(cp - 0xFEE0))
        elif cp == 0x3000:
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def normalize_whitespace(text: str) -> str:
    r"""\\r\\n→\\n, tabs→space, collapse runs of spaces, strip each line."""
    text = text.replace("\r\n", "\n")
    text = text.replace("\t", " ")
    text = re.sub(r"[^\S\n]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


def remove_accents_nfd(text: str) -> str:
    """NFD-normalize then strip combining marks (Mn) — e.g. é→e."""
    nfd = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


_opencc_converter: Optional[object] = None


def traditional_to_simplified(text: str) -> str:
    """Convert traditional Chinese characters to simplified (via OpenCC)."""
    global _opencc_converter
    if _opencc_converter is None:
        _opencc_converter = opencc.OpenCC("t2s")
    return _opencc_converter.convert(text)


def normalize_text(text: str, cfg: CleanConfig) -> str:
    """Phase-1 normalization (before filtering). Preserves case & punctuation."""
    if cfg.fullwidth_to_halfwidth:
        text = fullwidth_to_halfwidth(text)
    if cfg.normalize_whitespace:
        text = normalize_whitespace(text)
    if cfg.nfd_normalize:
        text = remove_accents_nfd(text)
    if cfg.traditional_to_simplified:
        if not HAS_OPENCC:
            raise RuntimeError("--traditional-to-simplified requires `opencc-python-reimplemented`")
        text = traditional_to_simplified(text)
    return text


# ============================================================
# Phase 3: Final Normalization
# ============================================================

def finalize_text(text: str, cfg: CleanConfig) -> str:
    """Phase-3 normalization (after filtering)."""
    if cfg.to_lowercase:
        text = text.lower()
    if cfg.remove_punctuation:
        text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))
        text = re.sub(r" +", " ", text).strip()
    return text


# ============================================================
# Phase 2: Filter Functions  (True → keep, False → discard)
# ============================================================

def _char_script(ch: str) -> str:
    """Return a coarse script label for a character."""
    cp = ord(ch)
    # CJK Unified Ideographs + Extension A/B + Compatibility
    if (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF
            or 0x20000 <= cp <= 0x2A6DF or 0xF900 <= cp <= 0xFAFF):
        return "CJK"
    # Basic Latin letters
    if ('A' <= ch <= 'Z') or ('a' <= ch <= 'z'):
        return "LATIN"
    # Fullwidth Latin (should have been normalized, but just in case)
    if 0xFF21 <= cp <= 0xFF3A or 0xFF41 <= cp <= 0xFF5A:
        return "LATIN"
    # Latin Extended
    if 0x00C0 <= cp <= 0x024F:
        return "LATIN"
    return "OTHER"


def filter_language(text: str, allowed_scripts: List[str] = None) -> bool:
    """Keep if all letter characters belong to the allowed scripts."""
    if allowed_scripts is None:
        allowed_scripts = ["CJK", "LATIN"]
    allowed = {s.upper() for s in allowed_scripts}
    scripts_found = set()
    for ch in text:
        s = _char_script(ch)
        if s != "OTHER":
            scripts_found.add(s)
        elif ch.isalpha():
            # alphabetic char not in any known allowed script
            return False
    if not scripts_found:
        return True
    return scripts_found.issubset(allowed)


def filter_punctuation_ratio(text: str, max_ratio: float = 0.3) -> bool:
    if not text:
        return True
    punct_count = sum(1 for ch in text if unicodedata.category(ch).startswith("P"))
    return punct_count / len(text) <= max_ratio


def filter_message_length(text: str, min_len: int = 2, max_len: int = 2000) -> bool:
    n = len(text.strip())
    return min_len <= n <= max_len


def filter_not_mostly_uppercase(text: str, max_ratio: float = 0.7,
                                min_alpha_len: int = 10) -> bool:
    alpha = [ch for ch in text if ch.isalpha()]
    if len(alpha) < min_alpha_len:
        return True
    return sum(1 for ch in alpha if ch.isupper()) / len(alpha) <= max_ratio


def filter_not_purely_numeric(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    non_ws = [ch for ch in stripped if not ch.isspace()]
    if not non_ws:
        return True
    return not all(ch.isdigit() for ch in non_ws)


def filter_no_urls(text: str) -> bool:
    return not re.search(r"https?://\S+|www\.\S+", text, re.IGNORECASE)


def filter_no_ad_keywords(text: str, keywords: List[str]) -> bool:
    text_lower = text.lower()
    return not any(kw.lower() in text_lower for kw in keywords)


def filter_not_short_login(text: str, max_len: int = 10,
                           prefixes: List[str] = None) -> bool:
    if prefixes is None:
        prefixes = DEFAULT_LOGIN_PREFIXES
    s = text.strip()
    if len(s) > max_len:
        return True
    s_lower = s.lower()
    return not any(s_lower.startswith(p.lower()) for p in prefixes)


# ============================================================
# Conversation-level pipeline
# ============================================================

def process_conversation(conv: dict, cfg: CleanConfig):
    """Normalize → filter → finalize one conversation.

    Returns (cleaned_conv, None) on success or (None, reason) if filtered.
    """
    messages = conv.get("messages", [])
    if not messages:
        return None, "empty_messages"

    # --- Phase 1 -------------------------------------------------
    norm_msgs = []
    for msg in messages:
        text = msg["content"][0]["text"]
        text = normalize_text(text, cfg)
        norm_msgs.append({"role": msg["role"], "content": [{"text": text}]})

    # --- Phase 2 -------------------------------------------------
    # 2a  language (concatenated for accuracy)
    if cfg.filter_language:
        all_text = " ".join(m["content"][0]["text"] for m in norm_msgs)
        if not filter_language(all_text, cfg.allowed_scripts):
            return None, "non_target_language"

    # 2b  per-message rules
    combined_kw = cfg.ad_keywords + cfg.extra_ad_keywords
    for idx, msg in enumerate(norm_msgs):
        text = msg["content"][0]["text"]

        if cfg.filter_punctuation:
            if not filter_punctuation_ratio(text, cfg.max_punct_ratio):
                return None, f"high_punctuation_ratio@msg{idx}"

        if cfg.filter_length:
            if not filter_message_length(text, cfg.min_msg_len, cfg.max_msg_len):
                return None, f"message_length@msg{idx}"

        if cfg.filter_uppercase:
            if not filter_not_mostly_uppercase(text, cfg.max_upper_ratio,
                                               cfg.min_alpha_for_upper):
                return None, f"mostly_uppercase@msg{idx}"

        if cfg.filter_numeric:
            if not filter_not_purely_numeric(text):
                return None, f"purely_numeric@msg{idx}"

        if cfg.filter_urls:
            if not filter_no_urls(text):
                return None, f"contains_url@msg{idx}"

        if cfg.filter_keywords:
            if not filter_no_ad_keywords(text, combined_kw):
                return None, f"ad_keyword@msg{idx}"

        if cfg.filter_short_login:
            if not filter_not_short_login(text, prefixes=cfg.login_prefixes):
                return None, f"short_login@msg{idx}"

    # --- Phase 3 -------------------------------------------------
    final_msgs = []
    for msg in norm_msgs:
        text = msg["content"][0]["text"]
        text = finalize_text(text, cfg)
        if not text.strip():
            return None, "empty_after_finalize"
        final_msgs.append({"role": msg["role"], "content": [{"text": text}]})

    cleaned = dict(conv)
    cleaned["messages"] = final_msgs
    return cleaned, None


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Clean and filter Bedrock-format conversation JSONL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Default (Chinese + English + mixed, SFT-safe):
  python basic_clean_LLM_SFT_data.py

  # Chinese corpus only, convert T→S:
  python basic_clean_LLM_SFT_data.py \\
      --input zh_data.jsonl --output zh_clean.jsonl \\
      --traditional-to-simplified --allowed-scripts CJK

  # Latin-script only (English, etc.):
  python basic_clean_LLM_SFT_data.py --allowed-scripts LATIN

  # Aggressive normalisation for a retrieval index:
  python basic_clean_LLM_SFT_data.py --to-lowercase --remove-punctuation

  # Disable keyword filter, keep URLs:
  python basic_clean_LLM_SFT_data.py --no-filter-keywords --no-filter-urls
""",
    )

    io = p.add_argument_group("I/O")
    io.add_argument("-i", "--input", default="./daily-dialog-bedrock-all.jsonl",
                    help="Input JSONL path (default: %(default)s)")
    io.add_argument("-o", "--output", default="./daily-dialog-bedrock-cleaned.jsonl",
                    help="Output JSONL path (default: %(default)s)")

    norm = p.add_argument_group("Phase 1 — text normalization")
    norm.add_argument("--fullwidth-to-halfwidth", action=argparse.BooleanOptionalAction,
                      default=True,
                      help="Convert full-width ASCII → half-width (default: on)")
    norm.add_argument("--normalize-whitespace", action=argparse.BooleanOptionalAction,
                      default=True,
                      help="Normalize \\r\\n, tabs, consecutive spaces (default: on)")
    norm.add_argument("--nfd-normalize", action=argparse.BooleanOptionalAction,
                      default=True,
                      help="NFD Unicode normalization + remove accents (default: on)")
    norm.add_argument("--traditional-to-simplified", action=argparse.BooleanOptionalAction,
                      default=False,
                      help="Convert traditional Chinese → simplified (default: off)")

    fin = p.add_argument_group("Phase 3 — final normalization (after filtering)")
    fin.add_argument("--to-lowercase", action=argparse.BooleanOptionalAction,
                     default=False,
                     help="Lowercase all text (default: off for SFT)")
    fin.add_argument("--remove-punctuation", action=argparse.BooleanOptionalAction,
                     default=False,
                     help="Remove all punctuation (default: off for SFT)")

    flt = p.add_argument_group("Phase 2 — rule-based filtering")

    flt.add_argument("--filter-language", action=argparse.BooleanOptionalAction,
                     default=True, help="Enable language/script filter (default: on)")
    flt.add_argument("--allowed-scripts", nargs="+", default=["CJK", "LATIN"],
                     help="Allowed Unicode scripts: CJK, LATIN (default: CJK LATIN)")

    flt.add_argument("--filter-punctuation", action=argparse.BooleanOptionalAction,
                     default=True, help="Enable punctuation-ratio filter (default: on)")
    flt.add_argument("--max-punct-ratio", type=float, default=0.3,
                     help="Max fraction of punctuation chars (default: %(default)s)")

    flt.add_argument("--filter-length", action=argparse.BooleanOptionalAction,
                     default=True, help="Enable message-length filter (default: on)")
    flt.add_argument("--min-msg-len", type=int, default=2,
                     help="Min message length in chars (default: %(default)s)")
    flt.add_argument("--max-msg-len", type=int, default=2000,
                     help="Max message length in chars (default: %(default)s)")

    flt.add_argument("--filter-uppercase", action=argparse.BooleanOptionalAction,
                     default=True, help="Enable mostly-uppercase filter (default: on)")
    flt.add_argument("--max-upper-ratio", type=float, default=0.7,
                     help="Max fraction of uppercase alpha chars (default: %(default)s)")
    flt.add_argument("--min-alpha-for-upper", type=int, default=10,
                     help="Min alpha chars to trigger uppercase check (default: %(default)s)")

    flt.add_argument("--filter-numeric", action=argparse.BooleanOptionalAction,
                     default=True, help="Enable purely-numeric filter (default: on)")
    flt.add_argument("--filter-urls", action=argparse.BooleanOptionalAction,
                     default=True, help="Enable URL filter (default: on)")

    flt.add_argument("--filter-keywords", action=argparse.BooleanOptionalAction,
                     default=True, help="Enable ad-keyword filter (default: on)")
    flt.add_argument("--extra-ad-keywords", nargs="*", default=[],
                     help="Additional ad keywords to filter (space-separated)")

    flt.add_argument("--filter-short-login", action=argparse.BooleanOptionalAction,
                     default=True, help="Enable short login/register filter (default: on)")

    return p


def args_to_config(args: argparse.Namespace) -> CleanConfig:
    """Map parsed CLI args → CleanConfig, converting hyphens to underscores."""
    return CleanConfig(
        input=args.input,
        output=args.output,
        fullwidth_to_halfwidth=args.fullwidth_to_halfwidth,
        normalize_whitespace=args.normalize_whitespace,
        nfd_normalize=args.nfd_normalize,
        traditional_to_simplified=args.traditional_to_simplified,
        to_lowercase=args.to_lowercase,
        remove_punctuation=args.remove_punctuation,
        filter_language=args.filter_language,
        allowed_scripts=args.allowed_scripts,
        filter_punctuation=args.filter_punctuation,
        max_punct_ratio=args.max_punct_ratio,
        filter_length=args.filter_length,
        min_msg_len=args.min_msg_len,
        max_msg_len=args.max_msg_len,
        filter_uppercase=args.filter_uppercase,
        max_upper_ratio=args.max_upper_ratio,
        min_alpha_for_upper=args.min_alpha_for_upper,
        filter_numeric=args.filter_numeric,
        filter_urls=args.filter_urls,
        filter_keywords=args.filter_keywords,
        extra_ad_keywords=args.extra_ad_keywords or [],
        filter_short_login=args.filter_short_login,
    )


# ============================================================
# Main
# ============================================================

def run(cfg: CleanConfig):
    """Execute the full pipeline with the given config."""
    stats = {"total": 0, "kept": 0, "filtered": 0, "reasons": {}}
    # Per-reason sample collector: store up to 3 example conversations per reason
    reason_examples = {}

    # Prepare output directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    debug_dir = os.path.join(script_dir, "debug-log")
    report_dir = os.path.join(script_dir, "report")
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_log_path = os.path.join(debug_dir, f"basic_clean_debug_{timestamp}.log")
    report_path = os.path.join(report_dir, f"basic_clean_report_{timestamp}.json")
    report_txt_path = os.path.join(report_dir, f"basic_clean_report_{timestamp}.txt")

    with open(cfg.input, "r", encoding="utf-8") as fin, \
         open(cfg.output, "w", encoding="utf-8") as fout, \
         open(debug_log_path, "w", encoding="utf-8") as debug_f:

        debug_f.write(f"=== Basic Clean Debug Log ===\n")
        debug_f.write(f"Timestamp: {timestamp}\n")
        debug_f.write(f"Input:  {cfg.input}\n")
        debug_f.write(f"Output: {cfg.output}\n")
        debug_f.write(f"{'=' * 60}\n\n")

        for line in fin:
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1
            conv = json.loads(line)
            result, reason = process_conversation(conv, cfg)
            if result is not None:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                stats["kept"] += 1
            else:
                stats["filtered"] += 1
                stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
                # Write debug info for every filtered conversation
                debug_f.write(f"[FILTERED #{stats['filtered']}] reason={reason}\n")
                debug_f.write(f"  conversation={json.dumps(conv, ensure_ascii=False)}\n\n")
                # Collect up to 3 examples per reason for the report
                if reason not in reason_examples:
                    reason_examples[reason] = []
                if len(reason_examples[reason]) < 3:
                    # Extract a short preview of the conversation
                    msgs = conv.get("messages", [])
                    preview = []
                    for m in msgs[:3]:
                        text = m.get("content", [{}])[0].get("text", "")
                        if len(text) > 100:
                            text = text[:100] + "..."
                        preview.append({"role": m.get("role", ""), "text": text})
                    reason_examples[reason].append(preview)

        debug_f.write(f"\n{'=' * 60}\n")
        debug_f.write(f"Debug log complete. Total filtered: {stats['filtered']}\n")

    # --- Build report ---
    sorted_reasons = sorted(stats["reasons"].items(), key=lambda x: -x[1])

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
        },
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
            "fullwidth_to_halfwidth": cfg.fullwidth_to_halfwidth,
            "normalize_whitespace": cfg.normalize_whitespace,
            "nfd_normalize": cfg.nfd_normalize,
            "traditional_to_simplified": cfg.traditional_to_simplified,
            "to_lowercase": cfg.to_lowercase,
            "remove_punctuation": cfg.remove_punctuation,
            "filter_language": cfg.filter_language,
            "allowed_scripts": cfg.allowed_scripts,
            "filter_punctuation": cfg.filter_punctuation,
            "max_punct_ratio": cfg.max_punct_ratio,
            "filter_length": cfg.filter_length,
            "min_msg_len": cfg.min_msg_len,
            "max_msg_len": cfg.max_msg_len,
            "filter_uppercase": cfg.filter_uppercase,
            "max_upper_ratio": cfg.max_upper_ratio,
            "filter_numeric": cfg.filter_numeric,
            "filter_urls": cfg.filter_urls,
            "filter_keywords": cfg.filter_keywords,
            "filter_short_login": cfg.filter_short_login,
        },
    }
    with open(report_path, "w", encoding="utf-8") as rf:
        json.dump(report_data, rf, ensure_ascii=False, indent=2)

    # Text report
    with open(report_txt_path, "w", encoding="utf-8") as tf:
        tf.write("=" * 60 + "\n")
        tf.write("  Basic Clean LLM SFT Data — Processing Report\n")
        tf.write("=" * 60 + "\n\n")
        tf.write(f"Timestamp : {timestamp}\n")
        tf.write(f"Input     : {cfg.input}\n")
        tf.write(f"Output    : {cfg.output}\n\n")

        tf.write("-" * 40 + "\n")
        tf.write("  Sample Statistics\n")
        tf.write("-" * 40 + "\n")
        tf.write(f"  Total samples   : {stats['total']}\n")
        tf.write(f"  Kept samples    : {stats['kept']}\n")
        tf.write(f"  Filtered samples: {stats['filtered']}\n")
        keep_pct = round(stats["kept"] / stats["total"] * 100, 2) if stats["total"] else 0
        filt_pct = round(stats["filtered"] / stats["total"] * 100, 2) if stats["total"] else 0
        tf.write(f"  Keep rate       : {keep_pct}%\n")
        tf.write(f"  Filter rate     : {filt_pct}%\n\n")

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

    # Console output
    print(f"Total:    {stats['total']}")
    print(f"Kept:     {stats['kept']}")
    print(f"Filtered: {stats['filtered']}")
    if stats["reasons"]:
        print("Filter reasons:")
        for reason, count in sorted_reasons:
            print(f"  {reason}: {count}")
    print(f"\nDebug log saved to: {debug_log_path}")
    print(f"Report saved to:    {report_txt_path}")
    print(f"Report (JSON):      {report_path}")
    return stats


def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = args_to_config(args)
    run(cfg)


if __name__ == "__main__":
    main()
