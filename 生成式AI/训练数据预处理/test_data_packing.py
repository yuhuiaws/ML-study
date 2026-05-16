#!/usr/bin/env python3
"""
Tests for data_packing.py

Coverage:
  1.  extract_messages — system prompt, multi-turn, missing system
  2.  messages_to_plain_text — format correctness
  3.  apply_chat_template — with mock tokenizer, fallback on error
  4.  apply_plain_text_template — multiple conversations
  5.  resolve_packing_separator — explicit, eos, fallback
  6.  pack_samples — basic packing, oversized, drop_short_tail
  7.  pack_samples_no_tokenizer — character-based packing
  8.  tokenize_texts — padding, labels -100 for pad
  9.  _tokenized_to_arrow_table — schema, empty input
  10. _texts_to_arrow_table — column check
  11. export_arrow — file written, readable
  12. export_parquet — file written, compression variants, readable
  13. CLI parsing — all arguments, defaults
  14. args_to_config — correct mapping
  15. End-to-end: run() with mock tokenizer, parquet output
  16. End-to-end: run() text-only (no tokenization)
  17. End-to-end: run() arrow format
  18. End-to-end: run() with packing disabled
  19. End-to-end: run() with real tokenizer (Qwen3)
  20. Report generation — JSON and TXT files created
"""

import json
import os
import tempfile
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_packing import (
    HAS_PYARROW,
    HAS_TRANSFORMERS,
    IGNORE_INDEX,
    PackConfig,
    _probe_chat_markers,
    _texts_to_arrow_table,
    _tokenized_to_arrow_table,
    _write_report_json,
    _write_report_txt,
    apply_chat_template,
    apply_plain_text_template,
    args_to_config,
    build_parser,
    export_arrow,
    export_parquet,
    extract_messages,
    messages_to_plain_text,
    pack_samples,
    pack_samples_no_tokenizer,
    pack_token_samples,
    pad_token_samples,
    resolve_packing_separator,
    run,
    tokenize_conversation_with_labels,
    tokenize_texts,
)


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def _conv(texts, roles=None, system_text="sys"):
    """Build a minimal Bedrock conversation dict."""
    if roles is None:
        roles = ["user" if i % 2 == 0 else "assistant" for i in range(len(texts))]
    return {
        "schemaVersion": "bedrock-conversation-2024",
        "system": [{"text": system_text}],
        "messages": [
            {"role": r, "content": [{"text": t}]}
            for r, t in zip(roles, texts)
        ],
    }


def _conv_no_system(texts, roles=None):
    """Bedrock conversation without a system field."""
    if roles is None:
        roles = ["user" if i % 2 == 0 else "assistant" for i in range(len(texts))]
    return {
        "schemaVersion": "bedrock-conversation-2024",
        "messages": [
            {"role": r, "content": [{"text": t}]}
            for r, t in zip(roles, texts)
        ],
    }


def _write_jsonl(path, conversations):
    with open(path, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")


def _read_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


class MockTokenizer:
    """Lightweight mock tokenizer for testing without HuggingFace downloads."""

    def __init__(self, vocab_size=1000, eos_token="<|eos|>", pad_token="<|pad|>"):
        self.vocab_size = vocab_size
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.pad_token_id = 0
        self.eos_token_id = 1

    def encode(self, text, add_special_tokens=True):
        """Approximate tokenization: one token per ~4 characters."""
        n = max(1, len(text) // 4)
        return list(range(10, 10 + n))

    def __call__(self, text, truncation=False, max_length=None, padding=None,
                 return_attention_mask=False):
        ids = self.encode(text)
        if truncation and max_length and len(ids) > max_length:
            ids = ids[:max_length]
        attn = [1] * len(ids)
        if padding == "max_length" and max_length:
            pad_len = max_length - len(ids)
            ids = ids + [self.pad_token_id] * pad_len
            attn = attn + [0] * pad_len
        result = {"input_ids": ids, "attention_mask": attn}
        return result

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        """Simple template: <|role|>\\ncontent for each message."""
        parts = []
        for m in messages:
            parts.append(f"<|{m['role']}|>\n{m['content']}")
        text = "\n".join(parts)
        if add_generation_prompt:
            text += "\n<|assistant|>\n"
        return text


# ----------------------------------------------------------------
# 1. extract_messages
# ----------------------------------------------------------------

class TestExtractMessages:
    def test_basic_multi_turn(self):
        conv = _conv(["hello", "hi there", "how are you", "fine"],
                     roles=["user", "assistant", "user", "assistant"],
                     system_text="be helpful")
        msgs = extract_messages(conv)
        assert len(msgs) == 5  # system + 4 messages
        assert msgs[0] == {"role": "system", "content": "be helpful"}
        assert msgs[1] == {"role": "user", "content": "hello"}
        assert msgs[4] == {"role": "assistant", "content": "fine"}

    def test_no_system(self):
        conv = _conv_no_system(["q", "a"])
        msgs = extract_messages(conv)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"

    def test_empty_system(self):
        conv = _conv(["q", "a"], system_text="")
        msgs = extract_messages(conv)
        # Empty system should be omitted
        assert msgs[0]["role"] == "user"
        assert len(msgs) == 2

    def test_empty_messages(self):
        conv = {
            "schemaVersion": "bedrock-conversation-2024",
            "system": [{"text": "sys"}],
            "messages": [],
        }
        msgs = extract_messages(conv)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"

    def test_multi_part_content(self):
        conv = {
            "schemaVersion": "bedrock-conversation-2024",
            "system": [],
            "messages": [
                {"role": "user", "content": [{"text": "part1"}, {"text": "part2"}]},
                {"role": "assistant", "content": [{"text": "answer"}]},
            ],
        }
        msgs = extract_messages(conv)
        assert msgs[0]["content"] == "part1 part2"


# ----------------------------------------------------------------
# 2. messages_to_plain_text
# ----------------------------------------------------------------

class TestMessagesToPlainText:
    def test_format(self):
        msgs = [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        text = messages_to_plain_text(msgs)
        assert "<|system|>" in text
        assert "<|user|>" in text
        assert "<|assistant|>" in text
        assert "be nice" in text
        assert "hi" in text
        assert "hello" in text

    def test_empty(self):
        assert messages_to_plain_text([]) == ""


# ----------------------------------------------------------------
# 3. apply_chat_template
# ----------------------------------------------------------------

class TestApplyChatTemplate:
    def test_with_mock_tokenizer(self):
        tok = MockTokenizer()
        convs = [_conv(["q", "a"])]
        results = apply_chat_template(convs, tok)
        assert len(results) == 1
        assert "<|user|>" in results[0]
        assert "q" in results[0]

    def test_add_generation_prompt(self):
        tok = MockTokenizer()
        convs = [_conv(["q", "a"])]
        results = apply_chat_template(convs, tok, add_generation_prompt=True)
        assert results[0].endswith("<|assistant|>\n")

    def test_fallback_on_error(self):
        """If apply_chat_template raises, fall back to plain text."""
        tok = MockTokenizer()
        tok.apply_chat_template = mock.MagicMock(side_effect=RuntimeError("boom"))
        convs = [_conv(["q", "a"])]
        results = apply_chat_template(convs, tok)
        assert len(results) == 1
        # Should still contain the text via fallback
        assert "q" in results[0]

    def test_multiple_conversations(self):
        tok = MockTokenizer()
        convs = [_conv(["q1", "a1"]), _conv(["q2", "a2"])]
        results = apply_chat_template(convs, tok)
        assert len(results) == 2


# ----------------------------------------------------------------
# 4. apply_plain_text_template
# ----------------------------------------------------------------

class TestApplyPlainTextTemplate:
    def test_basic(self):
        convs = [_conv(["hi", "hello"]), _conv(["bye", "cya"])]
        results = apply_plain_text_template(convs)
        assert len(results) == 2
        assert "hi" in results[0]
        assert "bye" in results[1]


# ----------------------------------------------------------------
# 5. resolve_packing_separator
# ----------------------------------------------------------------

class TestResolvePackingSeparator:
    def test_explicit(self):
        cfg = PackConfig(packing_separator="<SEP>")
        assert resolve_packing_separator(cfg) == "<SEP>"

    def test_from_tokenizer_eos(self):
        cfg = PackConfig(packing_separator="")
        tok = MockTokenizer(eos_token="<|eos|>")
        assert resolve_packing_separator(cfg, tok) == "<|eos|>"

    def test_fallback(self):
        cfg = PackConfig(packing_separator="")
        tok = MockTokenizer(eos_token=None)
        assert resolve_packing_separator(cfg, tok) == "<|endoftext|>"

    def test_fallback_no_tokenizer(self):
        cfg = PackConfig(packing_separator="")
        assert resolve_packing_separator(cfg, None) == "<|endoftext|>"


# ----------------------------------------------------------------
# 6. pack_samples (token-based)
# ----------------------------------------------------------------

class TestPackSamples:
    def test_basic_packing(self):
        """Short texts should be merged into fewer packed samples."""
        tok = MockTokenizer()
        # Each "abcd" → ~1 token.  With max_seq_len=10 we can fit many.
        texts = ["abcd"] * 20
        packed = pack_samples(texts, tok, max_seq_len=10, separator="<|eos|>")
        assert len(packed) < len(texts)
        for p in packed:
            assert "<|eos|>" in p or len(p) <= 8  # separator present or single

    def test_single_oversized(self):
        """A single text larger than max_seq_len should be emitted as-is."""
        tok = MockTokenizer()
        huge = "x" * 10000  # will produce ~2500 tokens
        texts = ["short", huge, "short2"]
        packed = pack_samples(texts, tok, max_seq_len=100, separator="<SEP>")
        assert huge in packed  # oversized text present verbatim

    def test_drop_short_tail(self):
        tok = MockTokenizer()
        # One text that fits, one tiny text
        texts = ["a" * 200, "b" * 4]  # ~50 tokens, ~1 token
        packed_keep = pack_samples(
            texts, tok, max_seq_len=60, separator="<SEP>",
            drop_short_tail=False,
        )
        # With drop_short_tail and high ratio, the tiny remainder should be dropped
        packed_drop = pack_samples(
            texts, tok, max_seq_len=60, separator="<SEP>",
            drop_short_tail=True, min_packing_ratio=0.9,
        )
        assert len(packed_drop) <= len(packed_keep)

    def test_empty_input(self):
        tok = MockTokenizer()
        assert pack_samples([], tok, max_seq_len=100, separator="<SEP>") == []

    def test_all_fit_in_one(self):
        """If all texts fit in one bin, result should have length 1."""
        tok = MockTokenizer()
        texts = ["hi", "yo", "ok"]  # ~1 token each
        packed = pack_samples(texts, tok, max_seq_len=1000, separator="<SEP>")
        assert len(packed) == 1
        assert "<SEP>" in packed[0]


# ----------------------------------------------------------------
# 7. pack_samples_no_tokenizer
# ----------------------------------------------------------------

class TestPackSamplesNoTokenizer:
    def test_basic(self):
        texts = ["hello world"] * 5
        packed = pack_samples_no_tokenizer(
            texts, max_seq_len=20, separator="|||", chars_per_token=1.0,
        )
        assert len(packed) < len(texts) or len(packed) >= 1

    def test_oversized(self):
        huge = "x" * 10000
        texts = ["short", huge]
        packed = pack_samples_no_tokenizer(
            texts, max_seq_len=50, separator="|||", chars_per_token=1.0,
        )
        assert huge in packed

    def test_empty(self):
        assert pack_samples_no_tokenizer([], 100, "|||") == []


# ----------------------------------------------------------------
# 8. tokenize_texts
# ----------------------------------------------------------------

class TestTokenizeTexts:
    def test_basic(self):
        tok = MockTokenizer()
        results = tokenize_texts(["hello world", "foo bar"], tok, max_seq_len=32)
        assert len(results) == 2
        for r in results:
            assert "input_ids" in r
            assert "attention_mask" in r
            assert "labels" in r
            assert len(r["input_ids"]) == 32
            assert len(r["attention_mask"]) == 32
            assert len(r["labels"]) == 32

    def test_labels_padding(self):
        """Padding positions in labels should be -100."""
        tok = MockTokenizer()
        results = tokenize_texts(["hi"], tok, max_seq_len=64)
        r = results[0]
        for tid, mask, label in zip(r["input_ids"], r["attention_mask"], r["labels"]):
            if mask == 0:
                assert label == -100
            else:
                assert label == tid

    def test_truncation(self):
        tok = MockTokenizer()
        # Very long text should be truncated to max_seq_len
        long_text = "x" * 100000
        results = tokenize_texts([long_text], tok, max_seq_len=16)
        assert len(results[0]["input_ids"]) == 16

    def test_empty(self):
        tok = MockTokenizer()
        assert tokenize_texts([], tok, max_seq_len=32) == []


# ----------------------------------------------------------------
# 9. _tokenized_to_arrow_table
# ----------------------------------------------------------------

class TestTokenizedToArrowTable:
    def test_basic(self):
        records = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]},
            {"input_ids": [4, 5, 0], "attention_mask": [1, 1, 0], "labels": [4, 5, -100]},
        ]
        table = _tokenized_to_arrow_table(records)
        assert table.num_rows == 2
        assert "input_ids" in table.column_names
        assert "attention_mask" in table.column_names
        assert "labels" in table.column_names

    def test_empty(self):
        table = _tokenized_to_arrow_table([])
        assert table.num_rows == 0
        assert "input_ids" in table.column_names


# ----------------------------------------------------------------
# 10. _texts_to_arrow_table
# ----------------------------------------------------------------

class TestTextsToArrowTable:
    def test_basic(self):
        table = _texts_to_arrow_table(["hello", "world"])
        assert table.num_rows == 2
        assert table.column_names == ["text"]

    def test_empty(self):
        table = _texts_to_arrow_table([])
        assert table.num_rows == 0


# ----------------------------------------------------------------
# 11. export_arrow
# ----------------------------------------------------------------

class TestExportArrow:
    def test_write_and_read(self):
        table = pa.table({"text": ["a", "b", "c"]})
        with tempfile.TemporaryDirectory() as td:
            path = export_arrow(table, os.path.join(td, "test"))
            assert path.endswith(".arrow")
            assert os.path.isfile(path)
            # Read back
            with pa.OSFile(path, "rb") as f:
                reader = pa.ipc.open_file(f)
                read_table = reader.read_all()
            assert read_table.num_rows == 3

    def test_already_has_extension(self):
        table = pa.table({"x": [1]})
        with tempfile.TemporaryDirectory() as td:
            path = export_arrow(table, os.path.join(td, "out.arrow"))
            assert path.endswith(".arrow")
            assert not path.endswith(".arrow.arrow")


# ----------------------------------------------------------------
# 12. export_parquet
# ----------------------------------------------------------------

class TestExportParquet:
    def test_write_and_read_snappy(self):
        table = pa.table({"text": ["a", "b"]})
        with tempfile.TemporaryDirectory() as td:
            path = export_parquet(table, os.path.join(td, "test"), compression="snappy")
            assert path.endswith(".parquet")
            read_table = pq.read_table(path)
            assert read_table.num_rows == 2

    def test_write_gzip(self):
        table = pa.table({"text": ["x"]})
        with tempfile.TemporaryDirectory() as td:
            path = export_parquet(table, os.path.join(td, "test"), compression="gzip")
            assert os.path.isfile(path)

    def test_write_none_compression(self):
        table = pa.table({"text": ["x"]})
        with tempfile.TemporaryDirectory() as td:
            path = export_parquet(table, os.path.join(td, "test"), compression="none")
            assert os.path.isfile(path)
            read_table = pq.read_table(path)
            assert read_table.num_rows == 1

    def test_already_has_extension(self):
        table = pa.table({"x": [1]})
        with tempfile.TemporaryDirectory() as td:
            path = export_parquet(table, os.path.join(td, "out.parquet"))
            assert path.endswith(".parquet")
            assert not path.endswith(".parquet.parquet")


# ----------------------------------------------------------------
# 13. CLI parsing
# ----------------------------------------------------------------

class TestCLIParsing:
    def test_defaults(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.output_format == "parquet"
        assert args.enable_chat_template is True
        assert args.enable_packing is True
        assert args.enable_tokenization is True
        assert args.max_seq_len == 4096
        assert args.model_id == "Qwen/Qwen3-8B"

    def test_no_flags(self):
        parser = build_parser()
        args = parser.parse_args([
            "--no-chat-template",
            "--no-packing",
            "--no-tokenization",
        ])
        assert args.enable_chat_template is False
        assert args.enable_packing is False
        assert args.enable_tokenization is False

    def test_custom_values(self):
        parser = build_parser()
        args = parser.parse_args([
            "--input", "my.jsonl",
            "--output", "my_out",
            "--output-format", "arrow",
            "--model-id", "meta-llama/Llama-2-7b-chat-hf",
            "--max-seq-len", "2048",
            "--packing-separator", "<SEP>",
            "--parquet-compression", "gzip",
            "--version-tag", "v2.0",
            "--add-generation-prompt",
            "--drop-short-tail",
            "--min-packing-ratio", "0.8",
        ])
        assert args.input == "my.jsonl"
        assert args.output == "my_out"
        assert args.output_format == "arrow"
        assert args.model_id == "meta-llama/Llama-2-7b-chat-hf"
        assert args.max_seq_len == 2048
        assert args.packing_separator == "<SEP>"
        assert args.parquet_compression == "gzip"
        assert args.version_tag == "v2.0"
        assert args.add_generation_prompt is True
        assert args.drop_short_tail is True
        assert args.min_packing_ratio == 0.8


# ----------------------------------------------------------------
# 14. args_to_config
# ----------------------------------------------------------------

class TestArgsToConfig:
    def test_mapping(self):
        parser = build_parser()
        args = parser.parse_args([
            "--input", "in.jsonl",
            "--output", "out",
            "--output-format", "arrow",
            "--model-id", "test/model",
            "--max-seq-len", "512",
            "--version-tag", "v1",
        ])
        cfg = args_to_config(args)
        assert isinstance(cfg, PackConfig)
        assert cfg.input == "in.jsonl"
        assert cfg.output == "out"
        assert cfg.output_format == "arrow"
        assert cfg.model_id == "test/model"
        assert cfg.max_seq_len == 512
        assert cfg.version_tag == "v1"


# ----------------------------------------------------------------
# 15. End-to-end: run() with mock tokenizer → parquet
# ----------------------------------------------------------------

class TestEndToEndMockParquet:
    @mock.patch("data_packing.AutoTokenizer")
    def test_run_parquet(self, mock_auto_tok):
        mock_auto_tok.from_pretrained.return_value = MockTokenizer()

        with tempfile.TemporaryDirectory() as td:
            inp = os.path.join(td, "input.jsonl")
            out = os.path.join(td, "output")
            _write_jsonl(inp, [
                _conv(["q1", "a1"]),
                _conv(["q2", "a2"]),
                _conv(["q3", "a3"]),
            ])

            cfg = PackConfig(
                input=inp,
                output=out,
                output_format="parquet",
                parquet_compression="snappy",
                enable_chat_template=True,
                enable_packing=True,
                enable_tokenization=True,
                max_seq_len=64,
                model_id="test/model",
                version_tag="v1-test",
            )
            stats = run(cfg)

            assert stats["summary"]["total_input_samples"] == 3
            assert stats["summary"]["chat_template_applied"] is True
            assert stats["summary"]["packing_enabled"] is True
            assert stats["summary"]["tokenization_enabled"] is True
            assert stats["version_tag"] == "v1-test"

            # Output file exists and is readable
            out_path = stats["output_path"]
            assert os.path.isfile(out_path)
            table = pq.read_table(out_path)
            assert table.num_rows > 0
            assert "input_ids" in table.column_names


# ----------------------------------------------------------------
# 16. End-to-end: run() text-only (no tokenization)
# ----------------------------------------------------------------

class TestEndToEndTextOnly:
    @mock.patch("data_packing.AutoTokenizer")
    def test_run_text_only(self, mock_auto_tok):
        mock_auto_tok.from_pretrained.return_value = MockTokenizer()

        with tempfile.TemporaryDirectory() as td:
            inp = os.path.join(td, "input.jsonl")
            out = os.path.join(td, "output")
            _write_jsonl(inp, [
                _conv(["q1", "a1"]),
                _conv(["q2", "a2"]),
            ])

            cfg = PackConfig(
                input=inp,
                output=out,
                output_format="parquet",
                enable_chat_template=True,
                enable_packing=True,
                enable_tokenization=False,
                max_seq_len=256,
            )
            stats = run(cfg)

            assert stats["summary"]["tokenization_enabled"] is False
            out_path = stats["output_path"]
            table = pq.read_table(out_path)
            assert "text" in table.column_names


# ----------------------------------------------------------------
# 17. End-to-end: arrow format
# ----------------------------------------------------------------

class TestEndToEndArrow:
    @mock.patch("data_packing.AutoTokenizer")
    def test_run_arrow(self, mock_auto_tok):
        mock_auto_tok.from_pretrained.return_value = MockTokenizer()

        with tempfile.TemporaryDirectory() as td:
            inp = os.path.join(td, "input.jsonl")
            out = os.path.join(td, "output")
            _write_jsonl(inp, [_conv(["q", "a"])])

            cfg = PackConfig(
                input=inp,
                output=out,
                output_format="arrow",
                enable_chat_template=True,
                enable_packing=False,
                enable_tokenization=True,
                max_seq_len=32,
            )
            stats = run(cfg)

            out_path = stats["output_path"]
            assert out_path.endswith(".arrow")
            assert os.path.isfile(out_path)


# ----------------------------------------------------------------
# 18. End-to-end: packing disabled
# ----------------------------------------------------------------

class TestEndToEndNoPacking:
    @mock.patch("data_packing.AutoTokenizer")
    def test_run_no_packing(self, mock_auto_tok):
        mock_auto_tok.from_pretrained.return_value = MockTokenizer()

        with tempfile.TemporaryDirectory() as td:
            inp = os.path.join(td, "input.jsonl")
            out = os.path.join(td, "output")
            _write_jsonl(inp, [
                _conv(["q1", "a1"]),
                _conv(["q2", "a2"]),
                _conv(["q3", "a3"]),
            ])

            cfg = PackConfig(
                input=inp,
                output=out,
                enable_packing=False,
                enable_tokenization=True,
                max_seq_len=64,
            )
            stats = run(cfg)

            assert stats["summary"]["packing_enabled"] is False
            # Without packing, sample count should match input
            assert stats["summary"]["samples_after_packing"] == 3


# ----------------------------------------------------------------
# 19. End-to-end: real Qwen3 tokenizer
# ----------------------------------------------------------------

class TestEndToEndRealTokenizer:
    @pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
    def test_run_with_qwen3(self):
        """Integration test using the actual Qwen/Qwen3-8B tokenizer."""
        with tempfile.TemporaryDirectory() as td:
            inp = os.path.join(td, "input.jsonl")
            out = os.path.join(td, "output")
            _write_jsonl(inp, [
                _conv(["什么是Python？", "Python是一种编程语言。"],
                      system_text="你是一个有用的AI助手。"),
                _conv(["What is REST?", "REST is an architectural style."],
                      system_text="You are a helpful assistant."),
                _conv(["hello", "hi"], system_text="sys"),
                _conv(["a", "b"], system_text="sys"),
                _conv(["x", "y"], system_text="sys"),
            ])

            cfg = PackConfig(
                input=inp,
                output=out,
                output_format="parquet",
                parquet_compression="snappy",
                enable_chat_template=True,
                enable_packing=True,
                enable_tokenization=True,
                max_seq_len=512,
                model_id="Qwen/Qwen3-8B",
                version_tag="qwen3-test",
            )
            stats = run(cfg)

            assert stats["summary"]["total_input_samples"] == 5
            assert stats["summary"]["chat_template_applied"] is True
            assert stats["summary"]["model_id"] == "Qwen/Qwen3-8B"

            out_path = stats["output_path"]
            assert os.path.isfile(out_path)
            table = pq.read_table(out_path)
            assert table.num_rows > 0
            assert "input_ids" in table.column_names

            # Verify every row has the expected max_seq_len length
            for i in range(table.num_rows):
                ids = table.column("input_ids")[i].as_py()
                assert len(ids) == 512


# ----------------------------------------------------------------
# 20. Report generation
# ----------------------------------------------------------------

class TestReportGeneration:
    def test_json_report(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "report.json")
            stats = {"timestamp": "test", "summary": {"count": 5}}
            _write_report_json(path, stats)
            assert os.path.isfile(path)
            with open(path) as f:
                data = json.load(f)
            assert data["summary"]["count"] == 5

    def test_txt_report(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "report.txt")
            stats = {
                "timestamp": "20260214",
                "input_file": "in.jsonl",
                "output_path": "out.parquet",
                "output_format": "parquet",
                "version_tag": "v1",
                "summary": {
                    "total_input_samples": 100,
                    "chat_template_applied": True,
                    "model_id": "test",
                    "packing_enabled": True,
                    "samples_after_packing": 25,
                    "packing_ratio": "100:25 (4.00x)",
                    "tokenization_enabled": True,
                    "max_seq_len": 4096,
                    "final_sample_count": 25,
                },
            }
            _write_report_txt(path, stats)
            assert os.path.isfile(path)
            with open(path) as f:
                content = f.read()
            assert "100" in content
            assert "parquet" in content

    @mock.patch("data_packing.AutoTokenizer")
    def test_reports_created_by_run(self, mock_auto_tok):
        mock_auto_tok.from_pretrained.return_value = MockTokenizer()

        with tempfile.TemporaryDirectory() as td:
            inp = os.path.join(td, "input.jsonl")
            out = os.path.join(td, "output")
            _write_jsonl(inp, [_conv(["q", "a"])])

            cfg = PackConfig(input=inp, output=out, max_seq_len=32)
            stats = run(cfg)

            # Check report files were created in the script's report dir
            script_dir = os.path.dirname(os.path.abspath(__file__))
            report_dir = os.path.join(script_dir, "report")
            ts = stats["timestamp"]
            assert os.path.isfile(os.path.join(report_dir, f"data_packing_report_{ts}.json"))
            assert os.path.isfile(os.path.join(report_dir, f"data_packing_report_{ts}.txt"))


# ----------------------------------------------------------------
# 21a. _probe_chat_markers
# ----------------------------------------------------------------

class TestProbeChatMarkers:
    def test_mock_tokenizer(self):
        tok = MockTokenizer()
        asst_suffix, eot = _probe_chat_markers(tok)
        # MockTokenizer template: <|role|>\ncontent
        # assistant header suffix should end with "assistant\n..."
        # eot should be None or something depending on mock
        # The mock doesn't add end-of-turn markers, so eot is None or empty
        assert asst_suffix is None or "assistant" in asst_suffix

    @pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
    def test_qwen3_tokenizer(self):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
        asst_suffix, eot = _probe_chat_markers(tok)
        assert asst_suffix is not None
        assert "assistant" in asst_suffix
        assert eot is not None
        assert "im_end" in eot


# ----------------------------------------------------------------
# 21b. tokenize_conversation_with_labels
# ----------------------------------------------------------------

class TestTokenizeConversationWithLabels:
    def test_assistant_only_loss(self):
        """System and user tokens should be IGNORE_INDEX; assistant tokens
        should carry the real token IDs."""
        tok = MockTokenizer()
        conv = _conv(
            ["hello there, how are you today?",
             "I am doing well, thank you for asking!"],
            roles=["user", "assistant"],
            system_text="You are a helpful assistant",
        )
        ids, labels = tokenize_conversation_with_labels(conv, tok)

        assert len(ids) == len(labels)
        assert len(ids) > 0

        # There must be some IGNORE_INDEX labels (system + user)
        ignored = [l for l in labels if l == IGNORE_INDEX]
        assert len(ignored) > 0

        # There must be some real labels (assistant)
        real = [(idx, l) for idx, l in enumerate(labels) if l != IGNORE_INDEX]
        assert len(real) > 0

        # Real labels must match the corresponding input_ids
        for idx, l in real:
            assert l == ids[idx]

    def test_multi_turn(self):
        """Multiple assistant turns should all have real labels."""
        tok = MockTokenizer()
        conv = _conv(
            ["q1", "a1 is a longer answer here", "q2", "a2 is another answer"],
            roles=["user", "assistant", "user", "assistant"],
            system_text="sys",
        )
        ids, labels = tokenize_conversation_with_labels(conv, tok)

        assert len(ids) == len(labels)
        real_count = sum(1 for l in labels if l != IGNORE_INDEX)
        assert real_count > 0

    def test_no_assistant(self):
        """A conversation with no assistant turn should produce all-ignored labels."""
        tok = MockTokenizer()
        conv = {
            "schemaVersion": "bedrock-conversation-2024",
            "system": [{"text": "sys"}],
            "messages": [
                {"role": "user", "content": [{"text": "hello"}]},
            ],
        }
        ids, labels = tokenize_conversation_with_labels(conv, tok)

        assert len(ids) == len(labels)
        assert all(l == IGNORE_INDEX for l in labels)

    def test_empty_conversation(self):
        tok = MockTokenizer()
        conv = {
            "schemaVersion": "bedrock-conversation-2024",
            "messages": [],
        }
        ids, labels = tokenize_conversation_with_labels(conv, tok)
        assert ids == []
        assert labels == []

    def test_no_system(self):
        """Conversation without system prompt should still mask user correctly."""
        tok = MockTokenizer()
        conv = _conv_no_system(
            ["user question here please",
             "assistant answer here okay"],
        )
        ids, labels = tokenize_conversation_with_labels(conv, tok)

        ignored = sum(1 for l in labels if l == IGNORE_INDEX)
        real = sum(1 for l in labels if l != IGNORE_INDEX)
        assert ignored > 0
        assert real > 0


# ----------------------------------------------------------------
# 22. pack_token_samples
# ----------------------------------------------------------------

class TestPackTokenSamples:
    def test_basic_packing(self):
        """Two short token sequences should be packed into one."""
        s1 = ([10, 11, 12], [IGNORE_INDEX, IGNORE_INDEX, 12])
        s2 = ([20, 21, 22], [IGNORE_INDEX, 21, 22])
        sep_ids = [99]

        packed = pack_token_samples([s1, s2], max_seq_len=20, separator_ids=sep_ids)
        assert len(packed) == 1

        ids, lbls = packed[0]
        # s1(3) + sep(1) + s2(3) = 7 tokens
        assert len(ids) == 7
        assert len(lbls) == 7
        # Separator should have IGNORE_INDEX label
        assert lbls[3] == IGNORE_INDEX
        # Original labels preserved
        assert lbls[0] == IGNORE_INDEX
        assert lbls[2] == 12
        assert lbls[4] == IGNORE_INDEX
        assert lbls[5] == 21

    def test_overflow_creates_new_bin(self):
        """When adding a sample exceeds max_seq_len, it goes to a new bin."""
        s1 = (list(range(5)), [IGNORE_INDEX] * 3 + [3, 4])
        s2 = (list(range(10, 15)), [IGNORE_INDEX] * 2 + [12, 13, 14])
        sep_ids = [99]

        # max_seq_len=7: s1(5) + sep(1) + s2(5) = 11 > 7
        packed = pack_token_samples([s1, s2], max_seq_len=7, separator_ids=sep_ids)
        assert len(packed) == 2

    def test_oversized_sample_truncated(self):
        """A single sample exceeding max_seq_len should be truncated."""
        big = (list(range(100)), [IGNORE_INDEX] * 50 + list(range(50, 100)))
        packed = pack_token_samples([big], max_seq_len=10, separator_ids=[99])
        assert len(packed) == 1
        assert len(packed[0][0]) == 10
        assert len(packed[0][1]) == 10

    def test_drop_short_tail(self):
        s1 = (list(range(8)), [IGNORE_INDEX] * 4 + list(range(4, 8)))
        s2 = ([90], [90])
        sep_ids = [99]

        # Without drop: both emitted
        packed_keep = pack_token_samples(
            [s1, s2], max_seq_len=9, separator_ids=sep_ids,
            drop_short_tail=False,
        )
        packed_drop = pack_token_samples(
            [s1, s2], max_seq_len=9, separator_ids=sep_ids,
            drop_short_tail=True, min_packing_ratio=0.9,
        )
        assert len(packed_drop) <= len(packed_keep)

    def test_empty_input(self):
        assert pack_token_samples([], max_seq_len=10, separator_ids=[99]) == []


# ----------------------------------------------------------------
# 23. pad_token_samples
# ----------------------------------------------------------------

class TestPadTokenSamples:
    def test_padding(self):
        samples = [([10, 11, 12], [IGNORE_INDEX, 11, 12])]
        results = pad_token_samples(samples, max_seq_len=6, pad_token_id=0)

        assert len(results) == 1
        r = results[0]
        assert len(r["input_ids"]) == 6
        assert len(r["attention_mask"]) == 6
        assert len(r["labels"]) == 6

        # Real tokens
        assert r["input_ids"][:3] == [10, 11, 12]
        assert r["attention_mask"][:3] == [1, 1, 1]
        assert r["labels"][:3] == [IGNORE_INDEX, 11, 12]

        # Padding
        assert r["input_ids"][3:] == [0, 0, 0]
        assert r["attention_mask"][3:] == [0, 0, 0]
        assert r["labels"][3:] == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]

    def test_truncation(self):
        samples = [(list(range(20)), list(range(20)))]
        results = pad_token_samples(samples, max_seq_len=5, pad_token_id=0)
        assert len(results[0]["input_ids"]) == 5

    def test_exact_fit(self):
        samples = [([1, 2, 3], [IGNORE_INDEX, 2, 3])]
        results = pad_token_samples(samples, max_seq_len=3, pad_token_id=0)
        r = results[0]
        assert r["input_ids"] == [1, 2, 3]
        assert r["attention_mask"] == [1, 1, 1]
        assert r["labels"] == [IGNORE_INDEX, 2, 3]


# ----------------------------------------------------------------
# 24. End-to-end: verify assistant-only labels in output
# ----------------------------------------------------------------

class TestEndToEndAssistantOnlyLabels:
    @mock.patch("data_packing.AutoTokenizer")
    def test_labels_mask_system_user(self, mock_auto_tok):
        """Verify that after the full pipeline, labels contain IGNORE_INDEX
        for system/user positions and real token IDs for assistant positions."""
        mock_auto_tok.from_pretrained.return_value = MockTokenizer()

        with tempfile.TemporaryDirectory() as td:
            inp = os.path.join(td, "input.jsonl")
            out = os.path.join(td, "output")
            _write_jsonl(inp, [
                _conv(
                    ["Tell me about Python programming language in detail",
                     "Python is a high-level interpreted programming language"],
                    roles=["user", "assistant"],
                    system_text="You are a knowledgeable AI assistant",
                ),
            ])

            cfg = PackConfig(
                input=inp,
                output=out,
                output_format="parquet",
                enable_chat_template=True,
                enable_packing=False,
                enable_tokenization=True,
                max_seq_len=128,
            )
            stats = run(cfg)

            out_path = stats["output_path"]
            table = pq.read_table(out_path)
            assert table.num_rows == 1

            ids = table.column("input_ids")[0].as_py()
            labels = table.column("labels")[0].as_py()
            attn = table.column("attention_mask")[0].as_py()

            assert len(ids) == 128
            assert len(labels) == 128

            # There must be some ignored labels (system + user)
            ignored_real = [
                l for l, m in zip(labels, attn) if l == IGNORE_INDEX and m == 1
            ]
            assert len(ignored_real) > 0, "Expected some real tokens with IGNORE_INDEX (system/user)"

            # There must be some real labels (assistant)
            real_labels = [l for l in labels if l != IGNORE_INDEX]
            assert len(real_labels) > 0, "Expected some real labels for assistant tokens"

            # Real labels must match corresponding input_ids
            for i, (tid, lbl) in enumerate(zip(ids, labels)):
                if lbl != IGNORE_INDEX:
                    assert lbl == tid, f"Label mismatch at position {i}"

    @mock.patch("data_packing.AutoTokenizer")
    def test_labels_with_packing(self, mock_auto_tok):
        """Verify label masking is preserved through token-level packing."""
        mock_auto_tok.from_pretrained.return_value = MockTokenizer()

        with tempfile.TemporaryDirectory() as td:
            inp = os.path.join(td, "input.jsonl")
            out = os.path.join(td, "output")
            _write_jsonl(inp, [
                _conv(["q1 question text", "a1 answer text"],
                      system_text="sys prompt"),
                _conv(["q2 question text", "a2 answer text"],
                      system_text="sys prompt"),
            ])

            cfg = PackConfig(
                input=inp,
                output=out,
                output_format="parquet",
                enable_chat_template=True,
                enable_packing=True,
                enable_tokenization=True,
                max_seq_len=256,
            )
            stats = run(cfg)

            out_path = stats["output_path"]
            table = pq.read_table(out_path)

            for row_idx in range(table.num_rows):
                ids = table.column("input_ids")[row_idx].as_py()
                labels = table.column("labels")[row_idx].as_py()
                attn = table.column("attention_mask")[row_idx].as_py()

                # Padding labels must be IGNORE_INDEX
                for tid, lbl, m in zip(ids, labels, attn):
                    if m == 0:
                        assert lbl == IGNORE_INDEX
                    # If label is real, it must match input_id
                    if lbl != IGNORE_INDEX:
                        assert lbl == tid


# ----------------------------------------------------------------
# 25. Qwen3 multi-turn label masking (regression test for <think> tag issue)
# ----------------------------------------------------------------

class TestQwen3MultiTurnLabels:
    @pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
    def test_multi_turn_no_bleed(self):
        """Verify that in a multi-turn conversation the label mask does NOT
        bleed from one assistant turn into the next user turn.

        Qwen3's chat template adds ``<think>`` tags only to the *last*
        assistant turn.  The old prefix-comparison method produced incorrect
        boundaries because ``apply_chat_template`` on a partial conversation
        (ending with the first assistant turn) rendered it *with* think tags
        (since it was the last turn in that partial view), while the full
        conversation rendered the same turn *without* think tags.
        """
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-8B", trust_remote_code=True,
        )

        conv = {
            "schemaVersion": "bedrock-conversation-2024",
            "system": [{"text": "You are a helpful assistant."}],
            "messages": [
                {"role": "user", "content": [{"text": "What is REST?"}]},
                {"role": "assistant", "content": [
                    {"text": "REST is an architectural style for APIs."}
                ]},
                {"role": "user", "content": [{"text": "Give me an example."}]},
                {"role": "assistant", "content": [
                    {"text": "GET /users returns a list of users."}
                ]},
            ],
        }

        ids, labels = tokenize_conversation_with_labels(conv, tok)

        # Decode the tokens that have loss labels
        loss_ids = [tid for tid, lbl in zip(ids, labels) if lbl != IGNORE_INDEX]
        loss_text = tok.decode(loss_ids)

        # The loss text should contain both assistant answers
        assert "REST is an architectural style" in loss_text
        assert "GET /users returns a list" in loss_text

        # The loss text must NOT contain the user turn "Give me an example."
        assert "Give me an example" not in loss_text
        # And must NOT contain the user turn "What is REST?"
        assert "What is REST" not in loss_text
        # And must NOT contain the system prompt
        assert "helpful assistant" not in loss_text

    @pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
    def test_single_turn_includes_think_tags(self):
        """For a single-turn conversation the ``<think>`` tags should be
        part of the loss span (the model must learn to produce them)."""
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-8B", trust_remote_code=True,
        )

        conv = {
            "schemaVersion": "bedrock-conversation-2024",
            "system": [{"text": "sys"}],
            "messages": [
                {"role": "user", "content": [{"text": "hello"}]},
                {"role": "assistant", "content": [{"text": "hi there"}]},
            ],
        }

        ids, labels = tokenize_conversation_with_labels(conv, tok)
        loss_ids = [tid for tid, lbl in zip(ids, labels) if lbl != IGNORE_INDEX]
        loss_text = tok.decode(loss_ids)

        assert "hi there" in loss_text
        # The think tags should be in the loss
        assert "<think>" in loss_text or "think" in loss_text


# ----------------------------------------------------------------
# Additional edge-case tests
# ----------------------------------------------------------------

class TestEdgeCases:
    def test_single_sample_no_packing_needed(self):
        tok = MockTokenizer()
        texts = ["hello world"]
        packed = pack_samples(texts, tok, max_seq_len=1000, separator="<SEP>")
        assert packed == ["hello world"]

    def test_packing_preserves_all_content(self):
        tok = MockTokenizer()
        texts = ["aaa", "bbb", "ccc"]
        packed = pack_samples(texts, tok, max_seq_len=10000, separator="|||")
        combined = "|||".join(packed)
        for t in texts:
            assert t in combined

    @mock.patch("data_packing.AutoTokenizer")
    def test_no_chat_template_no_tokenization(self, mock_auto_tok):
        """Test run with everything disabled — plain text to parquet."""
        mock_auto_tok.from_pretrained.return_value = MockTokenizer()

        with tempfile.TemporaryDirectory() as td:
            inp = os.path.join(td, "input.jsonl")
            out = os.path.join(td, "output")
            _write_jsonl(inp, [_conv(["q", "a"])])

            cfg = PackConfig(
                input=inp,
                output=out,
                enable_chat_template=False,
                enable_packing=False,
                enable_tokenization=False,
            )
            stats = run(cfg)

            out_path = stats["output_path"]
            table = pq.read_table(out_path)
            assert "text" in table.column_names

    def test_tokenize_empty_text(self):
        tok = MockTokenizer()
        results = tokenize_texts([""], tok, max_seq_len=16)
        assert len(results) == 1
        assert len(results[0]["input_ids"]) == 16
