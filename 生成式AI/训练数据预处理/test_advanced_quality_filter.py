#!/usr/bin/env python3
"""
Tests for advanced_quality_filter.py

Coverage:
  1. check_empty_turns — empty text, whitespace-only, normal passes
  2. check_repeated_assistant — identical adjacent assistant, different passes
  3. check_role_alternation — correct alternation, consecutive same-role, first-not-user
  4. check_truncation — truncation markers, suspicious cutoff length, normal passes
  5. check_turn_count — below min, above max, within range
  6. check_user_low_info — all low-info fails, mixed passes, no user edge case
  7. check_stalling — assistant repeats, progressive passes
  8. llm_check_quality — high score pass, low score fail, parse error fail-safe
  9. llm_check_moderation — safe pass, unsafe reject, API error reject
  10. llm_check_pii — no PII pass, has PII reject, API error reject
  11. parse_json_response — valid JSON, markdown code block, invalid -> empty dict
  12. format_conversation_for_llm — system included, labels correct
  13. End-to-end — mixed good/bad conversations, verify counts and reasons
  14. End-to-end with LLM — mock Bedrock client, verify filters called
  15. CLI parsing — defaults, custom values, toggle flags
  16. JSON report — verify report file written correctly
"""

import io
import json
import os
import tempfile
from unittest import mock

import pytest

from advanced_quality_filter import (
    FilterConfig,
    _char_overlap_ratio,
    _get_msg_text,
    args_to_config,
    build_parser,
    check_empty_turns,
    check_repeated_assistant,
    check_role_alternation,
    check_stalling,
    check_truncation,
    check_turn_count,
    check_user_low_info,
    format_conversation_for_llm,
    llm_check_moderation,
    llm_check_pii,
    llm_check_quality,
    parse_json_response,
    run,
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


def _write_jsonl(path, conversations):
    with open(path, "w", encoding="utf-8") as f:
        for c in conversations:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def _read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _mock_invoke_response(text):
    """Build a mock return value for client.invoke_model()."""
    body_bytes = json.dumps({
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
    }).encode("utf-8")
    return {"body": io.BytesIO(body_bytes)}


# ================================================================
# 1. check_empty_turns
# ================================================================

class TestCheckEmptyTurns:
    def test_empty_text_rejected(self):
        conv = _conv(["hello", ""])
        passed, reason = check_empty_turns(conv)
        assert not passed
        assert "empty_turn" in reason

    def test_whitespace_only_rejected(self):
        conv = _conv(["hello", "   \t\n  "])
        passed, reason = check_empty_turns(conv)
        assert not passed
        assert "empty_turn" in reason

    def test_normal_passes(self):
        conv = _conv(["hello", "world"])
        passed, reason = check_empty_turns(conv)
        assert passed
        assert reason is None

    def test_empty_messages_list(self):
        conv = {"messages": []}
        passed, reason = check_empty_turns(conv)
        assert passed
        assert reason is None


# ================================================================
# 2. check_repeated_assistant
# ================================================================

class TestCheckRepeatedAssistant:
    def test_identical_adjacent_rejected(self):
        conv = _conv(["q1", "same answer", "q2", "same answer"])
        passed, reason = check_repeated_assistant(conv)
        assert not passed
        assert "repeated_assistant" in reason

    def test_different_responses_pass(self):
        conv = _conv(["q1", "answer one", "q2", "answer two"])
        passed, reason = check_repeated_assistant(conv)
        assert passed
        assert reason is None

    def test_single_turn_passes(self):
        conv = _conv(["hello", "world"])
        passed, reason = check_repeated_assistant(conv)
        assert passed
        assert reason is None

    def test_whitespace_variation_passes(self):
        """Different after strip => passes."""
        conv = _conv(["q1", "answer A", "q2", "answer B "])
        passed, reason = check_repeated_assistant(conv)
        assert passed

    def test_strips_before_compare(self):
        """Same after strip => rejected."""
        conv = _conv(["q1", "same ", "q2", " same"])
        passed, reason = check_repeated_assistant(conv)
        assert not passed


# ================================================================
# 3. check_role_alternation
# ================================================================

class TestCheckRoleAlternation:
    def test_correct_alternation_passes(self):
        conv = _conv(["q", "a", "q2", "a2"])
        passed, reason = check_role_alternation(conv)
        assert passed
        assert reason is None

    def test_consecutive_same_role_rejected(self):
        conv = _conv(["q1", "q2", "a1", "a2"],
                     roles=["user", "user", "assistant", "assistant"])
        passed, reason = check_role_alternation(conv)
        assert not passed
        assert "role_alternation" in reason

    def test_first_message_not_user_rejected(self):
        conv = _conv(["a1", "q1"],
                     roles=["assistant", "user"])
        passed, reason = check_role_alternation(conv)
        assert not passed
        assert reason == "role_alternation@msg0"

    def test_empty_messages_passes(self):
        conv = {"messages": []}
        passed, reason = check_role_alternation(conv)
        assert passed

    def test_single_user_message_passes(self):
        conv = _conv(["hello"], roles=["user"])
        passed, reason = check_role_alternation(conv)
        assert passed


# ================================================================
# 4. check_truncation
# ================================================================

class TestCheckTruncation:
    def test_ends_with_dots_rejected(self):
        conv = _conv(["question", "This is an incomplete answer..."])
        passed, reason = check_truncation(conv)
        assert not passed
        assert "truncation" in reason

    def test_ends_with_ellipsis_rejected(self):
        conv = _conv(["question", "This is incomplete…"])
        passed, reason = check_truncation(conv)
        assert not passed
        assert "truncation" in reason

    def test_ends_with_chinese_marker_rejected(self):
        conv = _conv(["question", "这个回答还没完成[截断]"])
        passed, reason = check_truncation(conv)
        assert not passed
        assert "truncation" in reason

    def test_normal_text_passes(self):
        conv = _conv(["question", "This is a complete answer."])
        passed, reason = check_truncation(conv)
        assert passed
        assert reason is None

    def test_exact_length_cutoff_detected(self):
        text = "x" * 4096
        conv = _conv(["question", text])
        passed, reason = check_truncation(conv)
        assert not passed
        assert "truncation" in reason

    def test_length_8192_detected(self):
        text = "y" * 8192
        conv = _conv(["question", text])
        passed, reason = check_truncation(conv)
        assert not passed

    def test_non_suspicious_length_passes(self):
        text = "a" * 500
        conv = _conv(["question", text])
        passed, reason = check_truncation(conv)
        assert passed

    def test_custom_markers(self):
        conv = _conv(["question", "answer ends with CUSTOM_END"])
        passed, reason = check_truncation(conv, markers=["CUSTOM_END"])
        assert not passed


# ================================================================
# 5. check_turn_count
# ================================================================

class TestCheckTurnCount:
    def test_below_min_rejected(self):
        conv = _conv(["single"], roles=["user"])
        passed, reason = check_turn_count(conv, min_turns=2, max_turns=100)
        assert not passed
        assert "turn_count:1" == reason

    def test_above_max_rejected(self):
        texts = [f"msg{i}" for i in range(10)]
        roles = ["user" if i % 2 == 0 else "assistant" for i in range(10)]
        conv = _conv(texts, roles=roles)
        passed, reason = check_turn_count(conv, min_turns=2, max_turns=5)
        assert not passed
        assert "turn_count:10" == reason

    def test_within_range_passes(self):
        conv = _conv(["q", "a", "q2", "a2"])
        passed, reason = check_turn_count(conv, min_turns=2, max_turns=10)
        assert passed
        assert reason is None

    def test_exact_min_passes(self):
        conv = _conv(["q", "a"])
        passed, reason = check_turn_count(conv, min_turns=2, max_turns=10)
        assert passed

    def test_exact_max_passes(self):
        texts = [f"msg{i}" for i in range(4)]
        roles = ["user" if i % 2 == 0 else "assistant" for i in range(4)]
        conv = _conv(texts, roles=roles)
        passed, reason = check_turn_count(conv, min_turns=2, max_turns=4)
        assert passed


# ================================================================
# 6. check_user_low_info
# ================================================================

class TestCheckUserLowInfo:
    def test_all_low_info_rejected(self):
        conv = _conv(["嗯", "回答1", "哦", "回答2", "好的", "回答3"])
        passed, reason = check_user_low_info(conv)
        assert not passed
        assert "user_low_info" in reason

    def test_mixed_passes(self):
        conv = _conv(["嗯", "回答1", "请详细解释一下这个概念", "回答2",
                      "我还想了解更多信息", "回答3"])
        passed, reason = check_user_low_info(conv)
        assert passed
        assert reason is None

    def test_no_user_messages_passes(self):
        conv = {"messages": [
            {"role": "assistant", "content": [{"text": "hello"}]}
        ]}
        passed, reason = check_user_low_info(conv)
        assert passed

    def test_short_text_counts_as_low_info(self):
        """Text <= 3 chars is low-info regardless of pattern match."""
        conv = _conv(["ab", "回答1", "cd", "回答2", "ef", "回答3"])
        passed, reason = check_user_low_info(conv)
        assert not passed

    def test_custom_patterns(self):
        conv = _conv(["CUSTOM", "answer", "CUSTOM", "answer2"])
        passed, reason = check_user_low_info(
            conv, patterns=["CUSTOM"], max_ratio=0.5)
        assert not passed

    def test_ratio_boundary(self):
        """Exactly at threshold should pass (> not >=)."""
        conv = _conv(["嗯", "a1", "哦", "a2", "好的", "a3",
                      "详细的问题", "a4", "另一个详细问题", "a5"])
        # 3 low-info out of 5 user msgs = 0.6 ratio, threshold is 0.6 -> passes
        passed, reason = check_user_low_info(conv, max_ratio=0.6)
        assert passed


# ================================================================
# 7. check_stalling
# ================================================================

class TestCheckStalling:
    def test_repeated_assistant_text_rejected(self):
        conv = _conv(["q1", "same answer here about the topic",
                      "q2", "same answer here about the topic",
                      "q3", "same answer here about the topic"])
        passed, reason = check_stalling(conv)
        assert not passed
        assert "stalling" in reason

    def test_progressive_conversation_passes(self):
        conv = _conv([
            "what is python", "Python is a programming language",
            "what about java", "Java is another popular language",
            "and rust", "Rust focuses on memory safety",
        ])
        passed, reason = check_stalling(conv)
        assert passed
        assert reason is None

    def test_single_assistant_passes(self):
        conv = _conv(["hello", "world"])
        passed, reason = check_stalling(conv)
        assert passed

    def test_no_assistant_passes(self):
        conv = {"messages": [
            {"role": "user", "content": [{"text": "hello"}]}
        ]}
        passed, reason = check_stalling(conv)
        assert passed


# ================================================================
# 8. llm_check_quality
# ================================================================

class TestLLMCheckQuality:
    def test_high_score_passes(self):
        mock_client = mock.MagicMock()
        response_json = json.dumps({
            "scores": {"完整性": 8, "信息密度": 7, "知识准确性": 8, "自然度": 7},
            "average": 7.5
        })
        mock_client.invoke_model.return_value = _mock_invoke_response(response_json)

        conv = _conv(["what is AI", "AI is artificial intelligence"])
        passed, detail = llm_check_quality(conv, mock_client, "model-id",
                                           threshold=6.0)
        assert passed
        assert detail["average"] == 7.5

    def test_low_score_rejected(self):
        mock_client = mock.MagicMock()
        response_json = json.dumps({
            "scores": {"完整性": 3, "信息密度": 2, "知识准确性": 3, "自然度": 2},
            "average": 2.5
        })
        mock_client.invoke_model.return_value = _mock_invoke_response(response_json)

        conv = _conv(["what", "ok"])
        passed, detail = llm_check_quality(conv, mock_client, "model-id",
                                           threshold=6.0)
        assert not passed
        assert detail["average"] == 2.5

    def test_parse_error_rejects(self):
        mock_client = mock.MagicMock()
        mock_client.invoke_model.return_value = _mock_invoke_response(
            "I cannot evaluate this conversation properly")

        conv = _conv(["hello", "world"])
        passed, detail = llm_check_quality(conv, mock_client, "model-id")
        assert not passed
        assert "error" in detail

    def test_api_error_rejects(self):
        mock_client = mock.MagicMock()
        mock_client.invoke_model.side_effect = RuntimeError("API error")

        conv = _conv(["hello", "world"])
        passed, detail = llm_check_quality(conv, mock_client, "model-id")
        assert not passed
        assert "error" in detail

    def test_multi_turn_dimensions(self):
        """Multi-turn (>=4 msgs) should request more dimensions."""
        mock_client = mock.MagicMock()
        response_json = json.dumps({
            "scores": {
                "完整性": 8, "连贯性": 7, "信息密度": 8, "自然度": 7,
                "上下文一致性": 8, "知识准确性": 7, "回答深度": 8, "对话推进": 7
            },
            "average": 7.5
        })
        mock_client.invoke_model.return_value = _mock_invoke_response(response_json)

        conv = _conv(["q1", "a1", "q2", "a2"])
        passed, detail = llm_check_quality(conv, mock_client, "model-id")
        assert passed
        # Verify the prompt sent includes multi-turn dimensions
        call_args = mock_client.invoke_model.call_args
        body = json.loads(call_args[1]["body"] if "body" in call_args[1]
                          else call_args[0][0])
        prompt_text = body["messages"][0]["content"]
        assert "连贯性" in prompt_text
        assert "对话推进" in prompt_text


# ================================================================
# 9. llm_check_moderation
# ================================================================

class TestLLMCheckModeration:
    def test_safe_passes(self):
        mock_client = mock.MagicMock()
        response_json = json.dumps({"safe": True, "categories": []})
        mock_client.invoke_model.return_value = _mock_invoke_response(response_json)

        conv = _conv(["how to cook", "Here is a recipe"])
        passed, detail = llm_check_moderation(conv, mock_client, "model-id")
        assert passed
        assert detail["safe"] is True

    def test_unsafe_rejected(self):
        mock_client = mock.MagicMock()
        response_json = json.dumps({
            "safe": False,
            "categories": ["涉暴"],
            "evidence": "contains violent content"
        })
        mock_client.invoke_model.return_value = _mock_invoke_response(response_json)

        conv = _conv(["bad question", "bad answer"])
        passed, detail = llm_check_moderation(conv, mock_client, "model-id")
        assert not passed
        assert "涉暴" in detail["categories"]

    def test_api_error_rejects(self):
        mock_client = mock.MagicMock()
        mock_client.invoke_model.side_effect = RuntimeError("API error")

        conv = _conv(["hello", "world"])
        passed, detail = llm_check_moderation(conv, mock_client, "model-id")
        assert not passed
        assert "error" in detail

    def test_parse_error_rejects(self):
        mock_client = mock.MagicMock()
        mock_client.invoke_model.return_value = _mock_invoke_response(
            "This seems safe to me")

        conv = _conv(["hello", "world"])
        passed, detail = llm_check_moderation(conv, mock_client, "model-id")
        assert not passed
        assert "error" in detail


# ================================================================
# 10. llm_check_pii
# ================================================================

class TestLLMCheckPII:
    def test_no_pii_passes(self):
        mock_client = mock.MagicMock()
        response_json = json.dumps({"has_pii": False, "items": []})
        mock_client.invoke_model.return_value = _mock_invoke_response(response_json)

        conv = _conv(["what is AI", "AI is artificial intelligence"])
        passed, detail = llm_check_pii(conv, mock_client, "model-id")
        assert passed
        assert detail["has_pii"] is False

    def test_has_pii_rejected(self):
        mock_client = mock.MagicMock()
        response_json = json.dumps({
            "has_pii": True,
            "items": [{"type": "手机号", "value": "138****"}]
        })
        mock_client.invoke_model.return_value = _mock_invoke_response(response_json)

        conv = _conv(["my phone number is 13800138000", "noted"])
        passed, detail = llm_check_pii(conv, mock_client, "model-id")
        assert not passed
        assert detail["has_pii"] is True

    def test_api_error_rejects(self):
        mock_client = mock.MagicMock()
        mock_client.invoke_model.side_effect = RuntimeError("API error")

        conv = _conv(["hello", "world"])
        passed, detail = llm_check_pii(conv, mock_client, "model-id")
        assert not passed
        assert "error" in detail

    def test_parse_error_rejects(self):
        mock_client = mock.MagicMock()
        mock_client.invoke_model.return_value = _mock_invoke_response(
            "No PII found in this conversation")

        conv = _conv(["hello", "world"])
        passed, detail = llm_check_pii(conv, mock_client, "model-id")
        assert not passed
        assert "error" in detail


# ================================================================
# 11. parse_json_response
# ================================================================

class TestParseJsonResponse:
    def test_valid_json(self):
        result = parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_in_markdown_code_block(self):
        text = '```json\n{"key": "value"}\n```'
        result = parse_json_response(text)
        assert result == {"key": "value"}

    def test_json_in_plain_code_block(self):
        text = '```\n{"key": "value"}\n```'
        result = parse_json_response(text)
        assert result == {"key": "value"}

    def test_invalid_text_returns_empty(self):
        result = parse_json_response("This is not JSON at all")
        assert result == {}

    def test_json_embedded_in_text(self):
        text = 'Here is the result: {"safe": true, "categories": []} hope this helps'
        result = parse_json_response(text)
        assert result == {"safe": True, "categories": []}

    def test_empty_string(self):
        result = parse_json_response("")
        assert result == {}


# ================================================================
# 12. format_conversation_for_llm
# ================================================================

class TestFormatConversationForLLM:
    def test_system_included(self):
        conv = _conv(["question", "answer"], system_text="You are helpful.")
        result = format_conversation_for_llm(conv)
        assert "[系统] You are helpful." in result

    def test_labels_correct(self):
        conv = _conv(["question here", "answer here"])
        result = format_conversation_for_llm(conv)
        assert "用户: question here" in result
        assert "助手: answer here" in result

    def test_multi_turn(self):
        conv = _conv(["q1", "a1", "q2", "a2"])
        result = format_conversation_for_llm(conv)
        lines = result.split("\n")
        # system + 4 message lines
        assert len(lines) == 5
        assert lines[0].startswith("[系统]")
        assert lines[1].startswith("用户:")
        assert lines[2].startswith("助手:")

    def test_empty_system(self):
        conv = _conv(["q", "a"], system_text="")
        result = format_conversation_for_llm(conv)
        assert "[系统]" not in result

    def test_no_system_key(self):
        conv = {"messages": [
            {"role": "user", "content": [{"text": "hello"}]},
            {"role": "assistant", "content": [{"text": "hi"}]},
        ]}
        result = format_conversation_for_llm(conv)
        assert "用户: hello" in result
        assert "助手: hi" in result


# ================================================================
# 13. End-to-end pipeline (rule-based)
# ================================================================

class TestEndToEnd:
    def test_mixed_conversations(self):
        """Mix of good and bad conversations, verify correct filtering."""
        good = _conv(["What is Python?", "Python is a programming language."])
        empty_turn = _conv(["hello", ""])
        bad_alternation = _conv(
            ["q1", "q2", "a1"],
            roles=["user", "user", "assistant"]
        )
        truncated = _conv(["question", "incomplete answer..."])
        too_short = _conv(["single"], roles=["user"])

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, "in.jsonl")
            out = os.path.join(tmpdir, "out.jsonl")
            _write_jsonl(inp, [good, empty_turn, bad_alternation,
                               truncated, too_short])

            cfg = FilterConfig(input=inp, output=out)
            stats = run(cfg)

            result = _read_jsonl(out)
            assert stats["total"] == 5
            assert stats["kept"] == 1
            assert stats["filtered"] == 4
            assert len(result) == 1
            # Verify the survivor is the good conversation
            assert result[0]["messages"][0]["content"][0]["text"] == "What is Python?"

    def test_all_good_conversations(self):
        convs = [
            _conv(["What is AI?", "AI is artificial intelligence."]),
            _conv(["Explain ML", "ML is machine learning."]),
            _conv(["What is Python?", "A language", "How about Java?", "Another one"]),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, "in.jsonl")
            out = os.path.join(tmpdir, "out.jsonl")
            _write_jsonl(inp, convs)

            cfg = FilterConfig(input=inp, output=out)
            stats = run(cfg)

            assert stats["kept"] == 3
            assert stats["filtered"] == 0

    def test_filter_reasons_counted(self):
        convs = [
            _conv(["hello", ""]),           # empty_turn
            _conv(["hello", "   "]),         # empty_turn
            _conv(["q", "a", "q2", "a..."]), # truncation
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, "in.jsonl")
            out = os.path.join(tmpdir, "out.jsonl")
            _write_jsonl(inp, convs)

            cfg = FilterConfig(input=inp, output=out)
            stats = run(cfg)

            assert stats["reasons"].get("empty_turn", 0) == 2
            assert stats["reasons"].get("truncation", 0) == 1

    def test_disable_specific_filters(self):
        """Disabling a filter allows previously-rejected conversations through."""
        truncated = _conv(["question", "incomplete..."])
        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, "in.jsonl")
            out = os.path.join(tmpdir, "out.jsonl")
            _write_jsonl(inp, [truncated])

            # With truncation filter on -> rejected
            cfg = FilterConfig(input=inp, output=out, filter_truncation=True)
            stats = run(cfg)
            assert stats["kept"] == 0

            # With truncation filter off -> kept
            cfg = FilterConfig(input=inp, output=out, filter_truncation=False)
            stats = run(cfg)
            assert stats["kept"] == 1

    def test_stalling_filter(self):
        # Use near-identical but not exact assistant responses to avoid
        # being caught by the repeated_assistant filter first.
        conv = _conv([
            "What is the topic?",
            "This is the answer about the topic at hand with details and context",
            "Tell me more please",
            "This is the answer about the topic at hand with details and background",
            "Can you elaborate?",
            "This is the answer about the topic at hand with details and overview",
        ])
        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, "in.jsonl")
            out = os.path.join(tmpdir, "out.jsonl")
            _write_jsonl(inp, [conv])

            cfg = FilterConfig(input=inp, output=out)
            stats = run(cfg)
            assert stats["filtered"] == 1
            assert "stalling" in stats["reasons"]

    def test_low_info_filter(self):
        conv = _conv(["嗯", "回答1", "哦", "回答2", "好的", "回答3",
                      "是的", "回答4"])
        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, "in.jsonl")
            out = os.path.join(tmpdir, "out.jsonl")
            _write_jsonl(inp, [conv])

            cfg = FilterConfig(input=inp, output=out)
            stats = run(cfg)
            assert stats["filtered"] == 1
            assert "user_low_info" in stats["reasons"]


# ================================================================
# 14. End-to-end with LLM
# ================================================================

class TestEndToEndWithLLM:
    def test_moderation_filter(self):
        """Conversations failing moderation are rejected."""
        good = _conv(["What is Python?", "Python is a programming language."])
        bad = _conv(["bad content", "also bad content"])

        mock_client = mock.MagicMock()
        call_count = [0]

        def mock_invoke(modelId, contentType, accept, body):
            call_count[0] += 1
            # First call -> safe, second call -> unsafe
            if call_count[0] == 1:
                return _mock_invoke_response(
                    json.dumps({"safe": True, "categories": []}))
            else:
                return _mock_invoke_response(
                    json.dumps({"safe": False, "categories": ["涉暴"]}))

        mock_client.invoke_model.side_effect = mock_invoke

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, "in.jsonl")
            out = os.path.join(tmpdir, "out.jsonl")
            _write_jsonl(inp, [good, bad])

            cfg = FilterConfig(
                input=inp, output=out,
                enable_llm_moderation=True,
            )
            stats = run(cfg, bedrock_client=mock_client)

            assert stats["kept"] == 1
            assert stats["filtered"] == 1
            assert stats["llm_calls"] == 2
            assert "llm_moderation" in stats["reasons"]

    def test_pii_filter(self):
        """Conversations with PII are rejected."""
        good = _conv(["What is AI?", "AI is artificial intelligence."])

        mock_client = mock.MagicMock()
        mock_client.invoke_model.return_value = _mock_invoke_response(
            json.dumps({"has_pii": True,
                        "items": [{"type": "手机号", "value": "138****"}]}))

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, "in.jsonl")
            out = os.path.join(tmpdir, "out.jsonl")
            _write_jsonl(inp, [good])

            cfg = FilterConfig(input=inp, output=out, enable_llm_pii=True)
            stats = run(cfg, bedrock_client=mock_client)

            assert stats["kept"] == 0
            assert "llm_pii" in stats["reasons"]

    def test_quality_filter(self):
        """Low-quality conversations are rejected by LLM quality check."""
        conv = _conv(["What is AI?", "AI is artificial intelligence."])

        mock_client = mock.MagicMock()
        mock_client.invoke_model.return_value = _mock_invoke_response(
            json.dumps({
                "scores": {"完整性": 3, "信息密度": 2,
                           "知识准确性": 3, "自然度": 2},
                "average": 2.5
            }))

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, "in.jsonl")
            out = os.path.join(tmpdir, "out.jsonl")
            _write_jsonl(inp, [conv])

            cfg = FilterConfig(input=inp, output=out,
                               enable_llm_quality=True,
                               llm_quality_threshold=6.0)
            stats = run(cfg, bedrock_client=mock_client)

            assert stats["kept"] == 0
            assert "llm_quality" in stats["reasons"]

    def test_all_llm_filters_combined(self):
        """All LLM filters enabled; only fully-passing conversations survive."""
        conv = _conv(["What is Python?", "Python is a programming language."])

        mock_client = mock.MagicMock()
        call_count = [0]

        def mock_invoke(modelId, contentType, accept, body):
            call_count[0] += 1
            req = json.loads(body)
            prompt = req["messages"][0]["content"]
            if "安全审核" in prompt:
                return _mock_invoke_response(
                    json.dumps({"safe": True, "categories": []}))
            elif "隐私信息" in prompt:
                return _mock_invoke_response(
                    json.dumps({"has_pii": False, "items": []}))
            elif "质量" in prompt:
                return _mock_invoke_response(
                    json.dumps({"scores": {"完整性": 8}, "average": 8.0}))
            return _mock_invoke_response("{}")

        mock_client.invoke_model.side_effect = mock_invoke

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, "in.jsonl")
            out = os.path.join(tmpdir, "out.jsonl")
            _write_jsonl(inp, [conv])

            cfg = FilterConfig(
                input=inp, output=out,
                enable_llm_moderation=True,
                enable_llm_pii=True,
                enable_llm_quality=True,
                llm_quality_threshold=6.0,
            )
            stats = run(cfg, bedrock_client=mock_client)

            assert stats["kept"] == 1
            assert stats["filtered"] == 0
            assert stats["llm_calls"] == 3

    def test_rule_filter_skips_llm(self):
        """Conversations failing rule filters should NOT trigger LLM calls."""
        empty_conv = _conv(["hello", ""])

        mock_client = mock.MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, "in.jsonl")
            out = os.path.join(tmpdir, "out.jsonl")
            _write_jsonl(inp, [empty_conv])

            cfg = FilterConfig(
                input=inp, output=out,
                enable_llm_moderation=True,
                enable_llm_pii=True,
            )
            stats = run(cfg, bedrock_client=mock_client)

            assert stats["filtered"] == 1
            assert stats["llm_calls"] == 0
            mock_client.invoke_model.assert_not_called()


# ================================================================
# 15. CLI parsing
# ================================================================

class TestCLIParsing:
    def _parse(self, argv):
        parser = build_parser()
        args = parser.parse_args(argv)
        return args_to_config(args)

    def test_defaults(self):
        cfg = self._parse([])
        assert cfg.input == "./zh_mixed_deduped.jsonl"
        assert cfg.output == "./zh_mixed_filtered.jsonl"
        assert cfg.filter_empty_turns is True
        assert cfg.filter_repeated_assistant is True
        assert cfg.filter_role_alternation is True
        assert cfg.filter_truncation is True
        assert cfg.filter_turn_count is True
        assert cfg.min_turns == 2
        assert cfg.max_turns == 100
        assert cfg.filter_user_low_info is True
        assert cfg.max_low_info_ratio == 0.6
        assert cfg.filter_stalling is True
        assert cfg.max_repeat_ratio == 0.5
        assert cfg.enable_llm_quality is False
        assert cfg.llm_quality_threshold == 6.0
        assert cfg.enable_llm_moderation is False
        assert cfg.enable_llm_pii is False
        assert cfg.bedrock_model_id == "us.anthropic.claude-opus-4-6-v1:0"
        assert cfg.bedrock_region == "us-east-1"

    def test_custom_io(self):
        cfg = self._parse(["-i", "in.jsonl", "-o", "out.jsonl"])
        assert cfg.input == "in.jsonl"
        assert cfg.output == "out.jsonl"

    def test_disable_filters(self):
        cfg = self._parse([
            "--no-filter-empty-turns",
            "--no-filter-truncation",
            "--no-filter-turn-count",
        ])
        assert cfg.filter_empty_turns is False
        assert cfg.filter_truncation is False
        assert cfg.filter_turn_count is False
        # Others still on
        assert cfg.filter_role_alternation is True

    def test_enable_llm(self):
        cfg = self._parse([
            "--enable-llm-quality",
            "--enable-llm-moderation",
            "--enable-llm-pii",
        ])
        assert cfg.enable_llm_quality is True
        assert cfg.enable_llm_moderation is True
        assert cfg.enable_llm_pii is True

    def test_custom_thresholds(self):
        cfg = self._parse([
            "--min-turns", "4",
            "--max-turns", "50",
            "--max-low-info-ratio", "0.8",
            "--max-repeat-ratio", "0.3",
            "--llm-quality-threshold", "7.0",
        ])
        assert cfg.min_turns == 4
        assert cfg.max_turns == 50
        assert cfg.max_low_info_ratio == 0.8
        assert cfg.max_repeat_ratio == 0.3
        assert cfg.llm_quality_threshold == 7.0

    def test_bedrock_options(self):
        cfg = self._parse([
            "--bedrock-model-id", "my-model",
            "--bedrock-region", "us-west-2",
        ])
        assert cfg.bedrock_model_id == "my-model"
        assert cfg.bedrock_region == "us-west-2"

    def test_report_json(self):
        cfg = self._parse(["--report-json", "report.json"])
        assert cfg.report_json == "report.json"


# ================================================================
# 16. JSON report
# ================================================================

class TestJSONReport:
    def test_report_written(self):
        good = _conv(["What is Python?", "Python is a programming language."])
        bad = _conv(["hello", ""])

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, "in.jsonl")
            out = os.path.join(tmpdir, "out.jsonl")
            report = os.path.join(tmpdir, "report.json")
            _write_jsonl(inp, [good, bad])

            cfg = FilterConfig(input=inp, output=out, report_json=report)
            stats = run(cfg)

            assert os.path.exists(report)
            with open(report) as f:
                report_data = json.load(f)
            assert report_data["stats"]["total"] == 2
            assert report_data["stats"]["kept"] == 1
            assert len(report_data["entries"]) == 2
            # First entry should be kept
            assert report_data["entries"][0]["status"] == "kept"
            # Second entry should be filtered
            assert report_data["entries"][1]["status"] == "filtered"
            assert report_data["entries"][1]["reason"] is not None


# ================================================================
# 17. Helper functions
# ================================================================

class TestHelpers:
    def test_get_msg_text(self):
        msg = {"content": [{"text": "hello"}]}
        assert _get_msg_text(msg) == "hello"

    def test_get_msg_text_empty_content(self):
        msg = {"content": []}
        assert _get_msg_text(msg) == ""

    def test_get_msg_text_no_content(self):
        msg = {}
        assert _get_msg_text(msg) == ""

    def test_char_overlap_ratio_identical(self):
        ratio = _char_overlap_ratio("hello world test", "hello world test")
        assert ratio == 1.0

    def test_char_overlap_ratio_different(self):
        ratio = _char_overlap_ratio("abcdefghij", "klmnopqrst")
        assert ratio == 0.0

    def test_char_overlap_ratio_short_identical(self):
        ratio = _char_overlap_ratio("ab", "ab")
        assert ratio == 1.0

    def test_char_overlap_ratio_short_different(self):
        ratio = _char_overlap_ratio("ab", "cd")
        assert ratio == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
