#!/usr/bin/env python3
"""
Tests for data_distribution_eval.py

Coverage:
  1.  _get_msg_text — content extraction from Bedrock message format
  2.  extract_input_text / extract_output_text / extract_full_text — text concatenation
  3.  count_turns / is_multi_turn — turn counting logic
  4.  detect_language_heuristic — CJK, Latin, mixed, empty, other scripts
  5.  compute_char_length — Unicode character length
  5b. classify_length_category — length category boundaries
  6.  compute_percentiles — percentile computation, edge cases
  7.  parse_json_response — valid JSON, markdown block, invalid → empty dict
  8.  annotate_sample — LLM annotation with mock, fallback on error
  9.  extract_metadata — existing metadata, heuristic fallback, length_category
  10. compute_language_distribution — counts, percentages, chars
  11. compute_task_distribution — task counts, multi-turn ratio
  12. compute_length_distribution — input/output/total stats and percentiles
  13. compute_difficulty_distribution — difficulty counts and percentages
  13b. compute_length_category_distribution — length category counts and percentages
  14. compute_cross_distribution — cross-dimensional counts (incl. length_category)
  15. check_language_alerts — under-represented language alerts
  16. check_task_alerts — dominance, under-represented, multi-turn alerts
  17. check_difficulty_alerts — deviation from target ratio
  17b. check_length_category_alerts — deviation from target length category ratio
  18. check_cross_alerts — sparse cross-dimension cells
  19. stratified_split — stratified train/val splitting (4 dimensions)
  20. End-to-end pipeline — mock LLM, full run, verify report
  21. CLI parsing — defaults, custom values, toggle flags
"""

import io
import json
import math
import os
import tempfile
from unittest import mock

import pytest

from data_distribution_eval import (
    EvalConfig,
    _call_bedrock,
    _fallback_annotation,
    _get_msg_text,
    annotate_sample,
    args_to_config,
    build_parser,
    check_cross_alerts,
    check_difficulty_alerts,
    check_language_alerts,
    check_length_category_alerts,
    check_task_alerts,
    classify_length_category,
    compute_4d_cross_distribution,
    compute_char_length,
    compute_cross_distribution,
    compute_difficulty_distribution,
    compute_language_distribution,
    compute_length_category_distribution,
    compute_length_distribution,
    compute_percentiles,
    compute_task_distribution,
    count_turns,
    detect_language_heuristic,
    extract_full_text,
    extract_input_text,
    extract_metadata,
    extract_output_text,
    is_multi_turn,
    parse_json_response,
    run,
    stratified_split,
    validate_bedrock_connection,
)


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def _conv(texts, roles=None, system_text="sys", metadata=None):
    """Build a minimal Bedrock conversation dict."""
    if roles is None:
        roles = ["user" if i % 2 == 0 else "assistant" for i in range(len(texts))]
    conv = {
        "schemaVersion": "bedrock-conversation-2024",
        "system": [{"text": system_text}],
        "messages": [
            {"role": r, "content": [{"text": t}]}
            for r, t in zip(roles, texts)
        ],
    }
    if metadata:
        conv["metadata"] = metadata
    return conv


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
# 1. _get_msg_text
# ================================================================

class TestGetMsgText:
    def test_normal_content(self):
        msg = {"role": "user", "content": [{"text": "hello"}]}
        assert _get_msg_text(msg) == "hello"

    def test_empty_content_list(self):
        msg = {"role": "user", "content": []}
        assert _get_msg_text(msg) == ""

    def test_missing_content_key(self):
        msg = {"role": "user"}
        assert _get_msg_text(msg) == ""

    def test_non_list_content(self):
        msg = {"role": "user", "content": "plain text"}
        assert _get_msg_text(msg) == ""

    def test_missing_text_in_content(self):
        msg = {"role": "user", "content": [{"type": "image"}]}
        assert _get_msg_text(msg) == ""


# ================================================================
# 2. extract_input_text / extract_output_text / extract_full_text
# ================================================================

class TestTextExtraction:
    def test_extract_input_text(self):
        conv = _conv(["你好", "hello", "再见", "bye"])
        result = extract_input_text(conv)
        assert "你好" in result
        assert "再见" in result
        assert "hello" not in result

    def test_extract_output_text(self):
        conv = _conv(["你好", "hello", "再见", "bye"])
        result = extract_output_text(conv)
        assert "hello" in result
        assert "bye" in result
        assert "你好" not in result

    def test_extract_full_text(self):
        conv = _conv(["你好", "hello", "再见", "bye"])
        result = extract_full_text(conv)
        assert "你好" in result
        assert "hello" in result
        assert "再见" in result
        assert "bye" in result

    def test_extract_empty_messages(self):
        conv = {"messages": []}
        assert extract_input_text(conv) == ""
        assert extract_output_text(conv) == ""
        assert extract_full_text(conv) == ""


# ================================================================
# 3. count_turns / is_multi_turn
# ================================================================

class TestTurnCounting:
    def test_count_turns_multi(self):
        conv = _conv(["q1", "a1", "q2", "a2"])
        assert count_turns(conv) == 4

    def test_count_turns_single(self):
        conv = _conv(["q", "a"])
        assert count_turns(conv) == 2

    def test_count_turns_empty(self):
        conv = {"messages": []}
        assert count_turns(conv) == 0

    def test_is_multi_turn_true(self):
        conv = _conv(["q1", "a1", "q2", "a2"])
        assert is_multi_turn(conv) is True

    def test_is_multi_turn_false(self):
        conv = _conv(["q", "a"])
        assert is_multi_turn(conv) is False

    def test_is_multi_turn_three_messages(self):
        """3 messages is multi-turn (>2)."""
        conv = _conv(["q", "a", "q2"], roles=["user", "assistant", "user"])
        assert is_multi_turn(conv) is True


# ================================================================
# 4. detect_language_heuristic
# ================================================================

class TestDetectLanguageHeuristic:
    def test_chinese_text(self):
        assert detect_language_heuristic("这是一段中文文本") == "中文"

    def test_english_text(self):
        assert detect_language_heuristic("This is an English text") == "英文"

    def test_mixed_text(self):
        # CJK and Latin both present, neither > 70%
        result = detect_language_heuristic("你好世界这是hello测试")
        assert result == "中英混合"

    def test_empty_text(self):
        assert detect_language_heuristic("") == "其他"

    def test_whitespace_only(self):
        assert detect_language_heuristic("   \t\n  ") == "其他"

    def test_numbers_only(self):
        assert detect_language_heuristic("123456") == "其他"

    def test_predominantly_chinese(self):
        assert detect_language_heuristic("这是一段很长的中文文本内容，只包含少量的ab") == "中文"

    def test_predominantly_english(self):
        assert detect_language_heuristic("This is mostly English with 中文") == "英文"


# ================================================================
# 5. compute_char_length
# ================================================================

class TestComputeCharLength:
    def test_ascii(self):
        assert compute_char_length("hello") == 5

    def test_unicode_chinese(self):
        assert compute_char_length("你好世界") == 4

    def test_empty(self):
        assert compute_char_length("") == 0

    def test_mixed(self):
        assert compute_char_length("hello你好") == 7


# ================================================================
# 5b. classify_length_category
# ================================================================

class TestClassifyLengthCategory:
    def test_short(self):
        assert classify_length_category(100, short_max=200, long_min=1000) == "短"

    def test_short_boundary(self):
        assert classify_length_category(200, short_max=200, long_min=1000) == "短"

    def test_medium(self):
        assert classify_length_category(500, short_max=200, long_min=1000) == "中"

    def test_medium_just_above_short(self):
        assert classify_length_category(201, short_max=200, long_min=1000) == "中"

    def test_medium_just_below_long(self):
        assert classify_length_category(999, short_max=200, long_min=1000) == "中"

    def test_long(self):
        assert classify_length_category(2000, short_max=200, long_min=1000) == "长"

    def test_long_boundary(self):
        assert classify_length_category(1000, short_max=200, long_min=1000) == "长"

    def test_zero_length(self):
        assert classify_length_category(0, short_max=200, long_min=1000) == "短"

    def test_custom_boundaries(self):
        assert classify_length_category(50, short_max=100, long_min=500) == "短"
        assert classify_length_category(300, short_max=100, long_min=500) == "中"
        assert classify_length_category(600, short_max=100, long_min=500) == "长"

    def test_default_params(self):
        """Test with default parameter values."""
        assert classify_length_category(100) == "短"
        assert classify_length_category(500) == "中"
        assert classify_length_category(2000) == "长"


# ================================================================
# 6. compute_percentiles
# ================================================================

class TestComputePercentiles:
    def test_basic(self):
        values = list(range(1, 101))  # 1..100
        result = compute_percentiles(values, [50])
        assert result["P50"] == pytest.approx(50.5, abs=0.1)

    def test_percentiles_quartiles(self):
        values = list(range(1, 101))
        result = compute_percentiles(values, [25, 50, 75])
        assert result["P25"] == pytest.approx(25.75, abs=0.5)
        assert result["P50"] == pytest.approx(50.5, abs=0.5)
        assert result["P75"] == pytest.approx(75.25, abs=0.5)

    def test_single_value(self):
        result = compute_percentiles([42.0], [10, 50, 90])
        assert result["P10"] == 42.0
        assert result["P50"] == 42.0
        assert result["P90"] == 42.0

    def test_empty_list(self):
        result = compute_percentiles([], [50])
        assert result["P50"] == 0.0

    def test_two_values(self):
        result = compute_percentiles([10.0, 20.0], [50])
        assert result["P50"] == pytest.approx(15.0, abs=0.1)

    def test_p0_and_p100(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = compute_percentiles(values, [0, 100])
        assert result["P0"] == 1.0
        assert result["P100"] == 5.0


# ================================================================
# 7. parse_json_response
# ================================================================

class TestParseJsonResponse:
    def test_valid_json(self):
        result = parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_markdown_code_block(self):
        text = '```json\n{"key": "value"}\n```'
        result = parse_json_response(text)
        assert result == {"key": "value"}

    def test_json_embedded_in_text(self):
        text = 'Here is the result: {"key": "value"} and more text'
        result = parse_json_response(text)
        assert result == {"key": "value"}

    def test_invalid_json(self):
        result = parse_json_response("not json at all")
        assert result == {}

    def test_empty_string(self):
        result = parse_json_response("")
        assert result == {}


# ================================================================
# 8. annotate_sample
# ================================================================

class TestAnnotateSample:
    def test_successful_annotation(self):
        conv = _conv(["什么是机器学习？", "机器学习是人工智能的一个分支..."])
        mock_client = mock.MagicMock()
        mock_client.invoke_model.return_value = _mock_invoke_response(
            '{"language": "中文", "task_type": "QA问答", "difficulty": "中"}'
        )
        result = annotate_sample(conv, mock_client, "test-model")
        assert result["language"] == "中文"
        assert result["task_type"] == "QA问答"
        assert result["difficulty"] == "中"

    def test_annotation_fallback_on_error(self):
        conv = _conv(["Hello", "World"])
        mock_client = mock.MagicMock()
        mock_client.invoke_model.side_effect = Exception("API Error")
        result = annotate_sample(conv, mock_client, "test-model")
        # Should fallback to heuristic
        assert result["language"] == "英文"
        assert result["task_type"] == "其他"
        assert result["difficulty"] == "中"

    def test_annotation_fallback_on_bad_json(self):
        conv = _conv(["你好", "你好啊"])
        mock_client = mock.MagicMock()
        mock_client.invoke_model.return_value = _mock_invoke_response("not json")
        result = annotate_sample(conv, mock_client, "test-model")
        # Falls back to heuristic for language
        assert result["language"] == "中文"
        assert result["task_type"] == "其他"

    def test_annotation_missing_fields(self):
        conv = _conv(["test", "response"])
        mock_client = mock.MagicMock()
        mock_client.invoke_model.return_value = _mock_invoke_response(
            '{"language": "英文"}'  # missing task_type and difficulty
        )
        result = annotate_sample(conv, mock_client, "test-model")
        # Should fallback because task_type and difficulty are missing
        assert result["task_type"] == "其他"


# ================================================================
# 8b. validate_bedrock_connection
# ================================================================

class TestValidateBedrockConnection:
    def test_success(self):
        mock_client = mock.MagicMock()
        mock_client.invoke_model.return_value = _mock_invoke_response("OK")
        # Should not raise
        validate_bedrock_connection(mock_client, "test-model")
        mock_client.invoke_model.assert_called_once()

    def test_failure_raises_runtime_error(self):
        mock_client = mock.MagicMock()
        mock_client.invoke_model.side_effect = Exception("Connection refused")
        with pytest.raises(RuntimeError, match="Bedrock 连通性测试失败"):
            validate_bedrock_connection(mock_client, "test-model")

    def test_error_message_includes_model_id(self):
        mock_client = mock.MagicMock()
        mock_client.invoke_model.side_effect = Exception("bad creds")
        with pytest.raises(RuntimeError, match="my-model-id"):
            validate_bedrock_connection(mock_client, "my-model-id")

    def test_error_message_includes_error_type(self):
        mock_client = mock.MagicMock()
        mock_client.invoke_model.side_effect = ValueError("invalid region")
        with pytest.raises(RuntimeError, match="ValueError"):
            validate_bedrock_connection(mock_client, "test-model")


# ================================================================
# 8c. _fallback_annotation
# ================================================================

class TestFallbackAnnotation:
    def test_chinese_conv(self):
        conv = _conv(["你好世界", "你好！"])
        result = _fallback_annotation(conv)
        assert result["language"] == "中文"
        assert result["task_type"] == "其他"
        assert result["difficulty"] == "中"

    def test_english_conv(self):
        conv = _conv(["Hello world", "Hi there!"])
        result = _fallback_annotation(conv)
        assert result["language"] == "英文"
        assert result["task_type"] == "其他"
        assert result["difficulty"] == "中"

    def test_empty_conv(self):
        conv = {"messages": []}
        result = _fallback_annotation(conv)
        assert result["language"] == "其他"
        assert result["task_type"] == "其他"
        assert result["difficulty"] == "中"

    def test_returns_all_required_keys(self):
        conv = _conv(["test", "reply"])
        result = _fallback_annotation(conv)
        assert set(result.keys()) == {"language", "task_type", "difficulty"}


# ================================================================
# 9. extract_metadata
# ================================================================

class TestExtractMetadata:
    def test_with_existing_metadata(self):
        conv = _conv(["你好", "你好啊"], metadata={
            "language": "中文",
            "task_type": "多轮闲聊",
            "difficulty": "低",
        })
        meta = extract_metadata(conv)
        assert meta["language"] == "中文"
        assert meta["task_type"] == "多轮闲聊"
        assert meta["difficulty"] == "低"
        assert meta["input_char_len"] == 2  # "你好"
        assert meta["output_char_len"] == 3  # "你好啊"
        assert meta["num_turns"] == 2
        assert meta["is_multi_turn"] is False
        # total = "你好\n你好啊" = 6 chars, which is ≤ 200 → "短"
        assert meta["length_category"] == "短"

    def test_without_metadata_uses_heuristic(self):
        conv = _conv(["Hello world", "Hi there"])
        meta = extract_metadata(conv)
        assert meta["language"] == "英文"
        assert meta["task_type"] == "未标注"
        assert meta["difficulty"] == "未标注"
        assert meta["input_char_len"] == 11  # "Hello world"
        assert meta["output_char_len"] == 8  # "Hi there"
        assert meta["length_category"] == "短"  # total=20 ≤ 200

    def test_multi_turn_metadata(self):
        conv = _conv(["q1", "a1", "q2", "a2"])
        meta = extract_metadata(conv)
        assert meta["num_turns"] == 4
        assert meta["is_multi_turn"] is True
        assert "length_category" in meta

    def test_partial_metadata(self):
        conv = _conv(["test", "response"], metadata={"language": "英文"})
        meta = extract_metadata(conv)
        assert meta["language"] == "英文"
        assert meta["task_type"] == "未标注"
        assert meta["difficulty"] == "未标注"
        assert "length_category" in meta

    def test_length_category_with_custom_boundaries(self):
        """Test that custom boundaries are passed through."""
        # Create a conv with total_char_len ~500 chars
        long_text = "a" * 300
        conv = _conv([long_text, long_text])
        meta = extract_metadata(conv, length_short_max=100, length_long_min=800)
        assert meta["length_category"] == "中"  # 601 chars, between 100 and 800

    def test_length_category_long(self):
        long_text = "x" * 600
        conv = _conv([long_text, long_text])
        meta = extract_metadata(conv, length_short_max=200, length_long_min=1000)
        # total = "xxx...600\nxxx...600" = 1201 chars
        assert meta["length_category"] == "长"


# ================================================================
# 10. compute_language_distribution
# ================================================================

class TestLanguageDistribution:
    def test_basic(self):
        metadata_list = [
            {"language": "中文", "total_char_len": 100},
            {"language": "中文", "total_char_len": 200},
            {"language": "英文", "total_char_len": 150},
        ]
        result = compute_language_distribution(metadata_list)
        assert result["total_samples"] == 3
        assert result["distribution"]["中文"]["count"] == 2
        assert result["distribution"]["中文"]["percentage"] == pytest.approx(66.67, abs=0.01)
        assert result["distribution"]["中文"]["total_chars"] == 300
        assert result["distribution"]["英文"]["count"] == 1
        assert result["distribution"]["英文"]["percentage"] == pytest.approx(33.33, abs=0.01)

    def test_single_language(self):
        metadata_list = [
            {"language": "中文", "total_char_len": 50},
            {"language": "中文", "total_char_len": 60},
        ]
        result = compute_language_distribution(metadata_list)
        assert result["distribution"]["中文"]["percentage"] == 100.0

    def test_empty_list(self):
        result = compute_language_distribution([])
        assert result["total_samples"] == 0
        assert result["distribution"] == {}


# ================================================================
# 11. compute_task_distribution
# ================================================================

class TestTaskDistribution:
    def test_basic(self):
        metadata_list = [
            {"task_type": "QA问答", "is_multi_turn": False},
            {"task_type": "QA问答", "is_multi_turn": False},
            {"task_type": "多轮闲聊", "is_multi_turn": True},
        ]
        result = compute_task_distribution(metadata_list)
        assert result["total_samples"] == 3
        assert result["distribution"]["QA问答"]["count"] == 2
        assert result["distribution"]["多轮闲聊"]["count"] == 1
        assert result["multi_turn_count"] == 1
        assert result["multi_turn_percentage"] == pytest.approx(33.33, abs=0.01)

    def test_all_multi_turn(self):
        metadata_list = [
            {"task_type": "多轮闲聊", "is_multi_turn": True},
            {"task_type": "QA问答", "is_multi_turn": True},
        ]
        result = compute_task_distribution(metadata_list)
        assert result["multi_turn_percentage"] == 100.0

    def test_no_multi_turn(self):
        metadata_list = [
            {"task_type": "QA问答", "is_multi_turn": False},
        ]
        result = compute_task_distribution(metadata_list)
        assert result["multi_turn_percentage"] == 0.0

    def test_empty(self):
        result = compute_task_distribution([])
        assert result["total_samples"] == 0
        assert result["multi_turn_percentage"] == 0.0


# ================================================================
# 12. compute_length_distribution
# ================================================================

class TestLengthDistribution:
    def test_basic(self):
        metadata_list = [
            {"input_char_len": 10, "output_char_len": 20, "total_char_len": 30},
            {"input_char_len": 30, "output_char_len": 40, "total_char_len": 70},
            {"input_char_len": 50, "output_char_len": 60, "total_char_len": 110},
        ]
        result = compute_length_distribution(metadata_list, [50])
        assert result["input"]["mean"] == pytest.approx(30.0, abs=0.01)
        assert result["input"]["min"] == 10
        assert result["input"]["max"] == 50
        assert result["output"]["mean"] == pytest.approx(40.0, abs=0.01)
        assert result["total"]["mean"] == pytest.approx(70.0, abs=0.01)

    def test_single_sample(self):
        metadata_list = [
            {"input_char_len": 42, "output_char_len": 84, "total_char_len": 126},
        ]
        result = compute_length_distribution(metadata_list, [50])
        assert result["input"]["min"] == 42
        assert result["input"]["max"] == 42
        assert result["input"]["mean"] == 42.0
        assert result["input"]["percentiles"]["P50"] == 42

    def test_empty(self):
        result = compute_length_distribution([], [50])
        assert result["input"]["mean"] == 0.0
        assert result["input"]["min"] == 0
        assert result["input"]["max"] == 0


# ================================================================
# 13. compute_difficulty_distribution
# ================================================================

class TestDifficultyDistribution:
    def test_basic(self):
        metadata_list = [
            {"difficulty": "低"},
            {"difficulty": "低"},
            {"difficulty": "中"},
            {"difficulty": "中"},
            {"difficulty": "中"},
            {"difficulty": "高"},
        ]
        result = compute_difficulty_distribution(metadata_list)
        assert result["total_samples"] == 6
        assert result["distribution"]["低"]["count"] == 2
        assert result["distribution"]["中"]["count"] == 3
        assert result["distribution"]["高"]["count"] == 1
        assert result["distribution"]["低"]["percentage"] == pytest.approx(33.33, abs=0.01)

    def test_single_difficulty(self):
        metadata_list = [{"difficulty": "中"}] * 5
        result = compute_difficulty_distribution(metadata_list)
        assert result["distribution"]["中"]["percentage"] == 100.0

    def test_empty(self):
        result = compute_difficulty_distribution([])
        assert result["total_samples"] == 0


# ================================================================
# 13b. compute_length_category_distribution
# ================================================================

class TestLengthCategoryDistribution:
    def test_basic(self):
        metadata_list = [
            {"length_category": "短"},
            {"length_category": "短"},
            {"length_category": "中"},
            {"length_category": "中"},
            {"length_category": "中"},
            {"length_category": "长"},
        ]
        result = compute_length_category_distribution(metadata_list)
        assert result["total_samples"] == 6
        assert result["distribution"]["短"]["count"] == 2
        assert result["distribution"]["中"]["count"] == 3
        assert result["distribution"]["长"]["count"] == 1
        assert result["distribution"]["短"]["percentage"] == pytest.approx(33.33, abs=0.01)

    def test_single_category(self):
        metadata_list = [{"length_category": "中"}] * 5
        result = compute_length_category_distribution(metadata_list)
        assert result["distribution"]["中"]["percentage"] == 100.0

    def test_empty(self):
        result = compute_length_category_distribution([])
        assert result["total_samples"] == 0


# ================================================================
# 14. compute_cross_distribution
# ================================================================

class TestCrossDistribution:
    def test_basic(self):
        metadata_list = [
            {"language": "中文", "task_type": "QA问答"},
            {"language": "中文", "task_type": "QA问答"},
            {"language": "英文", "task_type": "多轮闲聊"},
            {"language": "中文", "task_type": "多轮闲聊"},
        ]
        result = compute_cross_distribution(metadata_list, "language", "task_type")
        assert result["dim_a"] == "language"
        assert result["dim_b"] == "task_type"
        assert result["cross"]["中文"]["QA问答"] == 2
        assert result["cross"]["中文"]["多轮闲聊"] == 1
        assert result["cross"]["英文"]["多轮闲聊"] == 1

    def test_single_cell(self):
        metadata_list = [
            {"language": "中文", "task_type": "QA问答"},
        ]
        result = compute_cross_distribution(metadata_list, "language", "task_type")
        assert result["cross"]["中文"]["QA问答"] == 1

    def test_empty(self):
        result = compute_cross_distribution([], "language", "task_type")
        assert result["cross"] == {}


# ================================================================
# 14b. compute_4d_cross_distribution
# ================================================================

class TestCompute4dCrossDistribution:
    def test_basic(self):
        metadata_list = [
            {"language": "中文", "task_type": "QA问答",
             "length_category": "短", "difficulty": "低"},
            {"language": "中文", "task_type": "QA问答",
             "length_category": "短", "difficulty": "低"},
            {"language": "英文", "task_type": "多轮闲聊",
             "length_category": "长", "difficulty": "高"},
        ]
        result = compute_4d_cross_distribution(metadata_list)
        assert result["dimensions"] == [
            "language", "task_type", "length_category", "difficulty"]
        assert result["total_samples"] == 3
        assert result["num_cells"] == 2
        # Sorted by count desc
        assert result["cells"][0]["count"] == 2
        assert result["cells"][0]["language"] == "中文"
        assert result["cells"][1]["count"] == 1
        assert result["cells"][1]["language"] == "英文"

    def test_empty(self):
        result = compute_4d_cross_distribution([])
        assert result["total_samples"] == 0
        assert result["num_cells"] == 0
        assert result["cells"] == []

    def test_single_sample(self):
        metadata_list = [
            {"language": "中文", "task_type": "翻译",
             "length_category": "中", "difficulty": "中"},
        ]
        result = compute_4d_cross_distribution(metadata_list)
        assert result["num_cells"] == 1
        assert result["cells"][0]["count"] == 1
        assert result["cells"][0]["language"] == "中文"
        assert result["cells"][0]["task_type"] == "翻译"
        assert result["cells"][0]["length_category"] == "中"
        assert result["cells"][0]["difficulty"] == "中"

    def test_all_dimensions_present(self):
        metadata_list = [
            {"language": "中文", "task_type": "QA问答",
             "length_category": "短", "difficulty": "低"},
            {"language": "英文", "task_type": "代码生成",
             "length_category": "长", "difficulty": "高"},
        ]
        result = compute_4d_cross_distribution(metadata_list)
        for cell in result["cells"]:
            assert "language" in cell
            assert "task_type" in cell
            assert "length_category" in cell
            assert "difficulty" in cell
            assert "count" in cell


# ================================================================
# 15. check_language_alerts
# ================================================================

class TestLanguageAlerts:
    def test_no_alert_when_sufficient(self):
        lang_dist = {
            "total_samples": 10000,
            "distribution": {
                "中文": {"count": 7000, "percentage": 70.0, "total_chars": 0},
                "英文": {"count": 3000, "percentage": 30.0, "total_chars": 0},
            },
        }
        cfg = EvalConfig()
        alerts = check_language_alerts(lang_dist, cfg)
        assert len(alerts) == 0

    def test_alert_when_low_count_and_pct(self):
        lang_dist = {
            "total_samples": 100000,
            "distribution": {
                "中文": {"count": 90000, "percentage": 90.0, "total_chars": 0},
                "英文": {"count": 9500, "percentage": 9.5, "total_chars": 0},
                "日文": {"count": 500, "percentage": 0.5, "total_chars": 0},
            },
        }
        cfg = EvalConfig(lang_min_pct=1.0, lang_min_abs=5000)
        alerts = check_language_alerts(lang_dist, cfg)
        assert len(alerts) == 1
        assert "日文" in alerts[0]["message"]

    def test_no_alert_if_pct_ok(self):
        """If percentage is above threshold, no alert even if count is low."""
        lang_dist = {
            "total_samples": 1000,
            "distribution": {
                "中文": {"count": 900, "percentage": 90.0, "total_chars": 0},
                "英文": {"count": 100, "percentage": 10.0, "total_chars": 0},
            },
        }
        cfg = EvalConfig(lang_min_pct=1.0, lang_min_abs=5000)
        alerts = check_language_alerts(lang_dist, cfg)
        assert len(alerts) == 0

    def test_other_language_ignored(self):
        """'其他' language is skipped for alerts."""
        lang_dist = {
            "total_samples": 10000,
            "distribution": {
                "中文": {"count": 9900, "percentage": 99.0, "total_chars": 0},
                "其他": {"count": 100, "percentage": 1.0, "total_chars": 0},
            },
        }
        cfg = EvalConfig(lang_min_pct=5.0, lang_min_abs=5000)
        alerts = check_language_alerts(lang_dist, cfg)
        # "其他" should not trigger alert
        assert len(alerts) == 0


# ================================================================
# 16. check_task_alerts
# ================================================================

class TestTaskAlerts:
    def test_dominance_alert(self):
        task_dist = {
            "total_samples": 1000,
            "distribution": {
                "QA问答": {"count": 600, "percentage": 60.0},
                "多轮闲聊": {"count": 400, "percentage": 40.0},
            },
            "multi_turn_count": 400,
            "multi_turn_percentage": 40.0,
        }
        cfg = EvalConfig(task_max_single_pct=50.0)
        alerts = check_task_alerts(task_dist, cfg)
        dominance = [a for a in alerts if "占比过高" in a["message"]]
        assert len(dominance) == 1
        assert "QA问答" in dominance[0]["message"]

    def test_low_count_alert(self):
        task_dist = {
            "total_samples": 5000,
            "distribution": {
                "QA问答": {"count": 3000, "percentage": 60.0},
                "数学推理": {"count": 100, "percentage": 2.0},
                "多轮闲聊": {"count": 1900, "percentage": 38.0},
            },
            "multi_turn_count": 1900,
            "multi_turn_percentage": 38.0,
        }
        cfg = EvalConfig(task_min_abs=2000)
        alerts = check_task_alerts(task_dist, cfg)
        low_count = [a for a in alerts if "样本不足" in a["message"]]
        assert any("数学推理" in a["message"] for a in low_count)

    def test_multi_turn_alert(self):
        task_dist = {
            "total_samples": 1000,
            "distribution": {
                "QA问答": {"count": 1000, "percentage": 100.0},
            },
            "multi_turn_count": 100,
            "multi_turn_percentage": 10.0,
        }
        cfg = EvalConfig(multi_turn_min_pct=20.0)
        alerts = check_task_alerts(task_dist, cfg)
        mt_alerts = [a for a in alerts if "多轮对话占比过低" in a["message"]]
        assert len(mt_alerts) == 1

    def test_no_alerts(self):
        task_dist = {
            "total_samples": 10000,
            "distribution": {
                "QA问答": {"count": 3000, "percentage": 30.0},
                "多轮闲聊": {"count": 3000, "percentage": 30.0},
                "代码生成": {"count": 2000, "percentage": 20.0},
                "翻译": {"count": 2000, "percentage": 20.0},
            },
            "multi_turn_count": 3000,
            "multi_turn_percentage": 30.0,
        }
        cfg = EvalConfig(task_max_single_pct=50.0, task_min_abs=2000,
                         multi_turn_min_pct=20.0)
        alerts = check_task_alerts(task_dist, cfg)
        assert len(alerts) == 0

    def test_unlabeled_and_other_ignored(self):
        """'未标注' and '其他' tasks should not trigger alerts."""
        task_dist = {
            "total_samples": 1000,
            "distribution": {
                "未标注": {"count": 500, "percentage": 50.0},
                "其他": {"count": 500, "percentage": 50.0},
            },
            "multi_turn_count": 500,
            "multi_turn_percentage": 50.0,
        }
        cfg = EvalConfig(task_max_single_pct=40.0)
        alerts = check_task_alerts(task_dist, cfg)
        # Neither "未标注" nor "其他" should generate dominance or low-count alerts
        task_specific = [a for a in alerts if "占比过高" in a["message"] or "样本不足" in a["message"]]
        assert len(task_specific) == 0


# ================================================================
# 17. check_difficulty_alerts
# ================================================================

class TestDifficultyAlerts:
    def test_no_alert_near_target(self):
        diff_dist = {
            "total_samples": 100,
            "distribution": {
                "低": {"count": 30, "percentage": 30.0},
                "中": {"count": 50, "percentage": 50.0},
                "高": {"count": 20, "percentage": 20.0},
            },
        }
        cfg = EvalConfig(difficulty_tolerance=15.0)
        alerts = check_difficulty_alerts(diff_dist, cfg)
        assert len(alerts) == 0

    def test_alert_when_deviated(self):
        diff_dist = {
            "total_samples": 100,
            "distribution": {
                "低": {"count": 80, "percentage": 80.0},
                "中": {"count": 15, "percentage": 15.0},
                "高": {"count": 5, "percentage": 5.0},
            },
        }
        cfg = EvalConfig(difficulty_tolerance=15.0)
        alerts = check_difficulty_alerts(diff_dist, cfg)
        # 低: 80 vs 30 = 50% deviation, 中: 15 vs 50 = 35% deviation
        assert len(alerts) >= 2
        low_alerts = [a for a in alerts if "'低'" in a["message"]]
        assert len(low_alerts) == 1
        assert "过高" in low_alerts[0]["message"]

    def test_missing_difficulty_level(self):
        diff_dist = {
            "total_samples": 100,
            "distribution": {
                "低": {"count": 50, "percentage": 50.0},
                "中": {"count": 50, "percentage": 50.0},
                # "高" is missing (0%)
            },
        }
        cfg = EvalConfig(
            difficulty_target={"低": 30.0, "中": 50.0, "高": 20.0},
            difficulty_tolerance=15.0,
        )
        alerts = check_difficulty_alerts(diff_dist, cfg)
        # 高: 0 vs 20 = 20% deviation (>15), 低: 50 vs 30 = 20% (>15)
        assert any("'高'" in a["message"] for a in alerts)
        assert any("'低'" in a["message"] for a in alerts)


# ================================================================
# 17b. check_length_category_alerts
# ================================================================

class TestLengthCategoryAlerts:
    def test_no_alert_near_target(self):
        len_cat_dist = {
            "total_samples": 100,
            "distribution": {
                "短": {"count": 25, "percentage": 25.0},
                "中": {"count": 50, "percentage": 50.0},
                "长": {"count": 25, "percentage": 25.0},
            },
        }
        cfg = EvalConfig(length_cat_tolerance=15.0)
        alerts = check_length_category_alerts(len_cat_dist, cfg)
        assert len(alerts) == 0

    def test_alert_when_deviated(self):
        len_cat_dist = {
            "total_samples": 100,
            "distribution": {
                "短": {"count": 70, "percentage": 70.0},
                "中": {"count": 25, "percentage": 25.0},
                "长": {"count": 5, "percentage": 5.0},
            },
        }
        cfg = EvalConfig(length_cat_tolerance=15.0)
        alerts = check_length_category_alerts(len_cat_dist, cfg)
        # 短: 70 vs 25 = 45% deviation, 中: 25 vs 50 = 25% deviation,
        # 长: 5 vs 25 = 20% deviation
        assert len(alerts) == 3
        short_alerts = [a for a in alerts if "'短'" in a["message"]]
        assert len(short_alerts) == 1
        assert "过高" in short_alerts[0]["message"]

    def test_missing_category(self):
        len_cat_dist = {
            "total_samples": 100,
            "distribution": {
                "短": {"count": 50, "percentage": 50.0},
                "中": {"count": 50, "percentage": 50.0},
                # "长" is missing (0%)
            },
        }
        cfg = EvalConfig(
            length_cat_target={"短": 25.0, "中": 50.0, "长": 25.0},
            length_cat_tolerance=15.0,
        )
        alerts = check_length_category_alerts(len_cat_dist, cfg)
        assert any("'长'" in a["message"] for a in alerts)
        assert any("'短'" in a["message"] for a in alerts)

    def test_alert_dimension_field(self):
        len_cat_dist = {
            "total_samples": 10,
            "distribution": {
                "短": {"count": 10, "percentage": 100.0},
            },
        }
        cfg = EvalConfig(length_cat_tolerance=10.0)
        alerts = check_length_category_alerts(len_cat_dist, cfg)
        for a in alerts:
            assert a["dimension"] == "length_category"
            assert a["level"] == "warning"


# ================================================================
# 18. check_cross_alerts
# ================================================================

class TestCrossAlerts:
    def test_sparse_cell_alert(self):
        cross_4d = {
            "dimensions": ["language", "task_type", "length_category", "difficulty"],
            "cells": [
                {"language": "中文", "task_type": "QA问答",
                 "length_category": "短", "difficulty": "低", "count": 5000},
                {"language": "英文", "task_type": "QA问答",
                 "length_category": "短", "difficulty": "低", "count": 3000},
                {"language": "中文", "task_type": "数学推理",
                 "length_category": "中", "difficulty": "高", "count": 50},
            ],
            "total_samples": 8050,
            "num_cells": 3,
        }
        alerts = check_cross_alerts(cross_4d, min_count=100)
        assert len(alerts) == 1
        assert "数学推理" in alerts[0]["message"]
        assert "中文" in alerts[0]["message"]
        assert alerts[0]["dimension"] == "language×task_type×length_category×difficulty"

    def test_no_alerts(self):
        cross_4d = {
            "dimensions": ["language", "task_type", "length_category", "difficulty"],
            "cells": [
                {"language": "中文", "task_type": "QA问答",
                 "length_category": "短", "difficulty": "低", "count": 500},
                {"language": "英文", "task_type": "QA问答",
                 "length_category": "短", "difficulty": "低", "count": 500},
            ],
            "total_samples": 1000,
            "num_cells": 2,
        }
        alerts = check_cross_alerts(cross_4d, min_count=100)
        assert len(alerts) == 0

    def test_all_sparse(self):
        cross_4d = {
            "dimensions": ["language", "task_type", "length_category", "difficulty"],
            "cells": [
                {"language": "中文", "task_type": "QA问答",
                 "length_category": "短", "difficulty": "低", "count": 10},
                {"language": "英文", "task_type": "翻译",
                 "length_category": "长", "difficulty": "高", "count": 20},
            ],
            "total_samples": 30,
            "num_cells": 2,
        }
        alerts = check_cross_alerts(cross_4d, min_count=50)
        assert len(alerts) == 2


# ================================================================
# 19. stratified_split
# ================================================================

class TestStratifiedSplit:
    def _meta(self, lang="中文", task="QA问答", diff="中", lencat="中"):
        return {"language": lang, "task_type": task,
                "difficulty": diff, "length_category": lencat}

    def test_basic_split(self):
        metadata_list = (
            [self._meta("中文", "QA问答", "中", "短")] * 5
            + [self._meta("英文", "多轮闲聊", "低", "长")] * 5
        )
        train_idx, val_idx = stratified_split(metadata_list, val_ratio=0.2, seed=42)

        # All indices accounted for
        assert sorted(train_idx + val_idx) == list(range(10))

        # No overlap
        assert set(train_idx) & set(val_idx) == set()

        # Val set should have ~20% from each stratum
        assert len(val_idx) == 2  # 1 from each stratum

    def test_reproducible_with_seed(self):
        metadata_list = [self._meta()] * 20
        t1, v1 = stratified_split(metadata_list, val_ratio=0.2, seed=42)
        t2, v2 = stratified_split(metadata_list, val_ratio=0.2, seed=42)
        assert t1 == t2
        assert v1 == v2

    def test_different_seeds_differ(self):
        metadata_list = [self._meta()] * 20
        _, v1 = stratified_split(metadata_list, val_ratio=0.2, seed=42)
        _, v2 = stratified_split(metadata_list, val_ratio=0.2, seed=123)
        assert v1 != v2

    def test_single_sample_per_stratum(self):
        metadata_list = [
            self._meta("中文", "QA问答", "低", "短"),
            self._meta("英文", "翻译", "高", "长"),
        ]
        train_idx, val_idx = stratified_split(metadata_list, val_ratio=0.2, seed=42)
        # Each stratum has 1 sample, should go to train
        assert sorted(train_idx) == [0, 1]
        assert val_idx == []

    def test_val_ratio_zero(self):
        metadata_list = [self._meta()] * 10
        train_idx, val_idx = stratified_split(metadata_list, val_ratio=0.0, seed=42)
        # val_ratio=0.0 → max(1, 0) = 1, so still get 1 val sample from the stratum
        assert len(val_idx) >= 1

    def test_many_strata_4d(self):
        """Stratification on all 4 dimensions: 3 langs × 3 tasks × 1 diff × 1 lencat."""
        metadata_list = []
        for lang in ["中文", "英文", "日文"]:
            for task in ["QA", "翻译", "闲聊"]:
                for _ in range(10):
                    metadata_list.append(self._meta(lang, task, "中", "中"))
        train_idx, val_idx = stratified_split(metadata_list, val_ratio=0.1, seed=42)
        assert sorted(train_idx + val_idx) == list(range(len(metadata_list)))
        # 9 strata (3 lang × 3 task × 1 diff × 1 lencat), 10 each → 9 val
        assert len(val_idx) == 9

    def test_full_cross_strata(self):
        """All 4 dimensions vary: 2 × 2 × 2 × 2 = 16 strata, 4 samples each."""
        metadata_list = []
        for lang in ["中文", "英文"]:
            for task in ["QA", "闲聊"]:
                for diff in ["低", "高"]:
                    for lc in ["短", "长"]:
                        for _ in range(4):
                            metadata_list.append(self._meta(lang, task, diff, lc))
        train_idx, val_idx = stratified_split(metadata_list, val_ratio=0.25, seed=42)
        assert sorted(train_idx + val_idx) == list(range(len(metadata_list)))
        # 16 strata × 4 each, val_ratio=0.25 → 1 per stratum → 16 val
        assert len(val_idx) == 16


# ================================================================
# 20. End-to-end pipeline
# ================================================================

class TestEndToEnd:
    def test_full_pipeline_no_annotation(self):
        """Run pipeline without LLM annotation, using heuristic language detection."""
        conversations = [
            _conv(["你好", "你好，有什么可以帮您的吗？"], metadata={
                "language": "中文", "task_type": "多轮闲聊", "difficulty": "低"}),
            _conv(["请解释量子计算的基本原理", "量子计算利用量子力学的叠加态和纠缠态..."], metadata={
                "language": "中文", "task_type": "QA问答", "difficulty": "高"}),
            _conv(["Hello", "Hi! How can I help you?"], metadata={
                "language": "英文", "task_type": "多轮闲聊", "difficulty": "低"}),
            _conv(["Write a Python function to sort a list",
                   "Here's a Python function:\ndef sort_list(lst):\n    return sorted(lst)"],
                  metadata={"language": "英文", "task_type": "代码生成", "difficulty": "中"}),
            _conv(["帮我把这段话翻译成英文：今天天气很好",
                   "The weather is nice today."],
                  metadata={"language": "中英混合", "task_type": "翻译", "difficulty": "低"}),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.jsonl")
            report_path = os.path.join(tmpdir, "report.json")
            plot_dir = os.path.join(tmpdir, "plots")

            _write_jsonl(input_path, conversations)

            cfg = EvalConfig(
                input=input_path,
                output_annotated=os.path.join(tmpdir, "annotated.jsonl"),
                report_json=report_path,
                report_dir=plot_dir,
                enable_annotation=False,
                enable_val_split=False,
                enable_plots=False,
                lang_min_abs=1,
                task_min_abs=1,
            )

            report = run(cfg)

            # Verify report structure
            assert report["total_samples"] == 5
            assert "language_distribution" in report
            assert "task_distribution" in report
            assert "length_distribution" in report
            assert "difficulty_distribution" in report
            assert "length_category_distribution" in report
            assert "cross_distribution_4d" in report
            assert "alerts" in report
            assert "config" in report

            # Verify language distribution
            lang = report["language_distribution"]["distribution"]
            assert lang["中文"]["count"] == 2
            assert lang["英文"]["count"] == 2
            assert lang["中英混合"]["count"] == 1

            # Verify task distribution
            task = report["task_distribution"]["distribution"]
            assert task["多轮闲聊"]["count"] == 2
            assert task["QA问答"]["count"] == 1

            # Verify length category distribution exists
            len_cat = report["length_category_distribution"]
            assert len_cat["total_samples"] == 5
            # All samples are short (< 200 chars)
            assert "短" in len_cat["distribution"]

            # Verify 4D cross distribution structure
            cross_4d = report["cross_distribution_4d"]
            assert cross_4d["dimensions"] == [
                "language", "task_type", "length_category", "difficulty"]
            assert isinstance(cross_4d["cells"], list)
            assert cross_4d["num_cells"] > 0
            assert cross_4d["total_samples"] == 5
            # Each cell has all dimension keys + count
            for cell in cross_4d["cells"]:
                assert "language" in cell
                assert "task_type" in cell
                assert "length_category" in cell
                assert "difficulty" in cell
                assert "count" in cell

            # Verify config boundaries are in report
            assert report["config"]["length_short_max"] == 200
            assert report["config"]["length_long_min"] == 1000

            # Verify JSON report was written
            assert os.path.exists(report_path)
            with open(report_path) as f:
                saved_report = json.load(f)
            assert saved_report["total_samples"] == 5

    def test_pipeline_with_val_split(self):
        """Run pipeline with validation set splitting."""
        conversations = []
        for i in range(20):
            lang = "中文" if i % 2 == 0 else "英文"
            task = "QA问答" if i % 3 == 0 else "多轮闲聊"
            diff = ["低", "中", "高"][i % 3]
            conversations.append(_conv(
                [f"question_{i}", f"answer_{i}"],
                metadata={"language": lang, "task_type": task, "difficulty": diff},
            ))

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.jsonl")
            _write_jsonl(input_path, conversations)

            cfg = EvalConfig(
                input=input_path,
                output_annotated=os.path.join(tmpdir, "annotated.jsonl"),
                report_json=os.path.join(tmpdir, "report.json"),
                report_dir=os.path.join(tmpdir, "plots"),
                enable_annotation=False,
                enable_val_split=True,
                val_ratio=0.2,
                val_output=os.path.join(tmpdir, "val.jsonl"),
                train_output=os.path.join(tmpdir, "train.jsonl"),
                enable_plots=False,
                lang_min_abs=1,
                task_min_abs=1,
            )

            report = run(cfg)

            assert report["split_info"]["train_count"] + report["split_info"]["val_count"] == 20

            # Verify files written
            train_data = _read_jsonl(os.path.join(tmpdir, "train.jsonl"))
            val_data = _read_jsonl(os.path.join(tmpdir, "val.jsonl"))
            assert len(train_data) + len(val_data) == 20

    def test_pipeline_with_annotation(self):
        """Run pipeline with mocked LLM annotation."""
        conversations = [
            _conv(["什么是Python？", "Python是一种编程语言。"]),
            _conv(["Hello", "Hi there!"]),
        ]

        mock_client = mock.MagicMock()
        responses = [
            # 1st call: validate_bedrock_connection sends "请回复OK"
            _mock_invoke_response("OK"),
            # 2nd call: annotate sample 0
            _mock_invoke_response(json.dumps({
                "language": "中文", "task_type": "QA问答", "difficulty": "低"
            })),
            # 3rd call: annotate sample 1
            _mock_invoke_response(json.dumps({
                "language": "英文", "task_type": "多轮闲聊", "difficulty": "低"
            })),
        ]
        mock_client.invoke_model.side_effect = responses

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.jsonl")
            _write_jsonl(input_path, conversations)

            cfg = EvalConfig(
                input=input_path,
                output_annotated=os.path.join(tmpdir, "annotated.jsonl"),
                report_json=os.path.join(tmpdir, "report.json"),
                report_dir=os.path.join(tmpdir, "plots"),
                enable_annotation=True,
                enable_val_split=False,
                enable_plots=False,
                lang_min_abs=1,
                task_min_abs=1,
            )

            report = run(cfg, bedrock_client=mock_client)

            assert report["total_samples"] == 2
            # 1 validation call + 2 annotation calls = 3
            assert mock_client.invoke_model.call_count == 3

            # Verify annotated file was written
            annotated = _read_jsonl(os.path.join(tmpdir, "annotated.jsonl"))
            assert len(annotated) == 2
            assert annotated[0]["metadata"]["language"] == "中文"
            assert annotated[1]["metadata"]["language"] == "英文"

    def test_pipeline_skips_annotation_when_metadata_present(self):
        """If all samples have metadata, annotation should be skipped."""
        conversations = [
            _conv(["你好", "你好"],
                  metadata={"language": "中文", "task_type": "QA问答", "difficulty": "低"}),
        ]

        mock_client = mock.MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.jsonl")
            _write_jsonl(input_path, conversations)

            cfg = EvalConfig(
                input=input_path,
                output_annotated=os.path.join(tmpdir, "annotated.jsonl"),
                report_json=os.path.join(tmpdir, "report.json"),
                report_dir=os.path.join(tmpdir, "plots"),
                enable_annotation=True,
                enable_val_split=False,
                enable_plots=False,
                lang_min_abs=1,
                task_min_abs=1,
            )

            report = run(cfg, bedrock_client=mock_client)

            # LLM should not be called since metadata is already present
            mock_client.invoke_model.assert_not_called()


# ================================================================
# 21. CLI parsing
# ================================================================

class TestCLI:
    def test_defaults(self):
        parser = build_parser()
        args = parser.parse_args([])
        cfg = args_to_config(args)
        assert cfg.input == "./zh_mixed_filtered.jsonl"
        assert cfg.enable_annotation is True
        assert cfg.enable_val_split is False
        assert cfg.val_ratio == 0.1
        assert cfg.lang_min_abs == 5000
        assert cfg.task_max_single_pct == 50.0
        assert cfg.difficulty_tolerance == 15.0
        assert cfg.length_short_max == 200
        assert cfg.length_long_min == 1000
        assert cfg.length_cat_tolerance == 15.0
        assert cfg.cross_min_count == 100

    def test_custom_input(self):
        parser = build_parser()
        args = parser.parse_args(["-i", "custom.jsonl"])
        cfg = args_to_config(args)
        assert cfg.input == "custom.jsonl"

    def test_disable_annotation(self):
        parser = build_parser()
        args = parser.parse_args(["--no-enable-annotation"])
        cfg = args_to_config(args)
        assert cfg.enable_annotation is False

    def test_enable_val_split(self):
        parser = build_parser()
        args = parser.parse_args(["--enable-val-split", "--val-ratio", "0.2"])
        cfg = args_to_config(args)
        assert cfg.enable_val_split is True
        assert cfg.val_ratio == 0.2

    def test_custom_alert_thresholds(self):
        parser = build_parser()
        args = parser.parse_args([
            "--lang-min-abs", "1000",
            "--task-max-single-pct", "40.0",
            "--task-min-abs", "500",
            "--multi-turn-min-pct", "25.0",
            "--difficulty-tolerance", "10.0",
        ])
        cfg = args_to_config(args)
        assert cfg.lang_min_abs == 1000
        assert cfg.task_max_single_pct == 40.0
        assert cfg.task_min_abs == 500
        assert cfg.multi_turn_min_pct == 25.0
        assert cfg.difficulty_tolerance == 10.0

    def test_disable_plots(self):
        parser = build_parser()
        args = parser.parse_args(["--no-enable-plots"])
        cfg = args_to_config(args)
        assert cfg.enable_plots is False

    def test_custom_seed(self):
        parser = build_parser()
        args = parser.parse_args(["--random-seed", "123"])
        cfg = args_to_config(args)
        assert cfg.random_seed == 123

    def test_custom_length_category_settings(self):
        parser = build_parser()
        args = parser.parse_args([
            "--length-short-max", "100",
            "--length-long-min", "500",
            "--length-cat-tolerance", "10.0",
        ])
        cfg = args_to_config(args)
        assert cfg.length_short_max == 100
        assert cfg.length_long_min == 500
        assert cfg.length_cat_tolerance == 10.0

    def test_custom_cross_min_count(self):
        parser = build_parser()
        args = parser.parse_args(["--cross-min-count", "50"])
        cfg = args_to_config(args)
        assert cfg.cross_min_count == 50


# ================================================================
# 22. Additional edge cases
# ================================================================

class TestEdgeCases:
    def test_conversation_with_no_assistant(self):
        """A conversation with only user messages."""
        conv = {"messages": [
            {"role": "user", "content": [{"text": "hello"}]},
        ]}
        meta = extract_metadata(conv)
        assert meta["output_char_len"] == 0
        assert meta["input_char_len"] == 5
        assert meta["num_turns"] == 1
        assert meta["is_multi_turn"] is False
        assert meta["length_category"] == "短"  # 5 chars

    def test_large_percentile_list(self):
        values = list(range(1, 1001))
        result = compute_percentiles(values, [1, 5, 10, 25, 50, 75, 90, 95, 99])
        assert result["P1"] < result["P5"] < result["P50"] < result["P95"] < result["P99"]

    def test_cross_distribution_missing_key(self):
        """If metadata has missing key, defaults to '未知'."""
        metadata_list = [
            {"task_type": "QA"},  # no "language" key
        ]
        result = compute_cross_distribution(metadata_list, "language", "task_type")
        assert "未知" in result["cross"]

    def test_cross_with_length_category(self):
        """Cross distribution works with length_category dimension."""
        metadata_list = [
            {"language": "中文", "length_category": "短"},
            {"language": "中文", "length_category": "中"},
            {"language": "中文", "length_category": "长"},
            {"language": "英文", "length_category": "短"},
        ]
        result = compute_cross_distribution(
            metadata_list, "language", "length_category")
        assert result["cross"]["中文"]["短"] == 1
        assert result["cross"]["中文"]["中"] == 1
        assert result["cross"]["中文"]["长"] == 1
        assert result["cross"]["英文"]["短"] == 1

    def test_alerts_level_field(self):
        """Verify alert dicts have required fields."""
        lang_dist = {
            "total_samples": 100000,
            "distribution": {
                "日文": {"count": 100, "percentage": 0.1, "total_chars": 0},
            },
        }
        cfg = EvalConfig()
        alerts = check_language_alerts(lang_dist, cfg)
        for a in alerts:
            assert "level" in a
            assert "dimension" in a
            assert "message" in a
            assert a["level"] in ("warning", "info")

    def test_length_distribution_all_same_length(self):
        metadata_list = [
            {"input_char_len": 100, "output_char_len": 200, "total_char_len": 300},
        ] * 10
        result = compute_length_distribution(metadata_list, [10, 50, 90])
        assert result["input"]["mean"] == 100.0
        assert result["input"]["min"] == 100
        assert result["input"]["max"] == 100
        assert result["input"]["percentiles"]["P10"] == 100
        assert result["input"]["percentiles"]["P90"] == 100
