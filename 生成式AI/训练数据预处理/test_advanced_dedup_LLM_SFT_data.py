#!/usr/bin/env python3
"""
Tests for advanced_dedup_LLM_SFT_data.py

Coverage:
  1. extract_full_text — system ignored, all messages concatenated
  2. extract_input_text — only user turns
  3. extract_assistant_texts — only assistant turns
  4. cosine_similarity — identical, orthogonal, opposite, zero, dimension mismatch
  5. UnionFind — basic, transitivity, groups
  6. embed_texts — batching, mock client
  7. cluster_by_cosine — controlled embeddings
  8. compute_heuristic_score — more turns / longer text scores higher
  9. parse_json_response — plain, markdown-wrapped, messy, empty
 10. format_conversation_for_judge — labels, system included, multi-turn
 11. score_single_turn_output — mock LLM, parse, error fallback
 12. score_multi_turn_output — mock LLM, weighting modes, partial failures
 13. score_output_quality — dispatch to single/multi
 14. select_best_per_group — heuristic and LLM paths
 15. End-to-end: full_sample_dedup only
 16. End-to-end: input_dedup only
 17. End-to-end: both stages
 18. End-to-end: heuristic scoring
 19. End-to-end: all unique (no removal)
 20. CLI parsing
"""

import io
import json
import math
import os
import tempfile
from typing import List
from unittest import mock

import pytest

from advanced_dedup_LLM_SFT_data import (
    AdvancedDedupConfig,
    UnionFind,
    _call_bedrock,
    _format_conv_full,
    _get_msg_text,
    args_to_config,
    build_parser,
    cluster_by_cosine,
    compute_heuristic_score,
    cosine_similarity,
    embed_texts,
    extract_assistant_texts,
    extract_full_text,
    extract_input_text,
    format_conversation_for_judge,
    parse_json_response,
    run,
    score_multi_turn_output,
    score_output_quality,
    score_single_turn_output,
    select_best_per_group,
)


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def _conv(texts, roles=None, system_text="sys"):
    """Build a minimal Bedrock conversation dict."""
    if roles is None:
        roles = ["user" if i % 2 == 0 else "assistant"
                 for i in range(len(texts))]
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


def _mock_llm_response(text):
    """Build a mock return value for client.invoke_model() (Claude API)."""
    body_bytes = json.dumps({
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
    }).encode("utf-8")
    return {"body": io.BytesIO(body_bytes)}


def _mock_embed_response(embeddings):
    """Build a mock return value for client.invoke_model() (Cohere Embed API)."""
    body_bytes = json.dumps({
        "embeddings": embeddings,
        "id": "mock",
        "texts": [],
    }).encode("utf-8")
    return {"body": io.BytesIO(body_bytes)}


# Pre-defined unit vectors for controlled cosine similarity testing.
# 3-dimensional for simplicity.
VEC_A = [1.0, 0.0, 0.0]   # unit x
VEC_B = [0.0, 1.0, 0.0]   # unit y — orthogonal to A
VEC_C = [-1.0, 0.0, 0.0]  # opposite to A
# A vector close to A (small angle ≈ cos(0.15) ≈ 0.989)
VEC_A_NEAR = [0.98, 0.2, 0.0]  # not unit but that's fine


# ================================================================
# 1. extract_full_text
# ================================================================

class TestExtractFullText:
    def test_all_messages_concatenated(self):
        conv = _conv(["hello", "world", "foo", "bar"])
        text = extract_full_text(conv)
        assert text == "hello\nworld\nfoo\nbar"

    def test_system_ignored(self):
        conv = _conv(["hi", "there"], system_text="You are helpful.")
        text = extract_full_text(conv)
        assert "helpful" not in text
        assert "hi" in text
        assert "there" in text

    def test_empty_messages(self):
        conv = {"messages": []}
        assert extract_full_text(conv) == ""

    def test_custom_separator(self):
        conv = _conv(["a", "b"])
        assert extract_full_text(conv, separator=" | ") == "a | b"

    def test_roles_not_in_text(self):
        conv = _conv(["hello", "world"])
        text = extract_full_text(conv)
        assert "user" not in text
        assert "assistant" not in text


# ================================================================
# 2. extract_input_text
# ================================================================

class TestExtractInputText:
    def test_only_user_turns(self):
        conv = _conv(["q1", "a1", "q2", "a2"])
        text = extract_input_text(conv)
        assert text == "q1\nq2"
        assert "a1" not in text
        assert "a2" not in text

    def test_single_turn(self):
        conv = _conv(["question", "answer"])
        text = extract_input_text(conv)
        assert text == "question"

    def test_empty_messages(self):
        conv = {"messages": []}
        assert extract_input_text(conv) == ""

    def test_system_ignored(self):
        conv = _conv(["hi", "there"], system_text="system prompt")
        text = extract_input_text(conv)
        assert "system" not in text


# ================================================================
# 3. extract_assistant_texts
# ================================================================

class TestExtractAssistantTexts:
    def test_basic(self):
        conv = _conv(["q1", "a1", "q2", "a2"])
        texts = extract_assistant_texts(conv)
        assert texts == ["a1", "a2"]

    def test_single_turn(self):
        conv = _conv(["q", "a"])
        texts = extract_assistant_texts(conv)
        assert texts == ["a"]

    def test_no_assistant(self):
        conv = {"messages": [
            {"role": "user", "content": [{"text": "hi"}]},
        ]}
        assert extract_assistant_texts(conv) == []

    def test_empty_messages(self):
        conv = {"messages": []}
        assert extract_assistant_texts(conv) == []


# ================================================================
# 4. cosine_similarity
# ================================================================

class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity(VEC_A, VEC_B) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert cosine_similarity(VEC_A, VEC_C) == pytest.approx(-1.0)

    def test_near_vectors(self):
        sim = cosine_similarity(VEC_A, VEC_A_NEAR)
        assert sim > 0.95

    def test_zero_vector(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_dimension_mismatch(self):
        with pytest.raises(ValueError):
            cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])

    def test_single_dimension(self):
        assert cosine_similarity([3.0], [5.0]) == pytest.approx(1.0)
        assert cosine_similarity([3.0], [-5.0]) == pytest.approx(-1.0)


# ================================================================
# 5. UnionFind
# ================================================================

class TestUnionFind:
    def test_basic_union(self):
        uf = UnionFind([1, 2, 3])
        uf.union(1, 2)
        assert uf.find(1) == uf.find(2)
        assert uf.find(1) != uf.find(3)

    def test_transitivity(self):
        uf = UnionFind([1, 2, 3])
        uf.union(1, 2)
        uf.union(2, 3)
        assert uf.find(1) == uf.find(3)

    def test_groups(self):
        uf = UnionFind([0, 1, 2, 3])
        uf.union(0, 1)
        uf.union(2, 3)
        groups = uf.groups()
        assert len(groups) == 2
        sizes = sorted(len(v) for v in groups.values())
        assert sizes == [2, 2]

    def test_all_separate(self):
        uf = UnionFind([0, 1, 2])
        groups = uf.groups()
        assert len(groups) == 3


# ================================================================
# 6. embed_texts
# ================================================================

class TestEmbedTexts:
    def test_single_batch(self):
        """All texts fit in one batch."""
        mock_client = mock.MagicMock()
        mock_client.invoke_model.return_value = _mock_embed_response(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

        result = embed_texts(["a", "b", "c"], mock_client, "model-id",
                             batch_size=96)
        assert len(result) == 3
        assert result[0] == [0.1, 0.2]
        mock_client.invoke_model.assert_called_once()

    def test_multiple_batches(self):
        """Texts split across multiple batches."""
        mock_client = mock.MagicMock()
        # First batch: 2 texts
        resp1 = _mock_embed_response([[1.0, 0.0], [0.0, 1.0]])
        # Second batch: 1 text
        resp2 = _mock_embed_response([[0.5, 0.5]])
        mock_client.invoke_model.side_effect = [resp1, resp2]

        result = embed_texts(["a", "b", "c"], mock_client, "model-id",
                             batch_size=2)
        assert len(result) == 3
        assert result[0] == [1.0, 0.0]
        assert result[2] == [0.5, 0.5]
        assert mock_client.invoke_model.call_count == 2

    def test_empty_input(self):
        mock_client = mock.MagicMock()
        result = embed_texts([], mock_client, "model-id")
        assert result == []
        mock_client.invoke_model.assert_not_called()

    def test_passes_correct_body(self):
        """Verify the request body sent to Bedrock."""
        mock_client = mock.MagicMock()
        mock_client.invoke_model.return_value = _mock_embed_response(
            [[0.1, 0.2]])

        embed_texts(["hello world"], mock_client, "my-model",
                    input_type="search_query", batch_size=96)

        call_args = mock_client.invoke_model.call_args
        body = json.loads(call_args[1]["body"])
        assert body["texts"] == ["hello world"]
        assert body["input_type"] == "search_query"
        assert body["truncate"] == "END"
        assert call_args[1]["modelId"] == "my-model"


# ================================================================
# 7. cluster_by_cosine
# ================================================================

class TestClusterByCosine:
    def test_identical_embeddings_clustered(self):
        embs = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        uf = cluster_by_cosine(embs, [0, 1, 2], threshold=0.9)
        groups = uf.groups()
        # 0 and 1 should be in the same group; 2 is separate
        assert uf.find(0) == uf.find(1)
        assert uf.find(0) != uf.find(2)

    def test_all_different(self):
        embs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        uf = cluster_by_cosine(embs, [0, 1, 2], threshold=0.5)
        groups = uf.groups()
        assert len(groups) == 3

    def test_all_similar(self):
        embs = [[1.0, 0.1, 0.0], [1.0, 0.2, 0.0], [1.0, 0.15, 0.0]]
        uf = cluster_by_cosine(embs, [0, 1, 2], threshold=0.9)
        groups = uf.groups()
        # All should be in one group (very similar)
        assert len(groups) == 1

    def test_threshold_boundary(self):
        """Two vectors at exactly 90 degrees should not cluster at threshold 0.1."""
        embs = [[1.0, 0.0], [0.0, 1.0]]
        uf = cluster_by_cosine(embs, [0, 1], threshold=0.1)
        groups = uf.groups()
        assert len(groups) == 2

    def test_non_sequential_indices(self):
        embs = [[1.0, 0.0], [1.0, 0.0]]
        uf = cluster_by_cosine(embs, [5, 10], threshold=0.9)
        assert uf.find(5) == uf.find(10)


# ================================================================
# 8. compute_heuristic_score
# ================================================================

class TestComputeHeuristicScore:
    def test_more_turns_higher(self):
        short = _conv(["hi", "ok"])
        long = _conv(["hi", "ok", "more q", "more a"])
        assert compute_heuristic_score(long) > compute_heuristic_score(short)

    def test_longer_text_higher(self):
        short = _conv(["hi", "ok"])
        long = _conv(["hi there my friend",
                       "I am doing very well today thank you"])
        assert compute_heuristic_score(long) > compute_heuristic_score(short)

    def test_positive(self):
        conv = _conv(["hello", "world"])
        assert compute_heuristic_score(conv) > 0

    def test_custom_weights(self):
        conv = _conv(["a", "b"])
        s1 = compute_heuristic_score(conv, w_completeness=1.0,
                                     w_info_density=0.0)
        s2 = compute_heuristic_score(conv, w_completeness=0.0,
                                     w_info_density=1.0)
        assert s1 != s2

    def test_empty_messages(self):
        conv = {"messages": []}
        assert compute_heuristic_score(conv) == 0.0


# ================================================================
# 9. parse_json_response
# ================================================================

class TestParseJsonResponse:
    def test_plain_json(self):
        result = parse_json_response('{"average": 7.5, "scores": {}}')
        assert result["average"] == 7.5

    def test_markdown_wrapped(self):
        text = '```json\n{"average": 8.0}\n```'
        result = parse_json_response(text)
        assert result["average"] == 8.0

    def test_markdown_no_lang(self):
        text = '```\n{"average": 6.0}\n```'
        result = parse_json_response(text)
        assert result["average"] == 6.0

    def test_json_in_text(self):
        text = 'Here is the result: {"average": 5.0} hope this helps.'
        result = parse_json_response(text)
        assert result["average"] == 5.0

    def test_empty(self):
        assert parse_json_response("no json here") == {}

    def test_invalid_json(self):
        assert parse_json_response("{broken json") == {}


# ================================================================
# 10. format_conversation_for_judge
# ================================================================

class TestFormatConversationForJudge:
    def test_labels(self):
        conv = _conv(["question", "answer"])
        result = format_conversation_for_judge(conv)
        assert "用户: question" in result
        assert "助手: answer" in result

    def test_system_included(self):
        conv = _conv(["q", "a"], system_text="You are helpful.")
        result = format_conversation_for_judge(conv)
        assert "[系统] You are helpful." in result

    def test_multi_turn(self):
        conv = _conv(["q1", "a1", "q2", "a2"])
        result = format_conversation_for_judge(conv)
        lines = result.split("\n")
        # system + 4 messages = 5 lines
        assert len(lines) == 5
        assert lines[1].startswith("用户:")
        assert lines[2].startswith("助手:")

    def test_no_system(self):
        conv = {"messages": [
            {"role": "user", "content": [{"text": "hi"}]},
            {"role": "assistant", "content": [{"text": "hello"}]},
        ]}
        result = format_conversation_for_judge(conv)
        assert "系统" not in result
        assert "用户: hi" in result


# ================================================================
# 11. score_single_turn_output
# ================================================================

class TestScoreSingleTurnOutput:
    def test_valid_response(self):
        mock_client = mock.MagicMock()
        mock_client.invoke_model.return_value = _mock_llm_response(
            '{"scores": {"完整性": 8}, "average": 7.5}')
        conv = _conv(["question", "answer"])
        score = score_single_turn_output(conv, mock_client, "model-id")
        assert score == pytest.approx(7.5)

    def test_api_error_returns_negative(self):
        mock_client = mock.MagicMock()
        mock_client.invoke_model.side_effect = RuntimeError("API down")
        conv = _conv(["q", "a"])
        score = score_single_turn_output(conv, mock_client, "model-id")
        assert score == -1.0

    def test_invalid_json_returns_negative(self):
        mock_client = mock.MagicMock()
        mock_client.invoke_model.return_value = _mock_llm_response(
            "I cannot parse this")
        conv = _conv(["q", "a"])
        score = score_single_turn_output(conv, mock_client, "model-id")
        assert score == -1.0

    def test_missing_average_returns_negative(self):
        mock_client = mock.MagicMock()
        mock_client.invoke_model.return_value = _mock_llm_response(
            '{"scores": {"完整性": 8}}')
        conv = _conv(["q", "a"])
        score = score_single_turn_output(conv, mock_client, "model-id")
        assert score == -1.0


# ================================================================
# 12. score_multi_turn_output
# ================================================================

class TestScoreMultiTurnOutput:
    def test_equal_weighting(self):
        """Two assistant turns with scores 6.0 and 8.0 -> average 7.0."""
        mock_client = mock.MagicMock()
        mock_client.invoke_model.side_effect = [
            _mock_llm_response('{"scores": {}, "average": 6.0}'),
            _mock_llm_response('{"scores": {}, "average": 8.0}'),
        ]
        conv = _conv(["q1", "a1", "q2", "a2"])
        score = score_multi_turn_output(conv, mock_client, "model-id",
                                        weight_mode="equal")
        assert score == pytest.approx(7.0)

    def test_linear_increasing_weighting(self):
        """Two turns with scores 6.0 and 8.0, weights 1 and 2.
        Weighted avg = (1*6 + 2*8) / (1+2) = 22/3 ≈ 7.333"""
        mock_client = mock.MagicMock()
        mock_client.invoke_model.side_effect = [
            _mock_llm_response('{"scores": {}, "average": 6.0}'),
            _mock_llm_response('{"scores": {}, "average": 8.0}'),
        ]
        conv = _conv(["q1", "a1", "q2", "a2"])
        score = score_multi_turn_output(conv, mock_client, "model-id",
                                        weight_mode="linear_increasing")
        assert score == pytest.approx(22.0 / 3.0)

    def test_partial_failure(self):
        """First turn fails, second succeeds -> uses only second turn."""
        mock_client = mock.MagicMock()
        mock_client.invoke_model.side_effect = [
            _mock_llm_response("invalid response"),
            _mock_llm_response('{"scores": {}, "average": 8.0}'),
        ]
        conv = _conv(["q1", "a1", "q2", "a2"])
        score = score_multi_turn_output(conv, mock_client, "model-id",
                                        weight_mode="equal")
        assert score == pytest.approx(8.0)

    def test_all_failures(self):
        mock_client = mock.MagicMock()
        mock_client.invoke_model.side_effect = RuntimeError("API down")
        conv = _conv(["q1", "a1", "q2", "a2"])
        score = score_multi_turn_output(conv, mock_client, "model-id")
        assert score == -1.0

    def test_no_assistant_turns(self):
        conv = {"messages": [
            {"role": "user", "content": [{"text": "hi"}]},
        ]}
        mock_client = mock.MagicMock()
        score = score_multi_turn_output(conv, mock_client, "model-id")
        assert score == -1.0

    def test_three_turns_linear_increasing(self):
        """Three turns with scores 4, 6, 9; weights 1, 2, 3.
        Weighted avg = (1*4 + 2*6 + 3*9) / (1+2+3) = 43/6 ≈ 7.167"""
        mock_client = mock.MagicMock()
        mock_client.invoke_model.side_effect = [
            _mock_llm_response('{"average": 4.0}'),
            _mock_llm_response('{"average": 6.0}'),
            _mock_llm_response('{"average": 9.0}'),
        ]
        conv = _conv(["q1", "a1", "q2", "a2", "q3", "a3"])
        score = score_multi_turn_output(conv, mock_client, "model-id",
                                        weight_mode="linear_increasing")
        assert score == pytest.approx(43.0 / 6.0)


# ================================================================
# 13. score_output_quality — dispatch
# ================================================================

class TestScoreOutputQuality:
    def test_dispatches_to_single_turn(self):
        """2 messages -> single-turn scoring."""
        mock_client = mock.MagicMock()
        mock_client.invoke_model.return_value = _mock_llm_response(
            '{"average": 7.0}')
        conv = _conv(["q", "a"])
        score = score_output_quality(conv, mock_client, "model-id")
        assert score == pytest.approx(7.0)
        # Only 1 API call (single turn)
        assert mock_client.invoke_model.call_count == 1

    def test_dispatches_to_multi_turn(self):
        """4 messages -> multi-turn scoring."""
        mock_client = mock.MagicMock()
        mock_client.invoke_model.side_effect = [
            _mock_llm_response('{"average": 6.0}'),
            _mock_llm_response('{"average": 8.0}'),
        ]
        conv = _conv(["q1", "a1", "q2", "a2"])
        score = score_output_quality(conv, mock_client, "model-id",
                                     weight_mode="equal")
        assert score == pytest.approx(7.0)
        # 2 API calls (one per assistant turn)
        assert mock_client.invoke_model.call_count == 2


# ================================================================
# 14. select_best_per_group
# ================================================================

class TestSelectBestPerGroup:
    def test_singleton_groups(self):
        """Singleton groups are all kept."""
        groups = {0: [0], 1: [1], 2: [2]}
        convs = [_conv(["a", "b"]), _conv(["c", "d"]), _conv(["e", "f"])]
        cfg = AdvancedDedupConfig(scoring_method="heuristic")
        keepers, _, _ = select_best_per_group(groups, convs, cfg,
                                               stage_name="test")
        assert keepers == {0, 1, 2}

    def test_heuristic_picks_longer(self):
        """Among heuristic-scored group, longer conversation wins."""
        short = _conv(["hi", "ok"])
        long = _conv(["hi there friend",
                       "I am doing very well today thank you for asking me"])
        groups = {0: [0, 1]}
        cfg = AdvancedDedupConfig(scoring_method="heuristic")
        keepers, _, _ = select_best_per_group(groups, [short, long], cfg,
                                               stage_name="full_sample_dedup")
        assert keepers == {1}

    def test_llm_scoring_for_input_dedup(self):
        """LLM scoring is only used when stage_name == 'input_dedup'."""
        mock_client = mock.MagicMock()
        # LLM judge picks first conv (returns higher score for it)
        mock_client.invoke_model.side_effect = [
            _mock_llm_response('{"average": 9.0}'),
            _mock_llm_response('{"average": 5.0}'),
        ]
        short = _conv(["q", "short answer"])
        long = _conv(["q",
                       "very long and detailed answer with lots of content"])
        groups = {0: [0, 1]}
        cfg = AdvancedDedupConfig(scoring_method="llm")
        keepers, _, _ = select_best_per_group(groups, [short, long], cfg,
                                               bedrock_client=mock_client,
                                               stage_name="input_dedup")
        # LLM said index 0 is better (score 9.0 > 5.0)
        assert keepers == {0}

    def test_llm_used_for_full_sample_stage(self):
        """With scoring_method='llm', full_sample_dedup also uses LLM judge."""
        mock_client = mock.MagicMock()
        # LLM judge says the short answer (index 0) is better
        mock_client.invoke_model.side_effect = [
            _mock_llm_response('{"average": 9.0}'),
            _mock_llm_response('{"average": 3.0}'),
        ]
        short = _conv(["q", "concise but perfect answer"])
        long = _conv(["q", "very long and detailed answer content here"])
        groups = {0: [0, 1]}
        cfg = AdvancedDedupConfig(scoring_method="llm")
        keepers, _, _ = select_best_per_group(groups, [short, long], cfg,
                                               bedrock_client=mock_client,
                                               stage_name="full_sample_dedup")
        # LLM picked index 0 (score 9.0 > 3.0), overriding heuristic
        assert keepers == {0}
        # LLM was called
        assert mock_client.invoke_model.call_count >= 1


# ================================================================
# Mock embed client factory for end-to-end tests
# ================================================================

def _build_mock_embed_client(text_to_vec: dict,
                             default_vec: List[float] = None):
    """Build a mock Bedrock client that returns pre-defined embeddings.

    text_to_vec: {text_string: [embedding_vector]}
    For unknown texts, returns default_vec or a random-ish vector.
    """
    if default_vec is None:
        default_vec = [0.0, 0.0, 0.0]

    def side_effect(*, modelId, contentType, accept, body):
        req = json.loads(body)
        texts = req["texts"]
        embeddings = []
        for t in texts:
            if t in text_to_vec:
                embeddings.append(text_to_vec[t])
            else:
                # Generate a deterministic but unique vector from text hash
                h = hash(t) % 10000
                embeddings.append([h / 10000.0, (h % 100) / 100.0,
                                   (h % 10) / 10.0])
        body_bytes = json.dumps({
            "embeddings": embeddings,
        }).encode("utf-8")
        return {"body": io.BytesIO(body_bytes)}

    client = mock.MagicMock()
    client.invoke_model.side_effect = side_effect
    return client


def _build_mock_combined_client(text_to_vec: dict, llm_responses: list):
    """Build a mock client that dispatches embed vs. LLM calls.

    Embed calls have 'texts' in the body; LLM calls have 'messages'.
    """
    llm_call_idx = [0]

    def side_effect(*, modelId, contentType, accept, body):
        req = json.loads(body)
        if "texts" in req:
            # Embedding call
            texts = req["texts"]
            embeddings = []
            for t in texts:
                if t in text_to_vec:
                    embeddings.append(text_to_vec[t])
                else:
                    h = hash(t) % 10000
                    embeddings.append([h / 10000.0, (h % 100) / 100.0,
                                       (h % 10) / 10.0])
            body_bytes = json.dumps({
                "embeddings": embeddings,
            }).encode("utf-8")
            return {"body": io.BytesIO(body_bytes)}
        else:
            # LLM call
            idx = llm_call_idx[0]
            llm_call_idx[0] += 1
            if idx < len(llm_responses):
                resp_text = llm_responses[idx]
            else:
                resp_text = '{"average": 5.0}'
            body_bytes = json.dumps({
                "content": [{"type": "text", "text": resp_text}],
                "stop_reason": "end_turn",
            }).encode("utf-8")
            return {"body": io.BytesIO(body_bytes)}

    client = mock.MagicMock()
    client.invoke_model.side_effect = side_effect
    return client


# ================================================================
# 15. End-to-end: full_sample_dedup only
# ================================================================

class TestEndToEndFullSampleDedup:
    def test_identical_full_texts_deduped(self):
        """Two convs with identical content -> one removed."""
        c1 = _conv(["hello", "world"], system_text="A")
        c2 = _conv(["hello", "world"], system_text="B")
        c3 = _conv(["different question", "different answer"])

        # Identical full texts get identical embeddings
        text_to_vec = {
            "hello\nworld": [1.0, 0.0, 0.0],
            "different question\ndifferent answer": [0.0, 1.0, 0.0],
        }
        mock_client = _build_mock_embed_client(text_to_vec)

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, "in.jsonl")
            out = os.path.join(tmpdir, "out.jsonl")
            _write_jsonl(inp, [c1, c2, c3])

            cfg = AdvancedDedupConfig(
                input=inp, output=out,
                enable_full_sample_dedup=True,
                enable_input_dedup=False,
                scoring_method="heuristic",
            )
            stats = run(cfg, embed_client=mock_client)

            result = _read_jsonl(out)
            assert len(result) == 2
            assert stats["stages"][0]["removed"] == 1
            assert stats["stages"][0]["dup_groups"] == 1

    def test_near_duplicate_full_texts(self):
        """Two convs with similar embeddings (cos > 0.85) -> deduped."""
        c1 = _conv(["similar question A", "similar answer A"])
        c2 = _conv(["similar question B", "similar answer B"])
        c3 = _conv(["totally different", "totally different answer"])

        # Near-identical embeddings for c1 and c2
        text_to_vec = {
            "similar question A\nsimilar answer A": [1.0, 0.05, 0.0],
            "similar question B\nsimilar answer B": [1.0, 0.06, 0.0],
            "totally different\ntotally different answer": [0.0, 0.0, 1.0],
        }
        mock_client = _build_mock_embed_client(text_to_vec)

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, "in.jsonl")
            out = os.path.join(tmpdir, "out.jsonl")
            _write_jsonl(inp, [c1, c2, c3])

            cfg = AdvancedDedupConfig(
                input=inp, output=out,
                enable_full_sample_dedup=True,
                enable_input_dedup=False,
                full_sample_threshold=0.85,
                scoring_method="heuristic",
            )
            stats = run(cfg, embed_client=mock_client)

            result = _read_jsonl(out)
            assert len(result) == 2
            assert stats["stages"][0]["dup_groups"] == 1


class TestEndToEndFullSampleDedupWithLLM:
    def test_llm_judge_picks_better_in_stage1(self):
        """Stage 1 with LLM scoring: LLM overrides heuristic choice."""
        # c1 is short but high quality; c2 is long but low quality
        c1 = _conv(["what is Python?", "A versatile programming language."],
                    system_text="A")
        c2 = _conv(["what is Python?", "A versatile programming language."],
                    system_text="B")
        c3 = _conv(["different topic", "different answer"])

        text_to_vec = {
            "what is Python?\nA versatile programming language.":
                [1.0, 0.0, 0.0],
            "different topic\ndifferent answer": [0.0, 1.0, 0.0],
        }
        # LLM scores: c1 gets 9.0, c2 gets 4.0 -> c1 wins
        llm_responses = [
            '{"average": 9.0}',
            '{"average": 4.0}',
        ]
        mock_client = _build_mock_combined_client(text_to_vec, llm_responses)

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, "in.jsonl")
            out = os.path.join(tmpdir, "out.jsonl")
            _write_jsonl(inp, [c1, c2, c3])

            cfg = AdvancedDedupConfig(
                input=inp, output=out,
                enable_full_sample_dedup=True,
                enable_input_dedup=False,
                scoring_method="llm",
            )
            stats = run(cfg, bedrock_client=mock_client,
                        embed_client=mock_client)

            result = _read_jsonl(out)
            assert len(result) == 2
            assert stats["stages"][0]["removed"] == 1
            assert stats["stages"][0]["dup_groups"] == 1


# ================================================================
# 16. End-to-end: input_dedup only
# ================================================================

class TestEndToEndInputDedup:
    def test_similar_inputs_different_outputs(self):
        """Two convs with same question but different quality answers."""
        c1 = _conv(["what is Python?", "It's a language."])
        c2 = _conv(["what is Python?",
                     "Python is a high-level programming language known for "
                     "its readability and versatility."])
        c3 = _conv(["what is Java?", "Java is a language."])

        text_to_vec = {
            # Stage 2: input texts (user turns only)
            "what is Python?": [1.0, 0.0, 0.0],
            "what is Java?": [0.0, 1.0, 0.0],
        }
        # LLM judge scores: c1 gets 5.0, c2 gets 9.0
        llm_responses = [
            '{"average": 5.0}',  # c1 output score
            '{"average": 9.0}',  # c2 output score
        ]
        mock_client = _build_mock_combined_client(text_to_vec, llm_responses)

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, "in.jsonl")
            out = os.path.join(tmpdir, "out.jsonl")
            _write_jsonl(inp, [c1, c2, c3])

            cfg = AdvancedDedupConfig(
                input=inp, output=out,
                enable_full_sample_dedup=False,
                enable_input_dedup=True,
                input_threshold=0.7,
                scoring_method="llm",
            )
            stats = run(cfg, bedrock_client=mock_client,
                        embed_client=mock_client)

            result = _read_jsonl(out)
            assert len(result) == 2
            # c2 (better answer) should survive, c1 should be removed
            assert stats["stages"][0]["dup_groups"] == 1
            assert stats["stages"][0]["removed"] == 1
            # Verify the better answer survived
            texts = [extract_full_text(r) for r in result]
            assert any("versatility" in t for t in texts)

    def test_no_duplicates_all_kept(self):
        """All different inputs -> nothing removed."""
        c1 = _conv(["question A", "answer A"])
        c2 = _conv(["question B", "answer B"])
        c3 = _conv(["question C", "answer C"])

        # All inputs get very different embeddings
        text_to_vec = {
            "question A": [1.0, 0.0, 0.0],
            "question B": [0.0, 1.0, 0.0],
            "question C": [0.0, 0.0, 1.0],
        }
        mock_client = _build_mock_embed_client(text_to_vec)

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, "in.jsonl")
            out = os.path.join(tmpdir, "out.jsonl")
            _write_jsonl(inp, [c1, c2, c3])

            cfg = AdvancedDedupConfig(
                input=inp, output=out,
                enable_full_sample_dedup=False,
                enable_input_dedup=True,
                input_threshold=0.7,
                scoring_method="heuristic",
            )
            stats = run(cfg, embed_client=mock_client)

            result = _read_jsonl(out)
            assert len(result) == 3
            assert stats["stages"][0]["removed"] == 0


# ================================================================
# 17. End-to-end: both stages
# ================================================================

class TestEndToEndBothStages:
    def test_two_stage_pipeline(self):
        """Stage 1 removes exact-content dup, stage 2 removes similar-input dup."""
        # c1 and c2 have identical full content -> stage 1 dedup
        c1 = _conv(["hello", "world"], system_text="A")
        c2 = _conv(["hello", "world"], system_text="B")
        # c3 and c4 have similar inputs but different outputs -> stage 2 dedup
        c3 = _conv(["what is AI?", "Short answer."])
        c4 = _conv(["what is AI?",
                     "Artificial Intelligence is a broad field encompassing "
                     "machine learning, natural language processing, and more."])
        # c5 is unique
        c5 = _conv(["how to cook?", "Use a recipe."])

        # Stage 1: full text embeddings
        full_text_vecs = {
            "hello\nworld": [1.0, 0.0, 0.0],
            "what is AI?\nShort answer.": [0.0, 0.5, 0.5],
            "what is AI?\nArtificial Intelligence is a broad field "
            "encompassing machine learning, natural language processing, "
            "and more.": [0.0, 0.6, 0.5],
            "how to cook?\nUse a recipe.": [0.0, 0.0, 1.0],
        }
        # Stage 2: input text embeddings
        input_text_vecs = {
            "what is AI?": [0.0, 1.0, 0.0],
            "how to cook?": [0.0, 0.0, 1.0],
        }
        # Merge: we need to handle both stages with one mock
        text_to_vec = {**full_text_vecs, **input_text_vecs}
        # For the surviving "hello\nworld" conv, its input is "hello"
        text_to_vec["hello"] = [1.0, 0.0, 0.0]

        llm_responses = [
            # Stage 1: LLM scores c1 and c2 (identical content dup group)
            '{"average": 7.0}',  # c1 output score
            '{"average": 6.0}',  # c2 output score (c1 wins)
            # Stage 2: LLM scores c3 and c4 (similar input dup group)
            '{"average": 4.0}',  # c3 output score
            '{"average": 9.0}',  # c4 output score (c4 wins)
        ]
        mock_client = _build_mock_combined_client(text_to_vec, llm_responses)

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, "in.jsonl")
            out = os.path.join(tmpdir, "out.jsonl")
            _write_jsonl(inp, [c1, c2, c3, c4, c5])

            cfg = AdvancedDedupConfig(
                input=inp, output=out,
                enable_full_sample_dedup=True,
                enable_input_dedup=True,
                full_sample_threshold=0.85,
                input_threshold=0.7,
                scoring_method="llm",
            )
            stats = run(cfg, bedrock_client=mock_client,
                        embed_client=mock_client)

            result = _read_jsonl(out)
            # Stage 1: c1,c2 deduped to 1 (LLM picks c1) -> 4 survive
            # Stage 2: c3,c4 deduped to 1 (LLM picks c4) -> 3 survive
            assert len(result) == 3
            assert stats["final"] == 3


# ================================================================
# 18. End-to-end: heuristic scoring
# ================================================================

class TestEndToEndHeuristicScoring:
    def test_heuristic_picks_best(self):
        """With heuristic scoring, longer/richer answer is kept."""
        c1 = _conv(["question", "ok"])
        c2 = _conv(["question",
                     "This is a very detailed and helpful answer that "
                     "covers many aspects of the topic thoroughly."])

        text_to_vec = {
            "question": [1.0, 0.0, 0.0],
        }
        mock_client = _build_mock_embed_client(text_to_vec)

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, "in.jsonl")
            out = os.path.join(tmpdir, "out.jsonl")
            _write_jsonl(inp, [c1, c2])

            cfg = AdvancedDedupConfig(
                input=inp, output=out,
                enable_full_sample_dedup=False,
                enable_input_dedup=True,
                input_threshold=0.7,
                scoring_method="heuristic",
            )
            stats = run(cfg, embed_client=mock_client)

            result = _read_jsonl(out)
            assert len(result) == 1
            # The longer answer should survive
            text = extract_full_text(result[0])
            assert "detailed" in text


# ================================================================
# 19. End-to-end: all unique
# ================================================================

class TestEndToEndAllUnique:
    def test_no_removal(self):
        """All conversations are unique -> nothing removed."""
        c1 = _conv(["q1", "a1"])
        c2 = _conv(["q2", "a2"])
        c3 = _conv(["q3", "a3"])

        text_to_vec = {
            "q1\na1": [1.0, 0.0, 0.0],
            "q2\na2": [0.0, 1.0, 0.0],
            "q3\na3": [0.0, 0.0, 1.0],
            "q1": [1.0, 0.0, 0.0],
            "q2": [0.0, 1.0, 0.0],
            "q3": [0.0, 0.0, 1.0],
        }
        mock_client = _build_mock_embed_client(text_to_vec)

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, "in.jsonl")
            out = os.path.join(tmpdir, "out.jsonl")
            _write_jsonl(inp, [c1, c2, c3])

            cfg = AdvancedDedupConfig(
                input=inp, output=out,
                enable_full_sample_dedup=True,
                enable_input_dedup=True,
                scoring_method="heuristic",
            )
            stats = run(cfg, embed_client=mock_client)

            result = _read_jsonl(out)
            assert len(result) == 3
            assert stats["final"] == 3
            for stage in stats["stages"]:
                assert stage["removed"] == 0


# ================================================================
# 20. End-to-end: single conversation
# ================================================================

class TestEndToEndSingleConversation:
    def test_single_conv_passes_through(self):
        """A single conversation should always survive."""
        c1 = _conv(["hello", "world"])

        text_to_vec = {
            "hello\nworld": [1.0, 0.0, 0.0],
            "hello": [1.0, 0.0, 0.0],
        }
        mock_client = _build_mock_embed_client(text_to_vec)

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, "in.jsonl")
            out = os.path.join(tmpdir, "out.jsonl")
            _write_jsonl(inp, [c1])

            cfg = AdvancedDedupConfig(
                input=inp, output=out,
                enable_full_sample_dedup=True,
                enable_input_dedup=True,
                scoring_method="heuristic",
            )
            stats = run(cfg, embed_client=mock_client)

            result = _read_jsonl(out)
            assert len(result) == 1
            assert stats["final"] == 1


# ================================================================
# 21. End-to-end: multi-turn conversations with LLM judge
# ================================================================

class TestEndToEndMultiTurn:
    def test_multi_turn_input_dedup(self):
        """Multi-turn convs with same user turns -> LLM picks best output."""
        c1 = _conv(["q1", "short a1", "q2", "short a2"])
        c2 = _conv(["q1", "very detailed and helpful first answer",
                     "q2", "very detailed and helpful second answer"])

        text_to_vec = {
            "q1\nq2": [1.0, 0.0, 0.0],
        }
        # LLM scores each assistant turn for both conversations:
        # c1 turn1=4.0, c1 turn2=4.0, c2 turn1=8.0, c2 turn2=9.0
        llm_responses = [
            '{"average": 4.0}',
            '{"average": 4.0}',
            '{"average": 8.0}',
            '{"average": 9.0}',
        ]
        mock_client = _build_mock_combined_client(text_to_vec, llm_responses)

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, "in.jsonl")
            out = os.path.join(tmpdir, "out.jsonl")
            _write_jsonl(inp, [c1, c2])

            cfg = AdvancedDedupConfig(
                input=inp, output=out,
                enable_full_sample_dedup=False,
                enable_input_dedup=True,
                input_threshold=0.7,
                scoring_method="llm",
                multi_turn_weight_mode="equal",
            )
            stats = run(cfg, bedrock_client=mock_client,
                        embed_client=mock_client)

            result = _read_jsonl(out)
            assert len(result) == 1
            # c2 should win (higher scores)
            text = extract_full_text(result[0])
            assert "detailed" in text


# ================================================================
# 22. End-to-end: Chinese/bilingual data
# ================================================================

class TestEndToEndChinese:
    def test_chinese_conversations(self):
        """Chinese conversations are handled correctly."""
        c1 = _conv(["什么是机器学习？", "机器学习是人工智能的一个分支。"])
        c2 = _conv(["什么是机器学习？",
                     "机器学习是人工智能的重要分支，它让计算机能够从数据中学习，"
                     "无需明确编程即可改善其性能。常见的方法包括监督学习、"
                     "无监督学习和强化学习。"])
        c3 = _conv(["如何学习编程？", "多练习。"])

        text_to_vec = {
            "什么是机器学习？": [1.0, 0.0, 0.0],
            "如何学习编程？": [0.0, 1.0, 0.0],
        }
        llm_responses = [
            '{"average": 5.0}',
            '{"average": 9.0}',
        ]
        mock_client = _build_mock_combined_client(text_to_vec, llm_responses)

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, "in.jsonl")
            out = os.path.join(tmpdir, "out.jsonl")
            _write_jsonl(inp, [c1, c2, c3])

            cfg = AdvancedDedupConfig(
                input=inp, output=out,
                enable_full_sample_dedup=False,
                enable_input_dedup=True,
                input_threshold=0.7,
                scoring_method="llm",
            )
            stats = run(cfg, bedrock_client=mock_client,
                        embed_client=mock_client)

            result = _read_jsonl(out)
            assert len(result) == 2
            texts = [extract_full_text(r) for r in result]
            # The detailed ML answer should survive
            assert any("监督学习" in t for t in texts)
            # The programming answer should also survive
            assert any("多练习" in t for t in texts)


# ================================================================
# 23. CLI parsing
# ================================================================

class TestCLIParsing:
    def _parse(self, argv):
        parser = build_parser()
        args = parser.parse_args(argv)
        return args_to_config(args)

    def test_defaults(self):
        cfg = self._parse([])
        assert cfg.enable_full_sample_dedup is True
        assert cfg.enable_input_dedup is True
        assert cfg.embed_model_id == "cohere.embed-multilingual-v3"
        assert cfg.embed_batch_size == 96
        assert cfg.full_sample_threshold == 0.85
        assert cfg.input_threshold == 0.7
        assert cfg.scoring_method == "llm"
        assert cfg.judge_model_id == "us.anthropic.claude-opus-4-6-v1"
        assert cfg.bedrock_region == "us-east-1"
        assert cfg.multi_turn_weight_mode == "linear_increasing"

    def test_io(self):
        cfg = self._parse(["-i", "in.jsonl", "-o", "out.jsonl"])
        assert cfg.input == "in.jsonl"
        assert cfg.output == "out.jsonl"

    def test_disable_stages(self):
        cfg = self._parse(["--no-enable-full-sample-dedup",
                            "--no-enable-input-dedup"])
        assert cfg.enable_full_sample_dedup is False
        assert cfg.enable_input_dedup is False

    def test_thresholds(self):
        cfg = self._parse(["--full-sample-threshold", "0.9",
                            "--input-threshold", "0.6"])
        assert cfg.full_sample_threshold == 0.9
        assert cfg.input_threshold == 0.6

    def test_scoring_heuristic(self):
        cfg = self._parse(["--scoring-method", "heuristic"])
        assert cfg.scoring_method == "heuristic"

    def test_custom_models(self):
        cfg = self._parse([
            "--embed-model-id", "my-embed-model",
            "--judge-model-id", "my-judge-model",
            "--bedrock-region", "us-west-2",
        ])
        assert cfg.embed_model_id == "my-embed-model"
        assert cfg.judge_model_id == "my-judge-model"
        assert cfg.bedrock_region == "us-west-2"

    def test_weight_mode(self):
        cfg = self._parse(["--multi-turn-weight-mode", "equal"])
        assert cfg.multi_turn_weight_mode == "equal"

    def test_embed_batch_size(self):
        cfg = self._parse(["--embed-batch-size", "32"])
        assert cfg.embed_batch_size == 32

    def test_embed_input_type(self):
        cfg = self._parse(["--embed-input-type", "search_query"])
        assert cfg.embed_input_type == "search_query"

    def test_heuristic_weights(self):
        cfg = self._parse(["--weight-completeness", "0.3",
                            "--weight-info-density", "0.7"])
        assert cfg.weight_completeness == 0.3
        assert cfg.weight_info_density == 0.7


# ================================================================
# 24. Edge cases
# ================================================================

class TestEdgeCases:
    def test_both_stages_disabled(self):
        """Both stages disabled -> output = input."""
        c1 = _conv(["q", "a"])
        c2 = _conv(["q", "a"])

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, "in.jsonl")
            out = os.path.join(tmpdir, "out.jsonl")
            _write_jsonl(inp, [c1, c2])

            cfg = AdvancedDedupConfig(
                input=inp, output=out,
                enable_full_sample_dedup=False,
                enable_input_dedup=False,
                scoring_method="heuristic",
            )
            # No clients needed since both stages are disabled
            stats = run(cfg, bedrock_client=mock.MagicMock(),
                        embed_client=mock.MagicMock())

            result = _read_jsonl(out)
            assert len(result) == 2
            assert stats["final"] == 2

    def test_empty_input_file(self):
        """Empty input file -> empty output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, "in.jsonl")
            out = os.path.join(tmpdir, "out.jsonl")
            with open(inp, "w") as f:
                f.write("")

            cfg = AdvancedDedupConfig(
                input=inp, output=out,
                enable_full_sample_dedup=True,
                enable_input_dedup=True,
                scoring_method="heuristic",
            )
            stats = run(cfg, bedrock_client=mock.MagicMock(),
                        embed_client=mock.MagicMock())

            result = _read_jsonl(out)
            assert len(result) == 0
            assert stats["final"] == 0

    def test_get_msg_text_empty_content(self):
        assert _get_msg_text({"content": []}) == ""
        assert _get_msg_text({}) == ""
        assert _get_msg_text({"content": "not a list"}) == ""

    def test_format_conv_full(self):
        conv = _conv(["hello", "world"], system_text="test system")
        result = _format_conv_full(conv)
        assert "系统" in result
        assert "用户" in result
        assert "助手" in result
        assert "turns" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
