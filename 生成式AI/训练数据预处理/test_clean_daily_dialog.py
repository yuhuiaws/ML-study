#!/usr/bin/env python3
"""
Tests for clean_daily_dialog.py

Coverage:
  1. Individual normalization functions (fullwidth, whitespace, NFD, T2S)
  2. Individual filter functions
  3. CleanConfig defaults & CLI argument parsing
  4. process_conversation with config-driven toggles
  5. PPL infrastructure (mocked — no torch required)
  6. End-to-end validation on generated cleaned file
"""

import json
import os
import re
import unicodedata
from unittest import mock

import pytest

from clean_daily_dialog import (
    CleanConfig,
    DEFAULT_AD_KEYWORDS,
    DEFAULT_LOGIN_PREFIXES,
    build_parser,
    args_to_config,
    fullwidth_to_halfwidth,
    normalize_whitespace,
    remove_accents_nfd,
    normalize_text,
    finalize_text,
    traditional_to_simplified,
    filter_language,
    filter_punctuation_ratio,
    filter_message_length,
    filter_not_mostly_uppercase,
    filter_not_purely_numeric,
    filter_no_urls,
    filter_no_ad_keywords,
    filter_not_short_login,
    process_conversation,
    HAS_OPENCC,
)


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def _cfg(**overrides) -> CleanConfig:
    """Build a CleanConfig with all filters disabled except those overridden."""
    defaults = dict(
        filter_language=False,
        filter_punctuation=False,
        filter_length=False,
        filter_uppercase=False,
        filter_numeric=False,
        filter_urls=False,
        filter_keywords=False,
        filter_short_login=False,
        filter_ppl=False,
    )
    defaults.update(overrides)
    return CleanConfig(**defaults)


def _conv(texts, roles=None):
    """Build a minimal Bedrock conversation dict."""
    if roles is None:
        roles = ["user" if i % 2 == 0 else "assistant" for i in range(len(texts))]
    return {
        "schemaVersion": "bedrock-conversation-2024",
        "system": [{"text": "sys"}],
        "messages": [
            {"role": r, "content": [{"text": t}]}
            for r, t in zip(roles, texts)
        ],
    }


# ================================================================
# 1. Normalization unit tests
# ================================================================

class TestFullwidthToHalfwidth:
    def test_letters(self):
        assert fullwidth_to_halfwidth("Ａｂ") == "Ab"

    def test_digits(self):
        assert fullwidth_to_halfwidth("０１２") == "012"

    def test_punctuation(self):
        assert fullwidth_to_halfwidth("！？（）") == "!?()"

    def test_ideographic_space(self):
        assert fullwidth_to_halfwidth("a\u3000b") == "a b"

    def test_no_change(self):
        assert fullwidth_to_halfwidth("Hello 123!") == "Hello 123!"

    def test_chinese_preserved(self):
        assert fullwidth_to_halfwidth("你好Ａ") == "你好A"

    def test_empty(self):
        assert fullwidth_to_halfwidth("") == ""


class TestNormalizeWhitespace:
    def test_crlf(self):
        assert normalize_whitespace("a\r\nb") == "a\nb"

    def test_tab(self):
        assert normalize_whitespace("a\tb") == "a b"

    def test_multi_spaces(self):
        assert normalize_whitespace("a   b") == "a b"

    def test_strip_lines(self):
        assert normalize_whitespace("  a  \n  b  ") == "a\nb"

    def test_preserve_double_newline(self):
        assert normalize_whitespace("a\n\nb") == "a\n\nb"

    def test_complex(self):
        assert normalize_whitespace("  a \t b  \r\n  c  ") == "a b\nc"


class TestRemoveAccentsNFD:
    def test_e_acute(self):
        assert remove_accents_nfd("é") == "e"

    def test_u_diaeresis(self):
        assert remove_accents_nfd("ü") == "u"

    def test_mixed_word(self):
        assert remove_accents_nfd("café résumé") == "cafe resume"

    def test_chinese(self):
        assert remove_accents_nfd("你好") == "你好"


@pytest.mark.skipif(not HAS_OPENCC, reason="opencc not installed")
class TestTraditionalToSimplified:
    def test_basic(self):
        assert traditional_to_simplified("國際") == "国际"

    def test_mixed(self):
        result = traditional_to_simplified("計算機科學")
        assert result == "计算机科学"

    def test_already_simplified(self):
        assert traditional_to_simplified("你好世界") == "你好世界"

    def test_english_unaffected(self):
        assert traditional_to_simplified("Hello World") == "Hello World"


class TestNormalizeText:
    def test_combined(self):
        text = "Ｈｅｌｌｏ\u3000ｗｏｒｌｄ！\r\n  café  "
        cfg = _cfg()
        assert normalize_text(text, cfg) == "Hello world!\ncafe"

    def test_all_disabled(self):
        text = "Ａ\r\n  é  "
        cfg = _cfg(fullwidth_to_halfwidth=False, normalize_whitespace=False,
                   nfd_normalize=False)
        assert normalize_text(text, cfg) == text

    @pytest.mark.skipif(not HAS_OPENCC, reason="opencc not installed")
    def test_with_t2s(self):
        cfg = _cfg(traditional_to_simplified=True)
        assert normalize_text("國際", cfg) == "国际"


class TestFinalizeText:
    def test_lowercase(self):
        cfg = _cfg(to_lowercase=True)
        assert finalize_text("Hello World", cfg) == "hello world"

    def test_remove_punct(self):
        cfg = _cfg(remove_punctuation=True)
        assert finalize_text("Hello, world!", cfg) == "Hello world"

    def test_both(self):
        cfg = _cfg(to_lowercase=True, remove_punctuation=True)
        assert finalize_text("Hello, World!", cfg) == "hello world"

    def test_neither(self):
        cfg = _cfg()
        assert finalize_text("Hello, World!", cfg) == "Hello, World!"


# ================================================================
# 2. Filter unit tests
# ================================================================

class TestFilterPunctuationRatio:
    def test_normal(self):
        assert filter_punctuation_ratio("Hello, how are you?") is True

    def test_high(self):
        assert filter_punctuation_ratio("!!!???...", max_ratio=0.3) is False

    def test_empty(self):
        assert filter_punctuation_ratio("") is True


class TestFilterMessageLength:
    def test_normal(self):
        assert filter_message_length("Hello!") is True

    def test_too_short(self):
        assert filter_message_length("H", min_len=2) is False

    def test_too_long(self):
        assert filter_message_length("x" * 2001, max_len=2000) is False


class TestFilterNotMostlyUppercase:
    def test_normal(self):
        assert filter_not_mostly_uppercase("Hello World") is True

    def test_all_upper_long(self):
        assert filter_not_mostly_uppercase("ABCDEFGHIJ KLMNOPQRST",
                                           max_ratio=0.7, min_alpha_len=10) is False

    def test_short_exempt(self):
        assert filter_not_mostly_uppercase("OK", max_ratio=0.7, min_alpha_len=10) is True


class TestFilterNotPurelyNumeric:
    def test_digits(self):
        assert filter_not_purely_numeric("123456") is False

    def test_digits_with_spaces(self):
        assert filter_not_purely_numeric("123 456") is False

    def test_price_kept(self):
        assert filter_not_purely_numeric("$ 50 .") is True

    def test_time_kept(self):
        assert filter_not_purely_numeric("9:00 !") is True


class TestFilterNoUrls:
    def test_clean(self):
        assert filter_no_urls("Hello world") is True

    def test_http(self):
        assert filter_no_urls("Visit http://example.com") is False

    def test_www(self):
        assert filter_no_urls("Go to www.google.com") is False


class TestFilterNoAdKeywords:
    def test_clean(self):
        assert filter_no_ad_keywords("A nice day.", DEFAULT_AD_KEYWORDS) is True

    def test_chinese_ad(self):
        assert filter_no_ad_keywords("请关注我们", DEFAULT_AD_KEYWORDS) is False

    def test_english_ad(self):
        assert filter_no_ad_keywords("Please subscribe", DEFAULT_AD_KEYWORDS) is False


class TestFilterNotShortLogin:
    def test_normal(self):
        assert filter_not_short_login("How are you doing today?") is True

    def test_short_login_zh(self):
        assert filter_not_short_login("登录系统") is False

    def test_short_register_en(self):
        assert filter_not_short_login("register") is False


# ================================================================
# 3. CleanConfig & CLI argument parsing
# ================================================================

class TestCleanConfigDefaults:
    def test_defaults(self):
        cfg = CleanConfig()
        assert cfg.fullwidth_to_halfwidth is True
        assert cfg.normalize_whitespace is True
        assert cfg.nfd_normalize is True
        assert cfg.traditional_to_simplified is False
        assert cfg.to_lowercase is False
        assert cfg.remove_punctuation is False
        assert cfg.filter_language is True
        assert cfg.target_lang == "en"
        assert cfg.filter_ppl is False
        assert cfg.max_ppl == 1500.0
        assert cfg.filter_keywords is True
        assert cfg.ad_keywords == DEFAULT_AD_KEYWORDS
        assert cfg.extra_ad_keywords == []

    def test_override(self):
        cfg = CleanConfig(to_lowercase=True, max_ppl=800.0, target_lang="zh-cn")
        assert cfg.to_lowercase is True
        assert cfg.max_ppl == 800.0
        assert cfg.target_lang == "zh-cn"


class TestCLIParsing:
    def _parse(self, argv: list) -> CleanConfig:
        parser = build_parser()
        args = parser.parse_args(argv)
        return args_to_config(args)

    def test_default_args(self):
        cfg = self._parse([])
        assert cfg.fullwidth_to_halfwidth is True
        assert cfg.to_lowercase is False
        assert cfg.filter_ppl is False

    def test_enable_lowercase(self):
        cfg = self._parse(["--to-lowercase"])
        assert cfg.to_lowercase is True

    def test_disable_fullwidth(self):
        cfg = self._parse(["--no-fullwidth-to-halfwidth"])
        assert cfg.fullwidth_to_halfwidth is False

    def test_enable_ppl(self):
        cfg = self._parse(["--filter-ppl", "--max-ppl", "800", "--ppl-model", "gpt2",
                            "--ppl-device", "cuda"])
        assert cfg.filter_ppl is True
        assert cfg.max_ppl == 800.0
        assert cfg.ppl_model == "gpt2"
        assert cfg.ppl_device == "cuda"

    def test_disable_multiple_filters(self):
        cfg = self._parse(["--no-filter-keywords", "--no-filter-urls",
                            "--no-filter-language"])
        assert cfg.filter_keywords is False
        assert cfg.filter_urls is False
        assert cfg.filter_language is False

    def test_custom_thresholds(self):
        cfg = self._parse(["--max-punct-ratio", "0.5", "--min-msg-len", "5",
                            "--max-msg-len", "500", "--max-upper-ratio", "0.9"])
        assert cfg.max_punct_ratio == 0.5
        assert cfg.min_msg_len == 5
        assert cfg.max_msg_len == 500
        assert cfg.max_upper_ratio == 0.9

    def test_extra_ad_keywords(self):
        cfg = self._parse(["--extra-ad-keywords", "promo", "discount"])
        assert cfg.extra_ad_keywords == ["promo", "discount"]

    def test_io_paths(self):
        cfg = self._parse(["-i", "in.jsonl", "-o", "out.jsonl"])
        assert cfg.input == "in.jsonl"
        assert cfg.output == "out.jsonl"

    def test_t2s(self):
        cfg = self._parse(["--traditional-to-simplified"])
        assert cfg.traditional_to_simplified is True

    def test_remove_punctuation(self):
        cfg = self._parse(["--remove-punctuation"])
        assert cfg.remove_punctuation is True


# ================================================================
# 4. process_conversation — config-driven toggles
# ================================================================

class TestProcessConversation:

    # --- kept ---

    def test_normal_kept(self):
        c = _conv(["How are you?", "I am fine, thanks!"])
        result, reason = process_conversation(c, _cfg())
        assert result is not None and reason is None

    def test_fullwidth_normalized(self):
        c = _conv(["Ｈｅｌｌｏ there friend？", "Fine thanks!"])
        result, _ = process_conversation(c, _cfg())
        assert result["messages"][0]["content"][0]["text"] == "Hello there friend?"

    def test_accents_removed(self):
        c = _conv(["This café is nice.", "Indeed it is!"])
        result, _ = process_conversation(c, _cfg())
        assert result["messages"][0]["content"][0]["text"] == "This cafe is nice."

    @pytest.mark.skipif(not HAS_OPENCC, reason="opencc not installed")
    def test_t2s_in_pipeline(self):
        c = _conv(["計算機科學很有趣", "是的非常有趣"])
        result, _ = process_conversation(c, _cfg(traditional_to_simplified=True))
        assert result["messages"][0]["content"][0]["text"] == "计算机科学很有趣"

    def test_ok_uppercase_exempt(self):
        c = _conv(["Is that fine?", "OK ."])
        result, reason = process_conversation(c, _cfg(filter_uppercase=True))
        assert result is not None, f"should be kept but got reason={reason}"

    def test_price_kept(self):
        c = _conv(["How much?", "$ 50 ."])
        result, reason = process_conversation(c, _cfg(filter_numeric=True))
        assert result is not None, f"should be kept but got reason={reason}"

    def test_schema_preserved(self):
        c = _conv(["Hi there, friend!", "Hello, how are you!"])
        result, _ = process_conversation(c, _cfg())
        assert result["schemaVersion"] == "bedrock-conversation-2024"
        assert result["system"] == c["system"]

    # --- filtered ---

    def test_empty_messages(self):
        c = {"schemaVersion": "x", "system": [{"text": "x"}], "messages": []}
        _, reason = process_conversation(c, _cfg())
        assert reason == "empty_messages"

    def test_high_punctuation(self):
        c = _conv(["!!!???...!!!", "ok response here"])
        _, reason = process_conversation(c, _cfg(filter_punctuation=True, max_punct_ratio=0.3))
        assert "high_punctuation_ratio" in reason

    def test_too_short_message(self):
        c = _conv(["A", "Hello there!"])
        _, reason = process_conversation(c, _cfg(filter_length=True, min_msg_len=2))
        assert "message_length" in reason

    def test_mostly_uppercase_long(self):
        c = _conv(["HELLO HOW ARE YOU DOING", "Fine thanks."])
        _, reason = process_conversation(c, _cfg(filter_uppercase=True))
        assert "mostly_uppercase" in reason

    def test_purely_numeric(self):
        c = _conv(["What is it?", "12345"])
        _, reason = process_conversation(c, _cfg(filter_numeric=True))
        assert "purely_numeric" in reason

    def test_url(self):
        c = _conv(["Check https://example.com ok", "Sure thing!"])
        _, reason = process_conversation(c, _cfg(filter_urls=True))
        assert "contains_url" in reason

    def test_ad_keyword(self):
        c = _conv(["Please subscribe to the channel", "Sure thing!"])
        _, reason = process_conversation(c, _cfg(filter_keywords=True))
        assert "ad_keyword" in reason

    def test_extra_ad_keywords(self):
        c = _conv(["Get a promo code now!", "Sure thing!"])
        _, reason = process_conversation(c, _cfg(filter_keywords=True,
                                                  extra_ad_keywords=["promo"]))
        assert "ad_keyword" in reason

    def test_short_login(self):
        c = _conv(["login now", "Welcome my friend!"])
        _, reason = process_conversation(c, _cfg(filter_short_login=True))
        assert "short_login" in reason

    # --- toggle off = no filtering ---

    def test_url_allowed_when_disabled(self):
        c = _conv(["Visit https://example.com please", "Sure thing!"])
        result, _ = process_conversation(c, _cfg(filter_urls=False))
        assert result is not None

    def test_keyword_allowed_when_disabled(self):
        c = _conv(["Please subscribe to the channel", "Sure thing!"])
        result, _ = process_conversation(c, _cfg(filter_keywords=False))
        assert result is not None

    def test_numeric_allowed_when_disabled(self):
        c = _conv(["What number?", "12345"])
        result, _ = process_conversation(c, _cfg(filter_numeric=False))
        assert result is not None

    def test_uppercase_allowed_when_disabled(self):
        c = _conv(["HELLO HOW ARE YOU DOING", "Fine thanks."])
        result, _ = process_conversation(c, _cfg(filter_uppercase=False))
        assert result is not None

    # --- finalize ---

    def test_finalize_lowercase(self):
        c = _conv(["Hello World friend", "Hi There friend"])
        result, _ = process_conversation(c, _cfg(to_lowercase=True))
        assert result["messages"][0]["content"][0]["text"] == "hello world friend"

    def test_finalize_remove_punct(self):
        c = _conv(["Hello, world friend!", "Hi there friend."])
        result, _ = process_conversation(c, _cfg(remove_punctuation=True))
        assert result["messages"][0]["content"][0]["text"] == "Hello world friend"


# ================================================================
# 5. PPL filter (mocked — no torch/transformers required)
# ================================================================

class TestPPLFilter:
    """Test PPL filtering logic by mocking the model inference."""

    def test_ppl_keeps_low_perplexity(self):
        c = _conv(["How are you doing today?", "I am fine thanks!"])
        cfg = _cfg(filter_ppl=True, max_ppl=1500.0)
        with mock.patch("clean_daily_dialog.compute_ppl", return_value=50.0):
            result, reason = process_conversation(c, cfg)
        assert result is not None and reason is None

    def test_ppl_rejects_high_perplexity(self):
        c = _conv(["asdf jkl qwerty zxcv", "random gibberish here"])
        cfg = _cfg(filter_ppl=True, max_ppl=500.0)
        with mock.patch("clean_daily_dialog.compute_ppl", return_value=2000.0):
            result, reason = process_conversation(c, cfg)
        assert result is None and reason == "high_ppl"

    def test_ppl_at_boundary(self):
        c = _conv(["Hello there friend", "Hi there friend"])
        cfg = _cfg(filter_ppl=True, max_ppl=1500.0)
        with mock.patch("clean_daily_dialog.compute_ppl", return_value=1500.0):
            result, _ = process_conversation(c, cfg)
        assert result is not None  # equal to threshold → keep

    def test_ppl_skipped_when_disabled(self):
        c = _conv(["Hello there friend", "Hi there friend"])
        cfg = _cfg(filter_ppl=False)
        with mock.patch("clean_daily_dialog.compute_ppl") as mock_ppl:
            result, _ = process_conversation(c, cfg)
        mock_ppl.assert_not_called()
        assert result is not None


# ================================================================
# 6. End-to-end validation of cleaned file
# ================================================================

CLEANED_PATH = "./daily-dialog-bedrock-cleaned.jsonl"


def _validate_bedrock(obj):
    errors = []
    if obj.get("schemaVersion") != "bedrock-conversation-2024":
        errors.append("wrong schemaVersion")
    msgs = obj.get("messages", [])
    if len(msgs) < 2:
        errors.append("fewer than 2 messages")
        return errors
    if msgs[0]["role"] != "user":
        errors.append("first message not user")
    if msgs[-1]["role"] != "assistant":
        errors.append("last message not assistant")
    for i in range(1, len(msgs)):
        if msgs[i]["role"] == msgs[i - 1]["role"]:
            errors.append(f"consecutive same role at {i - 1},{i}")
    for i, msg in enumerate(msgs):
        text = msg["content"][0]["text"]
        if not text.strip():
            errors.append(f"empty text at msg {i}")
    return errors


@pytest.mark.skipif(
    not os.path.exists(CLEANED_PATH),
    reason=f"{CLEANED_PATH} not found — run clean_daily_dialog.py first",
)
class TestCleanedFile:

    @pytest.fixture(scope="class")
    def data(self):
        rows = []
        with open(CLEANED_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def test_non_empty(self, data):
        assert len(data) > 0

    def test_reasonable_count(self, data):
        assert len(data) >= 9000

    def test_all_valid_schema(self, data):
        for i, obj in enumerate(data):
            errors = _validate_bedrock(obj)
            assert errors == [], f"Row {i}: {errors}"

    def test_no_urls(self, data):
        for i, obj in enumerate(data):
            for j, msg in enumerate(obj["messages"]):
                text = msg["content"][0]["text"]
                assert not re.search(r"https?://\S+|www\.\S+", text, re.IGNORECASE), \
                    f"Row {i} msg {j} has URL"

    def test_no_empty_text(self, data):
        for i, obj in enumerate(data):
            for j, msg in enumerate(obj["messages"]):
                assert msg["content"][0]["text"].strip(), f"Row {i} msg {j} empty"

    def test_no_fullwidth_ascii(self, data):
        for i, obj in enumerate(data):
            for j, msg in enumerate(obj["messages"]):
                for ch in msg["content"][0]["text"]:
                    cp = ord(ch)
                    assert not (0xFF01 <= cp <= 0xFF5E), \
                        f"Row {i} msg {j} has fullwidth U+{cp:04X}"

    def test_no_crlf(self, data):
        for i, obj in enumerate(data):
            for j, msg in enumerate(obj["messages"]):
                assert "\r\n" not in msg["content"][0]["text"]

    def test_no_purely_numeric_messages(self, data):
        for i, obj in enumerate(data):
            for j, msg in enumerate(obj["messages"]):
                text = msg["content"][0]["text"].strip()
                non_ws = [ch for ch in text if not ch.isspace()]
                if non_ws:
                    assert not all(ch.isdigit() for ch in non_ws), \
                        f"Row {i} msg {j} purely numeric: {text}"

    def test_no_high_punctuation(self, data):
        for i, obj in enumerate(data):
            for j, msg in enumerate(obj["messages"]):
                text = msg["content"][0]["text"]
                if text:
                    ratio = sum(1 for c in text
                                if unicodedata.category(c).startswith("P")) / len(text)
                    assert ratio <= 0.3, \
                        f"Row {i} msg {j} punct ratio {ratio:.2f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
