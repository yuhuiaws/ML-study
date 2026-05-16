#!/usr/bin/env python3
"""
Tests for convert_daily_dialog.py

Validates:
  1. Parsing logic (convo field splitting, respondent detection)
  2. Consecutive turn merging
  3. Role assignment (alternation, starts with user, ends with assistant)
  4. Edge cases (single turn, same-speaker runs, empty text)
  5. Output format matches Bedrock conversation schema
  6. End-to-end validation on generated JSONL files
"""

import json
import os
import pytest

from convert_daily_dialog import (
    parse_convo_turns,
    merge_consecutive_turns,
    convert_row,
    SYSTEM_PROMPT,
)

# ---------------------------------------------------------------------------
# 1. parse_convo_turns
# ---------------------------------------------------------------------------

class TestParseConvoTurns:

    def test_basic_two_turns(self):
        convo = "Person2:Hello .##Person1:Hi .##Person1:"
        turns, respondent = parse_convo_turns(convo)
        assert respondent == "Person1"
        assert turns == [("Person2", "Hello ."), ("Person1", "Hi .")]

    def test_single_marker_only(self):
        """Only a respondent marker, no prior content."""
        convo = "Person1:"
        turns, respondent = parse_convo_turns(convo)
        assert respondent == "Person1"
        assert turns == []

    def test_multiple_turns(self):
        convo = "Person2:A##Person1:B##Person2:C##Person1:D##Person2:"
        turns, respondent = parse_convo_turns(convo)
        assert respondent == "Person2"
        assert len(turns) == 4
        assert turns[0] == ("Person2", "A")
        assert turns[3] == ("Person1", "D")

    def test_colon_in_text(self):
        """Text containing colons should not break parsing."""
        convo = "Person1:Time is 10:30 .##Person2:"
        turns, respondent = parse_convo_turns(convo)
        assert respondent == "Person2"
        assert turns == [("Person1", "Time is 10:30 .")]

    def test_consecutive_same_speaker_in_convo(self):
        convo = "Person2:Look at the cloud .##Person2:"
        turns, respondent = parse_convo_turns(convo)
        assert respondent == "Person2"
        assert turns == [("Person2", "Look at the cloud .")]


# ---------------------------------------------------------------------------
# 2. merge_consecutive_turns
# ---------------------------------------------------------------------------

class TestMergeConsecutiveTurns:

    def test_no_merge_needed(self):
        turns = [("Person1", "A"), ("Person2", "B"), ("Person1", "C")]
        merged = merge_consecutive_turns(turns)
        assert merged == turns

    def test_merge_two_consecutive(self):
        turns = [("Person1", "A"), ("Person1", "B"), ("Person2", "C")]
        merged = merge_consecutive_turns(turns)
        assert merged == [("Person1", "A\nB"), ("Person2", "C")]

    def test_merge_three_consecutive(self):
        turns = [("Person2", "X"), ("Person2", "Y"), ("Person2", "Z")]
        merged = merge_consecutive_turns(turns)
        assert merged == [("Person2", "X\nY\nZ")]

    def test_empty_input(self):
        assert merge_consecutive_turns([]) == []

    def test_single_turn(self):
        turns = [("Person1", "solo")]
        assert merge_consecutive_turns(turns) == [("Person1", "solo")]


# ---------------------------------------------------------------------------
# 3. convert_row — valid cases
# ---------------------------------------------------------------------------

class TestConvertRowValid:

    def test_simple_two_turn(self):
        """Person2 speaks first, Person1 responds."""
        convo = "Person2:Good morning .##Person1:"
        response = "Good morning to you too ."
        result, reason = convert_row(convo, response)
        assert result is not None
        assert reason == ""
        msgs = result["messages"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"][0]["text"] == "Good morning ."
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"][0]["text"] == "Good morning to you too ."

    def test_four_turn_alternating(self):
        convo = "Person2:A##Person1:B##Person2:C##Person1:"
        response = "D"
        result, _ = convert_row(convo, response)
        assert result is not None
        msgs = result["messages"]
        assert len(msgs) == 4
        roles = [m["role"] for m in msgs]
        assert roles == ["user", "assistant", "user", "assistant"]
        texts = [m["content"][0]["text"] for m in msgs]
        assert texts == ["A", "B", "C", "D"]

    def test_merge_response_with_preceding_same_speaker(self):
        """
        Row 0 from the real data:
        Person2 speaks, Person1 speaks, then Person1 responds.
        Person1's two turns should be merged into one assistant turn.
        """
        convo = "Person2:Would you please take a seat ?##Person1:Thanks .##Person1:"
        response = "I see . All right ."
        result, _ = convert_row(convo, response)
        assert result is not None
        msgs = result["messages"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"][0]["text"] == "Would you please take a seat ?"
        assert msgs[1]["role"] == "assistant"
        assert "Thanks ." in msgs[1]["content"][0]["text"]
        assert "I see . All right ." in msgs[1]["content"][0]["text"]

    def test_long_conversation(self):
        """6 alternating turns."""
        convo = "Person1:A##Person2:B##Person1:C##Person2:D##Person1:E##Person2:"
        response = "F"
        result, _ = convert_row(convo, response)
        assert result is not None
        msgs = result["messages"]
        assert len(msgs) == 6
        roles = [m["role"] for m in msgs]
        assert roles == ["user", "assistant"] * 3

    def test_schema_fields(self):
        convo = "Person2:Hi##Person1:"
        response = "Hello"
        result, _ = convert_row(convo, response)
        assert result["schemaVersion"] == "bedrock-conversation-2024"
        assert result["system"] == [{"text": SYSTEM_PROMPT}]
        assert "messages" in result


# ---------------------------------------------------------------------------
# 4. convert_row — skip / edge cases
# ---------------------------------------------------------------------------

class TestConvertRowEdgeCases:

    def test_single_marker_no_context(self):
        """Only respondent marker, no prior turns → fewer_than_2_groups."""
        convo = "Person1:"
        response = "Okay ."
        result, reason = convert_row(convo, response)
        assert result is None
        assert reason == "fewer_than_2_groups"

    def test_same_speaker_only_two_turns(self):
        """
        Person2 says something and Person2 responds.
        After merge: 1 group → fewer_than_2_groups.
        """
        convo = "Person2:Look at the cloud .##Person2:"
        response = "Let's go quickly ."
        result, reason = convert_row(convo, response)
        assert result is None
        assert reason == "fewer_than_2_groups"

    def test_starts_with_assistant_trimmed(self):
        """
        Odd number of groups where respondent starts.
        The first group (respondent's opening) is dropped so the other speaker leads.
        Person2:A##Person1:B##Person2:C##Person1:D##Person2:E##Person1:F##Person2:G##Person2:
        Groups: P2(A), P1(B), P2(C), P1(D), P2(E), P1(F), P2(G+H) = 7 groups (odd)
        After trim: P1(B), P2(C), P1(D), P2(E), P1(F), P2(G+H) = 6 groups (even)
        Roles: user, asst, user, asst, user, asst ✓
        """
        convo = "Person2:A##Person1:B##Person2:C##Person1:D##Person2:E##Person1:F##Person2:G##Person2:"
        response = "H"
        result, reason = convert_row(convo, response)
        assert result is not None
        assert reason == ""
        msgs = result["messages"]
        assert len(msgs) == 6
        roles = [m["role"] for m in msgs]
        assert roles == ["user", "assistant"] * 3
        # First message is Person1's "B" (since Person2's "A" was trimmed)
        assert msgs[0]["content"][0]["text"] == "B"
        # Last assistant message contains merged G+H
        assert "G\nH" == msgs[-1]["content"][0]["text"]

    def test_starts_with_assistant_too_short_after_trim(self):
        """
        After trimming the respondent's opening, only 1 group remains → skip.
        Person1:X##Person2:Y##Person1: → respondent=Person1
        Groups: P1(X), P2(Y), P1(resp) = 3 groups
        P1 is respondent. roles from back: asst, user, asst → first=asst
        Trim first group: P2(Y), P1(resp) = 2 groups → user, asst ✓
        Actually this works. Let's make a case that really fails:
        Person1:X##Person1: → P1(X), P1(resp) → merged: P1(X+resp) = 1 group → fewer_than_2
        """
        convo = "Person1:X##Person1:"
        response = "Y"
        result, reason = convert_row(convo, response)
        assert result is None
        assert reason == "fewer_than_2_groups"


# ---------------------------------------------------------------------------
# 5. Output format validation helpers
# ---------------------------------------------------------------------------

def validate_bedrock_format(obj):
    """Validate a single Bedrock conversation object. Returns list of error strings."""
    errors = []

    if obj.get("schemaVersion") != "bedrock-conversation-2024":
        errors.append("Missing or wrong schemaVersion")

    system = obj.get("system")
    if not isinstance(system, list) or len(system) == 0:
        errors.append("system must be a non-empty list")
    elif not isinstance(system[0].get("text"), str) or not system[0]["text"]:
        errors.append("system[0].text must be a non-empty string")

    messages = obj.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        errors.append("messages must have at least 2 entries")
        return errors

    # First must be user, last must be assistant
    if messages[0]["role"] != "user":
        errors.append(f"First message role must be 'user', got '{messages[0]['role']}'")
    if messages[-1]["role"] != "assistant":
        errors.append(f"Last message role must be 'assistant', got '{messages[-1]['role']}'")

    # Strict alternation
    for i in range(1, len(messages)):
        if messages[i]["role"] == messages[i - 1]["role"]:
            errors.append(f"Messages {i-1} and {i} have same role '{messages[i]['role']}'")

    # Each message has content[0].text
    for i, msg in enumerate(messages):
        if msg["role"] not in ("user", "assistant"):
            errors.append(f"Message {i} has invalid role '{msg['role']}'")
        content = msg.get("content")
        if not isinstance(content, list) or len(content) == 0:
            errors.append(f"Message {i} content must be a non-empty list")
        elif not isinstance(content[0].get("text"), str) or not content[0]["text"]:
            errors.append(f"Message {i} content[0].text must be a non-empty string")

    return errors


class TestValidateBedrock:

    def test_valid_object(self):
        obj = {
            "schemaVersion": "bedrock-conversation-2024",
            "system": [{"text": "sys"}],
            "messages": [
                {"role": "user", "content": [{"text": "hi"}]},
                {"role": "assistant", "content": [{"text": "hello"}]},
            ]
        }
        assert validate_bedrock_format(obj) == []

    def test_invalid_alternation(self):
        obj = {
            "schemaVersion": "bedrock-conversation-2024",
            "system": [{"text": "sys"}],
            "messages": [
                {"role": "user", "content": [{"text": "a"}]},
                {"role": "user", "content": [{"text": "b"}]},
            ]
        }
        errors = validate_bedrock_format(obj)
        assert any("same role" in e for e in errors)


# ---------------------------------------------------------------------------
# 6. End-to-end: validate generated JSONL files
# ---------------------------------------------------------------------------

TRAIN_JSONL = "./daily-dialog-bedrock-train.jsonl"
TEST_JSONL = "./daily-dialog-bedrock-test.jsonl"


def _load_jsonl(path):
    """Load all JSON objects from a JSONL file."""
    objects = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                objects.append(json.loads(line))
            except json.JSONDecodeError as e:
                pytest.fail(f"Invalid JSON at line {line_num} in {path}: {e}")
    return objects


@pytest.mark.skipif(
    not os.path.exists(TRAIN_JSONL),
    reason=f"{TRAIN_JSONL} not found — run convert_daily_dialog.py first"
)
class TestGeneratedTrainFile:

    @pytest.fixture(scope="class")
    def train_data(self):
        return _load_jsonl(TRAIN_JSONL)

    def test_non_empty(self, train_data):
        assert len(train_data) > 0, "Train file should not be empty"

    def test_all_rows_valid_schema(self, train_data):
        """Every row must pass Bedrock format validation."""
        for i, obj in enumerate(train_data):
            errors = validate_bedrock_format(obj)
            assert errors == [], f"Row {i} has errors: {errors}"

    def test_sample_row_structure(self, train_data):
        """Spot-check the first row."""
        first = train_data[0]
        assert first["schemaVersion"] == "bedrock-conversation-2024"
        assert len(first["messages"]) >= 2
        assert first["messages"][0]["role"] == "user"
        assert first["messages"][-1]["role"] == "assistant"

    def test_no_empty_text(self, train_data):
        """No message should have empty text."""
        for i, obj in enumerate(train_data):
            for j, msg in enumerate(obj["messages"]):
                text = msg["content"][0]["text"]
                assert text.strip(), f"Row {i} msg {j} has empty text"

    def test_reasonable_count(self, train_data):
        """Should have at least 80% of the original 9450 rows."""
        assert len(train_data) >= 7500


@pytest.mark.skipif(
    not os.path.exists(TEST_JSONL),
    reason=f"{TEST_JSONL} not found — run convert_daily_dialog.py first"
)
class TestGeneratedTestFile:

    @pytest.fixture(scope="class")
    def test_data(self):
        return _load_jsonl(TEST_JSONL)

    def test_non_empty(self, test_data):
        assert len(test_data) > 0

    def test_all_rows_valid_schema(self, test_data):
        for i, obj in enumerate(test_data):
            errors = validate_bedrock_format(obj)
            assert errors == [], f"Row {i} has errors: {errors}"


# ---------------------------------------------------------------------------
# 7. Round-trip: convert known examples and verify exact output
# ---------------------------------------------------------------------------

class TestKnownExamples:

    def test_row0_like(self):
        """Simulates Row 0: Person2 speaks, Person1 speaks, Person1 responds."""
        convo = (
            "Person2:Would you please take a seat over there , madam ?##"
            "Person1:Thanks . I can wait here .##"
            "Person1:"
        )
        response = "I see . All right , then . Thanks ."
        result, _ = convert_row(convo, response)
        assert result is not None
        msgs = result["messages"]
        # Person2 = user (first speaker), Person1 = assistant (respondent)
        # P2 content + P1 content + P1 response → merged: P2(1 turn), P1(2 turns merged)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"][0]["text"] == "Would you please take a seat over there , madam ?"
        assert msgs[1]["role"] == "assistant"
        assert "Thanks . I can wait here ." in msgs[1]["content"][0]["text"]
        assert "I see . All right , then . Thanks ." in msgs[1]["content"][0]["text"]

    def test_row3_like(self):
        """Simulates Row 3: 6 content turns + Person1 response."""
        convo = (
            "Person2:Are you feeling better today ?##"
            "Person1:It's hard to say .##"
            "Person2:You should give up smoking .##"
            "Person1:You're right .##"
            "Person2:But you should make up your mind .##"
            "Person1:I need something to keep me awake .##"
            "Person1:"
        )
        response = "Thank you for your advice !"
        result, _ = convert_row(convo, response)
        assert result is not None
        msgs = result["messages"]
        # 6 alternating turns: P2,P1,P2,P1,P2,P1 + P1 response
        # After merge with response: P2,P1,P2,P1,P2,P1(merged with response) = 6 groups
        assert len(msgs) == 6
        roles = [m["role"] for m in msgs]
        assert roles == ["user", "assistant", "user", "assistant", "user", "assistant"]
        # Last assistant message should contain the response
        assert "Thank you for your advice !" in msgs[-1]["content"][0]["text"]

    def test_exact_output_format(self):
        """Verify the exact JSON structure matches Bedrock spec."""
        convo = "Person1:Hi there .##Person2:"
        response = "Hello !"
        result, _ = convert_row(convo, response)
        expected = {
            "schemaVersion": "bedrock-conversation-2024",
            "system": [{"text": SYSTEM_PROMPT}],
            "messages": [
                {"role": "user", "content": [{"text": "Hi there ."}]},
                {"role": "assistant", "content": [{"text": "Hello !"}]},
            ]
        }
        assert result == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
