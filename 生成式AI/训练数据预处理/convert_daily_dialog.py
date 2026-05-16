#!/usr/bin/env python3
"""
Convert DailyDialog HuggingFace dataset to Amazon Bedrock conversation fine-tuning format.

Input:  ./daily-dialog.txt  (HuggingFace datasets directory with 'convo' and 'response' fields)
Output: ./daily-dialog-bedrock-train.jsonl, ./daily-dialog-bedrock-test.jsonl

Each row has:
  - convo: "Person2:text##Person1:text##...##PersonX:"  (## separated, last part is respondent marker)
  - response: the respondent's reply text

Target format (one JSON object per line):
{
  "schemaVersion": "bedrock-conversation-2024",
  "system": [{"text": "..."}],
  "messages": [{"role": "user", "content": [{"text": "..."}]}, {"role": "assistant", "content": [{"text": "..."}]}, ...]
}

Constraints:
  - Messages must alternate user/assistant, starting with user, ending with assistant.
  - The 'response' field always becomes the final assistant turn.
"""

import json
import sys
from datasets import load_from_disk

SYSTEM_PROMPT = "You are a helpful conversational assistant. Engage in natural, friendly dialogue."


def parse_convo_turns(convo: str):
    """Parse the convo field into a list of (speaker, text) tuples and the respondent name.

    Returns:
        content_turns: list of (speaker, text) for turns with actual content
        respondent: name of the person who gives the response (e.g. 'Person1')
    """
    parts = convo.split("##")
    # Last part is the respondent marker, e.g. "Person1:"
    marker = parts[-1].strip()
    respondent = marker.rstrip(":")

    content_turns = []
    for part in parts[:-1]:
        part = part.strip()
        if not part:
            continue
        # Split on first ':'
        colon_idx = part.index(":")
        speaker = part[:colon_idx].strip()
        text = part[colon_idx + 1:].strip()
        if text:  # skip empty content turns
            content_turns.append((speaker, text))

    return content_turns, respondent


def merge_consecutive_turns(turns):
    """Merge consecutive turns from the same speaker into one, joining text with newline."""
    if not turns:
        return []
    merged = [list(turns[0])]
    for speaker, text in turns[1:]:
        if merged[-1][0] == speaker:
            merged[-1][1] += "\n" + text
        else:
            merged.append([speaker, text])
    return [(s, t) for s, t in merged]


def convert_row(convo: str, response: str):
    """Convert a single dataset row to Bedrock conversation format.

    Returns:
        dict: Bedrock conversation object, or None if the row cannot be converted.
        str:  Skip reason if None, else empty string.
    """
    content_turns, respondent = parse_convo_turns(convo)

    # Append response as the respondent's final turn
    all_turns = content_turns + [(respondent, response.strip())]

    # Merge consecutive same-speaker turns
    merged = merge_consecutive_turns(all_turns)

    if len(merged) < 2:
        # Need at least 1 user + 1 assistant turn
        return None, "fewer_than_2_groups"

    # Assign roles: last group = assistant, alternate backward
    n = len(merged)
    roles = [""] * n
    roles[-1] = "assistant"
    for i in range(n - 2, -1, -1):
        roles[i] = "user" if roles[i + 1] == "assistant" else "assistant"

    if roles[0] != "user":
        # Conversation starts with assistant (respondent spoke first, odd # of groups).
        # Fix: drop the first group (respondent's opening) so the other speaker leads.
        merged = merged[1:]
        if len(merged) < 2:
            return None, "fewer_than_2_groups_after_trim"

        # Re-assign roles after trimming
        n = len(merged)
        roles = [""] * n
        roles[-1] = "assistant"
        for i in range(n - 2, -1, -1):
            roles[i] = "user" if roles[i + 1] == "assistant" else "assistant"

        if roles[0] != "user":
            # Should not happen after dropping one group, but guard anyway
            return None, "starts_with_assistant"

    # Build messages
    messages = []
    for i, (_, text) in enumerate(merged):
        messages.append({
            "role": roles[i],
            "content": [{"text": text}]
        })

    return {
        "schemaVersion": "bedrock-conversation-2024",
        "system": [{"text": SYSTEM_PROMPT}],
        "messages": messages
    }, ""


def convert_split(dataset, output_path: str):
    """Convert an entire dataset split and write to JSONL file.

    Returns:
        stats dict with counts of converted, skipped, and skip reasons.
    """
    stats = {"converted": 0, "skipped": 0, "skip_reasons": {}}

    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(len(dataset)):
            row = dataset[i]
            result, reason = convert_row(row["convo"], row["response"])
            if result is None:
                stats["skipped"] += 1
                stats["skip_reasons"][reason] = stats["skip_reasons"].get(reason, 0) + 1
            else:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                stats["converted"] += 1

    return stats


def main():
    dataset_path = "./daily-dialog.txt"
    ds = load_from_disk(dataset_path)

    for split_name in ["train", "test"]:
        output_path = f"./daily-dialog-bedrock-{split_name}.jsonl"
        print(f"Converting {split_name} split ({len(ds[split_name])} rows)...")
        stats = convert_split(ds[split_name], output_path)
        print(f"  Output: {output_path}")
        print(f"  Converted: {stats['converted']}")
        print(f"  Skipped:   {stats['skipped']}")
        if stats["skip_reasons"]:
            for reason, count in stats["skip_reasons"].items():
                print(f"    - {reason}: {count}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
