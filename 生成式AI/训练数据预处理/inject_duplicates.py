#!/usr/bin/env python3
"""
Read zh_mixed_cleaned.jsonl, inject exact & near-duplicate copies of
selected conversations, write to zh_mixed_with_dups.jsonl.
"""
import copy
import json
import random

random.seed(42)

INPUT = "zh_mixed_cleaned.jsonl"
OUTPUT = "zh_mixed_with_dups.jsonl"

with open(INPUT, "r", encoding="utf-8") as f:
    conversations = [json.loads(line) for line in f if line.strip()]

original_count = len(conversations)
injected = []

# ── Pick 5 conversations for exact duplication (2-4 copies each) ──
exact_dup_indices = [0, 5, 10, 20, 30]
for idx in exact_dup_indices:
    n_copies = random.randint(2, 4)
    for _ in range(n_copies):
        dup = copy.deepcopy(conversations[idx])
        # Change system prompt to simulate different source but same content
        dup["system"] = [{"text": f"系统提示变体-{random.randint(1000,9999)}"}]
        injected.append(dup)

# ── Pick 5 conversations for near-duplication (2-3 variants each) ──
# We slightly modify the assistant's reply to create near-dups
substitutions = [
    ("非常", "特别"), ("可以", "能够"), ("建议", "推荐"),
    ("需要", "应该"), ("因此", "所以"), ("例如", "比如"),
    ("一些", "若干"), ("开始", "着手"), ("方法", "方式"),
    ("了解", "理解"), ("提供", "给出"), ("重要", "关键"),
]


def make_near_dup(conv):
    """Create a near-duplicate by applying 2-3 random substitutions to assistant text."""
    dup = copy.deepcopy(conv)
    subs = random.sample(substitutions, min(3, len(substitutions)))
    for msg in dup["messages"]:
        if msg["role"] == "assistant":
            text = msg["content"][0]["text"]
            for old, new in subs:
                text = text.replace(old, new, 1)
            msg["content"][0]["text"] = text
    dup["system"] = [{"text": f"近似变体-{random.randint(1000,9999)}"}]
    return dup


near_dup_indices = [2, 8, 15, 25, 40]
for idx in near_dup_indices:
    n_variants = random.randint(2, 3)
    for _ in range(n_variants):
        injected.append(make_near_dup(conversations[idx]))

# ── Combine and shuffle ──
all_conversations = conversations + injected
random.shuffle(all_conversations)

with open(OUTPUT, "w", encoding="utf-8") as f:
    for c in all_conversations:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

exact_total = sum(random.Random(42).randint(2, 4) for _ in exact_dup_indices)
print(f"Original:            {original_count}")
print(f"Injected exact dups: {sum(1 for _ in injected[:len(injected)])} total injected")
print(f"Total output:        {len(all_conversations)} -> {OUTPUT}")
print(f"Exact dup sources:   indices {exact_dup_indices}")
print(f"Near-dup sources:    indices {near_dup_indices}")
