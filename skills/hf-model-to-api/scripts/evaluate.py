#!/usr/bin/env python3
"""Correctness evaluation: replay a labeled dataset against a *running* local API.

This does NOT start or manage the server — run `uvicorn app.main:app` yourself first.
Standard-library only (urllib for HTTP), so it needs nothing beyond what's already
required to have the API running.

Input is JSONL (one JSON object per line). Expected shape depends on --task:

  text-classification:  {"text": "...", "label": "POSITIVE"}
  token-classification:  {"text": "...", "entities": [{"text": "Obama", "label": "PER"}, ...]}
  embeddings:             {"text_a": "...", "text_b": "...", "similar": true}

Usage:
    python evaluate.py --task text-classification --data eval.jsonl \\
        --base-url http://127.0.0.1:8000

    python evaluate.py --task embeddings --data pairs.jsonl \\
        --base-url http://127.0.0.1:8000 --similarity-threshold 0.6

image-generation has no automated correctness check here — grading a generated image
against a prompt needs a separate model (e.g. CLIP score) or human review, both out of
scope for this script. Skip evaluate.py for that task family.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field

DEFAULT_ENDPOINTS = {
    "text-classification": "/v1/classify",
    "token-classification": "/v1/extract-entities",
    "embeddings": "/v1/embed",
}


def post_json(url: str, payload: dict, timeout: float = 60.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from {url}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not reach {url} ({e.reason}). Is the server running "
            f"(uvicorn app.main:app --port ...)?"
        ) from e


def load_jsonl(path: str) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{line_no}: invalid JSON — {e}")
    return items


def chunked(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


@dataclass
class PRF1:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, predicted: set, expected: set) -> None:
        self.tp += len(predicted & expected)
        self.fp += len(predicted - expected)
        self.fn += len(expected - predicted)

    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    def f1(self) -> float:
        p, r = self.precision(), self.recall()
        return 2 * p * r / (p + r) if (p + r) else 0.0


def eval_text_classification(base_url: str, endpoint: str, data: list[dict], batch_size: int) -> None:
    url = base_url.rstrip("/") + endpoint
    correct = 0
    total = 0
    per_label = {}
    for batch in chunked(data, batch_size):
        texts = [item["text"] for item in batch]
        expected = [str(item["label"]) for item in batch]
        result = post_json(url, {"texts": texts})
        predicted = [r["label"] for r in result["results"]]
        for exp, pred in zip(expected, predicted):
            total += 1
            correct += int(exp == pred)
            bucket = per_label.setdefault(exp, {"correct": 0, "total": 0})
            bucket["total"] += 1
            bucket["correct"] += int(exp == pred)

    print(f"\n=== text-classification correctness: {total} examples ===")
    print(f"Accuracy: {correct}/{total} = {correct / total:.3f}" if total else "No examples.")
    print("\nPer-label accuracy:")
    for label, stats in sorted(per_label.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] else 0.0
        print(f"  {label:20s}  {stats['correct']:4d}/{stats['total']:<4d}  ({acc:.3f})")


def eval_token_classification(base_url: str, endpoint: str, data: list[dict], batch_size: int) -> None:
    url = base_url.rstrip("/") + endpoint
    overall = PRF1()
    for batch in chunked(data, batch_size):
        texts = [item["text"] for item in batch]
        expected_sets = [
            {(e["text"], e["label"]) for e in item.get("entities", [])} for item in batch
        ]
        result = post_json(url, {"texts": texts})
        for exp_set, predicted_entities in zip(expected_sets, result["results"]):
            pred_set = {(e["text"], e["label"]) for e in predicted_entities}
            overall.add(pred_set, exp_set)

    print(f"\n=== token-classification (NER) correctness: {len(data)} examples ===")
    print("Note: exact (text, label) match — no partial credit for boundary-only overlaps.")
    print(f"Precision: {overall.precision():.3f}")
    print(f"Recall:    {overall.recall():.3f}")
    print(f"F1:        {overall.f1():.3f}")


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def eval_embeddings(base_url: str, endpoint: str, data: list[dict], threshold: float) -> None:
    url = base_url.rstrip("/") + endpoint
    correct = 0
    similar_scores = []
    dissimilar_scores = []
    for item in data:
        result = post_json(url, {"texts": [item["text_a"], item["text_b"]]})
        vec_a, vec_b = result["embeddings"]
        sim = cosine(vec_a, vec_b)
        expected_similar = bool(item["similar"])
        predicted_similar = sim >= threshold
        correct += int(predicted_similar == expected_similar)
        (similar_scores if expected_similar else dissimilar_scores).append(sim)

    total = len(data)
    print(f"\n=== embeddings correctness: {total} pairs, threshold={threshold} ===")
    print(f"Accuracy: {correct}/{total} = {correct / total:.3f}" if total else "No examples.")
    if similar_scores:
        print(f"Mean cosine similarity for 'similar' pairs:    {sum(similar_scores) / len(similar_scores):.3f}")
    if dissimilar_scores:
        print(f"Mean cosine similarity for 'dissimilar' pairs: {sum(dissimilar_scores) / len(dissimilar_scores):.3f}")
    print(
        "A healthy embedding model should show a clear gap between these two means — "
        "if they're close, the threshold or the model may not fit this data well."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=list(DEFAULT_ENDPOINTS))
    parser.add_argument("--data", required=True, help="Path to a JSONL labeled dataset")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--endpoint", default=None, help="Override the default endpoint path")
    parser.add_argument("--batch-size", type=int, default=8, help="Must be <= the server's MAX_BATCH_SIZE")
    parser.add_argument("--similarity-threshold", type=float, default=0.6, help="embeddings task only")
    args = parser.parse_args()

    endpoint = args.endpoint or DEFAULT_ENDPOINTS[args.task]
    data = load_jsonl(args.data)
    if not data:
        print("No examples found in the dataset.", file=sys.stderr)
        return 1

    try:
        if args.task == "text-classification":
            eval_text_classification(args.base_url, endpoint, data, args.batch_size)
        elif args.task == "token-classification":
            eval_token_classification(args.base_url, endpoint, data, args.batch_size)
        elif args.task == "embeddings":
            eval_embeddings(args.base_url, endpoint, data, args.similarity_threshold)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
