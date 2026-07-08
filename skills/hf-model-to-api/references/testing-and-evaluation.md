# Testing and evaluation

Three distinct layers, don't conflate them:

1. **Unit/smoke tests** (`pytest`, generated into the project) — does the API behave correctly at
   all (loads, validates input, returns the right shape)? Runs in-process, no server needed.
2. **Correctness evaluation** (`evaluate.py`, a skill script, not part of the generated project) —
   does the *model* produce the right answers on data the user cares about? Needs a running server
   and labeled data.
3. **Performance benchmarking** (`benchmark.py`, same as above) — is it fast/scalable enough? Needs
   a running server, no labels needed.

## Layer 1: generated pytest tests

Uses `fastapi.testclient.TestClient`, which drives the app in-process — no real network calls, and
critically, `with TestClient(app) as client:` triggers the `lifespan` context manager, so the model
actually loads during tests. This is why the generated tests use the `with` form rather than bare
`TestClient(app)`, which would skip `lifespan` entirely and test against an app whose model was
never loaded.

```python
def test_classify_happy_path():
    with TestClient(app) as client:
        resp = client.post("/v1/classify", json={"texts": ["I love this product!"]})
        assert resp.status_code == 200
```

These tests actually download and run the real model (they're not mocked) — that's intentional for
a small model being wrapped for local use, but means `pytest` will be slow the first time
(downloading weights) and still non-trivial afterward (real inference). For `image-generation`,
the generated tests deliberately skip a happy-path run (see `image-generation.md`) since a real
generation is too slow for routine test runs.

Run with:
```bash
pytest
```

## Layer 2: `evaluate.py` — correctness against labeled data

Requires the server already running (`uvicorn app.main:app --port 8000`). This script is bundled
with the skill, not generated into the project — copy it in or run it directly from the skill's
`scripts/` directory, pointing `--base-url` at the running server.

```bash
python evaluate.py --task text-classification --data my_eval_set.jsonl \
    --base-url http://127.0.0.1:8000
```

Dataset format is JSONL, one labeled example per line — see the script's docstring for the exact
shape per task family. If the user doesn't have labeled data yet, don't force this step: correctness
evaluation needs ground truth, there's no way around getting some, even a handful of manually
labeled examples is enough for a useful first signal. Offer to help them write a small JSONL file
from a few examples they can eyeball themselves.

Output is accuracy/F1-style metrics printed to stdout. Interpretation:
- **text-classification**: overall accuracy + per-label accuracy. A big gap between labels usually
  means class imbalance in either the model's training data or your eval set — call it out.
- **token-classification**: precision/recall/F1 on exact (entity text, entity label) matches. This
  is a stricter check than span-overlap F1 (no partial credit for near-miss boundaries) — good
  enough as a first signal, but don't over-interpret small differences.
- **embeddings**: pairwise accuracy against a similarity threshold, plus the mean similarity score
  for "similar" vs "dissimilar" pairs. The gap between those two means matters more than the raw
  accuracy number — a small gap means the threshold (or the model) doesn't separate the classes
  well, even if accuracy looks OK on a small/lucky sample.
- **image-generation**: not supported — there's no automated way to grade "did this image match
  the prompt" without another model in the loop (e.g. CLIP score) or a human. Tell the user to
  review generated images manually via `/docs`, or skip this step for that task family.

## Layer 3: `benchmark.py` — latency and throughput

Also requires the server running. Fires concurrent requests at a single endpoint with a fixed
example payload and reports p50/p95/p99 latency plus throughput:

```bash
python benchmark.py --url http://127.0.0.1:8000/v1/classify \
    --payload '{"texts": ["example input"]}' --requests 100 --concurrency 4
```

This is a simple local load test, not a substitute for `Locust`/`k6` at real production scale — for
a locally-served single-model API that's the right level of rigor. A few things worth telling the
user when interpreting results:
- **p95/p99, not just the mean** — a mean can hide a long tail that ruins the actual user
  experience; always look at both.
- **Concurrency beyond CPU/GPU capacity won't help** — if latency degrades sharply as `--concurrency`
  increases, that's the hardware saturating, not a bug. Compare a couple of concurrency levels
  (e.g. 1, 4, 8) to find where it starts degrading, rather than picking one number blindly.
- **The first real measurement should already be warm** — `benchmark.py` sends untimed `--warmup`
  requests before measuring, on top of the server's own startup warm-up. If p50 still looks like an
  outlier-heavy distribution, something (not warm-up) is wrong.
