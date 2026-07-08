#!/usr/bin/env python3
"""Latency/throughput benchmark against a *running* local API.

This does NOT start or manage the server — run `uvicorn app.main:app` yourself first,
and make sure it's warmed up (the lifespan warm-up call handles this automatically).
Standard-library only (urllib + concurrent.futures), no load-testing framework needed
for a local single-machine benchmark.

Usage:
    python benchmark.py --url http://127.0.0.1:8000/v1/classify \\
        --payload '{"texts": ["This movie was great!"]}' \\
        --requests 100 --concurrency 4

    python benchmark.py --url http://127.0.0.1:8000/v1/embed \\
        --payload-file example_payload.json --requests 200 --concurrency 8
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def send_one(url: str, body: bytes, timeout: float) -> tuple[float, int | None, str | None]:
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
            elapsed = time.perf_counter() - start
            return elapsed, resp.status, None
    except urllib.error.HTTPError as e:
        elapsed = time.perf_counter() - start
        return elapsed, e.code, str(e)
    except urllib.error.URLError as e:
        elapsed = time.perf_counter() - start
        return elapsed, None, str(e.reason)


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (pct / 100)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True, help="Full endpoint URL, e.g. http://127.0.0.1:8000/v1/classify")
    payload_group = parser.add_mutually_exclusive_group(required=True)
    payload_group.add_argument("--payload", help="JSON request body as a string")
    payload_group.add_argument("--payload-file", help="Path to a file containing the JSON request body")
    parser.add_argument("--requests", type=int, default=100, help="Total requests to send")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent in-flight requests")
    parser.add_argument("--warmup", type=int, default=5, help="Untimed warm-up requests before measuring")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    if args.payload_file:
        with open(args.payload_file, encoding="utf-8") as f:
            payload_text = f.read()
    else:
        payload_text = args.payload
    try:
        json.loads(payload_text)  # validate it's well-formed JSON before sending
    except json.JSONDecodeError as e:
        print(f"--payload is not valid JSON: {e}", file=sys.stderr)
        return 1
    body = payload_text.encode("utf-8")

    print(f"Warming up with {args.warmup} untimed request(s)...")
    for _ in range(args.warmup):
        send_one(args.url, body, args.timeout)

    print(f"Sending {args.requests} requests at concurrency {args.concurrency}...")
    latencies: list[float] = []
    errors: list[str] = []
    status_counts: dict[int | None, int] = {}

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(send_one, args.url, body, args.timeout) for _ in range(args.requests)]
        for fut in as_completed(futures):
            elapsed, status, error = fut.result()
            latencies.append(elapsed)
            status_counts[status] = status_counts.get(status, 0) + 1
            if error:
                errors.append(error)
    wall_elapsed = time.perf_counter() - wall_start

    latencies.sort()
    success = status_counts.get(200, 0)

    print(f"\n=== Results: {args.requests} requests, concurrency={args.concurrency} ===")
    print(f"Wall time:      {wall_elapsed:.2f}s")
    print(f"Throughput:     {args.requests / wall_elapsed:.2f} req/s")
    print(f"Success (200):  {success}/{args.requests}")
    if len(status_counts) > 1 or None in status_counts:
        print(f"Status breakdown: {status_counts}")
    print("\nLatency (seconds):")
    print(f"  min:  {min(latencies):.3f}")
    print(f"  p50:  {percentile(latencies, 50):.3f}")
    print(f"  p95:  {percentile(latencies, 95):.3f}")
    print(f"  p99:  {percentile(latencies, 99):.3f}")
    print(f"  max:  {max(latencies):.3f}")
    print(f"  mean: {statistics.mean(latencies):.3f}")

    if errors:
        print(f"\n{len(errors)} error(s), first 5:")
        for e in errors[:5]:
            print(f"  {e}")

    return 0 if success == args.requests else 1


if __name__ == "__main__":
    sys.exit(main())
