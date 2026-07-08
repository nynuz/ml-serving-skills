#!/usr/bin/env python3
"""Inspect a Hugging Face Hub model_id and report which task family it belongs to.

Only talks to the public HF Hub REST API (https://huggingface.co/api/...) and, when needed,
fetches config.json for label info. Never downloads model weights. Standard-library only,
so it runs before the user has installed anything for the generated project.

Usage:
    python inspect_model.py <model_id>

For gated/private repos, provide a token via environment variable (preferred — keeps it
out of your shell history and the process list):

    HF_TOKEN=hf_xxx python inspect_model.py <model_id>

`--token` is also accepted as an override, but avoid it on shared machines.

Prints a single JSON object to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

HUB_API = "https://huggingface.co/api/models/{model_id}"
CONFIG_URL = "https://huggingface.co/{model_id}/resolve/main/config.json"

TEXT_CLASSIFICATION_ARCHS = ("ForSequenceClassification",)
TOKEN_CLASSIFICATION_ARCHS = ("ForTokenClassification",)


def fetch_json(url: str, token: str | None = None) -> dict | None:
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # The HF Hub API returns 401 for BOTH "repo does not exist" and "repo is
        # gated/private and you lack access" — it never distinguishes the two for
        # anonymous requests, to avoid leaking which private repos exist.
        if e.code in (401, 403):
            return {"__error__": "not_found_or_gated", "status": e.code}
        if e.code == 404:
            return {"__error__": "not_found_or_gated", "status": e.code}
        return {"__error__": "http_error", "status": e.code}
    except urllib.error.URLError as e:
        return {"__error__": "network_error", "reason": str(e.reason)}


def guess_task_family(model_info: dict, config: dict | None) -> str:
    pipeline_tag = (model_info.get("pipeline_tag") or "").lower()
    library_name = (model_info.get("library_name") or "").lower()
    architectures = []
    if config and isinstance(config.get("architectures"), list):
        architectures = config["architectures"]
    tags = [t.lower() for t in model_info.get("tags", [])]

    if library_name == "diffusers" or pipeline_tag in ("text-to-image", "image-to-image"):
        return "image-generation"

    if library_name == "sentence-transformers" or pipeline_tag in (
        "feature-extraction",
        "sentence-similarity",
    ) or "sentence-transformers" in tags:
        return "embeddings"

    if pipeline_tag == "token-classification" or any(
        arch.endswith(TOKEN_CLASSIFICATION_ARCHS) for arch in architectures
    ):
        return "token-classification"

    if pipeline_tag == "text-classification" or any(
        arch.endswith(TEXT_CLASSIFICATION_ARCHS) for arch in architectures
    ):
        return "text-classification"

    return "unknown"


def approx_params(model_info: dict) -> str | None:
    safetensors = model_info.get("safetensors")
    if isinstance(safetensors, dict):
        total = safetensors.get("total")
        if isinstance(total, int):
            if total >= 1_000_000_000:
                return f"~{total / 1_000_000_000:.1f}B params"
            if total >= 1_000_000:
                return f"~{total / 1_000_000:.0f}M params"
            return f"~{total} params"
    return None


def device_note(approx: str | None, task_family: str) -> str:
    if task_family == "image-generation":
        return (
            "Diffusion models are heavy — a CUDA GPU is strongly recommended for usable "
            "latency; CPU inference works but can take tens of seconds per image."
        )
    if approx and approx.startswith("~") and "B params" in approx:
        return "Billion-parameter model — expect slow CPU inference; prefer CUDA/MPS if available."
    return "Small/medium model — CPU inference is workable for local single-user use; use CUDA/MPS if available for lower latency."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id", help="Hugging Face Hub model id, e.g. bert-base-uncased")
    parser.add_argument(
        "--token",
        default=None,
        help="HF token for gated/private repos. Prefer the HF_TOKEN (or "
        "HUGGING_FACE_HUB_TOKEN) environment variable instead — a token on the command "
        "line is visible in shell history and the process list.",
    )
    args = parser.parse_args()

    # Token precedence: explicit --token overrides, otherwise fall back to the standard
    # HF environment variables so the token never has to appear on the command line.
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    model_info = fetch_json(HUB_API.format(model_id=args.model_id), token)
    if model_info is None or "__error__" in model_info:
        err = (model_info or {}).get("__error__", "unknown_error")
        result = {
            "model_id": args.model_id,
            "error": err,
            "message": {
                "not_found_or_gated": "The Hub API returned 401/403/404 for this model_id. This "
                "means EITHER the model_id is wrong/doesn't exist, OR the repo is gated/private "
                "and requires authentication — the public API deliberately doesn't distinguish "
                "the two. First double-check the model_id with the user (typos, wrong "
                "org/namespace). If it's correct, it's likely gated: ask the user to accept the "
                "license on the model page and pass --token, or run `huggingface-cli login`.",
                "network_error": "Could not reach huggingface.co. Check network access.",
                "http_error": "Hugging Face Hub API returned an unexpected error.",
            }.get(err, "Unknown error inspecting the model."),
        }
        print(json.dumps(result, indent=2))
        return 1

    config = fetch_json(CONFIG_URL.format(model_id=args.model_id), token)
    if config and "__error__" in config:
        config = None  # config.json missing/unreadable is not fatal, just less info

    task_family = guess_task_family(model_info, config)

    id2label = None
    num_labels = None
    if config and isinstance(config.get("id2label"), dict):
        id2label = config["id2label"]
        num_labels = len(id2label)

    approx = approx_params(model_info)

    result = {
        "model_id": args.model_id,
        "task_family": task_family,
        "pipeline_tag": model_info.get("pipeline_tag"),
        "library_name": model_info.get("library_name"),
        "architectures": (config or {}).get("architectures"),
        "id2label": id2label,
        "num_labels": num_labels,
        "approx_params": approx,
        "gated": bool(model_info.get("gated")),
        "requires_trust_remote_code": bool((config or {}).get("auto_map")),
        "device_recommendation": device_note(approx, task_family),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
