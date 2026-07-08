---
name: hf-model-to-api
description: Turns a Hugging Face model that already works in a notebook (text classifier, NER/token-classification, text embedder, or image generator) into a production-grade local FastAPI service, with layered app structure, lifespan model loading + warm-up, batching, input/output validation, health/readiness probes, and correctness + performance evaluation scripts. Use whenever the user wants to "wrap", "serve locally", "turn into an API", or "expose as a FastAPI service" a Hugging Face model or model_id (e.g. BERT classifiers, NER models, sentence-transformers embedders, Stable Diffusion / diffusers image generators), even if they only mention it works in a Jupyter notebook and don't say "FastAPI" explicitly. Does NOT cover frontend/client code, Docker, or cloud deployment — local FastAPI serving only.
compatibility: Python 3.10+, pip, and network access to huggingface.co for model discovery. Generated projects add their own dependencies (transformers/diffusers/sentence-transformers/torch/fastapi/uvicorn) depending on the task family.
metadata:
  author: antonio
  version: 1.0.0
  category: ml-engineering
---

# HF Model → FastAPI Builder

## What this skill does

Takes a Hugging Face `model_id` and generates a self-contained, local, production-grade FastAPI
project around it: layered app structure, model loaded once at startup (never per-request), a
task-appropriate inference endpoint, batching with explicit caps, input/output validation, health
and readiness probes, structured error handling, and optional evaluation/benchmark scripts.

It supports four task families, each with a different endpoint shape and underlying library:

| Task family | HF concept | Library | Reference file |
|---|---|---|---|
| Text classification | `AutoModelForSequenceClassification` | `transformers` | `references/text-classification.md` |
| Token classification (NER) | `AutoModelForTokenClassification` | `transformers` | `references/token-classification-ner.md` |
| Text embeddings | sentence embedding model | `sentence-transformers` | `references/embeddings.md` |
| Image generation | diffusion pipeline | `diffusers` | `references/image-generation.md` |

## Scope boundaries — read this before starting

- **Local only.** No Docker, no Kubernetes, no cloud deployment (Railway, AWS, etc.). The output
  is a project the user runs with `uvicorn` on their own machine.
- **Backend only.** Never generate frontend/client code (React, Next.js, fetch calls, etc.). If the
  user wants a UI to call the API, tell them that's a separate concern and stop at the API boundary.
  It is fine to enable permissive local CORS so a frontend dev server *can* call the API later —
  that's a backend config concern, not frontend code.
- **Serving, not training.** If the user wants to fine-tune or train a model, this skill is the
  wrong tool — say so.
- **Local, single-user threat model.** Generated services have NO authentication and NO rate
  limiting by design — they're meant to run on `localhost`. They do enforce input cost caps (batch
  size, per-string length, generation params, and a request-body size limit → 413). Always tell the
  user not to bind the service to `0.0.0.0` or expose it beyond localhost without first adding auth,
  rate limiting, and a hardened CORS policy — those are out of scope here.
- **Don't install or launch anything without asking.** Scaffold the project and give the user exact
  commands to run. Only `pip install` or start `uvicorn` yourself if the user explicitly asks you
  to verify it works — some of these dependencies (`torch`, `diffusers`) are large downloads.

## Workflow

### Step 1: Gather inputs

You need:
- **`model_id`** (required) — the Hugging Face Hub ID, e.g. `distilbert-base-uncased-finetuned-sst-2-english`.
- **Output directory** (ask if not given, otherwise default to `./<model-name-slug>-api` in the
  current working directory).
- Optional: a small labeled evaluation dataset path (for correctness evaluation later), a port
  (default `8000`).

If the user only describes what they did in a notebook ("I load it with `pipeline(...)`, it
classifies sentiment"), ask for the exact `model_id` if it isn't stated — everything downstream
depends on it.

### Step 2: Inspect the model

Run the bundled inspector — it only hits the public HF Hub API, it never downloads model weights:

```bash
python scripts/inspect_model.py <model_id>
```

It prints JSON with `task_family`, `pipeline_tag`, `library_name`, `architectures`, `id2label` (if
available), `num_labels`, an approximate parameter count, and a device recommendation note.

- If `task_family` comes back as one of the four supported families, proceed.
- If it comes back `"unknown"` or you disagree with the guess (some repos have missing/wrong
  metadata), **ask the user** which of the four task families applies — don't silently guess wrong,
  the whole endpoint shape depends on it.
- If the script reports `not_found_or_gated`, HF's public API returned 401/403/404 — this means
  the `model_id` is either wrong/misspelled, or the repo is gated/private. It cannot tell you
  which. First re-confirm the exact `model_id` with the user (typos and wrong org namespaces are
  the most common cause); only if it's confirmed correct should you treat it as gated and ask the
  user to accept the license on the model page and authenticate. Prefer passing the token via the
  `HF_TOKEN` environment variable (`HF_TOKEN=hf_xxx python scripts/inspect_model.py <model_id>`) or
  `huggingface-cli login` — avoid the `--token` flag, since a token on the command line leaks into
  shell history and the process list.

### Step 3: Read the matching reference file

Before scaffolding, read `references/<task-family>.md` for the task family from Step 2. It contains
the exact pipeline/model-loading code, the request/response schema, and task-specific gotchas
(e.g. label mapping for classification, entity aggregation for NER, pooling strategy for embeddings,
image encoding for generation). Also skim `references/architecture.md` once per session — it covers
the parts that are identical across all four task families (lifespan loading, health probes, CORS,
error handling) so you don't have to re-derive them each time.

Use the already-installed `transformers-huggingface` skill for anything about model loading,
tokenization, device placement (CPU/CUDA/MPS), or dtype choice that comes up beyond what the
reference file covers — it has the authoritative patterns for the `transformers` library itself.

### Step 4: Scaffold the project

Run the generator with the task family and any label/config info you got from Step 2:

```bash
python scripts/scaffold_project.py \
  --model-id "<model_id>" \
  --task <text-classification|token-classification|embeddings|image-generation> \
  --output-dir "<output_dir>" \
  --id2label-json "<path-to-json-or-inline-json>"   # optional, classification/NER only
```

Run `python scripts/scaffold_project.py --help` to see all flags (port, max batch size, max input
length / image size, etc.) — pass through anything relevant you learned in Step 2 instead of
leaving the generated defaults if you have better values.

This produces:

```
<output_dir>/
├── app/
│   ├── main.py            # FastAPI() + lifespan (load model once, warm up), CORS, router mount
│   ├── core/
│   │   ├── config.py      # pydantic-settings: MODEL_ID, DEVICE, MAX_BATCH_SIZE, etc. from env
│   │   └── model.py       # load_model() / inference function for the chosen task family
│   ├── api/
│   │   └── routes.py      # task endpoint + /health/live + /health/ready + /v1/info
│   └── schemas.py         # Pydantic request/response models with Field constraints
├── tests/
│   ├── test_health.py
│   └── test_inference.py  # happy path + a validation-error case
├── requirements.txt       # only what this task family needs
├── .env.example
├── README.md              # run instructions, endpoint docs, CORS note
└── .gitignore
```

The scaffolder fills in what it can determine programmatically (model id, task family, labels if
you passed them). **You still need to review the generated `app/core/model.py` and
`app/schemas.py`** against the reference file from Step 3 — the scaffolder does not call the model,
so it can't know things like the exact label set if you didn't pass `--id2label-json`, or whether an
NER model needs `aggregation_strategy="simple"`. Fill in any `# TODO` markers it leaves.

### Step 5: Verify the cross-cutting production concerns

Every generated project must have all of these — check them explicitly, they're easy to silently
drop when hand-editing generated code:

- Model loaded in `lifespan`, not at import time or per-request (`app/main.py`)
- A warm-up call (one dummy inference) right after loading, so the first real request isn't the
  slow one — see `references/architecture.md`
- `/health/live` (process alive, no dependency checks) is a **different** function from
  `/health/ready` (checks the model is loaded) — conflating them causes restart cascades, see
  `references/architecture.md`
- Every inference response includes the model id/version — non-negotiable, it's how you debug
  "which model produced this" later
- Batch input has a server-enforced max length/size (`Field(..., max_length=...)`) — an unbounded
  batch is an unbounded GPU/CPU spend. The scaffolder also caps each individual string
  (`StringConstraints(max_length=...)`, tune with `--max-chars-per-text`) and rejects oversized
  request bodies via a `413` middleware in `main.py` (`--max-request-mb`) — verify both survived any
  hand-editing, since the body-size check is the only guard that fires *before* the payload is
  parsed into memory
- CPU/GPU-bound inference runs via `anyio.to_thread.run_sync(...)`, never directly in an `async def`
  handler — otherwise it blocks the event loop for every other in-flight request
- Validation errors return 422 with a useful message; model/runtime errors are caught and mapped to
  a typed HTTP error, never an unhandled 500 with a stack trace leaking to the client

See `references/batching.md` for the batch-array pattern used by default, and when (rarely, for a
local single-user setup) dynamic micro-batching would be worth the added complexity.

### Step 6: Tests

The scaffolder generates `tests/test_health.py` and `tests/test_inference.py` using
`fastapi.testclient.TestClient` (see `references/testing-and-evaluation.md`). Read through them,
adjust the sample input/expected shape to match the actual model, and tell the user to run:

```bash
pytest
```

### Step 7: Evaluation and benchmarking (optional, only if the user wants it)

Two independent scripts are bundled — copy them into the generated project's directory (they're
not part of the FastAPI app itself, they're tools that call the *running* server):

- `scripts/evaluate.py` — correctness: runs a labeled dataset (CSV/JSONL) against the running API
  and reports accuracy/F1 (classification, NER) or cosine-similarity sanity checks (embeddings).
  Needs the user's labeled data — ask for a path, or skip this if they don't have one yet.
- `scripts/benchmark.py` — performance: fires concurrent requests at the running API and reports
  p50/p95/p99 latency and throughput. Needs no labeled data, just a few example inputs.

See `references/testing-and-evaluation.md` for exact usage and how to interpret the output.

### Step 8: Wrap up

Give the user the exact commands to run locally:

```bash
cd <output_dir>
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Point them at `/docs` for interactive testing (FastAPI's automatic OpenAPI UI). Explicitly state
what you did *not* do: no Docker image, no cloud deployment, no frontend code. If they want any of
those next, that's outside this skill.

## Troubleshooting

- **`inspect_model.py` returns `task_family: unknown`**: the repo's metadata is incomplete. Ask the
  user directly which of the four families it is, and pass `--task` explicitly to the scaffolder.
- **Model needs a `trust_remote_code=True` / custom code**: flag this to the user before scaffolding
  — it means arbitrary code from the Hub runs locally; get explicit confirmation before generating a
  project that sets that flag.
- **User wants a task family not in the table** (e.g. speech, translation, VQA): say clearly that
  this skill currently covers only the four listed families, and offer to build the FastAPI wrapper
  by hand using the same architectural patterns from `references/architecture.md` instead of forcing
  it into one of the four reference files.
- **Generated `requirements.txt` install is slow/huge**: `torch` and `diffusers` are large; warn the
  user before they run `pip install` if the task family pulls them in.
