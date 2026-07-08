# ML Serving Skills

A collection of [Agent Skills](https://agentskills.io) for Claude Code and other AI
coding assistants, focused on taking machine-learning models from a notebook to a
running service — and everything around it.

Each skill lives in its own folder under [`skills/`](skills/), is fully self-contained
(its own references and scripts), and can be installed independently.

## Installation

Install a single skill into your current project (auto-detects your AI coding agent):

```bash
npx skills add nynuz/ml-serving-skills@hf-model-to-api
```

Or, using the [agentskills.io](https://agentskills.io) CLI, install everything at once:

```bash
# all skills in this repo
npx add-skill nynuz/ml-serving-skills

# list what's available
npx add-skill nynuz/ml-serving-skills --list

# only specific skills
npx add-skill nynuz/ml-serving-skills --skill hf-model-to-api

# non-interactive (CI/CD)
npx add-skill nynuz/ml-serving-skills -y
```

## Available skills

| Skill | Description |
| ----- | ----------- |
| [hf-model-to-api](skills/hf-model-to-api/) | Turns a Hugging Face model (text classifier, NER, embedder, or image generator) that already works in a notebook into a production-grade **local FastAPI** service — layered structure, lifespan model loading + warm-up, batching, input/output validation, health/readiness probes, and correctness + performance evaluation scripts. Local serving only; no Docker, no cloud, no frontend. |

## `hf-model-to-api` at a glance

Give the skill a Hugging Face `model_id` and it scaffolds a self-contained FastAPI
project around it:

- Model loaded **once** at startup (lifespan) with a warm-up pass — never per request.
- Task-appropriate endpoint for one of four families: text classification,
  token classification (NER), text embeddings, image generation.
- Batching with explicit caps, per-string and request-body size limits, structured
  error handling, and separate `/health/live` vs `/health/ready` probes.
- Bundled scripts to inspect a model, evaluate correctness against labeled data, and
  benchmark latency/throughput.

It is deliberately **local and backend-only**: no Docker, no cloud deployment, and no
frontend/client code.

## Roadmap

- **`fastapi-frontend`** *(planned)* — scaffolds a React/Next.js frontend that talks to
  a FastAPI service produced by `hf-model-to-api`, so the two skills compose into a full
  model-serving stack.

## Repository structure

```
.
├── skills/
│   └── hf-model-to-api/
│       ├── SKILL.md
│       ├── references/
│       └── scripts/
├── README.md
└── LICENSE
```

New skills are added as sibling folders under `skills/`.

## Requirements

- Python 3.10+ and network access to `huggingface.co` (for the model-serving skills).
- An AI coding agent that supports Agent Skills (e.g. Claude Code).

## License

[MIT](LICENSE) © 2026 nynuz
