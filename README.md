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
npx skills add https://github.com/nynuz/ml-serving-skills

# list what's available
npx skills add https://github.com/nynuz/ml-serving-skills --list

# only specific skills
npx skills add https://github.com/nynuz/ml-serving-skills --skill hf-model-to-api

# non-interactive (CI/CD)
npx skills add https://github.com/nynuz/ml-serving-skills -y
```

## Usage

Once installed, invoke the skill from your AI coding agent by describing what you want,
or by naming the skill directly. A minimal example:

> Use the `hf-model-to-api` skill to turn the Hugging Face model `distilbert-base-uncased-finetuned-sst-2-english` into a local FastAPI service.

The agent will inspect the model, detect the task family, scaffold the project, and give
you the exact commands to run it locally:

```bash
cd distilbert-base-uncased-finetuned-sst-2-english-api
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Then open http://127.0.0.1:8000/docs to try the API interactively.

You don't have to name the skill explicitly — prompts like *"wrap this NER model as a
local API"* or *"serve sentence-transformers/all-MiniLM-L6-v2 with FastAPI"* will trigger
it too.

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

## Disclaimer

These skills drive an **AI coding agent that generates code**. That output is
non-deterministic and may contain bugs, insecure patterns, or security
vulnerabilities — regardless of the guardrails the skills try to enforce.

By using these skills you acknowledge that:

- **You are solely responsible** for reviewing, testing, and securing any generated
  code before you run it, install its dependencies, or expose it on a network.
- Generated services are intended for **local, single-user use only**. They ship
  **without authentication or rate limiting** and must not be exposed beyond
  `localhost` without additional hardening that is out of scope for these skills.
- Generated projects download and execute **third-party models and dependencies**
  (e.g. from the Hugging Face Hub and PyPI). Reviewing their licenses, provenance, and
  trustworthiness — including any model that requests `trust_remote_code` — is your
  responsibility.
- The authors and maintainers accept **no liability** for any damage, data loss,
  security incident, service disruption, or cost of any kind arising from the use of
  these skills or the code they generate.

The software is provided **"as is", without warranty of any kind**, as stated in the
[LICENSE](LICENSE).

## License

[MIT](LICENSE) © 2026 nynuz
