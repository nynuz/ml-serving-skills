# Batching: what the default pattern is, and when to reach for something else

## The default: capped batch arrays (what `scaffold_project.py` generates)

The client sends an array (`texts: list[str]`, capped at `max_batch_size`) in one request; the
server processes the whole array in a single forward pass and returns all results together. This
is "offline"/client-driven batching — simple, predictable, and the right default for a local,
single-user (or low-concurrency) service, which is what this skill targets.

Two things make this safe:
- **A server-enforced max length** (`Field(..., max_length=N)`) — GPUs and CPUs alike have a point
  where a bigger batch stops being free and starts being an OOM risk or a multi-minute request.
  Default caps in generated projects (8 for text tasks) are conservative starting points, not
  measured optima — tell the user to raise them once they've confirmed their hardware handles it,
  by watching memory use while sending progressively larger batches.
- **CPU/GPU work still runs off the event loop** via `anyio.to_thread.run_sync` — batching doesn't
  change this requirement, see `architecture.md`.

## Dynamic (server-side) micro-batching — not implemented by default, here's why

GPUs are throughput machines: processing 64 inputs often costs little more than processing one.
Dedicated inference servers (Triton, vLLM, TorchServe) exploit this by holding incoming requests
for a few milliseconds, batching whatever arrived in that window across *different* HTTP requests,
running one forward pass, then fanning results back to each caller — clients see normal
single-item request/response semantics while the server gets GPU-batching throughput.

This is real engineering complexity (a background batching loop, per-request futures, careful
timeout handling) that only pays off under sustained concurrent load from multiple simultaneous
callers. For the target use case of this skill — one developer running a model locally, calling it
from their own scripts or a local frontend — there typically isn't enough concurrent request
volume for micro-batching to matter, and the added complexity is not worth it by default.

**When to actually build it**: if the user describes multiple concurrent clients hammering the
same local server and wants to squeeze more throughput out of a GPU, that's the signal to
implement dynamic micro-batching — but treat it as a deliberate escalation the user asks for, not
a default. If you get there, the shape is: an `asyncio.Queue` requests are placed on, a background
task that drains the queue every few ms and runs one batched forward pass, and per-request
`asyncio.Future`s that get resolved with each item's slice of the batch result.

## Batch vs real-time as two different API shapes (not a spectrum)

For genuinely large offline work — re-scoring a whole dataset, nightly bulk embedding — don't force
it through the synchronous request/response endpoint at all, even with a big `max_batch_size`. A
huge synchronous request ties up an HTTP connection for however long the whole batch takes, with no
partial progress and no ability to check status. If a user describes wanting to process hundreds or
thousands of items, that's a different endpoint shape:  `POST /v1/batches` accepts a file/dataset
reference, returns `202 Accepted` + a job id immediately, and a separate `GET /v1/batches/{id}`
polls status — well beyond what this skill scaffolds by default, but worth naming explicitly if the
user's actual need is bulk processing rather than a "few items at a time" API.
