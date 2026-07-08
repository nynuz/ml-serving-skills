# Shared architecture: what's identical across all four task families

These patterns apply no matter which task family you're scaffolding. `scaffold_project.py`
already implements them — this file explains *why*, so you can review generated code
intelligently and adapt it when a model needs something non-standard.

## Model loading belongs in `lifespan`, never at import time or per-request

Per-request loading is catastrophic: seconds of disk/GPU transfer on every call, and concurrent
requests each loading their own copy until the process runs out of memory. Import-time loading
(top-level code in `model.py`) couples loading to module import — tests, linters, and any tool
that imports the app pays the cost, or crashes without a GPU. `lifespan` is correct: it runs once
per worker process before traffic arrives, and gives a clean shutdown hook.

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_bundle = load_model(settings)
    warm_up(app.state.model_bundle)
    app.state.model_ready = True
    yield
    app.state.model_bundle = None
    app.state.model_ready = False
```

## Warm-up is not optional

The first call to a freshly loaded model is almost always slower than subsequent ones — CUDA
kernel compilation, lazy weight initialization, tokenizer caching. Without a warm-up call, that
cost lands on whichever real user makes the first request. Always run one dummy inference through
the full pipeline right after loading, before `yield`.

## Liveness vs readiness are different questions with different remedies

```python
@router.get("/health/live")     # "is the process stuck?" -> restart me
def live():
    return {"status": "ok"}      # no dependency checks — just proves the process responds

@router.get("/health/ready")    # "can I serve traffic?" -> route to me
def ready(request: Request):
    if not getattr(request.app.state, "model_ready", False):
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready"}
```

Conflating the two is a classic outage pattern: a liveness check that also verifies the model is
loaded will restart a perfectly healthy process that's just still warming up, which resets warm-up
progress and can cascade. Keep liveness dependency-free.

## CPU/GPU-bound inference must never run directly in an `async def` handler

`model.predict(...)` or `pipeline(...)` are synchronous, CPU/GPU-bound calls. Running them
directly inside `async def` blocks the single event loop for the entire process — every other
in-flight request stalls until inference finishes. Always offload to a thread:

```python
from anyio import to_thread

@router.post("/v1/classify")
async def classify(payload: ClassifyIn, request: Request):
    result = await to_thread.run_sync(predict, request.app.state.model_bundle, payload.texts)
    ...
```

## Every response includes the model id

When output quality regresses, the first question is always "which model produced this?" — and
during any kind of side-by-side comparison, knowing which model answered matters. It costs one
field; always include it.

## Error handling: validation vs runtime failures are different

- **Bad input** (empty batch, text over the length cap, malformed JSON): let Pydantic reject it —
  FastAPI returns 422 automatically with a structured error body. Don't catch and re-wrap these.
- **Model/runtime failure** (OOM, unexpected shape, a bug in your inference code): catch broadly
  around the inference call and raise a typed `HTTPException(status_code=500, ...)`. Never let a
  raw stack trace reach the client — it can leak file paths and internals, and it's not
  actionable for the caller anyway.

```python
try:
    result = await to_thread.run_sync(predict, bundle, payload.texts)
except Exception as exc:
    raise HTTPException(status_code=500, detail="inference failed") from exc
```

## CORS is a backend concern, not frontend code

Enabling CORS so a local frontend dev server (React/Next.js on `localhost:3000`) can call this API
is configuration on *this* project, not frontend code. It's included by default in generated
projects, restricted to localhost origins. Widen or restrict it based on what the user tells you
about their setup — but don't write any client-side fetch/axios code here, that's out of scope for
this skill.

## Batch size and input length are cost controls, not just correctness

A `Field(..., max_length=N)` on a batch array or a string isn't just validation — it's a spending
limit. An unbounded batch is an unbounded GPU/CPU bill (or, on a local single-user machine, an
unbounded way to make the process unresponsive). Always set explicit caps; see `batching.md` for
how to choose sensible defaults per task family.
