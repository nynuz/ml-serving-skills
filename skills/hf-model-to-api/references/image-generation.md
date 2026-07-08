# Image generation (diffusion models)

Covers text-to-image diffusion models distributed via the `diffusers` library (model card shows
`library_name: diffusers`, `pipeline_tag: text-to-image`). This is the one task family in this
skill that isn't part of `transformers` at all — it's a different library with different loading
and inference conventions. `scaffold_project.py --task image-generation` generates the pattern
below.

## Loading pattern

```python
dtype = torch.float16 if device == "cuda" else torch.float32   # fp16 needs a CUDA GPU
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
pipe = pipe.to(device)
```

`DiffusionPipeline.from_pretrained` auto-detects the correct pipeline subclass (Stable Diffusion,
SDXL, etc.) from the repo's config — you don't need to know which specific pipeline class a model
uses ahead of time. Use `float16` only on CUDA; it's unsupported or much slower on CPU, and MPS
(Apple Silicon) support for fp16 is inconsistent across diffusers versions — default to `float32`
on anything that isn't CUDA.

## Inference is much more expensive per call than the other three task families

A single image generation is CPU/GPU work measured in seconds (GPU) to tens of seconds or minutes
(CPU) — several orders of magnitude more than a classification or embedding call. This changes
several defaults from the other task families:

- **No request batching by default.** The generated endpoint takes one `prompt` per request, not
  `list[str]`. `diffusers` pipelines *can* accept a list of prompts and batch them internally, but
  for a local single-user setup this rarely helps — you're usually waiting on one image at a time
  anyway, and batching multiple large images multiplies peak memory use, which is more likely to
  OOM a local GPU than to save meaningful wall-clock time. If the user explicitly wants bulk
  generation (e.g. "generate 50 variations overnight"), that's a batch/offline job, not something
  to force into the synchronous request/response endpoint — see `batching.md`.
- **The cost caps are generation parameters, not batch size.** `num_inference_steps`, `width`, and
  `height` directly control compute cost (more steps and larger images = proportionally slower).
  Cap them with `Field(ge=..., le=...)` the same way you'd cap batch size elsewhere — an
  unconstrained `width=8192` from a client is the image-generation equivalent of an unbounded
  batch array.
- **Warm-up is more expensive but more important.** The first generation on a fresh CUDA context
  compiles kernels; use a tiny 1-step, 64x64 warm-up call, not a full-quality image, so startup
  doesn't take as long as a real request.

## Returning the image

For a local API with no object storage set up, base64-encoding the PNG directly into the JSON
response is the simplest correct choice — the response is self-contained and a frontend can render
it directly as a data URI (`data:image/png;base64,<...>`) with no extra round-trip. This does mean
larger response payloads than returning a URL would; if the user later wants to generate many
images or serve them repeatedly, that's a sign to introduce file/object storage — out of scope for
this skill's "local only" mandate, mention it as a future step rather than building it now.

```python
buf = io.BytesIO()
image.save(buf, format="PNG")
encoded = base64.b64encode(buf.getvalue()).decode("ascii")
```

## Response shape

```python
class GenerateIn(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    num_inference_steps: int = Field(default=25, ge=1, le=100)
    width: int = Field(default=512, ge=64, le=1024)
    height: int = Field(default=512, ge=64, le=1024)

class GenerateOut(BaseModel):
    image_base64: str
    format: str = "png"
    model_id: str
```

## Testing note

Don't generate a happy-path test that actually runs the diffusion model — it's slow and often
needs a GPU, which makes `pytest` painfully slow or outright broken on a CPU-only CI/dev machine.
The scaffolder only generates validation-error tests (empty prompt, oversized dimensions) for this
task family; a real generation test is something to add manually once the project is confirmed
running locally, ideally marked so it can be skipped by default.
