# Text embeddings

Covers sentence/text embedding models — typically distributed via the `sentence-transformers`
library (model card mentions `sentence-transformers`, `pipeline_tag: feature-extraction` or
`sentence-similarity`). `scaffold_project.py --task embeddings` generates the pattern below.

## Loading and inference pattern

```python
model = SentenceTransformer(model_id, device=device)   # device: "cpu" | "cuda" | "mps"
vectors = model.encode(
    texts,
    convert_to_numpy=True,
    normalize_embeddings=True,   # L2-normalize so cosine similarity == dot product
    batch_size=len(texts),
)
```

`SentenceTransformer` already handles tokenization, pooling (mean/CLS/etc. depending on the
model's config), and batching internally — there's no separate tokenizer step to manage, unlike
the other task families.

`normalize_embeddings=True` is worth defaulting to: it means consumers can use a plain dot product
instead of computing cosine similarity explicitly, and it's what most vector databases expect.
Only turn it off if the user has a specific reason to want raw (non-normalized) vectors.

## Design this endpoint batch-first

Embedding models are throughput-friendly — encoding 32 texts costs little more than encoding one.
Unlike image generation, there's no reason to restrict this to single-item requests; the generated
`EmbedIn.texts: list[str]` with a `max_length` cap is the right default shape, not a special case.

## The model version problem — this is the single most important thing to get right

Vectors are only meaningful relative to the model that produced them: cosine similarity between
vectors from two different embedding models (or even two versions of the same model) is
meaningless noise, not a small error. This is why `EmbedOut` always includes `model_id` —
consumers must store it alongside every vector they persist. If the user later swaps the model,
every previously stored vector needs re-embedding; there's no way to make old and new vectors
comparable. Flag this to the user explicitly if they mention they already have stored embeddings
from a different model or a notebook experiment — mixing vector sources silently produces
nonsense similarity scores with no error to alert anyone.

## Response shape

```python
class EmbedOut(BaseModel):
    embeddings: list[list[float]]
    dimensions: int      # len(embeddings[0]) — lets consumers validate before storing
    model_id: str
```

Including `dimensions` explicitly (rather than making the consumer infer it from the array length)
matters because it's the first thing a vector database schema needs, and it's a cheap sanity check
against silently storing truncated or malformed vectors.
