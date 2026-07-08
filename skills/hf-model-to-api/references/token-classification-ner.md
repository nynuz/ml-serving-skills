# Token classification / Named Entity Recognition (NER)

Covers models loadable with `AutoModelForTokenClassification` — NER, part-of-speech tagging, any
task that labels individual tokens rather than the whole input. `scaffold_project.py --task
token-classification` generates the pattern below.

## Why this uses `pipeline(...)` instead of raw model + tokenizer calls

Unlike text classification, NER output requires reassembling sub-word tokens back into whole
words/entities and merging consecutive `B-`/`I-` tags (BIO tagging scheme) into single entity
spans with correct character offsets. Reimplementing this by hand is error-prone — the
`transformers` `pipeline("token-classification", aggregation_strategy="simple")` abstraction
already does it correctly, including handling the WordPiece/BPE sub-word merging. Use it instead
of calling the raw model, even though every other task family in this skill avoids the high-level
`pipeline()` wrapper in favor of explicit model+tokenizer calls.

```python
ner_pipeline = pipeline(
    "token-classification",
    model=model_id,
    aggregation_strategy="simple",   # merges B-/I- tags into whole entities
    device=device,                    # -1 for CPU, 0+ for CUDA index, "mps" for Apple Silicon
)
entities = ner_pipeline(text)
```

Note `pipeline()`'s `device` argument has different semantics than the `device` string used
elsewhere in this skill (`"cpu"`/`"cuda"`/`"mps"`) — it wants an int (`-1` = CPU, `0` = first CUDA
device) or the string `"mps"`. The generated `_resolve_device()` in `app/core/model.py` already
handles this conversion; don't reuse the string-based resolver from the other task families
verbatim.

## `aggregation_strategy` options

- `"simple"` (default in generated code): merges consecutive same-type tags into one entity,
  averaging scores. Good default for most NER use cases.
- `"first"` / `"average"` / `"max"`: different strategies for scoring merged multi-token entities.
  Only worth changing if the user reports specific entity-boundary or confidence-score issues —
  it's rarely worth exposing as an API parameter for a local single-model service.

## Response shape

```python
class Entity(BaseModel):
    text: str        # the surface text of the entity, e.g. "Barack Obama"
    label: str        # entity type, e.g. "PER", "ORG", "LOC" — vocabulary is model-specific
    start: int         # character offset into the original input string
    end: int
    confidence: float = Field(ge=0, le=1)

class NerOut(BaseModel):
    results: list[list[Entity]]   # one list of entities per input text
    model_id: str
```

`start`/`end` character offsets matter for any downstream consumer that wants to highlight
entities in the original text — always pass them through rather than just returning the extracted
substring, since the substring alone can't be re-located if it appears more than once in the input.

## Label vocabulary is model-specific — don't assume PER/ORG/LOC/MISC

That's the CoNLL-2003 label set used by many popular NER models (like `dslim/bert-base-NER`), but
biomedical, legal, or domain-specific NER models use entirely different label sets. Always surface
whatever `entity_group` the pipeline returns rather than hardcoding an expected vocabulary.
