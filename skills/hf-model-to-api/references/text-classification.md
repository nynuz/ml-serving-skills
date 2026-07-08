# Text classification (BERT-style classifiers)

Covers models loadable with `AutoModelForSequenceClassification` — sentiment analysis, topic
classification, toxicity detection, intent classification, etc. `scaffold_project.py --task
text-classification` generates the pattern below; this file explains the parts worth double
checking by hand.

## Loading pattern

```python
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
model.to(device)
model.eval()          # disables dropout etc. — always call this before inference
```

`model.config.id2label` is populated from the model's `config.json` and is usually correct. Trust
it by default; only override with `--id2label-json` when `inspect_model.py` couldn't find labels
or you know the config is wrong/generic (e.g. `{0: "LABEL_0", 1: "LABEL_1"}` — a sign the model
card documents real label names that never made it into `config.json`).

## Inference pattern

```python
inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
with torch.no_grad():
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    confidences, predicted_ids = probs.max(dim=-1)
```

- `padding=True` + `truncation=True` + explicit `max_length` together are what make batched
  inference on variable-length text work; without truncation a single long input can blow up
  memory or crash.
- `model.eval()` + `torch.no_grad()` together are mandatory for inference: `eval()` disables
  training-only layers (dropout, batchnorm updates), `no_grad()` skips gradient tracking (saves
  memory and time you don't need at inference).
- `torch.softmax` + `.max(dim=-1)` gives you both the predicted label id and its confidence in one
  pass — don't call `argmax` and `softmax` separately if you need both.

## Multi-label vs single-label

The default pattern above assumes single-label classification (softmax, one predicted class per
input). If the model card says multi-label (e.g. a toxicity model that can flag multiple
categories at once), swap `softmax` for `sigmoid` and return all labels above a threshold instead
of just the argmax:

```python
probs = torch.sigmoid(logits)  # independent per-class probabilities, not a distribution
predicted = [(bundle.id2label[i], p) for i, p in enumerate(probs[0].tolist()) if p > 0.5]
```

Check `model.config.problem_type` — if it's `"multi_label_classification"`, use this branch.

## Response shape

```python
class ClassificationResult(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)

class ClassifyOut(BaseModel):
    results: list[ClassificationResult]
    model_id: str
```

`confidence` bounded `[0, 1]` via `Field(ge=0, le=1)` isn't just documentation — it's a guardrail
that catches a broken softmax/sigmoid computation at the validation boundary instead of silently
shipping garbage confidence scores.
