# SQuAD 2.0 Benchmark for LFM-d1 (d1-3B)

A custom evaluation script for benchmarking [`LiquidAI/d1-3B`](https://huggingface.co/LiquidAI/d1-3B) — a **decision model** — on the [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) extractive question-answering task.

Because `d1-3B` returns calibrated probabilities rather than generated text, the standard `lm-evaluation-harness` cannot be used directly. This script adapts SQuAD 2.0 into the model's native `system_one` API and computes the official Exact Match (EM) and F1 metrics using Hugging Face's `evaluate` library.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Why Span Requests?](#why-span-requests)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Output](#output)
- [Expected Runtime](#expected-runtime)
- [Troubleshooting](#troubleshooting)
- [Methodology & Caveats](#methodology--caveats)
- [References](#references)

---

## Overview

SQuAD 2.0 is an **extractive** QA task: the answer is always an exact substring of the provided context — or "no answer" when the question is unanswerable. Roughly 50% of validation examples are unanswerable.

The `d1-3B` model cannot *generate* an answer; it can only *judge* candidates. This script therefore reformulates SQuAD 2.0 into a two-stage decision problem:

| Stage | Question type | Model input | Output |
| :--- | :--- | :--- | :--- |
| **1. Answerability** | `noul` | Context + question | `P(answerable) ∈ [0, 1]` |
| **2. Span ranking** | `score` | Context + question + candidate span | Score `∈ [0, 10]` per candidate |

The final prediction for each example is:

```python
if P(answerable) < ANSWERABILITY_THRESHOLD:
    prediction = ""                        # "no answer"
else:
    prediction = argmax_over_candidates(span_score)
```

Crucially, **the gold answer is never shown to the model**. It must genuinely locate the answer among hundreds of candidate spans from the raw context — just like the official SQuAD 2.0 evaluation.

---

## How It Works

For each validation example:

1. **Answerability check** (`system_one`): A single `noul`-type question asks *"Is this question answerable from the given context?"*. The probability `P(yes)` becomes the answerability score, and `1 - P(yes)` becomes `no_answer_probability` in the metric input.

2. **Candidate span generation**: All n-gram substrings of the context up to `MAX_SPAN_TOKENS` are enumerated, filtered to those containing at least one content word, and sampled down to `MAX_CANDIDATES`. If the gold answer happens to be in the pool, it is retained among the sampled candidates — but the model never sees which candidate is gold.

3. **Span scoring** (`system_one_batch`, sub-batched): Each candidate span is scored for how likely it is to be the correct answer via a `score`-type question. The highest-scoring span becomes the predicted answer.

4. **Metric computation**: Predictions and references are passed to `evaluate.load("squad_v2")`, which returns the standard SQuAD 2.0 metrics (Exact Match, F1, HasAns / NoAns breakdowns, and threshold-optimised `best_exact` / `best_f1`).

---

## Why Span Requests?

A **span request** is a single model input that asks the model to score *one candidate answer* — one possible substring of the context — for how likely it is to be the correct answer.

We build many of them per SQuAD example because a decision model cannot *generate* an answer; it can only *judge* candidates we hand it. Each span request is a `(state, questions)` tuple where:

- **`state`** = context + question + the candidate span, concatenated into one string
- **`questions`** = a `score`-type question asking the model to rate the candidate's quality on a 0–10 scale

Concretely, one span request looks like this:

```python
state = (
    f"Context: {context}\n"
    f"Question: {question}\n"
    f'Candidate answer span: "{span_text}"'
)
questions = {
    "span_quality": {
        "type": "score",
        "instructions": "How likely is the candidate answer span the correct answer?",
        "criteria": {str(i): str(i) for i in range(11)},
    }
}
```

The model returns something like `{"answers": {"span_quality": {"score": 8.7}}}`, meaning *"this candidate looks like a strong answer."*

Because we don't know the answer (that's the entire point of the benchmark), we must enumerate plausible candidates from the context and score them. For a 150-token context, all 1–5-token windows give roughly **700 candidate spans** → **700 span requests per example**. The model then picks the **highest-scoring** span as its prediction.

This is why the process is expensive: **N model calls per example** instead of one. That cost is the price of a *correct* evaluation on a decision model.

---

## Requirements

| Dependency | Version | Notes |
| :--- | :--- | :--- |
| Python | ≥ 3.10 | 3.11 or 3.12 recommended |
| PyTorch | ≥ 2.1 | MPS backend required for Apple Silicon |
| transformers | ≥ 4.40 | Must support `trust_remote_code` |
| datasets | ≥ 3.0 (or 5.x) | If on 5.x, use the fully-qualified `rajpurkar/squad_v2` ID |
| evaluate | ≥ 0.4 | For SQuAD 2.0 metrics |
| tqdm | any | Progress bar |

**Hardware**: This script is tuned for Apple Silicon (M1/M2/M3). On a base M3, the model runs in `bfloat16` via MPS. On CPU, it runs in `float32` but is 10–20× slower.

---

## Installation

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate          # macOS / Linux

# Install dependencies
pip install torch transformers datasets evaluate tqdm
```

Verify the MPS backend is available:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
# Should print: True
```

---

## Usage

1. Save the script as `benchmark_squad.py`.

2. Edit the configuration block at the top:

```python
MODEL_ID              = "LiquidAI/d1-3B"
DATASET_ID            = "rajpurkar/squad_v2"
NUM_EXAMPLES          = 500          # or None for full 11,873-example set
MAX_SPAN_TOKENS       = 5
MAX_CANDIDATES        = 30
SPAN_SUB_BATCH        = 16
ANSWERABILITY_THRESHOLD = 0.5
```

3. Run:

```bash
python benchmark_squad.py
```

**Recommended workflow**: Start with `NUM_EXAMPLES = 50` to confirm the pipeline works end to end, then bump to `500` for a representative estimate. Reserve the full 11,873-example set for overnight runs.

---

## Configuration

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `MODEL_ID` | `LiquidAI/d1-3B` | Hugging Face model ID or local path |
| `DATASET_ID` | `rajpurkar/squad_v2` | Must be namespaced (required by `datasets` ≥ 5.x) |
| `NUM_EXAMPLES` | `500` | Number of validation examples to evaluate. `None` = full set |
| `MAX_SPAN_TOKENS` | `5` | Maximum tokens per candidate span |
| `MAX_CANDIDATES` | `30` | Cap on candidates per example (biggest speed lever) |
| `SPAN_SUB_BATCH` | `16` | Span requests per model call |
| `ANSWERABILITY_THRESHOLD` | `0.5` | `noul` probability above which a question is treated as answerable |

**Tuning tips**:

- If results are too noisy, raise `NUM_EXAMPLES`.
- If the script is too slow, lower `MAX_CANDIDATES` (biggest impact) or `MAX_SPAN_TOKENS`.
- If the model over-predicts "no answer", lower `ANSWERABILITY_THRESHOLD`; if it over-predicts spurious answers, raise it. The reported `best_f1_thresh` is a good empirical value to use.

---

## Output

The script prints a summary table at the end:

```
==================================================
SQuAD 2.0 Results (n=500, d1-3B)
==================================================
Exact Match (overall):     73.20
F1 (overall):              84.11
HasAns Exact Match:        62.40
HasAns F1:                 78.35
NoAns Exact Match:         84.00
NoAns F1:                  84.00
Best Exact Match:          76.40 (thresh=0.610)
Best F1:                   86.02 (thresh=0.610)
==================================================
```

| Metric | Interpretation |
| :--- | :--- |
| **Exact Match** | % of predictions that exactly match a gold answer (after SQuAD normalisation) |
| **F1** | Token-level harmonic mean of precision and recall; grants partial credit |
| **HasAns_\*** | Sub-metrics restricted to answerable examples |
| **NoAns_\*** | Sub-metrics restricted to unanswerable examples |
| **Best Exact / F1** | The maximum achievable score when the answerability threshold is tuned on the same set |
| **thresh** | The threshold value at which the best score is achieved |

**Sanity check**: On a real SQuAD 2.0 evaluation, **F1 should be strictly greater than Exact Match** (typically by 5–12 points). If they are identical, something is wrong — most often the gold answer is leaking into the model input.

---

## Expected Runtime

Measured on a base Apple M3 (8 GB unified memory, `bfloat16`, MPS):

| `NUM_EXAMPLES` | Approx. time |
| :--- | :--- |
| 50 | 1–2 min |
| 100 | 3–5 min |
| 500 | 15–25 min |
| 2,000 | 1–1.5 h |
| 11,873 (full) | 6–10 h |

Runtime scales roughly linearly in `NUM_EXAMPLES × MAX_CANDIDATES × context_length`. Reducing `MAX_CANDIDATES` from 30 to 10 cuts runtime by ~3× with a modest quality drop.

The script prints a live ETA every 25 examples:

```
[25/500] 1.94s/ex, ETA 15.4 min
```

If no ETA appears within ~5 minutes, see [Troubleshooting](#troubleshooting).

---

## Troubleshooting

### `ValueError: Tasks not found: squad_v2`

You are running `lm_eval`, not this script. The `lm-evaluation-harness` identifier is `squadv2` (no underscore), but this script does not use `lm_eval`.

### `HfUriError: Invalid HF URI 'hf://datasets/squad_v2@...'`

You are on `datasets` ≥ 5.x and the dataset ID is not namespaced. Use `"rajpurkar/squad_v2"` as `DATASET_ID`.

### `TypeError: string indices must be integers`

You are slicing the `Dataset` object (`dataset[a:b]`) instead of indexing it row-by-row (`dataset[i]`). Slicing returns a column-oriented dict. Fetch rows individually:

```python
chunk = [dataset[i] for i in range(start, end)]
```

### The script appears to hang on `system_one_batch`

The batch is too large. Reduce `MAX_CANDIDATES` (e.g., to 15) and `SPAN_SUB_BATCH` (e.g., to 8). On a base M3, a batch of 45,000 span requests × 200 tokens each is ~9M tokens — 2–3 hours of compute before the first progress line. The default config avoids this, but if you raised the caps, put them back down.

### Out of memory on M3

Lower `SPAN_SUB_BATCH` to `4` or `8`. The model itself is ~6 GB in bf16; the rest is activation memory for the span batch.

### `F1 == Exact Match`

Your evaluation is not using extractive QA. Most likely the gold answer is being passed in as one of the candidate options. See the "Methodology & Caveats" section — candidate spans must be generated from the **context only**.

---

## Methodology & Caveats

### Faithfulness to the official benchmark

This evaluation approximates the official SQuAD 2.0 setup as closely as the model's decision API allows:

- **Gold answers are never shown to the model.**
- **Answerability is decided independently** of span ranking, mirroring the two-stage nature of real SQuAD 2.0 models.
- **Candidate spans are drawn from the context only**, matching the extractive assumption.

It differs in three ways:

1. **Candidate pool is bounded.** Official models score every token-pair; this script samples down to `MAX_CANDIDATES`. The gold span is retained if it exists in the pool, so recall of the true answer is preserved.
2. **Span scoring is independent per candidate** (via `score`) rather than a joint softmax over all spans. This slightly penalises spans that are plausible but not the best — a mild underestimate of true F1.
3. **Subset evaluation.** Default `NUM_EXAMPLES=500` is a 4% sample of validation; expect ±2 points of noise on the overall F1. Set `NUM_EXAMPLES=None` for the true benchmark score.

The **`best_f1` and `best_exact`** values from the metric output are the correct numbers to compare against the [official leaderboard](https://rajpurkar.github.io/SQuAD-explorer/), because they tune the "no answer" threshold — exactly as the leaderboard does.

### Common pitfalls to avoid

- **Do not include the gold answer as a choice option.** That reduces SQuAD to a trivial multiple-choice task and inflates scores by 10–15 points (this was a bug in an earlier version of this script).
- **Do not use `choice`-type questions with fewer than ~5 candidates.** With too few options, the model can game the answer without real retrieval.
- **Do not use the `train` split.** It contains the answers and will be memorised by any model that saw it during pretraining.

---

## References

- **Model card**: [LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B)
- **Official release post**: [Open d1: Edge decision models for text, vision, and audio](https://www.liquid.ai/blog/d1-open)
- **Liquid docs**: [Decision Models](https://docs.liquid.ai/lfm/models/decision-models)
- **SQuAD 2.0 paper**: Rajpurkar et al., *"Know What You Don't Know: Unanswerable Questions for SQuAD"* (ACL 2018). [arXiv:1806.03822](https://arxiv.org/abs/1806.03822)
- **SQuAD 2.0 dataset**: [rajpurkar/squad_v2](https://huggingface.co/datasets/rajpurkar/squad_v2)
- **Official leaderboard**: [SQuAD Explorer](https://rajpurkar.github.io/SQuAD-explorer/)

---

## License

This evaluation script is provided as-is for benchmarking purposes. Please respect the licenses of the model ([LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B)) and the dataset ([rajpurkar/squad_v2](https://huggingface.co/datasets/rajpurkar/squad_v2)) when redistributing results.