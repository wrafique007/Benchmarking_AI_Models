# SQuAD 2.0 Benchmark for LFM-d1 (d1-3B)

A custom evaluation script for benchmarking [`LiquidAI/d1-3B`](https://huggingface.co/LiquidAI/d1-3B) — a **decision model** — on the [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) extractive question-answering task.

Because `d1-3B` returns calibrated probabilities rather than generated text, the standard `lm-evaluation-harness` cannot be used directly. This script adapts SQuAD 2.0 into the model's native `system_one` API and computes the official Exact Match (EM) and F1 metrics using Hugging Face's `evaluate` library.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
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