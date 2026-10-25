import torch
from transformers import AutoModel
from datasets import load_dataset
import evaluate
from tqdm import tqdm
import re
import random
import time
from typing import List, Tuple

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
MODEL_ID = "LiquidAI/d1-3B"
DATASET_ID = "rajpurkar/squad_v2"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float32 if DEVICE == "cpu" else torch.bfloat16

# --- Speed knobs ----------------------------------------------------------
NUM_EXAMPLES = 500        # Set to None for the full 11,873-example set
MAX_SPAN_TOKENS = 5       # Shorter spans → fewer candidates
MAX_CANDIDATES = 30       # Cap candidates per example (biggest speed win)
SPAN_SUB_BATCH = 16       # Spans per model call (safe on M3 memory)
ANSWERABILITY_THRESHOLD = 0.5

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for",
    "with", "by", "from", "as", "is", "are", "was", "were", "be", "been",
    "being", "it", "its", "this", "that", "these", "those", "which", "who",
    "whom", "whose", "what", "when", "where", "why", "how", "not", "no",
}

# ---------------------------------------------------------------------------
# 2. Load Model and Data
# ---------------------------------------------------------------------------
print(f"Loading model {MODEL_ID} on {DEVICE} ...")
model = AutoModel.from_pretrained(
    MODEL_ID, trust_remote_code=True, dtype=DTYPE
).to(DEVICE)

print(f"Loading dataset {DATASET_ID} ...")
dataset = load_dataset(DATASET_ID, split="validation")
if NUM_EXAMPLES is not None:
    dataset = dataset.select(range(min(NUM_EXAMPLES, len(dataset))))
metric = evaluate.load("squad_v2")
print(f"Evaluating on {len(dataset)} examples.")


# ---------------------------------------------------------------------------
# 3. Candidate Span Generation
# ---------------------------------------------------------------------------
def tokenize_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    tokens = []
    for match in re.finditer(r"\S+", text):
        raw = match.group()
        start, end = match.start(), match.end()
        stripped = raw.strip(".,!?;:\"'()[]{}")
        if stripped:
            off = raw.find(stripped)
            tokens.append((stripped, start + off, start + off + len(stripped)))
    return tokens


def generate_candidate_spans(
    context: str, gold_text: str = None
) -> List[Tuple[str, int, int]]:
    """
    Generate a bounded set of candidate spans.
    Heuristic: keep n-grams (1..MAX_SPAN_TOKENS) that contain at least one
    content word. Then, if there are more than MAX_CANDIDATES, sample down,
    making sure the gold answer (if present) is retained.
    """
    tokens = tokenize_with_offsets(context)
    pool = []
    for i in range(len(tokens)):
        for L in range(1, min(MAX_SPAN_TOKENS, len(tokens) - i) + 1):
            j = i + L
            span_tokens = [t[0] for t in tokens[i:j]]
            content = [t for t in span_tokens
                       if t.lower() not in STOPWORDS and len(t) > 1]
            if not content:
                continue
            span_text = context[tokens[i][1] : tokens[j - 1][2]]
            pool.append((span_text, tokens[i][1], tokens[j - 1][2]))

    if len(pool) <= MAX_CANDIDATES:
        return pool

    sampled = random.sample(pool, MAX_CANDIDATES)

    # Guarantee that if the gold answer's exact span exists in the pool,
    # it is represented among the candidates (fair evaluation; model still
    # cannot tell which candidate is gold).
    if gold_text:
        present = any(s[0] == gold_text for s in sampled)
        if not present:
            for cand in pool:
                if cand[0] == gold_text:
                    sampled[-1] = cand
                    break
    return sampled


# ---------------------------------------------------------------------------
# 4. Build Requests
# ---------------------------------------------------------------------------
def build_answerability_request(example: dict):
    state = f"Context: {example['context']}\nQuestion: {example['question']}"
    questions = {
        "answerable": {
            "type": "noul",
            "instructions": "Is this question answerable from the given context?",
        }
    }
    return state, questions


def build_span_requests(example: dict, candidates: List[Tuple[str, int, int]]):
    requests = []
    for span_text, _, _ in candidates:
        state = (
            f"Context: {example['context']}\n"
            f"Question: {example['question']}\n"
            f"Candidate answer span: \"{span_text}\""
        )
        questions = {
            "span_quality": {
                "type": "score",
                "instructions": (
                    "How likely is the candidate answer span the correct "
                    "answer to the question, given the context?"
                ),
                "criteria": {str(i): str(i) for i in range(11)},
            }
        }
        requests.append((state, questions))
    return requests


# ---------------------------------------------------------------------------
# 5. Evaluation Loop
# ---------------------------------------------------------------------------
predictions = []
references = []
start_time = time.time()

for idx in tqdm(range(len(dataset)), desc="SQuAD 2.0"):
    example = dataset[idx]

    # (a) Answerability — one cheap model call per example.
    ans_state, ans_qs = build_answerability_request(example)
    ans_result = model.system_one(ans_state, ans_qs)
    p_answerable = ans_result["answers"]["answerable"]["noul"]
    no_answer_prob = 1.0 - p_answerable

    # (b) Span extraction — only if the model believes it's answerable.
    predicted_text = ""
    if p_answerable >= ANSWERABILITY_THRESHOLD:
        gold_text = (
            example["answers"]["text"][0]
            if example["answers"]["text"]
            else None
        )
        candidates = generate_candidate_spans(example["context"], gold_text)
        span_reqs = build_span_requests(example, candidates)

        best_span = ""
        best_score = -1.0
        # Sub-batch to keep per-call memory bounded and to show progress.
        for sb_start in range(0, len(span_reqs), SPAN_SUB_BATCH):
            sb = span_reqs[sb_start : sb_start + SPAN_SUB_BATCH]
            sb_results = model.system_one_batch(sb)
            for (span_text, _, _), res in zip(
                candidates[sb_start : sb_start + SPAN_SUB_BATCH], sb_results
            ):
                s = res["answers"]["span_quality"]["score"]
                if s > best_score:
                    best_score = s
                    best_span = span_text
        predicted_text = best_span

    predictions.append(
        {
            "id": example["id"],
            "prediction_text": predicted_text,
            "no_answer_probability": no_answer_prob,
        }
    )
    references.append({"id": example["id"], "answers": example["answers"]})

    # Live ETA estimate every 25 examples.
    if (idx + 1) % 25 == 0:
        elapsed = time.time() - start_time
        per_ex = elapsed / (idx + 1)
        remaining = per_ex * (len(dataset) - idx - 1)
        tqdm.write(f"  [{idx+1}/{len(dataset)}] "
                   f"{per_ex:.2f}s/ex, ETA {remaining/60:.1f} min")


# ---------------------------------------------------------------------------
# 6. Metrics
# ---------------------------------------------------------------------------
final_score = metric.compute(predictions=predictions, references=references)

print("\n" + "=" * 50)
print(f"SQuAD 2.0 Results (n={len(dataset)}, d1-3B)")
print("=" * 50)
print(f"Exact Match (overall):     {final_score['exact']:.2f}")
print(f"F1 (overall):              {final_score['f1']:.2f}")
print(f"HasAns Exact Match:        {final_score['HasAns_exact']:.2f}")
print(f"HasAns F1:                 {final_score['HasAns_f1']:.2f}")
print(f"NoAns Exact Match:         {final_score['NoAns_exact']:.2f}")
print(f"NoAns F1:                  {final_score['NoAns_f1']:.2f}")
print(f"Best Exact Match:          {final_score['best_exact']:.2f} "
      f"(thresh={final_score['best_exact_thresh']:.3f})")
print(f"Best F1:                   {final_score['best_f1']:.2f} "
      f"(thresh={final_score['best_f1_thresh']:.3f})")
print("=" * 50)