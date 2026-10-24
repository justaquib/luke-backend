# How Frontier Models Are Measured — Standard Techniques

> A reference for how GPT, GLM, Claude, Gemini and the open models are measured
> during and after training, and which of those techniques this app already
> uses in its small form. Read `docs/03-architecture.md` for how Luke trains
> and answers, `docs/04-training.md` for the trainer's workflow, and
> `docs/07-model-measurements.md` for this app's own measured numbers.

## 1. Signals during training

### Cross-entropy loss (next-token log loss)
The one number every frontier lab watches while training. The model reads a
sequence of tokens and is scored on how much probability it gave the token
that actually came next:

```
loss = -ln P(correct next token)
```

- 0.0 would mean it is always right; `ln(vocabulary_size)` is what pure
  guessing scores (e.g. `ln(100000)` ≈ 11.5 for a frontier tokenizer).
- Reported as an *exponentially smoothed* series, never as a single raw batch:
  a batch of a few thousand tokens is a noisy draw, and the labs plot the
  smoothed curve against a held-out sample of tokens.

### Perplexity
`e^(average loss over a text)` — the number of near-equally-likely tokens the
model hesitates between. Frontier papers use it for pretraining comparison on
fixed text corpora (The Pile, C4 equivalents), not as a product claim, because
it depends on the tokenizer: two models can publish wildly different
perplexity numbers over the same text and neither is wrong.

### Held-out evaluation, not training loss
The golden rule: the loss that matters is measured on data the model never
trained on, in `eval` mode (dropout off), averaged over many batches. A
falling training loss with a flat or rising eval loss is overfitting.
This app's checkpoint read (`neural/training.py::_check`) is exactly
this shape: a group split by response, fixed batches, resting state, averaged.

### Scaling laws
Labs publish how loss falls with model size, data and compute (Kaplan et al.
2020; Hoffmann et al. 2021, "Chinchilla"): for each extra parameter there is a
compute-optimal amount of training data (about 20 tokens per parameter in the
Chinchilla regime). The practical technique: hold small probe runs, fit the
curve, and predict the loss of the big run before paying for it. Several
passes over the data were long thought wasteful past ~4 epochs; small models
trained many epochs on curricula still keep improving — the measured behaviour
of this app's writer agrees with the small-model findings.

## 2. Benchmark suites (automatic, fixed-answer)

The standard way "how good" is reported in release notes (GPT-4 tech report,
GLM-4/4.5 technical reports, Claude/Gemini model cards, Llama release cards):

| Suite | Measures | Common form |
| --- | --- | --- |
| MMLU / MMLU-Pro | breadth of school+professional knowledge | multiple choice, accuracy |
| GPQA | graduate-level science questions | multiple choice, accuracy |
| GSM8K / MATH / AIME | maths word problems to olympiad level | exact-match accuracy |
| HumanEval / MBPP | write code from a docstring | `pass@k` — run unit tests, count passes |
| SWE-bench | fix real GitHub issues | resolved-or-not per issue |
| HellaSwag / ARC / WinoGrande | commonsense reasoning | multiple choice, accuracy |
| IFEval | can it follow explicit instructions | strict rule check of the output |
| MT-Bench / Arena-Hard | chat quality with multi-turn | 1–10 (LLM judge) |

Measurement conventions the labs standardised (and the open harness
`lm-evaluation-harness` implements, de-facto the field's ruler):

- **Fixed prompts and fixed few-shot counts** (0-shot / 5-shot / 25-shot),
  published with the number, because scores move with the prompt.
- **Scoring rules fixed in advance**: multiple choice by comparing the
  normalised log-probability of each answer option; generation tasks by exact
  match, `pass@k`, or F1.
- **Chain-of-thought allowed or noted**: "CoT" numbers are published separately
  because letting the model think aloud changes the score a lot.
- **Sample / self-consistency**: sample the same question many times and take
  the majority answer (or `pass@k` best-of) — big, honest gains on maths and
  code, and the number of samples is published.
- **Contamination checks**: run the same benchmarks into the training search —
  if a benchmark's exact questions are in the training corpus, the number is
  marked contaminated or dropped.
- **Same table, same variances**: release tables compare against the previous
  model and the leading rival, with the same harness for every model.

`pass@k` deserves the formula because it reappears everywhere:

```
pass@k = 1 - C(n-c, k) / C(n, k)   (unbiased estimator)
```

where n samples were drawn and c passed the tests; k answers are picked.

## 3. Preference and judge evaluation

Automatic benchmarks saturate (MMLU sits above 90% for the strongest models),
so quality is then measured the way people rank the models:

- **Chatbot Arena style**: two anonymised answers, the user picks the better.
  Millions of votes → each model gets an **Elo-style rating** computed with
  Bradley–Terry models of the votes. Arena Elo became the most-watched quality
  number for chat.
- **LLM-as-judge**: a strong fixed model scores outputs against a rubric
  (AlpacaEval 2, MT-Bench, Arena-Hard v1). Standards: judge temperature fixed,
  ordering randomised (position bias), style controls added because judges
  reward length and formatting, winner rates converted to a win percentage.
- **Human expert panels** for the hardest domains (safety, medicine, law),
  graded against written rubrics, reported with the numbers of graders and the
  agreement rate.

## 4. Honesty, grounding and calibration

The frontier labs measure several things beyond "answers correctly":

- **Truthfulness**: TruthfulQA (does it repeat common falsehoods?),
  FActScore (is every atomic fact of a long-form biography backed by a
  reference corpus?), and internal "hallucination rate" checks against a
  knowledge base.
- **Grounding / attributability** (the RAG setting, and what Luke's
  `_grounded` check is a small version of): citation-level faithfulness —
  every sentence of the answer must be supported by the retrieved snippet or
  document. Frontier products quote a support rate per answer.
- **Calibration**: does "I am not sure" mean something? Standard technique:
  bin the answers by the model's own confidence (its token probabilities) and
  compare with the actual accuracy — the Expected Calibration Error (ECE).
  A well-calibrated model's 80% confident answers are right ~80% of the time.
  This is what Luke's critique attaches when it says
  "I am not sure about the word …".
- **Refusal correctness**: false refusals (saying no to a fine request) and
  false compliance (answering what should be refused) are both counted against
  labelled sets, e.g. XSTest for false refusals.

## 5. Safety and behaviour evaluation

- **Red-teaming**: expert and crowd teams attack the model before release;
  findings graded by severity, and the release notes quote the policy-free
  violation rates against internal labelled sets.
- **Bias/demographic probes**: identical scenarios with only the protected
  attribute changed (BBQ, Bias Benchmark for QA), scored for parity.
- **Robustness / jailbreak suites**: published attacks re-run per new model.
- **Long-context evals**: "needle in a haystack" retrieval plus
  multi-hop document QA (LongBench, RULER) at the model's full context size,
  because quality degrades quietly with long context.

## 6. Serving-quality metrics (what users actually feel)

Reported separately from ability, because a smart model that answers slowly or
dies mid-sentence is not better:

- **Time to first token, tokens/second, end-to-end latency** at fixed hardware.
- **Uptime and rate-limit ceilings** at the offered scale.
- **Cost per million tokens** in/out, of the deployed model and of the
  distilled/smaller variants.

## 7. What gets published — the release-report convention

A frontier release (OpenAI System Card, GLM Technical Report, Claude Model
Card, Gemma/Llama release evals) is expected to contain:

1. Benchmark table: fixed suites, fixed settings, vs prior and rival models.
2. Training scale disclosures (parameters or compute), data cut-off, context
   length.
3. The eval harness and how it was run (contamination stated).
4. Safety eval results and red-team summary.
5. Known limitations, in paper form.

## 8. How this app maps to the conventions

| Frontier technique | Luke's small version |
| --- | --- |
| Held-out, averaged, eval-mode loss | `training._check` — fixed batches in resting state; shown on the brain page |
| Perplexity | the answer-token loss; lower knows your lessons better (see the brain panel) |
| Sample / best-of-k | the writer's critique-guided rewrite (`respond` writes again and keeps the better answer) |
| Repetition / IFEval-style rule checks | `quality.repeats_itself` blocks the loop answers; the grammar check refuses verb-less text |
| Grounding / attributability | `_grounded` — every word of the answer must come from the taught material, else the taught answer is used |
| Calibration ("I am not sure …") | token-confidence critique, the unsure-word note, and the loose-match flag on the chat path |
| Refusal correctness | refusing a request about something never taught instead of inventing (Area 3B refusal examples in training) |
| Contamination-free measurement | the instruction probe set is held out of training on purpose (`evaluation.py`) |
| Fixed harness, published numbers | the brain page and the training report are this app's release table |

The differences are inherent: a beam of GPUs versus one Mac. A frontier model
gives probability over 100,000 tokens; Luke gives one over the trainer's own
few thousand. What carries over unchanged is the measurement discipline:
fixed settings, averaged over many samples, on data the model was not trained
on, with the model's own uncertainty published.

## Sources

- Hoffmann, Borgeaud et al. — *Training Compute-Optimal Large Language
  Models* (DeepMind, 2022) — the Chinchilla scaling study.
- Kaplan, McCandlish et al. — *Scaling Laws for Neural Language Models*
  (OpenAI, 2020).
- Gao et al. — *A Framework for Few-Shot Language Model Evaluation*
  (EleutherAI `lm-evaluation-harness`).
- Hendrycks et al. — *Measuring Massive Multitask Language Understanding*
  (MMLU, 2021); *Detecting, Explaining, Mitigating Memorization* for
  contamination practice.
- Chen et al. — *Evaluating Large Language Models Trained on Code*
  (HumanEval and `pass@k`, 2021).
- Perez et al. — *TruthfulQA: Measuring How Models Mimic Human Falsehoods*
  (2021).
- Min et al. — *FActScore: Fine-grained Atomic Evaluation of Long-Form
  Factuality* (2023).
- Guo et al. — *Calibration of Pre-trained Transformers* (ECE, 2019).
- Ouyang, Röttger et al. — *XSTest* (false refusals, 2023).
- Zheng et al. — *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*
  (2023).
