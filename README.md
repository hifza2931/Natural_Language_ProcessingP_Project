# Specialist Self-Refine: A Three-Model Pipeline with Dual-Feedback Ranking for Improved Mathematical Reasoning in LLMs

> **Samar Iqbal · Hifza Umer · Shirin Sasna**
> Department of Artificial Intelligence & Data Science
> National University of Computer & Emerging Sciences, Islamabad, Pakistan

---

##  Overview

Large Language Models often produce sub-optimal first-attempt outputs — especially on multi-step mathematical reasoning tasks. The original **SELF-REFINE** framework (Madaan et al., NeurIPS 2023) mitigates this by looping a single model through generate → feedback → refine. However, when smaller open-source models are used, this single-model loop stalls: the same model that makes an arithmetic error is rarely capable of identifying its own mistake.

**Specialist Self-Refine** breaks this bottleneck by:
1. **Decoupling** the three pipeline stages and assigning each to a model best suited to that role.
2. **Introducing dual-feedback ranking** — sampling two independent critiques at different decoding temperatures and selecting the more actionable one before refinement.

On a controlled 10-problem GSM-8k evaluation, our pipeline improves solve rate from **60% → 100%** across 3 iterations, compared to **60% → 80%** for the single-model baseline.

---

##  Pipeline Architecture

```
Math Problem (GSM-8k)
        │
        ▼
┌───────────────────────┐
│  Model 1 — Generator  │  llama-3.1-8b-instant
│  generate solution()  │  (fast, reliable Python synthesis)
└──────────┬────────────┘
           │  initial Python solution y₀
           ▼
    ┌─────────────┐
    │ Correct?    │──── Yes ──▶ Final Answer ✅
    └──────┬──────┘
           │ No
           ▼
┌──────────────────────────────────────────┐
│        Model 2 — Critic & Ranker         │  llama-3.3-70b-versatile
│                                          │
│  Feedback A (T=0.0) — deterministic      │
│  Feedback B (T=0.4) — exploratory        │
│  → RANK by: accuracy · specificity ·     │
│             actionability                │
└──────────────────┬───────────────────────┘
                   │  winning feedback fb*
                   ▼
┌──────────────────────────────────────────┐
│        Model 3 — Refiner                 │  llama-4-scout-17b-16e-instruct
│  rewrite solution() using fb*            │  (MoE, strong multi-step reasoning)
└──────────────────┬───────────────────────┘
                   │  refined solution y_{t+1}
                   ▼
         Loop until correct or T=4
```


##  How It Differs from Original SELF-REFINE

| Feature | Original SELF-REFINE | **Specialist Self-Refine (Ours)** |
|---|---|---|
| Model roles | Single model for all 3 stages | 3 specialized models |
| Feedback samples per iteration | 1 | 2 (dual-temperature sampling) |
| Feedback selection | N/A | Explicit ranking call |
| Models used | GPT-3.5 / GPT-4 | Open-source via Groq API |
| Rate limit handling | N/A | Distributed across 3 models (~3× budget) |
| Training required | None | None (inference-time only) |

---

##  Key Contributions

1. **Faithful reproduction** of SELF-REFINE on 6 of 7 original tasks using `llama-3.1-8b-instant` via the Groq API — confirming qualitative claims hold under reduced model capability.

2. **Specialist Self-Refine** — a three-model pipeline where generation, feedback, and refinement are each assigned to a purpose-fit LLM.

3. **Dual-Feedback Ranking Mechanism** — the critic generates two feedbacks at `T=0.0` (deterministic) and `T=0.4` (exploratory), then ranks them on accuracy, specificity, and actionability before passing to the refiner.

4. **+40 percentage point improvement** on GSM-8k solve rate over the single-model baseline (60% → 100% vs. 60% → 80%).

5. **Released prompts, model assignments, and per-iteration accuracy trajectories** fully reproducible on the Groq free tier.

---

##  Results

### Per-Iteration Solve Rate on GSM-8k

| Iteration | Paper (GPT-3.5, n=1319) | Groq Baseline (8B) | **Ours (Specialist)** |
|:---------:|:-----------------------:|:------------------:|:---------------------:|
| 0 (initial) | 71.3% | 60.0% | 60.0% |
| 1 | 73.4% | 70.0% | **90.0%** |
| 2 | 75.1% | 70.0% | **90.0%** |
| 3 | 75.7% | 80.0% | **100.0%** |
| 4 (final) | 76.2% | 80.0% | **100.0%** |
| **Δ (0→4)** | +4.9% | +20.0% | **+40.0%** |

> Both Groq runs start from identical iteration-0 accuracy (same Model 1), **isolating the effect of the refinement mechanism**.

### Reproduction of SELF-REFINE on 6 Tasks (llama-3.1-8b vs. GPT-3.5/4)

| Task | Paper | Reproduced |
|------|-------|------------|
| Sentiment Reversal (pref.) | 84.7% | ~52% |
| Acronym Generation (pref.) | 23.5 | ~22.2 |
| Code Readability (var. ratio) | 51.3% | ~48% |
| Code Optimization (% optimized) | 15.6% | ~100% |
| Math Reasoning (solve rate) | 76.2% | ~80% |
| Constrained Generation (coverage) | 22.5 | ~19.8 |

### Feedback Ranking Statistics

| Feedback Variant | Temperature | Win Rate |
|------------------|-------------|----------|
| Feedback A — Deterministic | T = 0.0 | **67% (4/6)** |
| Feedback B — Exploratory | T = 0.4 | 33% (2/6) |

> The 33% win rate for Feedback B confirms the ranking step performs **non-trivial work** — defaulting to T=0.0 alone would have made sub-optimal choices in 2 of 6 refinement calls.

---

##  Method Details

### Role-Specialised Model Assignment

| Stage | Model | Rationale |
|-------|-------|-----------|
| **Generation** | `llama-3.1-8b-instant` | Fast, reliable `solution()` synthesis |
| **Feedback & Ranking** | `llama-3.3-70b-versatile` | Larger analytical model — better at pinpointing arithmetic errors |
| **Refinement** | `llama-4-scout-17b-16e-instruct` | MoE architecture with strong multi-step reasoning for faithful rewrites |

### Dual-Feedback Ranking (Formal)

At each iteration k:

```
fb*_k  =  RANK( M₂(yₖ, x; T=0.0),  M₂(yₖ, x; T=0.4) )
y_{k+1} =  M₃( yₖ, x, fb*_k )
```

Ranking criteria: **(i) accuracy of error identification · (ii) specificity of explanation · (iii) actionability for downstream rewriter**

### Stopping Criterion

The loop terminates when:
- Generated `solution()` returns a value within **1% relative error** of ground truth, OR
- Maximum iterations **T = 4** is reached.

---

## ⚙️ Implementation Details

### Hardware & Backend

| Component | Detail |
|-----------|--------|
| Runtime | Google Colab (free-tier T4 GPU) |
| LLM Inference | Groq Cloud API (LPU hardware, 300–500 tok/s) |
| GPU Usage | Not used directly — pure inference pipeline |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| GSM-8k problems evaluated | N = 10 |
| Maximum iterations | T = 4 |
| Correctness tolerance | \|pred − gt\| / \|gt\| < 0.01 |
| Generation temperature (M1) | 0.2 |
| Feedback A temperature (M2) | 0.0 |
| Feedback B temperature (M2) | 0.4 |
| Refinement temperature (M3) | 0.2 |
| Few-shot examples | k = 3 (chain-of-thought) |
| Rate limit retry policy | Exponential backoff, max 4 retries |
| Inter-call sleep | 1–2 seconds |

### Engineering Robustness

- **HTTP 429 handling**: Exponential backoff with jitter `(2ⁿ + U(0,1)s)` up to 4 retries.
- **Code extraction**: Multi-pattern routine strips Markdown fences and conversational preamble before executing `solution()`.
- **Non-executable code**: Automatically marked incorrect.
- **Rate limit benefit**: Calls distributed across 3 models, roughly **tripling the available request budget** vs. single-model setups.

---

##  Dataset

**GSM-8k** (Grade School Math 8K) — Cobbe et al., 2021
- Linguistically diverse grade-school math word problems
- Models produce a Python `solution()` function returning the numeric answer
- Problems loaded from the official SELF-REFINE repository
- Ground-truth parsed from `target` field with support for integers, floats, comma-formatted numbers, and `####` delimiters

---

## Reproducing This Work

All experiments are fully reproducible on the **Groq free tier**.

### Requirements

```bash
pip install groq datasets
```

### Models (via Groq API)

```python
GENERATOR_MODEL  = "llama-3.1-8b-instant"
CRITIC_MODEL     = "llama-3.3-70b-versatile"
REFINER_MODEL    = "llama-4-scout-17b-16e-instruct"
```

### Pipeline Pseudocode

```python
y0 = model1.generate(problem)                        # Initial solution

for t in range(MAX_ITER):
    if is_correct(y_t, ground_truth):
        break

    fb_A = model2.critique(y_t, problem, temperature=0.0)   # Deterministic
    fb_B = model2.critique(y_t, problem, temperature=0.4)   # Exploratory

    if fb_A == fb_B or fb_B is None:
        fb_star = fb_A
    else:
        fb_star = model2.rank(fb_A, fb_B)                   # Select best

    y_t1 = model3.refine(y_t, problem, fb_star)             # Refined solution
```

---

##  Limitations

- **Sample size**: Evaluation on 10 problems only (free-tier rate limits). Results are sensitive to single-problem outcomes.
- **Ceiling effect**: 100% solve rate likely partially explained by subset difficulty — not guaranteed to generalise at scale.
- **Specialisation vs. scale**: Cannot cleanly separate benefit of role specialisation from raw model scale without a single-70B ablation baseline.
- **Single failure mode**: When both feedbacks miss the root error, the pipeline has no escalation mechanism (observed on "Josh flipping a house" problem).
- **Prompt sensitivity**: Small formatting changes can alter outcomes, as in the original paper.

---

## 🔭 Future Work

| Direction | Description |
|-----------|-------------|
| Full-benchmark evaluation | Replicate on all 1,319 GSM-8k test problems with multiple seeds |
| Single-70B ablation | Evaluate `llama-3.3-70b-versatile` in all 3 roles to isolate specialisation benefit from scale |
| More feedback samples | Extend dual-feedback to k > 2 and analyse marginal gains |
| Adaptive stopping | Confidence-based or reward-model stopping to save iterations on easy problems |
| Cross-task generalisation | Apply specialist pipeline to all 7 SELF-REFINE tasks, especially language-heavy ones |

---

##  Citation

If you find this work useful, please cite:

```bibtex
@article{iqbal2024specialistSR,
  title     = {Specialist Self-Refine: A Three-Model Pipeline with Dual-Feedback
               Ranking for Improved Mathematical Reasoning in Large Language Models},
  author    = {Iqbal, Samar and Umer, Hifza and Sasna, Shirin},
  year      = {2024},
  institution = {National University of Computer and Emerging Sciences, Islamabad}
}
```

---

## Acknowledgements

This work builds upon and extends:
- **SELF-REFINE** — Madaan et al., NeurIPS 2023 ([arXiv:2303.17651](https://arxiv.org/abs/2303.17651))
- **GSM-8k** — Cobbe et al. ([arXiv:2110.14168](https://arxiv.org/abs/2110.14168))
- **Groq Cloud API** — for enabling free-tier LLM inference at production-grade throughput

---

<p align="center">
  <i>Department of AI & Data Science · FAST-NUCES Islamabad · 2026</i>
</p>
