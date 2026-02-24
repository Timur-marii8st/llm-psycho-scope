# PsychoScope 🧠

> **Do Language Models Build Implicit Psychological Models of Speakers?**  
> Evidence from Sparse Autoencoder Latents

[![arXiv](https://img.shields.io/badge/arXiv-TODO-b31b1b.svg)](https://arxiv.org/abs/TODO)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF.svg)](https://kaggle.com/TODO)

**Timur Sabitov** · Syurai AI · [syurai.ai](https://syurai.pages.dev)

---

## Overview

When a language model reads *"I have to check the kettle three times — what if the sensor fails?"*, does it internally represent the speaker's anxiety — even without the word "anxiety" ever appearing?

This paper tests the **Implicit Psychological Modeling (IPM) hypothesis**: LLMs performing next-token prediction develop internal representations of speaker psychological states detectable via Sparse Autoencoders (SAEs), even without explicit trait keywords.

**Key results:**
- All 6 personality traits (Big Five + Narcissism) show significant implicit activation — **p < 0.001** after Bonferroni correction
- Effect sizes **d = 0.68–2.39** (Cohen's d)
- Neuroticism latent generalises across **20/20 topics**
- Causal verification via **activation steering** confirms functional involvement

---

## Repository Structure

```
psycho-scope/
│
├── 📓 psycho-scope-dataset-gen.ipynb   # Dataset generation via Gemini 3 Flash
├── 📓 psycho-scope.ipynb               # Main experiment: SAE analysis + steering
├── 📓 validate-blind-psycho-scope.ipynb # Blind validation study (κ = 1.00)
│
├── 📁 data/
│   └── ...                             # Dataset CSVs
│
└── README.md
```

---

## Method in 60 seconds

```
Dataset (1,080 texts)                    SAE Analysis
┌─────────────────────────┐             ┌──────────────────────────────┐
│  A_explicit             │  discover   │  Contrastive latent discovery │
│  "He anxiously checks   │ ──────────► │  Δᵢ(t) = mean_A(t) - mean_A  │
│   the kettle..."        │             │              (other traits)   │
├─────────────────────────┤             └──────────────────────────────┘
│  B_implicit  ← TEST     │  measure              │
│  "I check 3 times,      │ ──────────────────────▼
│   what if it fails?"    │             ┌──────────────────────────────┐
├─────────────────────────┤             │  Hypothesis test: B > C?     │
│  C_baseline  ← CONTROL  │             │  Mann-Whitney U, Bonferroni  │
│  "I just pour and enjoy"│             └──────────────────────────────┘
└─────────────────────────┘                        │
  No trait keywords in B or C                      ▼
                                         Causal verification
                                         via decoder steering
```

---

## Results

| Trait | Latent | Key tokens | B mean | C mean | Ratio | Cohen's d | p (Bonf.) |
|-------|--------|-----------|--------|--------|-------|-----------|-----------|
| Neuroticism | 745 | *anguish, sadness, sobbing* | 209.5 | 19.2 | 10.9× | **2.39** | 7.5e-19 *** |
| Conscientiousness | 72 | *meticulously, painstakingly* | 90.0 | 4.1 | 22.1× | 1.65 | 6.4e-19 *** |
| Extraversion | 23505 | *laughter, playful, cheerful* | 47.1 | 0.6 | 76.4× | 1.53 | 1.4e-19 *** |
| Agreeableness | 273 | *compassion, kindness, empathy* | 107.3 | 18.5 | 5.8× | 1.12 | 4.3e-10 *** |
| Openness | 3415 | *artists, artworks, artistic* | 46.3 | 0.1 | 884× | 1.19 | 2.6e-15 *** |
| Narcissism | 2067 | *arrogant, arrogance* | 52.5 | 19.1 | 2.8× | 0.68 | 1.8e-06 *** |

The Neuroticism latent (745) shows ratio > 2× on **all 20/20 tested topics** — domestic, social, work, emotional, and planning contexts.

---

## Quickstart

### Requirements

```bash
pip install torch transformers safetensors huggingface_hub
pip install polars scipy scikit-learn matplotlib
```

You will also need:
- A [Hugging Face](https://huggingface.co) account with access to `google/gemma-3-4b-pt`
- GPU with ≥ 16GB VRAM (A100/V100 recommended; tested on Kaggle P100)

### 1. Generate the dataset

Open `psycho-scope-dataset-gen.ipynb`. You will need a Gemini API key (via OpenRouter or Google AI Studio).

```python
# The notebook generates texts across:
# 6 traits × 20 topics × 3 conditions × 3 reps = 1,080 texts
```

Or download the pre-generated dataset directly:

```python
import polars as pl
df = pl.read_csv("data/psych_trait_dataset_v2_clean.csv")
df = df.filter(pl.col("validation_passed") == True)
print(df.shape)  # (918, ...)
```

### 2. Run the main experiment

Open `psycho-scope.ipynb`. The notebook covers:

1. **Model + SAE loading** — Gemma 3 4B PT + Gemma Scope 2 (layer 22, 65k width)
2. **Activation extraction** — forward hooks on `model.model.language_model.layers[22]`
3. **Contrastive latent discovery** — isolates trait-specific latents vs. naive top-K
4. **Hypothesis testing** — Mann-Whitney U, effect sizes, Bonferroni correction
5. **Token-level heatmaps** — HTML visualisation of per-token activations
6. **Causal steering** — decoder vector injection during generation

### 3. Blind validation

Open `validate-blind-psycho-scope.ipynb` to reproduce the independent classifier study (accuracy = 100%, κ = 1.00 on 30 held-out B-condition texts).

---

## Data

The dataset is available on Kaggle:  
📦 [`marii8st/psycho-scope-dataset-traits`](https://kaggle.com/datasets/marii8st/psycho-scope-dataset-traits)

| File | Description |
|------|-------------|
| `psych_trait_dataset_v2_clean.csv` | Main dataset (1,080 texts, validated) |
| `activations_v2.parquet` | Pre-computed SAE activations (generated by notebook) |
| `statistical_results.csv` | Hypothesis test results |
| `steering_results_v2.csv` | Full steering experiment outputs |

**Dataset schema:**

| Column | Description |
|--------|-------------|
| `trait` | Personality trait (Neuroticism, Conscientiousness, ...) |
| `data_type` | A_explicit / B_implicit / C_baseline |
| `pole` | High / Low pole label |
| `topic` | Everyday situation (20 topics) |
| `rep` | Repetition index (0–2) |
| `text` | Generated text |
| `validation_passed` | Boolean — passed keyword + language checks |

---

## Key Implementation Details

### Contrastive Latent Discovery

Naive top-K selection by mean activation recovers **generic** high-magnitude latents (966, 839, 1263) shared across **all 6 traits** — they encode formal English register, not psychological traits. Our fix:

```python
# For each target trait t:
delta_i = mean_acts(A_explicit, trait=t) - mean_acts(A_explicit, trait≠t)
top_k_latents = delta_i.topk(K).indices
```

This reliably recovers semantically coherent trait-specific latents.

### Activation Steering

```python
def make_steering_hook(decoder_vec, coeff):
    def hook(mod, inp, out):
        if isinstance(out, tuple):          # prefill phase
            h = out[0]
            h = h + coeff * h.norm(dim=-1, keepdim=True) * decoder_vec
            return (h,) + out[1:]
        else:                               # decode phase (KV-cache)
            return out + coeff * out.norm(dim=-1, keepdim=True) * decoder_vec
    return hook
```

Note: handling both tuple and tensor outputs is required for Gemma 3's KV-cache behaviour during generation.

---

## Model & SAE Details

| Component | Value |
|-----------|-------|
| Base model | `google/gemma-3-4b-pt` |
| SAE | Gemma Scope 2, resid_post, layer 22 |
| SAE width | 65,536 features |
| SAE type | JumpReLU |
| d_model | 2,560 |
| Analysis layer | 22 / 34 (global attention block) |

**Why pre-trained (PT) not instruction-tuned (IT)?**  
Instruction tuning suppresses authorial style in favour of assistant persona — attenuating the psychological signal we aim to detect. PT models are the standard substrate for SAE interpretability research.

---

## Citation

If you use this work, please cite:

```bibtex
@article{sabitov2026psychoscope,
  title     = {Do Language Models Build Implicit Psychological Models of Speakers?
               Evidence from Sparse Autoencoder Latents},
  author    = {Sabitov, Timur},
  journal   = {arXiv preprint arXiv:TODO},
  year      = {2026},
  url       = {https://arxiv.org/abs/TODO}
}
```

---

## Related Work

- [Gemma Scope 2](https://deepmind.google/blog/gemma-scope-2) — SAEs used in this work
- [Onysk & Huys (2025)](https://arxiv.org/abs/2502.09487) — Supervised SAEs for depression detection
- [Rimsky et al. (2024)](https://aclanthology.org/2024.acl-long.828) — Contrastive activation addition
- [Templeton et al. (2024)](https://transformer-circuits.pub/2024/scaling-monosemanticity/) — Scaling monosemanticity

---

## License

MIT — see [LICENSE](LICENSE)

---

*Syurai Lab · [syurai.ai](https://syurai.pages.dev)*
