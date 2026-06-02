# ThaiFACTUAL 🇹🇭

**Debiasing Large Language Models in Thai Political Stance Detection via Counterfactual Calibration**

> Accepted at **EMNLP 2025 – Widening NLP (WiNLP) Workshop**, Suzhou, China  
> Author: **Teerapong Panboonyuen** (aka Kao Panboonyuen)  
> Affiliation: Chulalongkorn University & MARSAIL  

---

## 📄 Paper Abstract

Political stance detection in low-resource and culturally complex settings poses a critical challenge for large language models (LLMs). In the Thai political landscape—rich with indirect expressions, polarized figures, and sentiment-stance entanglement—LLMs often exhibit systematic biases, including sentiment leakage and entity favoritism.

We introduce **ThaiFACTUAL**, a lightweight, model-agnostic calibration framework that mitigates political bias **without fine-tuning LLMs**.

---

## 🗂️ Repository Structure

```
ThaiFACTUAL/
│
├── data/
│   ├── dataset_builder.py          # §A: Thai Political Stance Dataset Construction
│   ├── annotator_agreement.py      # §A.2: Fleiss' κ inter-annotator agreement
│   └── sample_dataset.json         # Sample data (270 tweets, 3 entities)
│
├── models/
│   ├── bias_measurement.py         # §3.1: RStd bias metric
│   ├── llm_inference.py            # §3: GPT-4 / LLaMA-3 inference wrapper
│   ├── counterfactual_builder.py   # §B: Counterfactual Construction Process
│   └── thaifactual_calibrator.py   # §4: ThaiFACTUAL Calibration Framework (CORE)
│
├── evaluation/
│   ├── metrics.py                  # §3.3: F1, OOD, Bias-SSC, RStd evaluation
│   ├── fairness_eval.py            # §3.3: Entity-level fairness analysis
│   └── ood_eval.py                 # §3.3: Out-of-distribution generalization
│
├── utils/
│   ├── prompts.py                  # Prompt templates (CoT, Debias, ThaiFACTUAL)
│   ├── thai_text_utils.py          # Thai language preprocessing utilities
│   └── logger.py                   # Logging and experiment tracking
│
├── scripts/
│   ├── run_baseline.sh             # Reproduce Table 1 baselines
│   ├── run_thaifactual.sh          # Reproduce Table 1 ThaiFACTUAL results
│   └── run_all.sh                  # Full pipeline reproduction
│
├── notebooks/
│   └── ThaiFACTUAL_Demo.ipynb     # Interactive demo notebook
│
├── results/                        # Auto-generated results (Table 1 reproduction)
├── figures/                        # Auto-generated figures
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API Keys
```bash
export OPENAI_API_KEY="your-openai-key"
export HUGGINGFACE_TOKEN="your-hf-token"
```

### 3. Run Full Pipeline (reproduces Table 1)
```bash
bash scripts/run_all.sh
```

### 4. Run ThaiFACTUAL Only
```bash
python models/thaifactual_calibrator.py --entity all --output results/
```

---

## 📊 Main Results (Table 1 in Paper)

| Model | Bias-SSC↓ | RStd↓ | F1↑ | OOD↑ |
|-------|-----------|-------|-----|------|
| GPT-4 (Raw) | 21.7 | 15.2 | 70.8 | 56.4 |
| GPT-4 (Debias Prompt) | 18.3 | 12.6 | 71.9 | 57.0 |
| LLaMA-3 (CoT Prompt) | 16.5 | 11.8 | 68.1 | 59.7 |
| **ThaiFACTUAL (Ours)** | **9.8** | **6.4** | **73.5** | **65.2** |

---

## 🔑 Key Concepts

- **Sentiment-Stance Entanglement**: LLMs use emotional tone as a proxy for stance (§3.1)
- **Entity Preference Bias**: Models favor/disfavor specific political actors (§3.1)
- **Counterfactual Calibration**: Swap entities, preserve sentiment, re-score (§B, §4)
- **RStd Metric**: Recall standard deviation across stance labels (§3.1, Eq. 1)

---

## 📚 Citation

```bibtex
@inproceedings{panboonyuen2025thaifactual,
  title     = {Debiasing Large Language Models in Thai Political Stance Detection via Counterfactual Calibration},
  author    = {Panboonyuen, Teerapong},
  booktitle = {Proceedings of the EMNLP 2025 Widening NLP Workshop (WiNLP)},
  year      = {2025},
  address   = {Suzhou, China}
}
```

---