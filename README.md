#  Sentiment Analysis on Stack Overflow using BERT & RoBERTa

> **Thesis** — Ionian University, Department of Informatics  
> **Author:** Stergios Moutzikos  
> **Supervisor:** Prof. Andreas Kanavos  
> **Date:** February 2026

---

##  Abstract

This project addresses the **extraction and analysis of sentiment from online platform texts** using modern Natural Language Processing (NLP) techniques. Two pretrained Transformer-based models are utilized:

- **BERT** (`monologg/bert-base-cased-goemotions-original`)
- **RoBERTa** (`SamLowe/roberta-base-go_emotions`)

Both models were fine-tuned on the **GoEmotions** dataset for classification of **28 distinct emotions** and applied to data from the **Stack Overflow** platform.

The evaluation framework includes agreement rates, Cohen's Kappa, McNemar's test, Pearson correlation, and independent samples t-test.

---

##  Repository Structure

```
.
├── Code.ipynb              # Main analysis notebook
├── Thesis_Moutzikos.pdf    # Full thesis document (Greek + English abstract)
├── requirements.txt        # Python dependencies
├── data/
│   ├── comments.csv        # 110,000 Stack Overflow comments
│   ├── posts_answers.csv   # 50,853 posts/answers
│   └── users.csv           # 54,836 user profiles
└── README.md
```

---

##  Dataset

Data was extracted from the **Stack Overflow BigQuery Public Dataset** via Google BigQuery. A 5% random sample was applied with a cap of 110,000 comments.

| Table | Records | Fields |
|-------|---------|--------|
| `comments.csv` | 110,000 | id, text, post_id, user_id, score |
| `posts_answers.csv` | 50,853 | id, comment_count, favorite_count, score |
| `users.csv` | 54,836 | id, display_name, age, location, reputation, up_votes, down_votes |

After preprocessing (HTML removal, URL stripping, emoji handling, deduplication), **108,687 clean comments** remained for analysis.

---

##  Models

### BERT — `monologg/bert-base-cased-goemotions-original`
- Architecture: 12 Transformer layers, 12 attention heads, 768 hidden dimensions
- Parameters: ~110M
- Pre-training: Masked Language Modeling (MLM) + Next Sentence Prediction (NSP)
- Fine-tuned on: GoEmotions (58,000 Reddit comments, 28 emotion labels)

### RoBERTa — `SamLowe/roberta-base-go_emotions`
- Architecture: 12 Transformer layers, 12 attention heads, 768 hidden dimensions
- Parameters: ~125M
- Key improvements over BERT: dynamic masking, no NSP, larger corpus (160GB), larger batch sizes
- Fine-tuned on: GoEmotions (same 28 emotion labels)

### Emotion → Sentiment Mapping

| Sentiment | Emotions |
|-----------|---------|
| **Positive** (12) | admiration, amusement, approval, caring, desire, excitement, gratitude, joy, love, optimism, pride, relief |
| **Negative** (11) | anger, annoyance, disappointment, disapproval, disgust, embarrassment, fear, grief, nervousness, remorse, sadness |
| **Neutral** (5) | confusion, curiosity, neutral, realization, surprise |

---

##  Key Results

### Model Agreement
| Metric | Value |
|--------|-------|
| Exact Emotion Agreement | **78.45%** |
| Sentiment Agreement (3-class) | **90.40%** |
| Cohen's Kappa (κ) | **0.669** — Substantial Agreement |
| McNemar p-value | < 0.0001 (statistically significant difference) |

### Sentiment Distribution
| Sentiment | RoBERTa | BERT |
|-----------|---------|------|
| Neutral | 73.7% | 74.0% |
| Positive | 20.3% | 20.3% |
| Negative | 5.9% | 5.8% |

### Confidence Scores
| Model | Mean Confidence | Std Dev |
|-------|----------------|---------|
| BERT | **93.8%** | 0.1284 |
| RoBERTa | **75.3%** | 0.1818 |

> **Interpretation:** BERT shows potential overconfidence (median confidence: 99.9%), while RoBERTa demonstrates better calibration with more realistic and varied confidence estimates.

### Top 5 Emotions (Both Models)
`neutral` › `gratitude` › `curiosity` › `confusion` › `approval`

---

##  Statistical Evaluation

| Method | Purpose | Result |
|--------|---------|--------|
| **Cohen's Kappa** | Inter-model agreement | κ = 0.669 (Substantial) |
| **McNemar's Test** | Systematic behavior difference | χ² = 11,919.92, p < 0.0001 |
| **Pearson Correlation** | Confidence vs. comment score | r ≈ 0.003 (negligible) |
| **Pearson Correlation** | BERT vs. RoBERTa confidence | r = 0.366 (moderate) |
| **Independent t-test** | Confidence score comparison | t = 274.12, p < 0.0001 |

---

##  Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the notebook
```bash
jupyter notebook Inf2021149_Emotional_Analysis.ipynb
```

>  **GPU recommended.** Running both BERT and RoBERTa over 108K samples is computationally intensive. The notebook was originally run on Kaggle with GPU acceleration. Without a GPU, inference will be significantly slower.

---

##  Requirements

```
pandas
numpy
seaborn
matplotlib
nltk
emoji
transformers
torch
scipy
statsmodels
scikit-learn
tqdm
ipywidgets
```

---

##  Methodology Overview

```
Stack Overflow (BigQuery)
        ↓
  Data Extraction (SQL + Python)
        ↓
  Preprocessing (HTML, URLs, emojis, dedup)
  108,687 clean comments
        ↓
  ┌─────────────┐     ┌──────────────┐
  │    BERT     │     │   RoBERTa    │
  │ GoEmotions  │     │  GoEmotions  │
  └──────┬──────┘     └──────┬───────┘
         └────────┬──────────┘
                  ↓
     28-Emotion Classification
                  ↓
     Sentiment Mapping (pos/neg/neu)
                  ↓
     Statistical Evaluation
     (Kappa, McNemar, Pearson, t-test)
                  ↓
     Analysis & Visualization
```

---

##  Geographic Analysis

The top contributing locations to the dataset are:

| Location | Comments |
|----------|----------|
| India | 1,539 |
| United States | 1,510 |
| Germany | 1,485 |
| London, United Kingdom | 1,410 |
| United Kingdom | 1,286 |

**Finding:** Emotion distribution is remarkably uniform across all geographic regions — `neutral` dominates everywhere, confirming the technical, task-focused nature of Stack Overflow regardless of user location.

---

##  Key Findings

1. **Technical communities skew neutral** — 74% of comments carry no strong emotional signal, reflecting the informational nature of Q&A platforms.
2. **Gratitude is the dominant positive emotion** — users frequently thank each other, making "gratitude" the second most common emotion after neutral.
3. **BERT vs. RoBERTa calibration gap** — BERT's near-perfect confidence scores (median 99.9%) suggest overconfidence. RoBERTa's wider spread (std = 0.18) indicates better-calibrated uncertainty.
4. **Model disagreements concentrate on borderline cases** — neutral↔curiosity, curiosity↔confusion, and admiration↔approval account for the majority of disagreements.
5. **No strong correlation between sentiment and popularity** — a comment's emotional tone (or confidence thereof) does not predict its upvote score (r ≈ 0).

---

##  Citation

If you use this work, please cite:

```bibtex
@thesis{moutzikos2026sentiment,
  author    = {Moutzikos, Stergios},
  title     = {Sentiment Extraction and Analysis Using BERT and RoBERTa on Texts from Online Platforms},
  school    = {Ionian University, Department of Informatics},
  year      = {2026},
  month     = {February},
  supervisor = {Kanavos, Andreas}
}
```

---

##  References

Key works referenced in this thesis:

- Devlin et al. (2018) — BERT: Pre-training of Deep Bidirectional Transformers
- Liu et al. (2019) — RoBERTa: A Robustly Optimized BERT Pretraining Approach
- Demszky et al. (2020) — GoEmotions: A Dataset of Fine-Grained Emotions
- Vaswani et al. (2017) — Attention Is All You Need
- Cohen (1960) — A Coefficient of Agreement for Nominal Scales
- McNemar (1947) — Note on the Sampling Error of the Difference Between Correlated Proportions

---

##  License

This project is submitted as an academic thesis. The code is made available for educational and research purposes. The Stack Overflow data is sourced from the [BigQuery Public Dataset](https://www.kaggle.com/datasets/stackoverflow/stackoverflow) and is subject to Stack Overflow's [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.

---

<p align="center">
  <b>Ionian University — Department of Informatics</b><br>
  Corfu, Greece · February 2026
</p>
