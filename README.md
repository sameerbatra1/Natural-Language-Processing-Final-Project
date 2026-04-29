# Jigsaw Toxic Comment Classification — NLP Final Project

**Team:** Sameer Batra · Dhruv Rai

Multi-label toxicity detection on Wikipedia talk-page comments using a fine-tuned BERT model.  
Detects 6 toxicity categories: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`.

---

## Project Structure

```
Natural-Language-Processing-Final-Project/
│
├── data/                        # Place Kaggle CSVs here (not committed)
│   ├── train.csv
│   └── test.csv
│
├── eda_outputs/                 # Auto-generated EDA plots
│
├── checkpoints/                 # Auto-generated model checkpoints
│   └── best_model.pt
│
├── eda.py                       # Exploratory data analysis
├── data_cleaner.py              # Text cleaning pipeline
├── dataset.py                   # PyTorch Dataset + DataLoader factory
├── focal_loss.py                # Class-weighted Focal Loss
├── model.py                     # JigsawBERTClassifier (BERT + linear head)
├── trainer.py                   # Training loop, metrics, checkpointing
├── train.py                     # Main training entry point
│
├── config.py                    # App constants (labels, model name, colours)
├── inference.py                 # Tokenise → infer → probabilities
├── app.py                       # Streamlit web application
│
├── requirements.txt             # Full training environment dependencies
├── requirements_app.txt         # Minimal inference/app dependencies
└── README.md
```

---

## Quickstart

### 1 — Get the data

Download from [Kaggle – Jigsaw Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) and place `train.csv` and `test.csv` inside a `data/` folder.

```
data/
├── train.csv
└── test.csv
```

### 2 — Install dependencies

**For training (full environment):**
```bash
pip install -r requirements.txt
```

**For the Streamlit app only (lighter):**
```bash
pip install -r requirements_app.txt
```

### 3 — Run EDA (optional)

```bash
python eda.py
```
Outputs 9 plots to `eda_outputs/`.

### 4 — Clean the data

```bash
python data_cleaner.py
```
Generates `data/train_clean.csv` and `data/test_clean.csv`.

### 5 — Train the model

```bash
python train.py
```
Saves the best checkpoint to `checkpoints/best_model.pt`.  
Expected training time: ~2–3 hours on a GPU, longer on CPU.

### 6 — Launch the Streamlit app

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**

> The app uses `unitary/toxic-bert` from Hugging Face by default (no local checkpoint needed).  
> To use your own fine-tuned model, change `MODEL_NAME` in `config.py` to your checkpoint path or HF repo.

---

## Model Details

| Component | Choice |
|-----------|--------|
| Base model | `bert-base-uncased` |
| Classification head | Linear(768 → 6) |
| Loss function | Focal Loss with class-imbalance alpha |
| Optimizer | AdamW, lr = 2e-5 |
| Scheduler | Linear warmup (10%) |
| Max sequence length | 256 tokens |
| Batch size | 32 |
| Epochs | 3 |
| Metric | ROC-AUC (per-label + macro mean) |

---

## Deploying to Hugging Face Spaces

1. Create a new Space → choose **Streamlit** SDK
2. Upload: `app.py`, `inference.py`, `config.py`, `requirements_app.txt`
3. The app runs without any code changes

---

## Results

| Label | AUC |
|-------|-----|
| toxic | — |
| severe_toxic | — |
| obscene | — |
| threat | — |
| insult | — |
| identity_hate | — |
| **Mean AUC** | **—** |

*(Fill in after training)*
