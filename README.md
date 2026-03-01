# NLP Project - Extractive Question Answering (SQuAD v2)

This repository contains a full QA pipeline built in multiple iterations:
- from-scratch models (`v1` to `v5`) on SQuAD v2,
- dedicated evaluation and comparison notebooks,
- a BERT fine-tuning notebook for stronger performance.

The project is optimized for Apple Silicon with MPS support:
```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

## 1) Project Goals

- Build and improve extractive QA models over time.
- Compare model versions with consistent EM/F1 metrics.
- Understand the impact of preprocessing, no-answer calibration, and decoding strategy.
- Provide reproducible notebooks for training, evaluation, and presentation.

## 2) Repository Structure

- `deep_NLP_project.ipynb`: original pipeline (baseline + improved model).
- `deep_NLP_project_v3.ipynb`: main from-scratch advanced pipeline (v3/v4/v5 sections).
- `deep_NLP_evaluation_v3.ipynb`: evaluation/tuning notebook for v3/v4 decoding and threshold search.
- `deep_NLP_compare_models.ipynb`: unified comparison table across model versions.
- `deep_NLP_bert_v1.ipynb`: BERT QA fine-tuning on SQuAD v2.
- `models.py`: shared model definitions (`QAModelV1`, `QAModelV2`, `QAModelV3`, `QAModelV5`).
- `data/train-v2.0.json`, `data/dev-v2.0.json`: SQuAD v2 dataset files.
- `*.pt`: model checkpoints.
- `*_config.json`: inference/calibration settings (for optimized versions).

## 3) Data

Dataset: **SQuAD v2.0**

Task format:
- Inputs: `question` + `context`.
- Output: answer span (`start`, `end`) in context, or **no-answer**.

Why SQuAD v2 is harder:
- It includes unanswerable questions (`is_impossible=True`).
- A QA system must learn both span extraction and no-answer detection.

## 4) Preprocessing Pipeline

### 4.1 Tokenization
- Baseline pipeline: lowercase + whitespace split.
- Improved pipeline: regex tokenization to better handle punctuation/numbers.

### 4.2 Span Alignment
- Ground-truth answers are provided at character level.
- We map char offsets to token indices `(start, end)`.
- If alignment is unreliable, the sample is skipped.

### 4.3 No-answer Encoding
- For unanswerable examples:
  - `has_answer = 0`
  - sentinel span `(0, 0)` with `<CLS>` convention in advanced models.

### 4.4 Vocabulary and Numerical Encoding
- Vocabulary built from train split with frequency threshold.
- Special tokens:
  - `<PAD>`
  - `<UNK>`
  - `<CLS>` (advanced models)
- Sequence shaping:
  - truncation + padding to fixed lengths (`MAX_CONTEXT_LEN`, `MAX_QUESTION_LEN`).

## 5) Model Evolution

### v1
- BiLSTM context/question encoder.
- Simple span heads.
- Baseline performance.

### v2
- Adds attention and stronger fusion.
- Better span quality than v1.

### v3
- Stronger architecture + no-answer head.
- Better handling of SQuAD v2 no-answer cases.

### v4
- Same core weights as v3, but optimized inference:
  - no-answer threshold calibration,
  - top-k span decoding,
  - length penalty.
- Significant EM/F1 gains without full retraining.

### v5
- Deeper from-scratch architecture (stronger modeling stack) with v4-style optimized inference.
- Trained with early stopping based on dev behavior.

### BERT v1
- `bert-base-uncased` fine-tuning for QA.
- Intended to push performance beyond from-scratch limits.

## 6) Main Notebooks and Usage

### A) Train from-scratch advanced model
Notebook: `deep_NLP_project_v3.ipynb`

Typical order:
1. Run imports/config/preprocessing sections.
2. Train v3.
3. Run v4 sections (optimized decoding/calibration).
4. Run v5 sections (if desired).
5. Export checkpoints/configs.

### B) Tune v3/v4 inference quickly
Notebook: `deep_NLP_evaluation_v3.ipynb`

Purpose:
- Sweep no-answer threshold.
- Sweep decoding params (`top-k`, max span length, length penalty).
- Produce baseline vs optimized table.

### C) Compare all model versions
Notebook: `deep_NLP_compare_models.ipynb`

Purpose:
- Evaluate available checkpoints (v1/v2/v3/v4/v5/BERT if present).
- Print one final EM/F1 comparison table.

### D) Fine-tune BERT
Notebook: `deep_NLP_bert_v1.ipynb`

Purpose:
- End-to-end Hugging Face QA fine-tuning and evaluation on SQuAD v2.
- Save best BERT model to `bert_qa_best/`.

## 7) Environment Setup

From project root:
```bash
cd /Users/albanecoiffe/Downloads/NLP
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch numpy
pip install transformers datasets evaluate accelerate sentencepiece
```

If using notebooks, ensure the active kernel points to:
- `/Users/albanecoiffe/Downloads/NLP/.venv/bin/python`

## 8) Running on Apple Silicon (MPS)

All major notebooks use:
```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

Notes:
- If MPS is unavailable, code falls back to CPU.
- Training speed and memory constraints may vary by model size and batch size.

## 9) Example Results Snapshot

Recent comparison (example run):

| Model | N evaluated | EM (%) | F1 (%) |
|---|---:|---:|---:|
| v1 | 2000 | 2.40 | 7.97 |
| v2 improved | 2000 | 10.55 | 18.73 |
| v3 | 1998 | 37.09 | 39.16 |
| v4 | 1998 | 49.25 | 49.70 |

Interpretation:
- v4 gains mainly come from inference calibration and decoding, not only architecture changes.
- from-scratch improvements are substantial, but BERT is expected to go higher.

## 10) Troubleshooting

### `ModuleNotFoundError: No module named 'datasets'`
Install dependencies in `.venv`:
```bash
.venv/bin/pip install datasets evaluate transformers accelerate sentencepiece
```

### `Trainer.__init__() got an unexpected keyword argument 'tokenizer'`
With `transformers 5.x`, use:
- `processing_class=tokenizer`
instead of:
- `tokenizer=tokenizer`

### `ImportError: cannot import name 'QAModelV5' from models`
Jupyter kernel cache issue. Fix:
1. Restart kernel, or
2. Reload module:
```python
import importlib, models
importlib.reload(models)
from models import QAModelV5
```

### Model comparison notebook runs but no final table
- Make sure final evaluation cell executed.
- Confirm checkpoints exist (`*.pt`, `bert_qa_best/`).

## 11) Reproducibility Tips

- Keep train/dev subsets explicit in notebook config.
- Keep one source of truth for model classes (`models.py`).
- Save inference configs (`*_config.json`) with checkpoints.
- Compare models with consistent sample size and filtering rules.

## 12) Git / Sharing

Initialize and push:
```bash
cd /Users/albanecoiffe/Downloads/NLP
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

Current `.gitignore` excludes large artifacts (including `*.pt`) by default.

## 13) Next Steps

- Integrate BERT metrics in final comparison table after full fine-tuning run.
- Add strict same-sample comparison mode across all models.
- Add slide-ready auto-report (tables + key plots) exported from notebook outputs.
