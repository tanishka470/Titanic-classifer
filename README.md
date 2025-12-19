# Titanic Survivor Classifier

A lightweight ML pipeline that trains a logistic regression classifier on the Titanic dataset.

## Quick Start

```bash
pip install -r requirements.txt
python run_pipeline.py --data-path data/titanic_sample.csv --output-dir outputs
```

Artifacts (metrics, model, predictions, confusion matrix) land in `outputs/`.

## Test

```bash
python -m pytest tests/test_pipeline_smoke.py -v
```

## Structure

- `data/` – Sample Titanic dataset
- `src/ds_pipeline/` – Data loading, preprocessing, model, metrics
- `tests/` – Smoke test for end-to-end training
- `run_pipeline.py` – Main training script
- `requirements.txt` – Dependencies
