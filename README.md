# Titanic Survivor Classifier

A reproducible, end-to-end **data science mini pipeline** built in Python to predict passenger survival on the Titanic dataset using a **logistic regression classifier**.  
The project focuses on clean pipeline design, reproducibility, and evaluation rather than model complexity.

---

## 🔍 Project Overview

This project demonstrates how a typical data science workflow is structured in an industry setting. It emphasizes building a reliable and maintainable pipeline instead of experimenting inside notebooks.

The pipeline covers data loading, preprocessing, model training, evaluation, and artifact generation, all executed through a **CLI-driven workflow**.

---

## ⚙️ Pipeline Workflow

The pipeline executes the following steps:

1. Load and validate the input dataset  
2. Perform data preprocessing and feature preparation  
3. Train a logistic regression classifier  
4. Evaluate model performance using standard metrics  
5. Save artifacts for reproducibility and inspection  

### Generated Artifacts

All outputs are saved in a timestamped directory under `outputs/`:

- Trained model (`.joblib`)
- Evaluation metrics (`.json`)
- Predictions (`.csv`)
- Confusion matrix (`.png`)

---

## 🚀 Quick Start

Install dependencies and run the pipeline:

```bash
pip install -r requirements.txt
python run_pipeline.py --data-path data/titanic_sample.csv --output-dir outputs

