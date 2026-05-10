![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Assessment | Full ML Pipeline: From Clustering to Deployment

## Overview

This assessment brings together everything you've learned in Unit 4.2 — unsupervised learning, model training and evaluation, and model deployment — into a single end-to-end machine learning pipeline. You'll work with the Palmer Penguins dataset, moving from exploratory clustering through supervised classification, rigorous evaluation, and finally a deployment prototype.

The goal is to demonstrate that you can execute the complete ML workflow independently: explore unlabeled structure, build and tune a supervised model, evaluate it with professional-grade diagnostics, and serve predictions through an API.

## Learning Goals

This assessment evaluates your ability to:

- Apply unsupervised techniques (PCA, t-SNE, K-Means, DBSCAN) for exploratory analysis.
- Build a complete supervised classification pipeline with preprocessing, model selection, and hyperparameter tuning.
- Produce thorough model evaluation artifacts (confusion matrices, ROC curves, learning curves, feature importances) and interpret them.
- Serialize a model and expose it through a minimal REST API.

## Prerequisites

- Python 3.9+
- Libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`, `flask`, `joblib`, `requests`

## Requirements

1. **Fork** this repository to your own GitHub account.
2. **Clone** the fork to your local machine.
3. Create a Jupyter Notebook called **`m4-10-assessment.ipynb`**.
4. **Commit regularly** as you work — at least once per task.

### Python environment

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy flask joblib requests
```

## Tasks

### Task 1 — Unsupervised Exploration

1. Load the Palmer Penguins dataset (`sns.load_dataset("penguins")`).
2. Clean the data: handle missing values, examine distributions, and note any anomalies.
3. Select the numeric features and scale them with `StandardScaler`.
4. Apply **PCA** (2 components) and **t-SNE** (2 components, `random_state=42`). Create a side-by-side plot colored by actual species.
5. Apply **K-Means** (k=3) and **DBSCAN** (experiment with at least 2 `eps`/`min_samples` combos) to the scaled data.
6. Evaluate each clustering result with the **silhouette score**.
7. Compare your best clustering labels to the actual species using **adjusted Rand score** and **normalized mutual information**. Visualize the comparison on a PCA projection.
8. In a markdown cell, summarize: How well did unsupervised methods recover the species structure? Where did they fail?

### Task 2 — Supervised Model Pipeline

1. Prepare the full dataset for supervised classification: target = `species`, features = all other columns.
2. Drop rows with missing values using `dropna()`, then build a preprocessing pipeline using `ColumnTransformer`:
   - Numeric features: scale (`StandardScaler`).
   - Categorical features: one-hot encode (`OneHotEncoder`).
3. Train and evaluate **at least 3 different models** (e.g., LogisticRegression, RandomForest, SVC) using **stratified 5-fold cross-validation**. Report accuracy, precision (macro), recall (macro), and F1 (macro) for each.
4. Select the best model based on F1 score.
5. Define a hyperparameter grid (at least 3 parameters) and run `GridSearchCV` with stratified 5-fold CV.
6. Report the best parameters, best CV score, and compare with the default model performance.

### Task 3 — Model Evaluation & Interpretation

Using the best tuned model from Task 2:

1. Train on the full training set and predict on a held-out test set (80/20 split, `random_state=42`).
2. Print the full **classification report**.
3. Plot the **confusion matrix** using `ConfusionMatrixDisplay`.
4. Plot **ROC curves** (one-vs-rest) for all three species on the same figure. Compute and display the AUC for each class. (If your model doesn't natively support `predict_proba`, wrap it in a `CalibratedClassifierCV` or switch to a model that does.)
5. Plot **learning curves** (training size from 10% to 100%).
6. Compute **permutation importances** on the test set and plot the top features.
7. In a markdown cell, write a comprehensive interpretation:
   - Is the model overfitting or underfitting?
   - Which species is hardest to classify and why?
   - Which features drive predictions the most?
   - Are there any signs of data leakage or evaluation issues?

### Task 4 — Model Deployment Prototype

1. Serialize the best tuned model (full pipeline including preprocessor) using `joblib.dump()`.
2. Create an **`app.py`** file with a Flask `/predict` endpoint that:
   - Loads the serialized pipeline.
   - Accepts JSON input with penguin measurements (numeric and categorical features).
   - Returns the predicted species and class probabilities.
   - Includes basic input validation.
3. Also add a `/health` endpoint.
4. Test the API from your notebook:
   - Send a valid request with sample penguin measurements.
   - Send an invalid request and verify error handling.
5. Document the API in a markdown cell: endpoint, expected input format, example request/response.

## Submission

### What to submit
- `m4-10-assessment.ipynb` — completed notebook with all four tasks.
- `app.py` — Flask application for Task 4.
- Any `.joblib` model artifacts.

### How to submit

```bash
git add .
git commit -m "assessment: complete full ML pipeline"
git push origin main
```

Then open a **Pull Request** on the original repository. Include a brief description of your approach in the PR body.

## Evaluation Criteria

| Criterion | Weight | Description |
|---|---|---|
| **Unsupervised Analysis** | 20% | Correct application of PCA, t-SNE, K-Means, DBSCAN with proper evaluation metrics and insightful comparison to true labels. |
| **Supervised Pipeline** | 25% | Well-structured preprocessing pipeline, at least 3 models compared with cross-validation, meaningful hyperparameter tuning. |
| **Evaluation & Interpretation** | 25% | Complete diagnostic suite (confusion matrix, ROC, learning curves, importances) with thoughtful, evidence-based interpretation. |
| **Deployment Prototype** | 15% | Working Flask API with prediction and health endpoints, input validation, and documented tests. |
| **Code Quality & Communication** | 15% | Clean, readable code; clear markdown explanations; notebook runs without errors; regular commits. |
