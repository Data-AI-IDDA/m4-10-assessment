# AGENTS.md

## Goal

Complete this ML assessment quickly and correctly. The required outputs are:

- m4-10-assessment.ipynb
- app.py
- trained .joblib model artifact

## Student workflow

The student wants the AI agent to inspect the full repository, create the required files, write clean notebook code, add markdown explanations, and keep the project ready for GitHub submission.

## Notebook requirements

The notebook must be clear and submission-ready:

- Markdown headings for each task.
- Code cells that run from top to bottom.
- Interpretation markdown after major outputs.
- All plots visible in the notebook.
- No hardcoded local machine paths.
- Random states fixed where required.

## Assessment tasks

The notebook must cover:

1. Unsupervised exploration:
   - Palmer Penguins dataset
   - cleaning
   - numeric scaling
   - PCA
   - t-SNE
   - K-Means
   - DBSCAN
   - silhouette score
   - ARI
   - NMI
   - PCA visualization

2. Supervised pipeline:
   - target = species
   - features = all other columns
   - dropna
   - ColumnTransformer
   - StandardScaler for numeric features
   - OneHotEncoder for categorical features
   - at least 3 models
   - stratified 5-fold CV
   - accuracy, precision_macro, recall_macro, f1_macro
   - select best model
   - GridSearchCV with at least 3 hyperparameters

3. Evaluation:
   - 80/20 train-test split, random_state=42
   - classification report
   - confusion matrix
   - ROC curves one-vs-rest
   - AUC per class
   - learning curves
   - permutation importances
   - comprehensive interpretation

4. Deployment:
   - save full trained pipeline with joblib
   - create Flask app.py
   - /health endpoint
   - /predict endpoint
   - JSON validation
   - return predicted species and probabilities
   - test valid and invalid API requests from notebook
   - document API in markdown

## Code style

- Keep code simple and readable.
- Prefer scikit-learn Pipeline and ColumnTransformer.
- Use relative paths.
- Save model as penguin_species_pipeline.joblib.
- If a model needs probabilities, prefer LogisticRegression, RandomForest, or SVC(probability=True).
- Avoid overengineering.

## Explanation style

- Notebook markdown should be in English.
- When explaining to the student in chat, use Russian.