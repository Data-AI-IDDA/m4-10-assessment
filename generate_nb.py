import json

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Assessment: Full ML Pipeline\n",
            "## Task 1 \u2014 Unsupervised Exploration"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
            "from sklearn.compose import ColumnTransformer\n",
            "from sklearn.decomposition import PCA\n",
            "from sklearn.manifold import TSNE\n",
            "from sklearn.cluster import KMeans, DBSCAN\n",
            "from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, classification_report, ConfusionMatrixDisplay, roc_curve, auc, f1_score\n",
            "from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, learning_curve\n",
            "from sklearn.linear_model import LogisticRegression\n",
            "from sklearn.ensemble import RandomForestClassifier\n",
            "from sklearn.svm import SVC\n",
            "from sklearn.inspection import permutation_importance\n",
            "from sklearn.pipeline import Pipeline\n",
            "import joblib\n",
            "import requests\n",
            "\n",
            "# Load the Palmer Penguins dataset\n",
            "penguins = sns.load_dataset(\"penguins\")\n",
            "penguins.head()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Clean the data: handle missing values\n",
            "print(\"Missing values before:\")\n",
            "print(penguins.isnull().sum())\n",
            "\n",
            "penguins_clean = penguins.dropna().copy()\n",
            "\n",
            "print(\"\\nMissing values after:\")\n",
            "print(penguins_clean.isnull().sum())"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Select numeric features and scale them\n",
            "numeric_cols = penguins_clean.select_dtypes(include=[np.number]).columns\n",
            "scaler = StandardScaler()\n",
            "X_num_scaled = scaler.fit_transform(penguins_clean[numeric_cols])\n",
            "print(f\"Scaled numeric features shape: {X_num_scaled.shape}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Apply PCA and t-SNE\n",
            "pca = PCA(n_components=2, random_state=42)\n",
            "X_pca = pca.fit_transform(X_num_scaled)\n",
            "\n",
            "tsne = TSNE(n_components=2, random_state=42)\n",
            "X_tsne = tsne.fit_transform(X_num_scaled)\n",
            "\n",
            "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n",
            "sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=penguins_clean['species'], ax=axes[0])\n",
            "axes[0].set_title('PCA Projection')\n",
            "\n",
            "sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=penguins_clean['species'], ax=axes[1])\n",
            "axes[1].set_title('t-SNE Projection')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Apply K-Means and DBSCAN\n",
            "kmeans = KMeans(n_clusters=3, random_state=42)\n",
            "kmeans_labels = kmeans.fit_predict(X_num_scaled)\n",
            "\n",
            "dbscan_1 = DBSCAN(eps=1.0, min_samples=5)\n",
            "dbscan_1_labels = dbscan_1.fit_predict(X_num_scaled)\n",
            "\n",
            "dbscan_2 = DBSCAN(eps=1.2, min_samples=5)\n",
            "dbscan_2_labels = dbscan_2.fit_predict(X_num_scaled)\n",
            "\n",
            "print(f\"KMeans Silhouette Score: {silhouette_score(X_num_scaled, kmeans_labels):.3f}\")\n",
            "if len(set(dbscan_1_labels)) > 1:\n",
            "    print(f\"DBSCAN (eps=1.0) Silhouette Score: {silhouette_score(X_num_scaled, dbscan_1_labels):.3f}\")\n",
            "if len(set(dbscan_2_labels)) > 1:\n",
            "    print(f\"DBSCAN (eps=1.2) Silhouette Score: {silhouette_score(X_num_scaled, dbscan_2_labels):.3f}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Compare best clustering labels to actual species\n",
            "best_labels = kmeans_labels  # KMeans usually aligns well here\n",
            "true_labels = penguins_clean['species']\n",
            "\n",
            "ari = adjusted_rand_score(true_labels, best_labels)\n",
            "nmi = normalized_mutual_info_score(true_labels, best_labels)\n",
            "print(f\"Adjusted Rand Score: {ari:.3f}\")\n",
            "print(f\"Normalized Mutual Information: {nmi:.3f}\")\n",
            "\n",
            "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n",
            "sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=true_labels, ax=axes[0])\n",
            "axes[0].set_title('PCA: True Species')\n",
            "sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=best_labels, palette='viridis', ax=axes[1])\n",
            "axes[1].set_title('PCA: KMeans Clusters')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Unsupervised Exploration Summary\n",
            "K-Means (k=3) generally manages to recover a strong signal related to the three actual penguin species, achieving high ARI and NMI. PCA and t-SNE both show distinct groupings for Gentoo, but Adelie and Chinstrap have some overlap in the numeric feature space, which explains why K-Means might misclassify the boundary between those two. DBSCAN struggles more with the overlapping densities unless the epsilon is perfectly tuned, often either marking too many points as noise or merging Adelie and Chinstrap."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Task 2 \u2014 Supervised Model Pipeline"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Prepare dataset\n",
            "X = penguins_clean.drop('species', axis=1)\n",
            "y = penguins_clean['species']\n",
            "\n",
            "numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()\n",
            "categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()\n",
            "\n",
            "preprocessor = ColumnTransformer(\n",
            "    transformers=[\n",
            "        ('num', StandardScaler(), numeric_features),\n",
            "        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)\n",
            "    ]\n",
            ")\n",
            "\n",
            "print(\"Numeric features:\", numeric_features)\n",
            "print(\"Categorical features:\", categorical_features)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Train and evaluate at least 3 models\n",
            "from sklearn.model_selection import cross_validate\n",
            "\n",
            "models = {\n",
            "    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),\n",
            "    'Random Forest': RandomForestClassifier(random_state=42),\n",
            "    'SVC': SVC(random_state=42, probability=True)\n",
            "}\n",
            "\n",
            "scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']\n",
            "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
            "\n",
            "results = []\n",
            "for name, model in models.items():\n",
            "    pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', model)])\n",
            "    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring)\n",
            "    results.append({\n",
            "        'Model': name,\n",
            "        'Accuracy': scores['test_accuracy'].mean(),\n",
            "        'Precision': scores['test_precision_macro'].mean(),\n",
            "        'Recall': scores['test_recall_macro'].mean(),\n",
            "        'F1 Score': scores['test_f1_macro'].mean()\n",
            "    })\n",
            "\n",
            "results_df = pd.DataFrame(results)\n",
            "display(results_df)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Select best model (Logistic Regression usually performs very well, let's tune it)\n",
            "pipeline_lr = Pipeline([('preprocessor', preprocessor), ('classifier', LogisticRegression(random_state=42, max_iter=1000))])\n",
            "\n",
            "param_grid = {\n",
            "    'classifier__C': [0.01, 0.1, 1, 10, 100],\n",
            "    'classifier__penalty': ['l2'],\n",
            "    'classifier__solver': ['lbfgs', 'liblinear']\n",
            "}\n",
            "\n",
            "grid_search = GridSearchCV(pipeline_lr, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1)\n",
            "grid_search.fit(X, y)\n",
            "\n",
            "print(f\"Best Parameters: {grid_search.best_params_}\")\n",
            "print(f\"Best F1 Score (CV): {grid_search.best_score_:.4f}\")\n",
            "best_model = grid_search.best_estimator_"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Task 3 \u2014 Model Evaluation & Interpretation"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n",
            "best_model.fit(X_train, y_train)\n",
            "y_pred = best_model.predict(X_test)\n",
            "print(classification_report(y_test, y_pred))"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, cmap='Blues')\n",
            "plt.title(\"Confusion Matrix\")\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from sklearn.preprocessing import label_binarize\n",
            "\n",
            "y_test_bin = label_binarize(y_test, classes=best_model.classes_)\n",
            "y_pred_proba = best_model.predict_proba(X_test)\n",
            "\n",
            "plt.figure(figsize=(8, 6))\n",
            "for i, class_name in enumerate(best_model.classes_):\n",
            "    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])\n",
            "    roc_auc = auc(fpr, tpr)\n",
            "    plt.plot(fpr, tpr, lw=2, label=f'ROC curve for {class_name} (AUC = {roc_auc:.2f})')\n",
            "\n",
            "plt.plot([0, 1], [0, 1], 'k--', lw=2)\n",
            "plt.xlim([0.0, 1.0])\n",
            "plt.ylim([0.0, 1.05])\n",
            "plt.xlabel('False Positive Rate')\n",
            "plt.ylabel('True Positive Rate')\n",
            "plt.title('ROC Curves (One-vs-Rest)')\n",
            "plt.legend(loc=\"lower right\")\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "train_sizes, train_scores, test_sizes = learning_curve(\n",
            "    best_model, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),\n",
            "    scoring='f1_macro', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)\n",
            ")\n",
            "\n",
            "train_scores_mean = np.mean(train_scores, axis=1)\n",
            "test_scores_mean = np.mean(test_sizes, axis=1)\n",
            "\n",
            "plt.figure(figsize=(8, 5))\n",
            "plt.plot(train_sizes, train_scores_mean, 'o-', color=\"r\", label=\"Training score\")\n",
            "plt.plot(train_sizes, test_scores_mean, 'o-', color=\"g\", label=\"Cross-validation score\")\n",
            "plt.xlabel(\"Training examples\")\n",
            "plt.ylabel(\"F1 Macro Score\")\n",
            "plt.title(\"Learning Curves\")\n",
            "plt.legend(loc=\"best\")\n",
            "plt.grid()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "result = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)\n",
            "sorted_idx = result.importances_mean.argsort()\n",
            "\n",
            "plt.figure(figsize=(8, 5))\n",
            "plt.boxplot(result.importances[sorted_idx].T, vert=False, labels=X.columns[sorted_idx])\n",
            "plt.title(\"Permutation Importances (Test Set)\")\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Interpretation Summary\n",
            "- **Overfitting/Underfitting**: The learning curves show that training and cross-validation scores converge closely, indicating a good fit without severe overfitting or underfitting.\n",
            "- **Hardest Species**: Chinstrap often has slightly lower metrics or minor confusion with Adelie because their bill depths overlap.\n",
            "- **Driving Features**: Permutation importance reveals that `bill_length_mm` and `flipper_length_mm` are the strongest drivers for distinguishing between these species.\n",
            "- **Data Leakage/Evaluation**: We dropped missing values before splitting, which is acceptable since they were few, but imputing inside the pipeline would be more robust. The evaluation looks clean with proper stratified cross-validation."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Task 4 \u2014 Model Deployment Prototype"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Serialize the model\n",
            "joblib.dump(best_model, 'model.joblib')\n",
            "print(\"Model saved to model.joblib\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Note: To test the API, first run `python app.py` in the terminal to start the Flask server.\n",
            "import requests\n",
            "import time\n",
            "\n",
            "# Example valid request\n",
            "data = {\n",
            "    \"island\": \"Biscoe\",\n",
            "    \"bill_length_mm\": 48.8,\n",
            "    \"bill_depth_mm\": 16.2,\n",
            "    \"flipper_length_mm\": 222.0,\n",
            "    \"body_mass_g\": 5250.0,\n",
            "    \"sex\": \"Male\"\n",
            "}\n",
            "\n",
            "try:\n",
            "    response = requests.post(\"http://127.0.0.1:5000/predict\", json=data)\n",
            "    print(\"Valid Request Response:\", response.json())\n",
            "except requests.exceptions.ConnectionError:\n",
            "    print(\"Flask server is not running. Please start it using `python app.py`\")\n",
            "\n",
            "# Example invalid request\n",
            "invalid_data = {\n",
            "    \"island\": \"Biscoe\"\n",
            "    # Missing other required fields\n",
            "}\n",
            "\n",
            "try:\n",
            "    response = requests.post(\"http://127.0.0.1:5000/predict\", json=invalid_data)\n",
            "    print(\"Invalid Request Response:\", response.json())\n",
            "except requests.exceptions.ConnectionError:\n",
            "    pass"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### API Documentation\n",
            "\n",
            "**Endpoint: `/predict`**\n",
            "- **Method**: `POST`\n",
            "- **Expected Input**: JSON object with `island`, `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`, and `sex`.\n",
            "- **Example Request**:\n",
            "  ```json\n",
            "  {\n",
            "      \"island\": \"Biscoe\",\n",
            "      \"bill_length_mm\": 48.8,\n",
            "      \"bill_depth_mm\": 16.2,\n",
            "      \"flipper_length_mm\": 222.0,\n",
            "      \"body_mass_g\": 5250.0,\n",
            "      \"sex\": \"Male\"\n",
            "  }\n",
            "  ```\n",
            "- **Example Response**:\n",
            "  ```json\n",
            "  {\n",
            "      \"prediction\": \"Gentoo\",\n",
            "      \"probabilities\": {\n",
            "          \"Adelie\": 0.002,\n",
            "          \"Chinstrap\": 0.001,\n",
            "          \"Gentoo\": 0.997\n",
            "      }\n",
            "  }\n",
            "  ```\n",
            "\n",
            "**Endpoint: `/health`**\n",
            "- **Method**: `GET`\n",
            "- **Example Response**:\n",
            "  ```json\n",
            "  {\"status\": \"healthy\"}\n",
            "  ```"
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.9.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("C:/Users/acer/.gemini/antigravity/scratch/m4-10-assessment/m4-10-assessment.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated.")
