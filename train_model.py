import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
import joblib

penguins = sns.load_dataset("penguins")
penguins_clean = penguins.dropna().copy()

X = penguins_clean.drop('species', axis=1)
y = penguins_clean['species']

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

pipeline_lr = Pipeline([
    ('preprocessor', preprocessor), 
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__penalty': ['l2'],
    'classifier__solver': ['lbfgs']
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(pipeline_lr, param_grid, cv=cv, scoring='f1_macro')
grid_search.fit(X, y)

best_model = grid_search.best_estimator_
joblib.dump(best_model, 'model.joblib')
print("Model trained and saved to model.joblib")
