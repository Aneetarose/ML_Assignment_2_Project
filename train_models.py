"""Train five classifiers on the Wisconsin Breast Cancer dataset and save model artifacts.

Run: python train_models.py
This script uses scikit-learn's copy of the public Wisconsin Diagnostic Breast Cancer dataset.
"""
from pathlib import Path
import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "model"
MODEL_DIR.mkdir(exist_ok=True)

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
full = X.copy()
full["target"] = y
full.to_csv(BASE / "dataset.csv", index=False)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
test = X_test.copy()
test["target"] = y_test.values
test.to_csv(BASE / "test_data.csv", index=False)

models = {
    "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=5000, random_state=42))]),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": Pipeline([("scaler", StandardScaler()), ("classifier", KNeighborsClassifier(n_neighbors=5))]),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": Pipeline([("scaler", StandardScaler()), ("classifier", SVC(probability=True, random_state=42))]),
}
rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_DIR / (name.lower().replace(" ", "_") + ".joblib"))
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    rows.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, prob),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, pred),
    })
pd.DataFrame(rows).to_csv(BASE / "metrics.csv", index=False)
print(pd.DataFrame(rows).round(4).to_string(index=False))
print("Saved models, dataset.csv, test_data.csv and metrics.csv.")
