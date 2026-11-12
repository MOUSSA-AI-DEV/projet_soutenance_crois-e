import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


def get_models(random_state=42):
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=random_state
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }


def train_and_compare_models(models, preprocessor, X_train, y_train, X_test, y_test):
    results = []
    trained_models = {}

    for model_name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model),
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        results.append({
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "F1-score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        })

        trained_models[model_name] = pipeline

    results_df = pd.DataFrame(results).sort_values(by="F1-score", ascending=False)
    best_model_name = results_df.iloc[0]["Model"]
    best_pipeline = trained_models[best_model_name]

    return results_df, trained_models, best_model_name, best_pipeline

