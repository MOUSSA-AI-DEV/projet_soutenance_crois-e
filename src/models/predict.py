from pathlib import Path
import joblib


def save_model(model, file_path="artifacts/models/best_mental_health_model.joblib"):
    project_root = Path(__file__).resolve().parent.parent.parent
    full_path = project_root / file_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, full_path)
    print("Modèle sauvegardé :", full_path)


def load_model(file_path="artifacts/models/best_mental_health_model.joblib"):
    project_root = Path(__file__).resolve().parent.parent.parent
    full_path = project_root / file_path
    return joblib.load(full_path)


def predict(model, X):
    return model.predict(X)


def predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return None
