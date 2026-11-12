import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def evaluate_predictions(y_test, y_pred, output_dict=False):
    if output_dict:
        return classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    report = classification_report(y_test, y_pred, zero_division=0)
    print(report)
    return report


def plot_confusion_matrix(y_test, y_pred, classes, title="Matrice de confusion"):
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap="Blues", values_format="d", ax=ax)
    plt.title(title)
    plt.tight_layout()
    return fig


def save_metrics(metrics, file_path="artifacts/metrics/evaluation_metrics.json"):
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print("Métriques sauvegardées :", path)
