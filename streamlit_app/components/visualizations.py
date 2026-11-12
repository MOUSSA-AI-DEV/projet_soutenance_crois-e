import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_numeric_distributions(df):
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    figures = []
    for column in numeric_columns:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        sns.histplot(data=df, x=column, kde=True, ax=ax)
        ax.set_title(f"Distribution de {column}")
        plt.tight_layout()
        figures.append((column, fig))
    return figures


def plot_categorical_distributions(df):
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if "Student_ID" in categorical_columns:
        categorical_columns.remove("Student_ID")

    n = len(categorical_columns)
    n_cols = 3
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(categorical_columns):
        sns.countplot(data=df, x=col, order=df[col].value_counts().index, ax=axes[i])
        axes[i].set_title(f"Distribution de {col}")
        axes[i].tick_params(axis="x", rotation=45)

    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    return fig


def plot_correlation_matrix(df):
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[numeric_columns].corr()
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Matrice de corrélation")
    plt.tight_layout()
    return fig


def plot_outliers_boxplots(df):
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    n = len(numeric_columns)
    n_cols = 3
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(numeric_columns):
        sns.boxplot(data=df, x=col, ax=axes[i])
        axes[i].set_title(f"Boxplot - {col}")

    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    return fig


def plot_target_relationships(df, target_column="Mental_Health_Risk"):
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if target_column in categorical_columns:
        categorical_columns.remove(target_column)
    if "Student_ID" in categorical_columns:
        categorical_columns.remove("Student_ID")

    n = len(categorical_columns)
    n_cols = 3
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(categorical_columns):
        cross_tab = pd.crosstab(df[col], df[target_column], normalize="index") * 100
        cross_tab.plot(kind="bar", stacked=True, ax=axes[i])
        axes[i].set_title(f"{col} selon le risque")
        axes[i].tick_params(axis="x", rotation=45)

    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    return fig


def plot_confusion_matrix_display(y_test, y_pred, classes):
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap="Blues", values_format="d", ax=ax)
    ax.set_title("Matrice de Confusion")
    plt.tight_layout()
    return fig

