import pandas as pd
from sklearn.model_selection import train_test_split


def split_features_target(df, target_column="Mental_Health_Risk"):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def check_class_distribution(y):
    class_counts = y.value_counts()
    class_percentages = y.value_counts(normalize=True) * 100
    return pd.DataFrame({
        "Nombre d observations": class_counts,
        "Pourcentage (%)": class_percentages.round(2),
    })


def split_data(X, y, test_size=0.20, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

