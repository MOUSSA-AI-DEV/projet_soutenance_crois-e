import numpy as np
import pandas as pd


def detect_outliers(df, numeric_columns=None):
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    outliers_summary = []
    for column in numeric_columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        outliers_summary.append({
            "Variable": column,
            "Q1": Q1,
            "Q3": Q3,
            "IQR": IQR,
            "Lower Bound": lower_bound,
            "Upper Bound": upper_bound,
            "Outliers Count": len(outliers),
        })

    return pd.DataFrame(outliers_summary)

