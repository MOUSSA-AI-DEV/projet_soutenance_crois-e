from pathlib import Path
import pandas as pd


def load_data(file_name="student_mental_health.csv"):
    project_root = Path(__file__).resolve().parent.parent.parent
    data_path = project_root / file_name

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset introuvable : {data_path}")

    df = pd.read_csv(data_path)
   
    print("Dimensions :", df.shape)
    return df

