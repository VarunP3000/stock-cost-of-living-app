from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

def load_cost_of_living(csv_name: str) -> pd.DataFrame:
    path = DATA_DIR / "raw" / csv_name
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df
