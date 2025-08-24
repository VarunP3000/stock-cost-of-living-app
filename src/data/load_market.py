from datetime import date
import pandas as pd
import yfinance as yf

def load_sp500_history(start="1990-01-01", end=None) -> pd.DataFrame:
    end = end or date.today().isoformat()
    df = yf.download("^GSPC", start=start, end=end, auto_adjust=True, progress=False)
    df = df.rename(columns=str.lower).reset_index().rename(columns={"date":"ds"})
    return df[["ds","close","open","high","low","volume"]]
