import streamlit as st, requests, pandas as pd

st.title("Inflation → SPX Forecast")

asof = st.date_input("As of (optional)", value=None)
params = {"asof": str(asof)} if asof else {}
r = requests.get("http://127.0.0.1:8000/forecast", params=params).json()
st.metric("Predicted next-month SPX return", f"{r['pred_spx_ret_next_month']:.2%}")

st.subheader("Model diagnostics")
m = requests.get("http://127.0.0.1:8000/metrics").json()
st.dataframe(pd.DataFrame(m).T)
