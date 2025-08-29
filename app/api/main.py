from fastapi import FastAPI
from app.api import forecast_ensemble, forecast_regional, artifacts

app = FastAPI(title="Inflation→SPX")

app.include_router(forecast_ensemble.router)
app.include_router(forecast_regional.router)
app.include_router(artifacts.router)

@app.get("/health")
def health():
    return {"ok": True}
