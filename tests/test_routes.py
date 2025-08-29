from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("ok") is True

def test_ensemble_default():
    r = client.get("/forecast/ensemble")
    assert r.status_code == 200
    j = r.json()
    for k in ["asof","prediction","feature_order","components","weights_used","metadata"]:
        assert k in j

def test_regional_americas():
    r = client.get("/forecast/regional", params={"region":"americas"})
    assert r.status_code == 200
    j = r.json()
    for k in ["asof","region","prediction","feature_order","metadata"]:
        assert k in j
