import json, os
from fastapi.testclient import TestClient
from app.api import app, THRESH, SCHEMA  # importa objetos expostos

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert isinstance(body["threshold"], float)

def test_score_one_minimal():
    payload = {
        "data": {
            "SK_ID_CURR": 100001,
            "NAME_CONTRACT_TYPE": "Cash loans",
            "AMT_CREDIT": 450000,
            "AMT_INCOME_TOTAL": 180000,
            "DAYS_EMPLOYED": -1200,
            "DAYS_BIRTH": -15000
        }
    }
    r = client.post("/score_one", json=payload)
    assert r.status_code == 200
    body = r.json()
    for k in ["pd_raw", "pd_calibrated", "band", "decision"]:
        assert k in body
    assert 0.0 <= body["pd_calibrated"] <= 1.0

def test_score_batch_basic():
    payload = {
        "rows": [
            {"SK_ID_CURR": 1, "AMT_CREDIT": 200000},
            {"SK_ID_CURR": 2, "AMT_CREDIT": 500000},
        ]
    }
    r = client.post("/score_batch", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 2
    assert len(body["rows"]) == 2

def test_score_file_csv(tmp_path):
    p = tmp_path / "mini.csv"
    p.write_text("SK_ID_CURR,AMT_CREDIT\n10,100000\n11,200000\n", encoding="utf-8")
    with p.open("rb") as f:
        r = client.post("/score-file", files={"file": ("mini.csv", f, "text/csv")})
    assert r.status_code == 200
    assert "text/csv" in r.headers.get("content-type", "")

