import io
import os

import pytest
import requests
from fastapi.testclient import TestClient

from app.api import app


def _ollama_up() -> bool:
    base = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434").rstrip("/")
    try:
        requests.get(f"{base}/api/tags", timeout=2).raise_for_status()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.integration


def test_healthz():
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["ok"] is True


@pytest.mark.skipif(not _ollama_up(), reason="Ollama not running or models not pulled")
def test_ingest_then_ask():
    client = TestClient(app)
    csv = "Customer,ARR,Region,ChurnRisk\nAcme,120000,West,Low\nGlobex,45000,East,High\n"
    files = {"csv": ("demo.csv", io.BytesIO(csv.encode("utf-8")), "text/csv")}
    r = client.post("/ingest", files=files)
    assert r.status_code == 200
    assert r.json()["indexed"] is True

    q = {"question": "Which customer has high churn risk?"}
    r = client.post("/ask", json=q)
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data and isinstance(data["answer"], str)
    assert "sources" in data and isinstance(data["sources"], list)
