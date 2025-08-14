"""Quick local check of API endpoints without starting a server.

Usage (from project root):
  uv run python fastapi_backend/dev_check.py
"""
from __future__ import annotations

from fastapi.testclient import TestClient
from fastapi_backend.app import app

client = TestClient(app)


def main():  # pragma: no cover
    print("/health ->", client.get("/health").json())
    print("/metadata ->", client.get("/metadata").json())
    payload = {"items": [{"lead_id": "demo1", "customer_summary": "Interested in quick purchase", "agent_summary": "Provided discount details"}]}
    resp = client.post("/predict", json=payload)
    print("/predict ->", resp.status_code, resp.json())


if __name__ == "__main__":  # pragma: no cover
    main()
