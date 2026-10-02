"""
Integration Tests for Person 2: FastAPI Endpoints.

Tests:
1. GET /health
2. GET /products (listing with summary and filters)
3. GET /products/{product_id} (existing product with full details)
4. GET /products/{product_id} with invalid ID -> 404
5. POST /decision (ad-hoc product evaluation)
6. POST /analyze (full inventory analysis, decision, AI explanation, alternatives)
7. POST /alternative-products (candidate alternative recommendations)
8. POST /approval & GET /approvals (Human-in-the-loop: APPROVE, MODIFY, REJECT)
9. GET /demo/scenarios (standard demonstration scenarios)
10. Input validation & error handling (invalid MOQ, negative values, missing optional fields)
"""

import os
import pytest
from fastapi.testclient import TestClient
from backend.main import app, load_handoff_data


@pytest.fixture(scope="module", autouse=True)
def ensure_standard_handoff():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_csv = os.path.join(root_dir, "data", "input", "customer_inventory_upload.csv")
    if os.path.exists(data_csv):
        from src.customer_data_adapter import adapt_customer_data
        from src.customer_forecasting import train_customer_forecasting_model, predict_customer_demand
        from src.product_intelligence import build_person2_handoff_json
        import backend.main
        df = adapt_customer_data(data_csv)
        m, _ = train_customer_forecasting_model(df)
        p_df = predict_customer_demand(m, df)
        build_person2_handoff_json(df, p_df, use_live_market_intel=False)
        backend.main._PRODUCTS_CACHE = backend.main.load_handoff_data()


@pytest.fixture(scope="module")
def client():
    # Use context manager to trigger startup events cleanly
    with TestClient(app) as test_client:
        yield test_client


class TestApiEndpoints:

    def test_json_response_headers(self, client):
        """Verifies that all API endpoints explicitly return application/json content-type headers."""
        endpoints = ["/health", "/products", "/demo/scenarios", "/approvals"]
        for ep in endpoints:
            res = client.get(ep)
            assert res.status_code == 200
            assert "application/json" in res.headers.get("content-type", "")

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "AI Inventory Decision Agent (Person 2)"
        assert data["handoff_exists"] is True
        assert data["total_products"] > 0

    def test_get_products_list(self, client):
        response = client.get("/products")
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")
        data = response.json()
        assert "summary" in data
        assert "products" in data
        assert data["summary"]["total"] >= 20
        assert len(data["products"]) >= 20


        # Verify product structure in list
        first = data["products"][0]
        assert "product_id" in first
        assert "product_name" in first
        assert "decision" in first
        assert "stockout_risk" in first
        assert "recommended_order_quantity" in first

    def test_get_products_with_filter(self, client):
        response = client.get("/products?decision_filter=INCREASE")
        assert response.status_code == 200
        data = response.json()
        for p in data["products"]:
            assert p["decision"] == "INCREASE"

    def test_get_product_detail_success(self, client):
        response = client.get("/products/P001")
        assert response.status_code == 200
        data = response.json()
        assert data["product"]["product_id"] == "P001"
        assert "decision" in data
        assert "ai_explanation" in data
        assert "alternative_products" in data
        assert len(data["ai_explanation"]) > 20

    def test_get_product_detail_not_found(self, client):
        response = client.get("/products/NONEXISTENT_999")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_post_decision_endpoint(self, client):
        payload = {
            "product_id": "CUSTOM-01",
            "product_name": "High-Demand Earphones",
            "forecast_7d": 140,
            "forecast_30d": 600,
            "sales_trend": "increasing",
            "current_inventory": 30,
            "reserved_inventory": 0,
            "incoming_inventory": 0,
            "lead_time_days": 7,
            "minimum_order_quantity": 50,
            "market_signal": "positive",
            "market_confidence": 0.85
        }
        response = client.post("/decision", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["product_id"] == "CUSTOM-01"
        assert data["decision"] == "INCREASE"
        assert data["stockout_risk"] == "HIGH"
        assert data["recommended_order_quantity"] > 0
        assert data["recommended_order_quantity"] % 50 == 0

    def test_post_analyze_endpoint(self, client):
        payload = {
            "product_id": "CUSTOM-02",
            "product_name": "Overstocked Keyboard",
            "forecast_7d": 14,
            "forecast_30d": 60,
            "sales_trend": "decreasing",
            "current_inventory": 2000,
            "reserved_inventory": 0,
            "incoming_inventory": 0,
            "lead_time_days": 10,
            "minimum_order_quantity": 25,
            "market_signal": "negative"
        }
        response = client.post("/analyze", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["decision"]["decision"] == "REDUCE"
        assert data["decision"]["recommended_order_quantity"] == 0
        assert "ai_explanation" in data
        assert isinstance(data["alternative_products"], list)

    def test_post_alternative_products(self, client):
        payload = {
            "product_id": "P001",
            "category": "Computer Accessories",
            "forecast_7d": 10,
            "sales_trend": "decreasing",
            "current_inventory": 5000
        }
        response = client.post("/alternative-products", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        if len(data) > 0:
            assert "product_id" in data[0]
            assert "reason" in data[0]

    def test_human_approval_workflow(self, client):
        # 1. Submit APPROVE
        approve_payload = {
            "product_id": "P019",
            "decision": "INCREASE",
            "action": "APPROVE",
            "notes": "Approved by Inventory Manager for seasonal restock"
        }
        r1 = client.post("/approval", json=approve_payload)
        assert r1.status_code == 200
        d1 = r1.json()
        assert d1["action"] == "APPROVE"
        assert d1["product_id"] == "P019"
        assert d1["final_quantity"] == d1["original_recommended_quantity"]

        # 2. Submit MODIFY
        modify_payload = {
            "product_id": "P019",
            "decision": "INCREASE",
            "action": "MODIFY",
            "modified_quantity": 250,
            "notes": "Adjusted higher due to anticipated marketing campaign"
        }
        r2 = client.post("/approval", json=modify_payload)
        assert r2.status_code == 200
        d2 = r2.json()
        assert d2["action"] == "MODIFY"
        assert d2["final_quantity"] == 250

        # 3. Submit REJECT
        reject_payload = {
            "product_id": "P019",
            "decision": "INCREASE",
            "action": "REJECT",
            "notes": "Holding orders until warehouse expansion completes"
        }
        r3 = client.post("/approval", json=reject_payload)
        assert r3.status_code == 200
        d3 = r3.json()
        assert d3["action"] == "REJECT"
        assert d3["final_quantity"] == 0

        # 4. Verify approvals list
        r4 = client.get("/approvals")
        assert r4.status_code == 200
        approvals = r4.json()
        assert len(approvals) >= 1
        assert any(a["product_id"] == "P019" for a in approvals)

    def test_human_approval_validation_errors(self, client):
        # Invalid action
        bad_action = {
            "product_id": "P001",
            "decision": "INCREASE",
            "action": "INVALID_ACTION"
        }
        r1 = client.post("/approval", json=bad_action)
        assert r1.status_code == 400

        # MODIFY missing modified_quantity
        missing_qty = {
            "product_id": "P001",
            "decision": "INCREASE",
            "action": "MODIFY"
        }
        r2 = client.post("/approval", json=missing_qty)
        assert r2.status_code == 400

    def test_demo_scenarios_endpoint(self, client):
        response = client.get("/demo/scenarios")
        assert response.status_code == 200
        scenarios = response.json()
        assert len(scenarios) == 3
        ids = [s["scenario_id"] for s in scenarios]
        assert "scenario_1" in ids
        assert "scenario_2" in ids
        assert "scenario_3" in ids

    def test_upload_and_run_with_missing_columns(self, client):
        import shutil
        from backend.main import HANDOFF_FILE_PATH, load_handoff_data
        backup_path = HANDOFF_FILE_PATH + ".bak"
        if os.path.exists(HANDOFF_FILE_PATH):
            shutil.copy(HANDOFF_FILE_PATH, backup_path)

        try:
            # Create CSV with missing optional/expected columns (e.g. missing category, brand, lead_time_days)
            csv_data = (
                "date,product_id,units_sold,price\n"
                "2026-01-01,P_TEST_01,15,100.0\n"
                "2026-01-02,P_TEST_01,20,100.0\n"
                "2026-01-03,P_TEST_01,18,100.0\n"
                "2026-01-01,P_TEST_02,5,50.0\n"
                "2026-01-02,P_TEST_02,4,50.0\n"
                "2026-01-03,P_TEST_02,6,50.0\n"
            )
            files = {"file": ("test_missing.csv", csv_data.encode("utf-8"), "text/csv")}
            response = client.post("/api/upload-and-run", files=files)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "validation_summary" in data
            assert "missing_columns" in data["validation_summary"]
            assert len(data["validation_summary"]["missing_columns"]) > 0
            assert "category" in data["validation_summary"]["missing_columns"]

            # Verify downloads work
            f_res = client.get("/api/download/forecasts")
            assert f_res.status_code == 200
            h_res = client.get("/api/download/handoff")
            assert h_res.status_code == 200
            d_res = client.get("/api/download/decisions")
            assert d_res.status_code == 200
        finally:
            if os.path.exists(backup_path):
                shutil.move(backup_path, HANDOFF_FILE_PATH)
                import backend.main
                backend.main._PRODUCTS_CACHE = load_handoff_data()


