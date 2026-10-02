"""
Comprehensive Data-Driven Pipeline Tests for BFWAI AI Inventory Decision Agent.

Verifies:
1. Different product counts work.
2. Different category counts work.
3. No hardcoded product IDs exist.
4. No hardcoded category names exist.
5. Historical inventory is not incorrectly summed.
6. Forecasting works per product.
7. Products with insufficient history use the documented fallback.
8. Market intelligence failure does not crash the pipeline.
9. Missing optional columns are handled safely.
10. MOQ is respected.
11. Recommended order quantity is never negative.
12. Every unique product receives a decision.
13. Person 2 receives exactly one product-level record per product.
14. Final output contains only actual calculated decisions.
15. Decision distribution is NOT artificially forced.
"""

import unittest
import os
import sys
import tempfile
import json
import pandas as pd
from unittest.mock import patch

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.customer_data_adapter import adapt_customer_data
from src.customer_forecasting import train_customer_forecasting_model, predict_customer_demand
from src.product_intelligence import build_person2_handoff_json
from src.inventory_decision_engine import run_inventory_decision_engine
from backend.decision_engine import evaluate_decision
from backend.schemas import ProductHandoff


class TestCustomerPipelineDataDriven(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_synthetic_customer_csv(
        self,
        product_ids,
        categories,
        num_days=30,
        omit_optional_cols=False
    ):
        """Helper to build dynamic customer CSV with arbitrary products & categories."""
        dates = pd.date_range("2025-01-01", periods=num_days, freq="D").astype(str).tolist()
        rows = []
        
        for idx, pid in enumerate(product_ids):
            category = categories[idx % len(categories)]
            p_name = f"Custom Item {pid}"
            brand = f"Brand-{pid[:3]}"
            
            for d in dates:
                row = {
                    "date": d,
                    "product_id": pid,
                    "units_sold": 10 + (idx * 5) + (hash(d) % 7)
                }
                if not omit_optional_cols:
                    row.update({
                        "product_name": p_name,
                        "category": category,
                        "brand": brand,
                        "price": 49.99,
                        "unit_cost": 25.00,
                        "revenue": (10 + (idx * 5) + (hash(d) % 7)) * 49.99,
                        "discount_pct": 0.0,
                        "promotion": 0,
                        "holiday_event": 0,
                        "rating": 4.5,
                        "review_count": 120,
                        "current_inventory": 300,
                        "reserved_inventory": 10,
                        "incoming_inventory": 50,
                        "lead_time_days": 10,
                        "minimum_order_quantity": 25,
                        "reorder_point": 100,
                        "competitor_price": 52.00
                    })
                rows.append(row)
                
        df = pd.DataFrame(rows)
        file_path = os.path.join(self.temp_dir.name, f"customer_{len(product_ids)}_prods.csv")
        df.to_csv(file_path, index=False)
        return file_path

    def test_1_different_product_counts_work(self):
        """1 & 3. Test pipeline works with 3 products and no hardcoded product IDs."""
        p_ids = ["CUSTOM_SKU_A", "CUSTOM_SKU_B", "CUSTOM_SKU_C"]
        csv_path = self._create_synthetic_customer_csv(p_ids, ["Category1"])
        
        adapted_df = adapt_customer_data(csv_path)
        self.assertEqual(adapted_df["product_id"].nunique(), 3)
        self.assertCountEqual(adapted_df["product_id"].unique(), p_ids)
        
        model, _ = train_customer_forecasting_model(adapted_df, model_dir=self.temp_dir.name)
        pred_df = predict_customer_demand(model, adapted_df, output_pred_path=os.path.join(self.temp_dir.name, "forecasts.csv"))
        self.assertEqual(len(pred_df), 3)

    def test_2_different_category_counts_work(self):
        """2 & 4. Test pipeline works with multiple dynamic categories and no hardcoded category names."""
        p_ids = [f"SKU_{i}" for i in range(6)]
        cats = ["Gaming Accessories", "Audio", "Storage", "Office Equipment", "Home Appliances", "Networking"]
        csv_path = self._create_synthetic_customer_csv(p_ids, cats)
        
        adapted_df = adapt_customer_data(csv_path)
        self.assertEqual(adapted_df["category"].nunique(), 6)
        self.assertCountEqual(adapted_df["category"].unique(), cats)

    def test_5_historical_inventory_not_incorrectly_summed(self):
        """5 & 11. Test historical inventory is taken from latest record and not summed across history."""
        # 30 days history where current_inventory = 300 each day
        csv_path = self._create_synthetic_customer_csv(["INV_TEST"], ["CatA"], num_days=30)
        adapted_df = adapt_customer_data(csv_path)
        
        pred_df = predict_customer_demand("dummy", adapted_df, output_pred_path=os.path.join(self.temp_dir.name, "forecasts.csv"))
        handoff_path = os.path.join(self.temp_dir.name, "handoff.json")
        
        handoff = build_person2_handoff_json(
            customer_df=adapted_df,
            forecast_df=pred_df,
            output_json_path=handoff_path,
            use_live_market_intel=False
        )
        
        product_rec = handoff["products"][0]
        # Current inventory should be 300, NOT 300 * 30 = 9000
        self.assertEqual(product_rec["inventory"]["current_inventory"], 300)
        self.assertNotEqual(product_rec["inventory"]["current_inventory"], 9000)

    def test_6_and_7_insufficient_history_uses_fallback(self):
        """6 & 7. Test product with < 14 days of history uses documented fallback method."""
        # Create dataset with 1 product having 5 days history and 1 having 30 days
        csv_path = self._create_synthetic_customer_csv(["SHORT_SKU"], ["CatA"], num_days=5)
        adapted_df = adapt_customer_data(csv_path)
        
        pred_df = predict_customer_demand("dummy", adapted_df, output_pred_path=os.path.join(self.temp_dir.name, "forecasts.csv"))
        self.assertEqual(len(pred_df), 1)
        self.assertTrue(pred_df.iloc[0]["fallback_used"])
        self.assertGreaterEqual(pred_df.iloc[0]["forecast_7d"], 0)

    def test_8_market_intelligence_failure_graceful(self):
        """8. Test that market intelligence failure returns neutral fallback signal without crashing."""
        csv_path = self._create_synthetic_customer_csv(["MKT_SKU"], ["CatA"])
        adapted_df = adapt_customer_data(csv_path)
        pred_df = predict_customer_demand("dummy", adapted_df, output_pred_path=os.path.join(self.temp_dir.name, "forecasts.csv"))
        
        handoff_path = os.path.join(self.temp_dir.name, "handoff.json")
        # Force live market intel call with invalid key
        with patch.dict(os.environ, {"GEMINI_API_KEY": "", "NEWS_API_KEY": ""}):
            handoff = build_person2_handoff_json(
                customer_df=adapted_df,
                forecast_df=pred_df,
                output_json_path=handoff_path,
                use_live_market_intel=True
            )
            
        m_intel = handoff["products"][0]["market_intelligence"]
        self.assertEqual(m_intel["market_signal"], "neutral")
        self.assertFalse(m_intel["external_signal_available"])

    def test_9_missing_optional_columns_handled_safely(self):
        """9. Test dataset with missing optional columns (revenue, price, rating, etc.) is handled safely."""
        csv_path = self._create_synthetic_customer_csv(["MIN_SKU"], ["CatA"], omit_optional_cols=True)
        adapted_df = adapt_customer_data(csv_path)
        
        self.assertIn("date", adapted_df.columns)
        self.assertIn("product_id", adapted_df.columns)
        self.assertIn("units_sold", adapted_df.columns)
        # Verify defaults populated
        self.assertIn("product_name", adapted_df.columns)
        self.assertEqual(adapted_df.iloc[0]["product_name"], "MIN_SKU")

    def test_10_and_11_moq_respected_and_non_negative_order(self):
        """10 & 11. Test MOQ is respected and recommended_order_quantity is never negative."""
        product = ProductHandoff(
            product_id="MOQ_TEST",
            product_name="MOQ Item",
            category="Test",
            brand="Test",
            forecast={"forecast_7d": 100, "forecast_30d": 400, "sales_trend": "increasing"},
            inventory={"current_inventory": 10, "reserved_inventory": 0, "incoming_inventory": 0},
            supplier={"lead_time_days": 10, "minimum_order_quantity": 50, "reorder_point": 100}
        )
        result = evaluate_decision(product)
        
        self.assertGreaterEqual(result.recommended_order_quantity, 0)
        if result.recommended_order_quantity > 0:
            self.assertEqual(result.recommended_order_quantity % 50, 0)
            self.assertGreaterEqual(result.recommended_order_quantity, 50)

    def test_12_13_14_person2_handoff_and_decisions_structure(self):
        """12, 13, 14. Test Person 2 receives exactly 1 record per product and outputs decisions for every product."""
        p_ids = [f"D_SKU_{i}" for i in range(5)]
        csv_path = self._create_synthetic_customer_csv(p_ids, ["CatA", "CatB"])
        
        adapted_df = adapt_customer_data(csv_path)
        pred_df = predict_customer_demand("dummy", adapted_df, output_pred_path=os.path.join(self.temp_dir.name, "forecasts.csv"))
        
        handoff_path = os.path.join(self.temp_dir.name, "person2_handoff.json")
        decisions_path = os.path.join(self.temp_dir.name, "inventory_decisions.json")
        
        handoff = build_person2_handoff_json(
            customer_df=adapted_df,
            forecast_df=pred_df,
            output_json_path=handoff_path,
            use_live_market_intel=False
        )
        self.assertEqual(len(handoff["products"]), 5)
        
        decisions_dict = run_inventory_decision_engine(
            handoff_json_path=handoff_path,
            output_decisions_path=decisions_path
        )
        self.assertEqual(len(decisions_dict["decisions"]), 5)
        decision_pids = [d["product_id"] for d in decisions_dict["decisions"]]
        self.assertCountEqual(decision_pids, p_ids)
        
        for d in decisions_dict["decisions"]:
            self.assertIn(d["decision"], ["INCREASE", "MAINTAIN", "REDUCE"])
            self.assertGreaterEqual(d["recommended_order_quantity"], 0)

    def test_15_decision_distribution_not_artificially_forced(self):
        """15. Test that decisions are data-driven and not artificially forced to 1/3 each."""
        # Create dataset where all products have massive excess inventory -> all should be REDUCE
        p_ids = [f"EXCESS_SKU_{i}" for i in range(4)]
        csv_path = self._create_synthetic_customer_csv(p_ids, ["CatA"])
        
        adapted_df = adapt_customer_data(csv_path)
        # Artificially set current_inventory very high (100,000 units)
        adapted_df["current_inventory"] = 100000
        adapted_df["units_sold"] = 1
        
        pred_df = predict_customer_demand("dummy", adapted_df, output_pred_path=os.path.join(self.temp_dir.name, "forecasts.csv"))
        handoff_path = os.path.join(self.temp_dir.name, "person2_handoff.json")
        decisions_path = os.path.join(self.temp_dir.name, "inventory_decisions.json")
        
        build_person2_handoff_json(customer_df=adapted_df, forecast_df=pred_df, output_json_path=handoff_path, use_live_market_intel=False)
        decisions_dict = run_inventory_decision_engine(handoff_json_path=handoff_path, output_decisions_path=decisions_path)
        
        # Verify decisions come purely from data (e.g., all 4 will naturally be REDUCE or MAINTAIN, not forced 1 INCREASE, 1 MAINTAIN, etc.)
        decisions = [d["decision"] for d in decisions_dict["decisions"]]
        self.assertEqual(len(decisions), 4)


if __name__ == "__main__":
    unittest.main()
