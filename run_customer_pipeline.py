"""
End-to-End Pipeline Execution Script for Customer-Data-Driven BFWAI AI Inventory Decision Agent.

Source of Truth:
Raw Customer Dataset: BFWAI/data/input/customer_inventory_upload.csv

Execution Flow:
1. Load & Validate Raw Customer Dataset (src/data_validation.py)
   ↓
2. Adapt Raw Customer Dataset & Save Processed Input (src/customer_data_adapter.py)
   ↓ Output: data/processed/customer_input.csv
3. Train Customer-Specific Forecasting Model & Predict (src/customer_forecasting.py)
   ↓ Output: outputs/predictions/customer_forecasts.csv
4. Gemini + NewsAPI Market Intelligence & Decision Input JSON (src/product_intelligence.py)
   ↓ Output: handoff/inventory_decision_input.json
5. Deterministic Inventory Decision Engine (src/inventory_decision_engine.py)
   ↓ Output: outputs/decisions/inventory_decisions.json
6. Verification & Test Suite Execution
"""

import os
import sys
import logging
import json
import pandas as pd
import pytest

# Add project root to sys.path
sys.path.append(os.path.dirname(__file__))

from src.data_validation import load_dataset, get_dataset_overview, run_validation_checks
from src.customer_data_adapter import adapt_customer_data
from src.customer_forecasting import (
    train_customer_forecasting_model,
    predict_customer_demand
)
from src.product_intelligence import build_inventory_decision_input_json
from src.inventory_decision_engine import run_inventory_decision_engine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BFWAI.PipelineRunner")


def clean_obsolete_prediction_files():
    """Removes old obsolete prediction files from outputs/predictions/."""
    obsolete_files = [
        "outputs/predictions/person2_handoff.json",
        "outputs/predictions/person2_handoff.csv",
        "outputs/predictions/person2_handoff_schema.json",
        "outputs/predictions/product_intelligence.csv",
        "outputs/predictions/product_intelligence_schema.json",
        "outputs/predictions/demand_predictions.csv",
        "outputs/predictions/market_intelligence_single_product_test.json"
    ]
    
    root_dir = os.path.dirname(__file__)
    removed_count = 0
    for rel_path in obsolete_files:
        full_path = os.path.join(root_dir, rel_path)
        if os.path.exists(full_path):
            try:
                os.remove(full_path)
                logger.info(f"Removed obsolete file: '{rel_path}'")
                removed_count += 1
            except Exception as e:
                logger.warning(f"Could not remove '{rel_path}': {e}")
                
    logger.info(f"Cleaned {removed_count} obsolete prediction files.")


def run_customer_pipeline(
    input_csv_path: str = None,
    use_live_market_intel: bool = True,
    run_tests_after: bool = True
):
    print("=" * 80)
    print("BFWAI AI Inventory Decision Agent - End-to-End Pipeline Execution")
    print("=" * 80)
    
    root_dir = os.path.dirname(__file__)
    if not input_csv_path:
        default_input = os.path.join(root_dir, "data", "input", "customer_inventory_upload.csv")
        default_upload = os.path.join(root_dir, "data", "raw", "customer_inventory_upload.csv")
        if os.path.exists(default_input):
            abs_input_path = default_input
        elif os.path.exists(default_upload):
            abs_input_path = default_upload
        else:
            abs_input_path = default_input
    else:
        abs_input_path = os.path.abspath(input_csv_path if os.path.isabs(input_csv_path) else os.path.join(root_dir, input_csv_path))
    
    # 1. Load Raw Customer File & Validate
    logger.info(f"Step 1: Inspecting & Validating Raw Customer File: '{abs_input_path}'")
    raw_df = pd.read_csv(abs_input_path)
    overview = get_dataset_overview(raw_df)
    validation_df = run_validation_checks(raw_df)
    
    print(f"-> Raw rows: {overview['num_rows']}")
    print(f"-> Unique products: {overview['num_unique_products']}")
    print(f"-> Categories: {overview['num_categories']}")
    print(f"-> Date range: {overview['date_min']} -> {overview['date_max']}")

    # 2. Adapt Customer Data & Save Processed Input CSV
    logger.info("Step 2: Adapting Customer Historical Data & Writing Processed CSV...")
    processed_csv_path = os.path.join(root_dir, "data", "processed", "customer_input.csv")
    customer_df = adapt_customer_data(abs_input_path, output_processed_path=processed_csv_path)
    print(f"-> Processed input CSV created: data/processed/customer_input.csv ({len(customer_df)} rows across {customer_df['product_id'].nunique()} products).")
    
    # 3. Customer Forecasting
    logger.info("Step 3: Training Customer-Specific Demand Forecasting Model & Predicting...")
    model, metrics = train_customer_forecasting_model(customer_data=customer_df)
    
    forecast_output_path = os.path.join(root_dir, "outputs", "predictions", "customer_forecasts.csv")
    pred_df = predict_customer_demand(model, customer_df, output_pred_path=forecast_output_path)
    
    fallback_count = pred_df["fallback_used"].sum() if "fallback_used" in pred_df.columns else 0
    ml_forecast_count = len(pred_df) - fallback_count
    
    print(f"-> Customer forecasts generated: outputs/predictions/customer_forecasts.csv ({len(pred_df)} products).")
    
    # 4. Market Intelligence & Decision Input JSON
    logger.info("Step 4: Running Market Intelligence & Generating Inventory Decision Input JSON...")
    decision_input_json_path = os.path.join(root_dir, "handoff", "inventory_decision_input.json")
    decision_input_dict = build_inventory_decision_input_json(
        customer_df=customer_df,
        forecast_df=pred_df,
        output_json_path=decision_input_json_path,
        use_live_market_intel=use_live_market_intel,
        source_file=abs_input_path
    )
    
    ext_signals_count = sum(1 for p in decision_input_dict["products"] if p["market_intelligence"]["external_signal_available"])
    no_signal_count = len(decision_input_dict["products"]) - ext_signals_count
    print(f"-> Inventory Decision Input JSON generated: handoff/inventory_decision_input.json ({len(decision_input_dict['products'])} products).")
    
    # 5. Inventory Decision Engine
    logger.info("Step 5: Executing Deterministic Inventory Decision Engine...")
    decisions_json_path = os.path.join(root_dir, "outputs", "decisions", "inventory_decisions.json")
    decisions_dict = run_inventory_decision_engine(
        handoff_json_path=decision_input_json_path,
        output_decisions_path=decisions_json_path
    )
    
    summary = decisions_dict["decision_summary"]
    inc_c = summary["INCREASE"]
    main_c = summary["MAINTAIN"]
    red_c = summary["REDUCE"]
    
    # 6. Clean obsolete files
    clean_obsolete_prediction_files()
    
    # 7. Optionally run tests
    tests_summary = "NOT RUN"
    if run_tests_after:
        logger.info("Step 7: Executing test suite...")
        class TestResultCollector:
            def __init__(self):
                self.passed = 0
                self.total = 0
            def pytest_runtest_logreport(self, report):
                if report.when == "call":
                    self.total += 1
                    if report.passed:
                        self.passed += 1

        collector = TestResultCollector()
        tests_dir = os.path.join(root_dir, "tests")
        pytest.main([tests_dir, "-q"], plugins=[collector])
        tests_summary = f"{collector.passed}/{collector.total} PASSED"

    # 8. Print Final Executive Summary Block
    print("\n" + "=" * 50)
    print("BFWAI PIPELINE COMPLETE")
    print("=" * 50)
    print(f"\nRaw rows: {overview['num_rows']}")
    print(f"Unique products detected: {overview['num_unique_products']}")
    print(f"Unique products processed: {customer_df['product_id'].nunique()}")
    print(f"Categories: {overview['num_categories']}")
    print(f"Date range: {overview['date_min']} -> {overview['date_max']}")
    print("\nForecasting:")
    print(f"Products forecasted: {len(pred_df)}")
    print(f"Fallback forecasts: {fallback_count}")
    print("\nMarket Intelligence:")
    print(f"Products processed: {len(decision_input_dict['products'])}")
    print(f"External signals available: {ext_signals_count}")
    print(f"No external signal (neutral): {no_signal_count}")
    print("\nInventory Decisions:")
    print(f"INCREASE: {inc_c}")
    print(f"MAINTAIN: {main_c}")
    print(f"REDUCE: {red_c}")
    print(f"Total recommended order quantity: {summary['total_recommended_order_quantity']}")
    print("\nGenerated Output Files:")
    print("  1. data/processed/customer_input.csv")
    print("  2. outputs/predictions/customer_forecasts.csv")
    print("  3. handoff/inventory_decision_input.json")
    print("  4. outputs/decisions/inventory_decisions.json")
    print(f"\nTests:")
    print(f"{tests_summary}")
    print("=" * 50)


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "data", "raw", "customer_inventory_upload_30_products_1000_days_balanced.csv")
    run_customer_pipeline(input_file, use_live_market_intel=True, run_tests_after=True)
