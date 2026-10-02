"""
Person 2 Inventory Decision Engine for BFWAI AI Inventory Decision Agent.

Input:
BFWAI/handoff/person2_handoff.json

Output:
BFWAI/outputs/decisions/inventory_decisions.json

Performs deterministic, rule-based inventory calculations for every unique product in customer dataset:
1. available_inventory = current_inventory - reserved_inventory + incoming_inventory
2. average_daily_demand = forecast_7d / 7 (or forecast_30d / 30)
3. inventory_coverage_days = available_inventory / average_daily_demand
4. lead_time_demand = average_daily_demand * lead_time_days
5. Evaluates decision (INCREASE, MAINTAIN, REDUCE) using coverage, lead time demand, reorder point, trend, market intelligence.
6. Calculates recommended order quantity respecting MOQ (never negative).
"""

from typing import Dict, Any, List, Optional
import os
import json
import logging
from datetime import datetime, timezone
import pandas as pd

from backend.schemas import ProductHandoff
from backend.decision_engine import evaluate_decision
from backend.inventory_analysis import analyze_inventory

logger = logging.getLogger("BFWAI.InventoryDecisionEngine")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

_DEFAULT_INPUT_HANDOFF_PATH = os.path.join(os.path.dirname(__file__), "..", "handoff", "inventory_decision_input.json")
_DEFAULT_OUTPUT_DECISIONS_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "decisions", "inventory_decisions.json")


def run_inventory_decision_engine(
    handoff_json_path: str = _DEFAULT_INPUT_HANDOFF_PATH,
    output_decisions_path: str = _DEFAULT_OUTPUT_DECISIONS_PATH
) -> Dict[str, Any]:
    """
    Reads inventory_decision_input.json, evaluates deterministic inventory decisions for all products,
    and outputs inventory_decisions.json.
    
    Args:
        handoff_json_path: Path to Inventory Decision Input JSON file.
        output_decisions_path: Path to save decision output JSON.
        
    Returns:
        Structured output dictionary containing decision results and metrics.
    """
    if not os.path.exists(handoff_json_path):
        legacy_path = os.path.join(os.path.dirname(__file__), "..", "handoff", "person2_handoff.json")
        if os.path.exists(legacy_path):
            handoff_json_path = legacy_path
        else:
            raise FileNotFoundError(f"Decision input JSON file not found at: '{handoff_json_path}'")
        
    logger.info(f"Loading Person 2 handoff dataset from '{handoff_json_path}'...")
    with open(handoff_json_path, "r", encoding="utf-8") as f:
        handoff_data = json.load(f)
        
    raw_products = handoff_data.get("products", [])
    if not raw_products:
        logger.warning(f"No products found in handoff file '{handoff_json_path}'.")
        
    logger.info(f"Processing deterministic inventory decisions for {len(raw_products)} products...")
    
    decisions_list = []
    increase_count = 0
    maintain_count = 0
    reduce_count = 0
    total_rec_order_qty = 0
    
    for raw_p in raw_products:
        # Validate / parse product through Pydantic model
        product = ProductHandoff(**raw_p)
        
        # Evaluate deterministic rule-based decision
        eval_result = evaluate_decision(product)
        
        # Count decision categories
        if eval_result.decision == "INCREASE":
            increase_count += 1
        elif eval_result.decision == "REDUCE":
            reduce_count += 1
        else:
            maintain_count += 1
            
        total_rec_order_qty += eval_result.recommended_order_quantity
        
        # Core calculations as required by specification
        curr_inv = max(0, product.inventory.current_inventory)
        res_inv = max(0, product.inventory.reserved_inventory)
        inc_inv = max(0, product.inventory.incoming_inventory)
        avail_inv = max(0, curr_inv - res_inv + inc_inv)
        
        avg_daily_demand = (
            float(product.forecast.forecast_7d) / 7.0
            if product.forecast.forecast_7d > 0
            else (float(product.forecast.forecast_30d) / 30.0 if product.forecast.forecast_30d > 0 else 0.0)
        )
        avg_daily_demand = round(avg_daily_demand, 3)
        
        coverage_days = (
            round(avail_inv / avg_daily_demand, 2)
            if avg_daily_demand > 0
            else (999.0 if avail_inv > 0 else 0.0)
        )
        
        lead_time_days = max(1, product.supplier.lead_time_days)
        lead_time_demand = round(avg_daily_demand * lead_time_days, 2)
        
        decision_obj = {
            "product_id": product.product_id,
            "product_name": product.product_name,
            "category": product.category,
            "brand": product.brand,
            "decision": eval_result.decision,
            "confidence": eval_result.confidence,
            "stockout_risk": eval_result.stockout_risk,
            "days_of_stock": eval_result.days_of_stock,
            "reorder_point": eval_result.reorder_point,
            "recommended_order_quantity": max(0, eval_result.recommended_order_quantity),
            "decision_score": eval_result.decision_score,
            "reasons": eval_result.reasons,
            "metrics": {
                "available_inventory": avail_inv,
                "average_daily_demand": avg_daily_demand,
                "inventory_coverage_days": coverage_days,
                "lead_time_demand": lead_time_demand,
                "safety_stock": eval_result.metrics.safety_stock,
                "minimum_order_quantity": max(1, product.supplier.minimum_order_quantity)
            }
        }
        decisions_list.append(decision_obj)
        
    output_dict = {
        "project": "BFWAI",
        "decision_engine_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_handoff": handoff_json_path,
        "total_products": len(decisions_list),
        "decision_summary": {
            "INCREASE": increase_count,
            "MAINTAIN": maintain_count,
            "REDUCE": reduce_count,
            "total_recommended_order_quantity": total_rec_order_qty
        },
        "decisions": decisions_list
    }
    
    os.makedirs(os.path.dirname(output_decisions_path), exist_ok=True)
    with open(output_decisions_path, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=2)
        
    logger.info(
        f"Inventory Decision Engine run complete. Processed {len(decisions_list)} products. "
        f"Summary: INCREASE={increase_count}, MAINTAIN={maintain_count}, REDUCE={reduce_count}. "
        f"Saved to '{output_decisions_path}'"
    )
    return output_dict


if __name__ == "__main__":
    run_inventory_decision_engine()
