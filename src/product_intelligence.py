"""
Product Intelligence Orchestrator for BFWAI AI Inventory Decision Agent.

Combines Customer Data, Customer Demand Forecasts, Latest Inventory/Supplier Metrics,
and Gemini + NewsAPI Market Intelligence into the final decision engine input format:
- BFWAI/handoff/inventory_decision_input.json
"""

from typing import Dict, Any, List, Optional
import os
import json
import logging
from datetime import datetime, timezone
import pandas as pd

from src.market_intelligence import (
    get_product_metadata,
    get_product_keywords,
    fetch_news,
    analyze_all_products_batch,
    _build_fallback_response
)

logger = logging.getLogger("BFWAI.ProductIntelligence")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

_DEFAULT_DECISION_INPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "handoff", "inventory_decision_input.json")


def build_inventory_decision_input_json(
    customer_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    output_json_path: str = _DEFAULT_DECISION_INPUT_PATH,
    use_live_market_intel: bool = True,
    source_file: str = "data/processed/customer_input.csv"
) -> Dict[str, Any]:
    """
    Generates the official Inventory Decision Input JSON (handoff/inventory_decision_input.json).
    
    Uses BATCHED Gemini API calls to stay within free-tier quota limits.
    
    Args:
        customer_df: Cleaned customer dataset from adapt_customer_data.
        forecast_df: Customer predictions DataFrame from predict_customer_demand.
        output_json_path: Destination path for inventory_decision_input.json.
        use_live_market_intel: If True, calls batch market intelligence for all products.
        source_file: Source customer CSV filename descriptor.
        
    Returns:
        Structured input dictionary saved to output_json_path.
    """
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    logger.info(f"Building official Inventory Decision Input JSON dataset at '{output_json_path}'...")
    
    # Get latest snapshot per product from customer dataset
    latest_customer_records = []
    for pid, group in customer_df.groupby("product_id"):
        sorted_grp = group.sort_values(by="date")
        latest_rec = sorted_grp.iloc[-1].to_dict()
        latest_customer_records.append(latest_rec)
        
    latest_df = pd.DataFrame(latest_customer_records)
    
    # Merge with forecast predictions
    merged = pd.merge(latest_df, forecast_df[["product_id", "forecast_7d", "forecast_30d", "sales_trend"]], on="product_id", how="left")
    
    # BATCH Market Intelligence: Fetch all news FIRST, then ONE Gemini call
    batch_mi_results: Dict[str, Dict[str, Any]] = {}
    
    if use_live_market_intel:
        logger.info("Step 4a: Fetching news articles for ALL products (using cache where available)...")
        products_with_articles = []
        
        for _, row in merged.iterrows():
            pid = str(row["product_id"])
            row_dict = row.to_dict()
            try:
                product_meta = get_product_metadata(row_dict)
                keywords = get_product_keywords(row_dict)
                articles = fetch_news(pid, keywords, use_cache=True)
                products_with_articles.append({
                    "product": product_meta,
                    "articles": articles
                })
            except Exception as e:
                logger.warning(f"Product {pid}: Failed to fetch news: {e}")
                products_with_articles.append({
                    "product": {"product_id": pid, "product_name": str(row.get("product_name", pid)),
                                "category": str(row.get("category", "General")), "brand": str(row.get("brand", "Generic"))},
                    "articles": []
                })
        
        logger.info(f"Step 4b: Sending ALL {len(products_with_articles)} products to Gemini in ONE batch call...")
        batch_mi_results = analyze_all_products_batch(products_with_articles)
        logger.info(f"Step 4c: Batch Gemini analysis complete. Got results for {len(batch_mi_results)} products.")

    # Assemble Final Products List
    products_list = []
    for _, row in merged.iterrows():
        pid = str(row["product_id"])
        p_name = str(row.get("product_name", pid))
        category = str(row.get("category", "General"))
        brand = str(row.get("brand", "Generic"))
        
        # Forecast info
        f_7d = int(row.get("forecast_7d", 0))
        f_30d = int(row.get("forecast_30d", 0))
        sales_trend = str(row.get("sales_trend", "stable"))
        
        # Inventory info
        curr_inv = int(row.get("current_inventory", 0))
        res_inv = int(row.get("reserved_inventory", 0))
        inc_inv = int(row.get("incoming_inventory", 0))
        
        # Supplier info
        lead_time = int(row.get("lead_time_days", 7))
        moq = int(row.get("minimum_order_quantity", 10))
        reorder_pt = int(row.get("reorder_point", 50))
        
        # Market Intelligence from batch results
        m_intel = batch_mi_results.get(pid, {})
            
        m_signal = str(m_intel.get("market_signal", "neutral")).lower()
        if m_signal not in ["positive", "neutral", "negative"]:
            m_signal = "neutral"
            
        m_impact = float(m_intel.get("market_impact", 0.0))
        m_conf = float(m_intel.get("confidence", m_intel.get("market_confidence", 0.0)))
        m_rel = float(m_intel.get("relevance", m_intel.get("market_relevance", 0.0)))
        ev_strength = float(m_intel.get("evidence_strength", 0.0))
        
        has_articles = bool(m_intel.get("articles_analyzed", 0) > 0)
        is_irrelevant = m_intel.get("evidence_type", []) == ["irrelevant"]
        
        if has_articles and not is_irrelevant:
            ext_available = True
            m_reason = str(m_intel.get("reason", m_intel.get("evidence_summary", "Analyzed external market news.")))
            key_events = list(m_intel.get("key_events", []))
            risks = list(m_intel.get("risks", []))
            opps = list(m_intel.get("opportunities", []))
        elif m_intel and (m_conf > 0.0 or m_impact > 0.0 or m_signal != "neutral"):
            ext_available = True
            m_reason = str(m_intel.get("reason", m_intel.get("evidence_summary", "Analyzed market intelligence evidence.")))
            key_events = list(m_intel.get("key_events", []))
            risks = list(m_intel.get("risks", []))
            opps = list(m_intel.get("opportunities", []))
        else:
            ext_available = False
            m_signal = "neutral"
            m_reason = str(m_intel.get("reason", "No external news articles or market intelligence available for this product."))
            key_events = []
            risks = []
            opps = []
            
        product_obj = {
            "product_id": pid,
            "product_name": p_name,
            "category": category,
            "brand": brand,
            "forecast": {
                "forecast_7d": f_7d,
                "forecast_30d": f_30d,
                "sales_trend": sales_trend
            },
            "inventory": {
                "current_inventory": curr_inv,
                "reserved_inventory": res_inv,
                "incoming_inventory": inc_inv
            },
            "supplier": {
                "lead_time_days": lead_time,
                "minimum_order_quantity": moq,
                "reorder_point": reorder_pt
            },
            "market_intelligence": {
                "market_signal": m_signal,
                "market_impact": m_impact,
                "market_confidence": m_conf,
                "market_relevance": m_rel,
                "evidence_strength": ev_strength,
                "external_signal_available": ext_available,
                "market_reason": m_reason,
                "key_events": key_events,
                "risks": risks,
                "opportunities": opps
            }
        }
        products_list.append(product_obj)
        
    decision_input_dict = {
        "project": "BFWAI",
        "handoff_version": "1.0",
        "source": source_file,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "products": products_list
    }
    
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(decision_input_dict, f, indent=2)
        
    # Also write to legacy handoff path for backward compatibility
    try:
        os.makedirs(os.path.dirname(_LEGACY_HANDOFF_PATH), exist_ok=True)
        with open(_LEGACY_HANDOFF_PATH, "w", encoding="utf-8") as f:
            json.dump(decision_input_dict, f, indent=2)
    except Exception:
        pass
        
    logger.info(f"Successfully generated Inventory Decision Input JSON ({len(products_list)} products) saved to '{output_json_path}'")
    return decision_input_dict


# Backwards compatibility alias
def build_person2_handoff_json(*args, **kwargs):
    return build_inventory_decision_input_json(*args, **kwargs)
