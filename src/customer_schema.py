"""
Customer Data Schema Definition for BFWAI AI Inventory Decision Agent.

Defines required columns, data types, forbidden/ignored columns, and schema validation rules
for customer-uploaded historical inventory and sales CSV datasets.
"""

from typing import List, Dict, Tuple
import pandas as pd

# Strictly required schema columns that MUST be present in customer upload for forecasting/processing
STRICT_REQUIRED_COLUMNS: List[str] = [
    "date",
    "product_id",
    "units_sold"
]

# Product metadata fields (with safe fallbacks if missing)
PRODUCT_INFO_COLUMNS: List[str] = [
    "product_name",
    "category",
    "brand"
]

# Inventory snapshot fields (with safe fallbacks if missing)
INVENTORY_COLUMNS: List[str] = [
    "current_inventory",
    "reserved_inventory",
    "incoming_inventory"
]

# Supplier specification fields (with safe fallbacks if missing)
SUPPLIER_COLUMNS: List[str] = [
    "lead_time_days",
    "minimum_order_quantity",
    "reorder_point"
]

# Optional business information fields
OPTIONAL_COLUMNS: List[str] = [
    "revenue",
    "price",
    "unit_cost",
    "discount_pct",
    "promotion",
    "holiday_event",
    "rating",
    "review_count",
    "competitor_price"
]

# Combined list of standard columns
REQUIRED_COLUMNS: List[str] = (
    STRICT_REQUIRED_COLUMNS + PRODUCT_INFO_COLUMNS + INVENTORY_COLUMNS + SUPPLIER_COLUMNS + OPTIONAL_COLUMNS
)

# Market intelligence or forecast fields supplied by customer that MUST BE IGNORED / DROPPED
DISCARD_COLUMNS: List[str] = [
    "market_signal",
    "market_impact",
    "market_confidence",
    "market_relevance",
    "evidence_strength",
    "news_available",
    "news_count",
    "search_interest_index",
    "demand_forecast",
    "forecast_7d",
    "forecast_30d",
    "inventory_decision",
    "recommended_order_quantity"
]

# Target data types for parsing and validation
COLUMN_TYPES: Dict[str, str] = {
    "date": "datetime",
    "product_id": "string",
    "product_name": "string",
    "category": "string",
    "brand": "string",
    "units_sold": "int",
    "revenue": "float",
    "price": "float",
    "unit_cost": "float",
    "discount_pct": "float",
    "promotion": "int",
    "holiday_event": "int",
    "rating": "float",
    "review_count": "int",
    "current_inventory": "int",
    "reserved_inventory": "int",
    "incoming_inventory": "int",
    "lead_time_days": "int",
    "minimum_order_quantity": "int",
    "reorder_point": "int",
    "competitor_price": "float"
}


def validate_customer_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validates a customer-uploaded DataFrame against required schema rules.
    Checks strictly required columns (date, product_id, units_sold).
    
    Args:
        df: Pandas DataFrame loaded from customer upload CSV.
        
    Returns:
        Tuple of (is_valid: bool, errors: List[str])
    """
    errors: List[str] = []
    
    if df is None or df.empty:
        return False, ["Customer dataset is empty or None."]
        
    missing_strict = [col for col in STRICT_REQUIRED_COLUMNS if col not in df.columns]
    if missing_strict:
        errors.append(f"Missing required columns in customer dataset: {missing_strict}")
        
    return len(errors) == 0, errors

