"""
Data validation module for BFWAI AI Inventory Decision Agent.

This module provides functions to load, inspect, validate, and report data quality
metrics for the inventory dataset.
"""

from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np


def load_dataset(file_path: str) -> pd.DataFrame:
    """Loads dataset from specified file path and parses date column."""
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def get_dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """Returns dataset dimensions, column types, date range, product and category counts."""
    pids = sorted([str(p) for p in df['product_id'].dropna().unique().tolist()]) if 'product_id' in df.columns else []
    cats = sorted([str(c) for c in df['category'].dropna().unique().tolist()]) if 'category' in df.columns else []
    
    if 'date' in df.columns and not df['date'].empty:
        parsed_dates = pd.to_datetime(df['date'], errors='coerce').dropna()
        date_min = parsed_dates.min().strftime('%Y-%m-%d') if not parsed_dates.empty else "N/A"
        date_max = parsed_dates.max().strftime('%Y-%m-%d') if not parsed_dates.empty else "N/A"
    else:
        date_min, date_max = "N/A", "N/A"

    
    return {
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()},
        "date_min": date_min,
        "date_max": date_max,
        "num_unique_products": len(pids),
        "product_ids": pids,
        "num_categories": len(cats),
        "categories": cats
    }


def run_validation_checks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executes comprehensive data quality and validation checks on inventory dataset.
    Returns a pandas DataFrame summarizing all validation check results.
    """
    results = []

    # 1. Required Columns Check
    from src.customer_schema import STRICT_REQUIRED_COLUMNS
    missing_strict = [c for c in STRICT_REQUIRED_COLUMNS if c not in df.columns]
    results.append({
        "check_name": "Required Core Columns",
        "category": "Schema",
        "status": "PASS" if len(missing_strict) == 0 else "FAIL",
        "issues_found": len(missing_strict),
        "details": "All core required columns present." if len(missing_strict) == 0 else f"Missing required columns: {missing_strict}"
    })

    # 2. Duplicate checks
    dup_rows = df.duplicated().sum()
    results.append({
        "check_name": "Duplicate Rows",
        "category": "Integrity",
        "status": "PASS" if dup_rows == 0 else "WARN",
        "issues_found": dup_rows,
        "details": "No exact duplicate rows found." if dup_rows == 0 else f"{dup_rows} exact duplicate rows detected."
    })

    if "product_id" in df.columns and "date" in df.columns:
        dup_pk = df.duplicated(subset=['product_id', 'date']).sum()
        results.append({
            "check_name": "Duplicate (product_id, date) Key Combinations",
            "category": "Integrity",
            "status": "PASS" if dup_pk == 0 else "WARN",
            "issues_found": dup_pk,
            "details": "Unique composite primary key (product_id, date) verified." if dup_pk == 0 else f"{dup_pk} duplicate key pairs detected."
        })

    # 3. Missing Values Check
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    null_cols = null_counts[null_counts > 0].to_dict()
    results.append({
        "check_name": "Missing Values",
        "category": "Completeness",
        "status": "INFO" if total_nulls > 0 else "PASS",
        "issues_found": total_nulls,
        "details": f"Missing values present in columns: {null_cols}" if total_nulls > 0 else "No missing values found."
    })

    # 4. Invalid Product IDs
    if "product_id" in df.columns:
        invalid_pids = df["product_id"].isna().sum() + (df["product_id"].astype(str).str.strip() == "").sum()
        results.append({
            "check_name": "Invalid Product IDs",
            "category": "Validity",
            "status": "PASS" if invalid_pids == 0 else "FAIL",
            "issues_found": invalid_pids,
            "details": "All product IDs are valid and non-empty." if invalid_pids == 0 else f"{invalid_pids} null or blank product IDs."
        })

    # 5. Numeric & Value Range Checks
    if "units_sold" in df.columns:
        neg_units = (pd.to_numeric(df['units_sold'], errors='coerce') < 0).sum()
        results.append({
            "check_name": "Negative Units Sold",
            "category": "Value Range",
            "status": "PASS" if neg_units == 0 else "WARN",
            "issues_found": neg_units,
            "details": "All units_sold values are non-negative." if neg_units == 0 else f"{neg_units} negative sales entries found."
        })

    if "revenue" in df.columns:
        neg_revenue = (pd.to_numeric(df['revenue'], errors='coerce') < 0).sum()
        results.append({
            "check_name": "Negative Revenue",
            "category": "Value Range",
            "status": "PASS" if neg_revenue == 0 else "WARN",
            "issues_found": neg_revenue,
            "details": "All revenue values are non-negative." if neg_revenue == 0 else f"{neg_revenue} negative revenue entries found."
        })

    if "price" in df.columns:
        invalid_price = (pd.to_numeric(df['price'], errors='coerce') < 0).sum()
        results.append({
            "check_name": "Negative Price",
            "category": "Value Range",
            "status": "PASS" if invalid_price == 0 else "WARN",
            "issues_found": invalid_price,
            "details": "All unit prices are non-negative." if invalid_price == 0 else f"{invalid_price} negative prices found."
        })

    inv_cols = [c for c in ['current_inventory', 'reserved_inventory', 'incoming_inventory'] if c in df.columns]
    if inv_cols:
        invalid_inventory = 0
        for col in inv_cols:
            invalid_inventory += (pd.to_numeric(df[col], errors='coerce') < 0).sum()
        results.append({
            "check_name": "Invalid Inventory Quantities (< 0)",
            "category": "Value Range",
            "status": "PASS" if invalid_inventory == 0 else "WARN",
            "issues_found": invalid_inventory,
            "details": "All current, reserved, and incoming inventory counts are non-negative." if invalid_inventory == 0 else f"{invalid_inventory} negative inventory levels."
        })

    if "lead_time_days" in df.columns:
        invalid_lead_time = (pd.to_numeric(df['lead_time_days'], errors='coerce') <= 0).sum()
        results.append({
            "check_name": "Invalid Lead Time (<= 0 days)",
            "category": "Value Range",
            "status": "PASS" if invalid_lead_time == 0 else "WARN",
            "issues_found": invalid_lead_time,
            "details": "All lead times are positive." if invalid_lead_time == 0 else f"{invalid_lead_time} invalid lead times."
        })

    if "minimum_order_quantity" in df.columns:
        invalid_moq = (pd.to_numeric(df['minimum_order_quantity'], errors='coerce') <= 0).sum()
        results.append({
            "check_name": "Invalid Minimum Order Quantity (<= 0)",
            "category": "Value Range",
            "status": "PASS" if invalid_moq == 0 else "WARN",
            "issues_found": invalid_moq,
            "details": "All MOQ values are positive." if invalid_moq == 0 else f"{invalid_moq} invalid MOQ values."
        })

    if "reorder_point" in df.columns:
        invalid_rop = (pd.to_numeric(df['reorder_point'], errors='coerce') < 0).sum()
        results.append({
            "check_name": "Invalid Reorder Point (< 0)",
            "category": "Value Range",
            "status": "PASS" if invalid_rop == 0 else "WARN",
            "issues_found": invalid_rop,
            "details": "All reorder points are non-negative." if invalid_rop == 0 else f"{invalid_rop} negative reorder points."
        })

    # 6. Granularity & Product Record Counts
    if "product_id" in df.columns:
        counts = df.groupby('product_id').size()
        min_c, max_c = counts.min(), counts.max()
        results.append({
            "check_name": "Product Historical Record Granularity",
            "category": "Granularity",
            "status": "PASS",
            "issues_found": 0,
            "details": f"{len(counts)} unique products with {min_c} to {max_c} historical daily records per product."
        })

    # 7. Chronological Continuity Check
    if "product_id" in df.columns and "date" in df.columns:
        gaps_count = 0
        df_temp = df.copy()
        df_temp['date'] = pd.to_datetime(df_temp['date'], errors='coerce')
        for pid, group in df_temp.groupby('product_id'):
            group_sorted = group.dropna(subset=['date']).sort_values('date')
            if len(group_sorted) > 1:
                diffs = group_sorted['date'].diff()
                missing_days = (diffs > pd.Timedelta(days=1)).sum()
                gaps_count += missing_days

        results.append({
            "check_name": "Chronological Continuity",
            "category": "Time Series",
            "status": "PASS" if gaps_count == 0 else "INFO",
            "issues_found": gaps_count,
            "details": "Chronological continuity verified." if gaps_count == 0 else f"{gaps_count} missing date gaps detected."
        })

    return pd.DataFrame(results)

