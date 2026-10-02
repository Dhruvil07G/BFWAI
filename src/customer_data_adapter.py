"""
Customer Data Adapter Module for BFWAI AI Inventory Decision Agent.

Loads customer-uploaded CSV files, validates required schema and data types,
strips/ignores customer-supplied market intelligence or forecast values,
handles missing data and duplicates, and sorts historical records.
"""

from typing import Optional
import os
import logging
import pandas as pd
import numpy as np

from src.customer_schema import (
    REQUIRED_COLUMNS,
    DISCARD_COLUMNS,
    COLUMN_TYPES,
    validate_customer_schema
)

logger = logging.getLogger("BFWAI.CustomerDataAdapter")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def adapt_customer_data(
    input_path: str,
    output_processed_path: str = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "customer_input.csv")
) -> pd.DataFrame:
    """
    Loads, cleans, validates, and adapts a customer-uploaded inventory CSV.
    Saves the processed input CSV to data/processed/customer_input.csv.
    
    Args:
        input_path: Absolute or relative file path to customer CSV upload.
        output_processed_path: Target path for saving cleaned processed customer CSV.
        
    Returns:
        Cleaned, validated, and sorted Pandas DataFrame ready for customer-specific forecasting.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Customer dataset file not found: '{input_path}'")
        
    logger.info(f"Loading customer dataset from '{input_path}'...")
    
    try:
        df = pd.read_csv(input_path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Customer CSV file '{input_path}' is completely empty.")
    except Exception as e:
        raise ValueError(f"Failed to read CSV file '{input_path}': {e}")
        
    if df is None or df.empty:
        raise ValueError(f"Customer dataset loaded from '{input_path}' is empty.")
        
    # 1. Automatically populate missing expected columns from schema with NaN/NULL
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            logger.info(f"Expected column '{col}' missing from uploaded dataset. Auto-populating with NULL.")
            df[col] = np.nan
        
    # 2. Ignore / Strip Customer-Supplied Market Intelligence or Forecast Columns
    dropped_cols = [c for c in DISCARD_COLUMNS if c in df.columns]
    if dropped_cols:
        logger.info(f"Dropping customer-supplied market intelligence/forecast columns: {dropped_cols}")
        df = df.drop(columns=dropped_cols)
        
    # 3. Clean & Auto-Impute Critical Key Fields (Product ID & Date & Units Sold)
    # Product ID Imputation
    if df["product_id"].isna().all() or (df["product_id"].astype(str).str.strip() == "").all():
        if "product_name" in df.columns and not df["product_name"].isna().all():
            df["product_id"] = "P_" + df["product_name"].astype(str).str.upper().str.replace(r'[^A-Z0-9]', '_', regex=True)
        else:
            df["product_id"] = [f"P{i+1:03d}" for i in range(len(df))]
    else:
        new_pids = []
        for idx, pid in enumerate(df["product_id"]):
            s_pid = str(pid).strip() if pd.notna(pid) else ""
            if not s_pid or s_pid.lower() in ["nan", "none", "null"]:
                new_pids.append(f"P{idx+1:03d}")
            else:
                new_pids.append(s_pid)
        df["product_id"] = new_pids

    # Date Parsing & Imputation
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().all():
        # Generate N daily dates ending today
        n_rows = len(df)
        end_date = pd.Timestamp.now().normalize()
        start_date = end_date - pd.Timedelta(days=n_rows - 1)
        df["date"] = pd.date_range(start=start_date, periods=n_rows, freq="D")
    else:
        # Interpolate missing dates
        df["date"] = df["date"].ffill().bfill()
        
    if df.empty:
        raise ValueError("Customer dataset contains no valid date records after date parsing.")
        
    # 4. Fill Missing Metadata Fields with Safe Defaults if absent
    if "product_name" not in df.columns:
        logger.info("Optional column 'product_name' missing. Defaulting to product_id.")
        df["product_name"] = df["product_id"]
    else:
        df["product_name"] = df["product_name"].fillna(df["product_id"]).astype(str).str.strip()

    if "category" not in df.columns:
        logger.info("Optional column 'category' missing. Defaulting to 'General'.")
        df["category"] = "General"
    else:
        df["category"] = df["category"].fillna("General").astype(str).str.strip()

    if "brand" not in df.columns:
        logger.info("Optional column 'brand' missing. Defaulting to 'Generic'.")
        df["brand"] = "Generic"
    else:
        df["brand"] = df["brand"].fillna("Generic").astype(str).str.strip()
            
    # 5. Fill Missing Inventory & Supplier Fields with Safe Defaults if absent
    if "current_inventory" not in df.columns:
        df["current_inventory"] = 0
    if "reserved_inventory" not in df.columns:
        df["reserved_inventory"] = 0
    if "incoming_inventory" not in df.columns:
        df["incoming_inventory"] = 0
    if "lead_time_days" not in df.columns:
        df["lead_time_days"] = 7
    if "minimum_order_quantity" not in df.columns:
        df["minimum_order_quantity"] = 1
    if "reorder_point" not in df.columns:
        df["reorder_point"] = 0

    # 6. Fill Missing Optional Business Columns with Safe Defaults if absent
    if "price" not in df.columns:
        df["price"] = 0.0
    if "unit_cost" not in df.columns:
        df["unit_cost"] = df["price"] * 0.7
    if "revenue" not in df.columns:
        df["revenue"] = df["units_sold"] * df["price"]
    if "discount_pct" not in df.columns:
        df["discount_pct"] = 0.0
    if "promotion" not in df.columns:
        df["promotion"] = 0
    if "holiday_event" not in df.columns:
        df["holiday_event"] = 0
    if "rating" not in df.columns:
        df["rating"] = 0.0
    if "review_count" not in df.columns:
        df["review_count"] = 0
    if "competitor_price" not in df.columns:
        df["competitor_price"] = df["price"]

    # 7. Convert and Clean Numeric Fields
    numeric_cols = [
        "units_sold", "revenue", "price", "unit_cost", "discount_pct",
        "promotion", "holiday_event", "rating", "review_count",
        "current_inventory", "reserved_inventory", "incoming_inventory",
        "lead_time_days", "minimum_order_quantity", "reorder_point",
        "competitor_price"
    ]
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
            
    # Impute missing numeric values logically
    df["units_sold"] = df["units_sold"].fillna(0).clip(lower=0)
    df["price"] = df["price"].fillna(0.0).clip(lower=0.0)
    df["unit_cost"] = df["unit_cost"].fillna(df["price"] * 0.7).clip(lower=0.0)
    df["revenue"] = df["revenue"].fillna(df["units_sold"] * df["price"]).clip(lower=0.0)
    df["discount_pct"] = df["discount_pct"].fillna(0.0).clip(0.0, 100.0)
    df["promotion"] = df["promotion"].fillna(0).astype(int)
    df["holiday_event"] = df["holiday_event"].fillna(0).astype(int)
    df["rating"] = df["rating"].fillna(0.0).clip(0.0, 5.0)
    df["review_count"] = df["review_count"].fillna(0).astype(int)
    df["competitor_price"] = df["competitor_price"].fillna(df["price"])
    df["current_inventory"] = df["current_inventory"].fillna(0).clip(lower=0)
    df["reserved_inventory"] = df["reserved_inventory"].fillna(0).clip(lower=0)
    df["incoming_inventory"] = df["incoming_inventory"].fillna(0).clip(lower=0)
    df["lead_time_days"] = df["lead_time_days"].fillna(7).clip(lower=1)
    df["minimum_order_quantity"] = df["minimum_order_quantity"].fillna(1).clip(lower=1)
    df["reorder_point"] = df["reorder_point"].fillna(0).clip(lower=0)
        
    # Forward-fill / back-fill inventory and supplier specs per product group (NEVER sum across rows)
    for prod_id, grp_idx in df.groupby("product_id").groups.items():
        for inv_col in ["current_inventory", "reserved_inventory", "incoming_inventory",
                        "lead_time_days", "minimum_order_quantity", "reorder_point"]:
            col_vals = df.loc[grp_idx, inv_col].ffill().bfill().fillna(0)
            df.loc[grp_idx, inv_col] = col_vals
                
    # 8. Deduplicate records based on (product_id, date)
    duplicate_count = df.duplicated(subset=["product_id", "date"]).sum()
    if duplicate_count > 0:
        logger.info(f"Removing {duplicate_count} duplicate records for (product_id, date).")
        df = df.drop_duplicates(subset=["product_id", "date"], keep="last")
        
    # 9. Sort Records chronologically by product_id and date
    df = df.sort_values(by=["product_id", "date"]).reset_index(drop=True)

    # 10. Save processed customer input CSV and sync forecasting_dataset.csv & product_catalog.csv
    if output_processed_path:
        os.makedirs(os.path.dirname(output_processed_path), exist_ok=True)
        df.to_csv(output_processed_path, index=False)
        logger.info(f"Saved processed customer input dataset to '{output_processed_path}'.")

    # Sync forecasting_dataset.csv
    processed_dir = os.path.dirname(output_processed_path) if output_processed_path else os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    forecasting_ds_path = os.path.join(processed_dir, "forecasting_dataset.csv")
    df.to_csv(forecasting_ds_path, index=False)
    logger.info(f"Updated '{forecasting_ds_path}' with {len(df)} records.")

    # Sync product_catalog.csv
    catalog_path = os.path.join(processed_dir, "product_catalog.csv")
    catalog_records = []
    for pid, grp in df.groupby("product_id"):
        latest_rec = grp.sort_values(by="date").iloc[-1]
        p_name = str(latest_rec.get("product_name", pid))
        cat = str(latest_rec.get("category", "General"))
        br = str(latest_rec.get("brand", "Generic"))
        kw = f"{p_name}, {br}, {cat}, {p_name} supply chain, {cat} market trends"
        catalog_records.append({
            "product_id": str(pid),
            "product_name": p_name,
            "category": cat,
            "brand": br,
            "news_keywords": kw
        })
    catalog_df = pd.DataFrame(catalog_records)
    catalog_df.to_csv(catalog_path, index=False)
    logger.info(f"Updated '{catalog_path}' with {len(catalog_df)} product entries.")

    logger.info(f"Successfully adapted customer dataset: {len(df)} records across {df['product_id'].nunique()} products.")
    return df


