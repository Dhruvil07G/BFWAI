"""
Customer-Specific Demand Forecasting Pipeline for BFWAI AI Inventory Decision Agent.

Trains a demand forecasting model exclusively on customer-uploaded historical sales data.
Generates 7-day and 30-day demand predictions along with sales trends per product.

Important Architecture Rules:
1. Baseline development model (models/demand_forecasting_model.pkl) is NEVER loaded or used.
2. Market Intelligence (market_signal, market_impact, Gemini, NewsAPI) is NEVER used as a forecasting feature.
3. Customer models are saved under models/customer/ separately.
"""

from typing import Dict, Any, Tuple, Optional
import os
import logging
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

logger = logging.getLogger("BFWAI.CustomerForecasting")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Default paths for customer model artifacts
_DEFAULT_CUSTOMER_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "customer")
_DEFAULT_CUSTOMER_METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "reports", "customer_forecasting_metrics.csv")
_DEFAULT_CUSTOMER_FORECAST_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "predictions", "customer_forecasts.csv")
_DEFAULT_CUSTOMER_PRED_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "predictions", "customer_demand_predictions.csv")


def extract_customer_forecasting_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs feature matrix from customer historical sales data.
    Dynamically handles present and missing optional business/pricing columns.
    
    Target variables:
    - target_7d_units: Cumulative 7-day future sales
    """
    df = df.copy()
    df = df.sort_values(by=["product_id", "date"]).reset_index(drop=True)
    
    # Temporal features
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    
    # Ensure default numeric features exist if missing
    for col, default_val in [
        ("price", 0.0), ("unit_cost", 0.0), ("discount_pct", 0.0),
        ("promotion", 0), ("holiday_event", 0), ("rating", 0.0),
        ("review_count", 0), ("competitor_price", 0.0)
    ]:
        if col not in df.columns:
            df[col] = default_val
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default_val)
    
    # Pricing & competitive ratios
    df["price_vs_competitor"] = np.where(
        df["competitor_price"] > 0,
        df["price"] / np.maximum(df["competitor_price"], 0.001),
        1.0
    )
    df["effective_price"] = df["price"] * (1.0 - (df["discount_pct"] / 100.0))
    
    # Lag and rolling features calculated per product
    feature_dfs = []
    for pid, group in df.groupby("product_id"):
        grp = group.copy()
        
        # Lag features
        grp["sales_lag_1"] = grp["units_sold"].shift(1)
        grp["sales_lag_7"] = grp["units_sold"].shift(7)
        grp["sales_lag_14"] = grp["units_sold"].shift(14)
        grp["sales_lag_30"] = grp["units_sold"].shift(30)
        
        # Rolling sales statistics
        grp["sales_rolling_mean_7"] = grp["units_sold"].shift(1).rolling(7, min_periods=1).mean()
        grp["sales_rolling_std_7"] = grp["units_sold"].shift(1).rolling(7, min_periods=1).std().fillna(0)
        grp["sales_rolling_mean_14"] = grp["units_sold"].shift(1).rolling(14, min_periods=1).mean()
        grp["sales_rolling_mean_30"] = grp["units_sold"].shift(1).rolling(30, min_periods=1).mean()
        
        # 7-day cumulative target horizon (forward rolling sum)
        forward_7d = grp["units_sold"].iloc[::-1].rolling(7, min_periods=1).sum().iloc[::-1].shift(-7)
        grp["target_7d_units"] = forward_7d
        
        feature_dfs.append(grp)
        
    full_feat_df = pd.concat(feature_dfs, ignore_index=True)
    return full_feat_df


def train_customer_forecasting_model(
    customer_data: pd.DataFrame,
    model_dir: str = _DEFAULT_CUSTOMER_MODEL_DIR,
    metrics_path: str = _DEFAULT_CUSTOMER_METRICS_PATH
) -> Tuple[Any, Dict[str, float]]:
    """
    Trains a customer-specific demand forecasting model using historical sales data.
    Saves trained model artifact under models/customer/ (separate from baseline model).
    """
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    
    logger.info("Extracting customer forecasting features...")
    feat_df = extract_customer_forecasting_features(customer_data)
    
    # Drop rows where target 7d horizon or critical lags are NaN
    train_df = feat_df.dropna(subset=["sales_lag_1", "target_7d_units"]).copy()
    
    feature_cols = [
        "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_30",
        "sales_rolling_mean_7", "sales_rolling_std_7", "sales_rolling_mean_14", "sales_rolling_mean_30",
        "price", "unit_cost", "discount_pct", "effective_price",
        "promotion", "holiday_event", "rating", "review_count",
        "competitor_price", "price_vs_competitor",
        "dayofweek", "month", "quarter", "is_weekend"
    ]
    
    if len(train_df) < 5:
        logger.warning("Very sparse customer dataset. Returning dummy regressor model.")
        from sklearn.dummy import DummyRegressor
        model = DummyRegressor(strategy="mean")
        if len(train_df) > 0:
            model.fit(train_df[feature_cols].fillna(0), train_df["target_7d_units"])
        else:
            dummy_x = pd.DataFrame(np.zeros((1, len(feature_cols))), columns=feature_cols)
            dummy_y = pd.Series([0.0])
            model.fit(dummy_x, dummy_y)
        metrics = {"dataset": "customer_historical", "train_samples": len(train_df), "test_samples": 0, "mae": 0.0, "rmse": 0.0, "mape_pct": 0.0}
        model_file = os.path.join(model_dir, "customer_demand_model.pkl")
        joblib.dump(model, model_file)
        return model, metrics
        
    X = train_df[feature_cols].fillna(0)
    y = train_df["target_7d_units"]
    
    # Chronological Train-Test Split (80% Train, 20% Test)
    split_idx = int(len(train_df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    if len(X_test) == 0:
        X_train, X_test = X, X
        y_train, y_test = y, y
        
    logger.info(f"Training HistGradientBoostingRegressor on {len(X_train)} customer training samples...")
    model = HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate Model
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    try:
        mape = mean_absolute_percentage_error(y_test + 1, y_pred + 1) * 100.0
    except Exception:
        mape = 0.0
        
    metrics = {
        "dataset": "customer_historical",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "mape_pct": round(float(mape), 2)
    }
    
    logger.info(f"Customer Forecasting Model Trained successfully. Metrics: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")
    
    # Save Customer Model
    model_file = os.path.join(model_dir, "customer_demand_model.pkl")
    joblib.dump(model, model_file)
    logger.info(f"Saved customer model artifact to '{model_file}'")
    
    # Save Metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved evaluation metrics to '{metrics_path}'")
    
    return model, metrics


def predict_customer_demand(
    model_or_path: Any,
    customer_data: pd.DataFrame,
    output_pred_path: str = _DEFAULT_CUSTOMER_FORECAST_PATH
) -> pd.DataFrame:
    """
    Generates 7-day and 30-day demand predictions and sales trends for each product in customer dataset.
    Uses documented fallback method (recent average demand) for products with insufficient history (< 14 days).
    
    Args:
        model_or_path: Trained model object or file path to customer model pickle.
        customer_data: Cleaned DataFrame adapted by adapt_customer_data.
        output_pred_path: File path to save customer_forecasts.csv.
        
    Returns:
        DataFrame containing product-level forecast results.
    """
    os.makedirs(os.path.dirname(output_pred_path), exist_ok=True)
    
    # Load model if path provided
    model = None
    if isinstance(model_or_path, str):
        if os.path.exists(model_or_path):
            try:
                model = joblib.load(model_or_path)
            except Exception as e:
                logger.warning(f"Could not load customer model from '{model_or_path}': {e}. Using fallback forecasting.")
        else:
            logger.info(f"Model path '{model_or_path}' does not exist. Using fallback forecasting method.")
    else:
        model = model_or_path

        
    logger.info("Generating features for customer demand prediction...")
    feat_df = extract_customer_forecasting_features(customer_data)
    
    feature_cols = [
        "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_30",
        "sales_rolling_mean_7", "sales_rolling_std_7", "sales_rolling_mean_14", "sales_rolling_mean_30",
        "price", "unit_cost", "discount_pct", "effective_price",
        "promotion", "holiday_event", "rating", "review_count",
        "competitor_price", "price_vs_competitor",
        "dayofweek", "month", "quarter", "is_weekend"
    ]
    
    results = []
    for pid, group in feat_df.groupby("product_id"):
        sorted_group = group.sort_values(by="date")
        latest_rec = sorted_group.iloc[-1:]
        
        p_name = latest_rec["product_name"].values[0] if "product_name" in latest_rec.columns else str(pid)
        category = latest_rec["category"].values[0] if "category" in latest_rec.columns else "General"
        brand = latest_rec["brand"].values[0] if "brand" in latest_rec.columns else "Generic"
        
        history_length = len(sorted_group)
        
        # Check if model is None or product has insufficient history (< 14 days of records)
        if model is None or history_length < 14 or sorted_group["units_sold"].isnull().all():
            fallback_used = True
            logger.info(f"Product '{pid}': Insufficient historical data ({history_length} days). Using fallback average demand.")
            
            # Documented Fallback Method: Recent average daily sales
            avg_daily_demand = float(sorted_group["units_sold"].tail(7).mean()) if history_length > 0 else 0.0
            forecast_7d = max(0, int(round(avg_daily_demand * 7.0)))
            forecast_30d = max(0, int(round(avg_daily_demand * 30.0)))
            
            # Trend calculation for fallback
            recent_7d_actual = float(sorted_group.tail(7)["units_sold"].sum()) if history_length >= 7 else float(sorted_group["units_sold"].sum())
            prev_7d_actual = float(sorted_group.head(max(1, history_length - 7))["units_sold"].sum()) if history_length > 7 else recent_7d_actual
            
            if prev_7d_actual > 0:
                trend_ratio = recent_7d_actual / prev_7d_actual
                if trend_ratio > 1.05:
                    trend = "increasing"
                elif trend_ratio < 0.95:
                    trend = "decreasing"
                else:
                    trend = "stable"
            else:
                trend = "stable"
        else:
            fallback_used = False
            # Prepare feature vector for latest state
            X_latest = latest_rec[feature_cols].fillna(0)
            
            try:
                pred_7d_raw = model.predict(X_latest)[0]
                forecast_7d = max(0, int(round(float(pred_7d_raw))))
            except Exception as e:
                logger.warning(f"Product '{pid}': Model prediction failed ({e}). Using average fallback.")
                avg_daily_demand = float(sorted_group["units_sold"].tail(14).mean())
                forecast_7d = max(0, int(round(avg_daily_demand * 7.0)))
                fallback_used = True

            # Extrapolate 30-day forecast
            forecast_30d = max(0, int(round(forecast_7d * (30.0 / 7.0))))
            
            # Compute Recent 7-day actual historical sales for trend comparison
            recent_7d_actual = sorted_group.tail(7)["units_sold"].sum()
            
            if recent_7d_actual > 0:
                ratio = forecast_7d / float(recent_7d_actual)
                if ratio > 1.05:
                    trend = "increasing"
                elif ratio < 0.95:
                    trend = "decreasing"
                else:
                    trend = "stable"
            else:
                trend = "stable"
            
        results.append({
            "product_id": str(pid),
            "product_name": str(p_name),
            "category": str(category),
            "brand": str(brand),
            "forecast_7d": forecast_7d,
            "forecast_30d": forecast_30d,
            "sales_trend": trend,
            "fallback_used": fallback_used
        })
        
    pred_df = pd.DataFrame(results)
    
    # Save output to customer_forecasts.csv as required by specification
    pred_df.to_csv(output_pred_path, index=False)
    logger.info(f"Saved customer forecasts for {len(pred_df)} products to '{output_pred_path}'")
    
    # Also save to customer_demand_predictions.csv for backwards compatibility
    alt_pred_path = _DEFAULT_CUSTOMER_PRED_PATH
    if output_pred_path != alt_pred_path:
        os.makedirs(os.path.dirname(alt_pred_path), exist_ok=True)
        pred_df.to_csv(alt_pred_path, index=False)
        
    return pred_df

