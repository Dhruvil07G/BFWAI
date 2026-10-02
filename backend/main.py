"""
FastAPI Backend Application for Person 2: AI Inventory Decision Agent.

Integrates:
- Person 1's handoff (handoff/person2_handoff.json)
- Inventory Analysis
- Deterministic Multi-Signal Decision Engine
- LLM / Deterministic Explanation Agent
- Alternative Product Re-Allocation Agent
- Human-in-the-Loop Approvals Workflow
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import uuid

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


from fastapi import FastAPI, HTTPException, Query, status, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import numpy as np
import io

from backend.schemas import (
    ProductHandoff,
    HandoffContainer,
    ProductInput,
    DecisionResult,
    FullAnalysisResponse,
    AlternativeProduct,
    ApprovalRequest,
    ApprovalRecord
)
from backend.inventory_analysis import analyze_inventory
from backend.decision_engine import evaluate_decision
from backend.explanation_agent import generate_llm_explanation, generate_deterministic_explanation
from backend.alternative_agent import find_alternative_products

from src.customer_schema import STRICT_REQUIRED_COLUMNS, REQUIRED_COLUMNS
from src.data_validation import get_dataset_overview, run_validation_checks
from src.customer_data_adapter import adapt_customer_data
from src.customer_forecasting import train_customer_forecasting_model, predict_customer_demand
from src.product_intelligence import build_person2_handoff_json
from src.inventory_decision_engine import run_inventory_decision_engine


# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("BFWAI.Person2Backend")

from contextlib import asynccontextmanager

def get_writeable_filepath(relative_path: str, for_write: bool = False) -> str:
    """
    Returns a writeable filepath. If running on Vercel or read-only filesystem,
    routes file write operations to /tmp/bfwai/...
    """
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    primary_path = os.path.join(root_dir, relative_path)
    tmp_path = os.path.join("/tmp", "bfwai", relative_path)
    
    if not for_write:
        if os.path.exists(primary_path):
            return primary_path
        if os.path.exists(tmp_path):
            return tmp_path
        return primary_path

    if os.getenv("VERCEL"):
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        return tmp_path

    try:
        os.makedirs(os.path.dirname(primary_path), exist_ok=True)
        test_file = primary_path + ".test_perm"
        with open(test_file, "w") as f:
            f.write("test")
        if os.path.exists(test_file):
            os.remove(test_file)
        return primary_path
    except (PermissionError, OSError):
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        return tmp_path

HANDOFF_FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "handoff", "inventory_decision_input.json")

# In-memory storage for human approval actions
_APPROVAL_RECORDS: Dict[str, ApprovalRecord] = {}
_PRODUCTS_CACHE: Dict[str, ProductHandoff] = {}


def load_handoff_data() -> Dict[str, ProductHandoff]:
    """Loads and validates the inventory decision input file."""
    target_path = get_writeable_filepath(os.path.join("handoff", "inventory_decision_input.json"), for_write=False)
    if not os.path.exists(target_path):
        logger.error(f"Decision input file not found at: {target_path}")
        return {}

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        products_dict: Dict[str, ProductHandoff] = {}
        for p_json in data.get("products", []):
            try:
                p_obj = ProductHandoff.model_validate(p_json)
                products_dict[p_obj.product_id] = p_obj
            except Exception as e:
                logger.warning(f"Failed to validate product item: {e}")
        return products_dict
    except Exception as e:
        logger.error(f"Failed to parse decision input JSON: {e}")
        return {}


@asynccontextmanager
async def lifespan(app: FastAPI):

    global _PRODUCTS_CACHE
    _PRODUCTS_CACHE = load_handoff_data()
    logger.info(f"Loaded {len(_PRODUCTS_CACHE)} products from Person 1 handoff.")
    yield


app = FastAPI(
    title="AI Inventory Decision Agent (Person 2)",
    description="Operational Decision Support Engine for Inventory Management with Deterministic Rules, LLM Explanations, and Human-in-the-Loop Approval.",
    version="1.0.0",
    default_response_class=JSONResponse,
    lifespan=lifespan
)


# Enable CORS for Frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint confirming API status and handoff availability."""
    global _PRODUCTS_CACHE
    handoff_exists = os.path.exists(HANDOFF_FILE_PATH)
    if not _PRODUCTS_CACHE and handoff_exists:
        _PRODUCTS_CACHE = load_handoff_data()
    return {
        "status": "healthy",
        "service": "AI Inventory Decision Agent (Person 2)",
        "handoff_loaded": len(_PRODUCTS_CACHE) > 0,
        "handoff_path": HANDOFF_FILE_PATH,
        "handoff_exists": handoff_exists,
        "total_products": len(_PRODUCTS_CACHE),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/products", tags=["Inventory & Decisions"])
def list_products(
    category: Optional[str] = None,
    decision_filter: Optional[str] = None,
    risk_filter: Optional[str] = None
):
    """
    Returns all products from the Person 1 handoff, enriched with real-time evaluated
    inventory metrics, stockout risks, decisions, and approval status.
    """
    global _PRODUCTS_CACHE
    if not _PRODUCTS_CACHE:
        _PRODUCTS_CACHE = load_handoff_data()

    results = []
    counts = {"total": 0, "increase": 0, "maintain": 0, "reduce": 0, "high_risk": 0}

    for pid, product in _PRODUCTS_CACHE.items():
        decision_res = evaluate_decision(product)
        
        d_val = decision_res.decision
        r_val = decision_res.stockout_risk
        
        # Tally metrics
        counts["total"] += 1
        if d_val == "INCREASE":
            counts["increase"] += 1
        elif d_val == "MAINTAIN":
            counts["maintain"] += 1
        elif d_val == "REDUCE":
            counts["reduce"] += 1
            
        if r_val == "HIGH":
            counts["high_risk"] += 1

        # Check filters
        if category and product.category.lower() != category.lower():
            continue
        if decision_filter and d_val.upper() != decision_filter.upper():
            continue
        if risk_filter and r_val.upper() != risk_filter.upper():
            continue

        approval = _APPROVAL_RECORDS.get(pid)

        item = {
            "product_id": product.product_id,
            "product_name": product.product_name,
            "category": product.category,
            "brand": product.brand,
            "current_inventory": product.inventory.current_inventory,
            "reserved_inventory": product.inventory.reserved_inventory,
            "incoming_inventory": product.inventory.incoming_inventory,
            "available_inventory": decision_res.metrics.available_inventory,
            "forecast_7d": product.forecast.forecast_7d,
            "forecast_30d": product.forecast.forecast_30d,
            "sales_trend": product.forecast.sales_trend,
            "average_daily_demand": decision_res.metrics.average_daily_demand,
            "days_of_stock": decision_res.days_of_stock,
            "supplier_lead_time": product.supplier.lead_time_days,
            "moq": product.supplier.minimum_order_quantity,
            "reorder_point": decision_res.reorder_point,
            "stockout_risk": decision_res.stockout_risk,
            "market_signal": product.market_intelligence.market_signal if product.market_intelligence else "neutral",
            "market_confidence": product.market_intelligence.market_confidence if product.market_intelligence else 0.0,
            "decision": decision_res.decision,
            "decision_score": decision_res.decision_score,
            "confidence": decision_res.confidence,
            "recommended_order_quantity": decision_res.recommended_order_quantity,
            "approval_status": approval.action if approval else "PENDING",
            "approval_record": approval
        }
        results.append(item)

    return {
        "summary": counts,
        "count": len(results),
        "products": results
    }


@app.get("/products/{product_id}", tags=["Inventory & Decisions"])
def get_product_details(product_id: str):
    """
    Returns comprehensive product details, complete inventory metrics,
    decision breakdown, AI explanation, alternative recommendations, and human approval status.
    """
    global _PRODUCTS_CACHE
    if not _PRODUCTS_CACHE:
        _PRODUCTS_CACHE = load_handoff_data()

    product = _PRODUCTS_CACHE.get(product_id)
    if not product:
        raise HTTPException(status_code=404, detail=f"Product with ID '{product_id}' not found in handoff data.")

    decision_res = evaluate_decision(product)
    explanation = generate_llm_explanation(product, decision_res)
    
    # Candidate alternative products
    catalog = list(_PRODUCTS_CACHE.values())
    alternatives = find_alternative_products(product, catalog, top_n=3)

    approval = _APPROVAL_RECORDS.get(product_id)

    return {
        "product": product,
        "decision": decision_res,
        "ai_explanation": explanation,
        "alternative_products": alternatives,
        "approval": approval
    }


@app.post("/decision", response_model=DecisionResult, tags=["Decision Engine"])
def compute_decision_endpoint(input_data: ProductInput):
    """
    Calculates inventory metrics, stockout risk, decision, recommended order quantity,
    and transparent reasons for an ad-hoc or custom product payload.
    """
    product = input_data.to_product_handoff()
    return evaluate_decision(product)


@app.post("/analyze", response_model=FullAnalysisResponse, tags=["Decision Engine"])
def analyze_endpoint(input_data: ProductInput):
    """
    Performs full end-to-end analysis:
    Inventory Analysis + Deterministic Decision + AI Explanation + Alternative Products.
    """
    global _PRODUCTS_CACHE
    if not _PRODUCTS_CACHE:
        _PRODUCTS_CACHE = load_handoff_data()

    product = input_data.to_product_handoff()
    decision_res = evaluate_decision(product)
    explanation = generate_llm_explanation(product, decision_res)

    catalog = list(_PRODUCTS_CACHE.values()) if _PRODUCTS_CACHE else [product]
    alternatives = find_alternative_products(product, catalog, top_n=3)

    return FullAnalysisResponse(
        product=product,
        decision=decision_res,
        ai_explanation=explanation,
        alternative_products=alternatives
    )


@app.post("/alternative-products", response_model=List[AlternativeProduct], tags=["Catalog Intelligence"])
def get_alternative_products_endpoint(input_data: ProductInput):
    """
    Returns candidate alternative products for capital re-allocation.
    """
    global _PRODUCTS_CACHE
    if not _PRODUCTS_CACHE:
        _PRODUCTS_CACHE = load_handoff_data()

    product = input_data.to_product_handoff()
    catalog = list(_PRODUCTS_CACHE.values())
    return find_alternative_products(product, catalog, top_n=3)


@app.post("/approval", response_model=ApprovalRecord, tags=["Human-in-the-Loop"])
def submit_approval(request: ApprovalRequest):
    """
    Records a human decision action: APPROVE, MODIFY, or REJECT.
    Ensures strict human-in-the-loop oversight without triggering direct external supplier orders.
    """
    action_upper = request.action.upper()
    if action_upper not in {"APPROVE", "MODIFY", "REJECT"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Action must be one of: 'APPROVE', 'MODIFY', 'REJECT'"
        )

    product = _PRODUCTS_CACHE.get(request.product_id)
    product_name = product.product_name if product else request.product_id
    
    # Calculate original recommended quantity
    if product:
        dec = evaluate_decision(product)
        original_qty = dec.recommended_order_quantity
    else:
        original_qty = request.modified_quantity or 0

    if action_upper == "APPROVE":
        final_qty = original_qty
    elif action_upper == "MODIFY":
        if request.modified_quantity is None or request.modified_quantity < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="For 'MODIFY' action, modified_quantity must be specified and non-negative."
            )
        final_qty = request.modified_quantity
    else: # REJECT
        final_qty = 0

    record_id = f"APP-{uuid.uuid4().hex[:8].upper()}"
    record = ApprovalRecord(
        id=record_id,
        product_id=request.product_id,
        product_name=product_name,
        decision=request.decision,
        action=action_upper,
        original_recommended_quantity=original_qty,
        final_quantity=final_qty,
        notes=request.notes,
        created_at=datetime.now(timezone.utc).isoformat()
    )

    _APPROVAL_RECORDS[request.product_id] = record
    logger.info(f"Recorded human approval for {request.product_id}: {action_upper} (Final Qty: {final_qty})")
    return record


@app.get("/approvals", response_model=List[ApprovalRecord], tags=["Human-in-the-Loop"])
def get_all_approvals():
    """Returns all recorded human approval decisions."""
    return list(_APPROVAL_RECORDS.values())


@app.get("/demo/scenarios", tags=["Demonstration & Presentation"])
def get_demo_scenarios():
    """
    Provides standard presentation scenarios showcasing:
    - Scenario 1: Increasing demand + low inventory -> INCREASE (e.g. P019 Robot Vacuum)
    - Scenario 2: Stable demand + healthy inventory -> MAINTAIN (e.g. P010 Smartwatch)
    - Scenario 3: Decreasing demand + high inventory -> REDUCE (e.g. P001 Wireless Gaming Mouse)
    """
    global _PRODUCTS_CACHE
    if not _PRODUCTS_CACHE:
        _PRODUCTS_CACHE = load_handoff_data()

    scenarios = [
        {
            "scenario_id": "scenario_1",
            "name": "Scenario 1: Surging Demand & Low Stock",
            "expected_decision": "INCREASE",
            "recommended_product_id": "P019",
            "description": "Product with increasing sales trend, limited on-hand inventory coverage below supplier lead time, and high stockout risk."
        },
        {
            "scenario_id": "scenario_2",
            "name": "Scenario 2: Stable Demand & Balanced Coverage",
            "expected_decision": "MAINTAIN",
            "recommended_product_id": "P010",
            "description": "Product with stable demand velocity, comfortable coverage matching lead time, and balanced stock."
        },
        {
            "scenario_id": "scenario_3",
            "name": "Scenario 3: Declining Demand & Inventory Overhang",
            "expected_decision": "REDUCE",
            "recommended_product_id": "P001",
            "description": "Product facing decreasing demand trend with substantial excess inventory coverage."
        }
    ]

    enriched_scenarios = []
    for s in scenarios:
        pid = s["recommended_product_id"]
        prod = _PRODUCTS_CACHE.get(pid)
        if prod:
            dec = evaluate_decision(prod)
            enriched_scenarios.append({
                **s,
                "product_name": prod.product_name,
                "actual_decision": dec.decision,
                "days_of_stock": dec.days_of_stock,
                "stockout_risk": dec.stockout_risk,
                "recommended_order_quantity": dec.recommended_order_quantity
            })
        else:
            enriched_scenarios.append(s)

    return enriched_scenarios


@app.post("/api/upload-and-run", tags=["Pipeline Execution"])
async def upload_and_run_pipeline(file: UploadFile = File(...)):
    """
    Accepts customer CSV file upload, inspects columns, automatically populates missing expected
    columns with NULL/NaN, creates processed input CSV at data/processed/customer_input.csv,
    runs forecasting, market intelligence, decision engine, verifies unique product counts,
    cleans temporary raw upload file, and returns complete workflow results.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a CSV file.")
        
    root_dir = os.path.join(os.path.dirname(__file__), "..")
    raw_temp_path = get_writeable_filepath(os.path.join("data", "raw", f"raw_upload_{uuid.uuid4().hex[:8]}_{file.filename}"), for_write=True)
    os.makedirs(os.path.dirname(raw_temp_path), exist_ok=True)
    
    try:
        contents = await file.read()
        with open(raw_temp_path, "wb") as f_out:
            f_out.write(contents)
        df_raw = pd.read_csv(raw_temp_path)
    except Exception as e:
        if os.path.exists(raw_temp_path):
            os.remove(raw_temp_path)
        raise HTTPException(status_code=400, detail=f"Failed to save or parse uploaded CSV file: {str(e)}")

    if df_raw is None or df_raw.empty:
        if os.path.exists(raw_temp_path):
            os.remove(raw_temp_path)
        raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.")

    # 1. Inspect raw columns against expected schema & count uploaded unique products
    present_cols = df_raw.columns.tolist()
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in present_cols]
    
    raw_pids = set()
    if "product_id" in df_raw.columns:
        raw_pids = set(df_raw["product_id"].dropna().astype(str).str.strip().unique().tolist())
        raw_pids.discard("")
        raw_pids.discard("nan")
        
    uploaded_product_count = len(raw_pids) if raw_pids else (df_raw["product_name"].nunique() if "product_name" in df_raw.columns else len(df_raw))

    # 2. Automatically populate ALL missing expected columns with NULL/NaN values
    for col in missing_cols:
        df_raw[col] = np.nan

    # Overwrite temporary raw file with imputed missing columns for adaptation step
    df_raw.to_csv(raw_temp_path, index=False)
    
    # 4. Generate validation summary before analysis
    overview = get_dataset_overview(df_raw)
    validation_df = run_validation_checks(df_raw)
    data_warnings = validation_df[validation_df['status'].isin(['WARN', 'FAIL'])]['details'].tolist()

    # 5. Adapt customer data & write to data/processed/customer_input.csv (Requirement 5)
    processed_input_path = get_writeable_filepath(os.path.join("data", "processed", "customer_input.csv"), for_write=True)
    from src.product_intelligence import build_inventory_decision_input_json
    adapted_df = adapt_customer_data(raw_temp_path, output_processed_path=processed_input_path)
    
    # 6. Customer Demand Forecasting -> outputs/predictions/customer_forecasts.csv (Requirement 1 & 3)
    model, metrics = train_customer_forecasting_model(adapted_df)
    forecast_output_path = get_writeable_filepath(os.path.join("outputs", "predictions", "customer_forecasts.csv"), for_write=True)
    pred_df = predict_customer_demand(model, adapted_df, output_pred_path=forecast_output_path)
    
    # 7. Market Intelligence & Decision Input JSON -> handoff/inventory_decision_input.json (Requirement 1, 2 & 3)
    decision_input_json_path = get_writeable_filepath(os.path.join("handoff", "inventory_decision_input.json"), for_write=True)
    decision_input_dict = build_inventory_decision_input_json(
        customer_df=adapted_df,
        forecast_df=pred_df,
        output_json_path=decision_input_json_path,
        use_live_market_intel=True,
        source_file=file.filename
    )
    
    # 8. Inventory Decision Engine -> outputs/decisions/inventory_decisions.json (Requirement 1 & 3)
    decisions_json_path = get_writeable_filepath(os.path.join("outputs", "decisions", "inventory_decisions.json"), for_write=True)
    decisions_dict = run_inventory_decision_engine(
        handoff_json_path=decision_input_json_path,
        output_decisions_path=decisions_json_path
    )
    
    # 9. Verify product processing parity (Requirement 8)
    processed_pids = set(adapted_df["product_id"].astype(str).str.strip().unique().tolist())
    processed_product_count = len(processed_pids)
    
    skipped_products = []
    if raw_pids:
        missing_pids = raw_pids - processed_pids
        for m_pid in missing_pids:
            skipped_products.append({
                "product_id": m_pid,
                "reason": "Insufficient valid historical date or sales records during parsing."
            })
            
    # 10. Clean up temporary uploaded CSV (Requirement 4)
    if os.path.exists(raw_temp_path):
        try:
            os.remove(raw_temp_path)
            logger.info("Successfully deleted temporary uploaded CSV file.")
        except Exception as e:
            logger.warning(f"Could not delete temporary CSV file: {e}")

    # 11. Refresh memory cache for endpoints
    global _PRODUCTS_CACHE
    _PRODUCTS_CACHE = load_handoff_data()
    
    products_response = list_products()["products"]

    return {
        "status": "success",
        "filename": file.filename,
        "validation_summary": {
            "num_rows": overview["num_rows"],
            "num_products": overview["num_unique_products"],
            "num_categories": overview["num_categories"],
            "date_min": overview["date_min"],
            "date_max": overview["date_max"],
            "present_columns": present_cols,
            "missing_columns": missing_cols,
            "data_warnings": data_warnings
        },
        "verification": {
            "uploaded_unique_products": uploaded_product_count,
            "processed_unique_products": processed_product_count,
            "skipped_products": skipped_products
        },
        "pipeline_summary": decisions_dict["decision_summary"],
        "decisions_count": len(decisions_dict["decisions"]),
        "products": products_response
    }


@app.get("/api/download/processed-customer-csv", tags=["Downloads"])
def download_processed_customer_csv():
    file_path = get_writeable_filepath(os.path.join("data", "processed", "customer_input.csv"), for_write=False)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Processed customer CSV file not found.")
    return FileResponse(file_path, filename="customer_input.csv", media_type="text/csv")


@app.get("/api/download/forecasts", tags=["Downloads"])
def download_forecasts_csv():
    file_path = get_writeable_filepath(os.path.join("outputs", "predictions", "customer_forecasts.csv"), for_write=False)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Forecast CSV file not found.")
    return FileResponse(file_path, filename="customer_forecasts.csv", media_type="text/csv")


@app.get("/api/download/decision-input", tags=["Downloads"])
def download_decision_input_json():
    file_path = get_writeable_filepath(os.path.join("handoff", "inventory_decision_input.json"), for_write=False)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Inventory decision input JSON file not found.")
    return FileResponse(file_path, filename="inventory_decision_input.json", media_type="application/json")


@app.get("/api/download/handoff", tags=["Downloads"])
def download_handoff_json():
    return download_decision_input_json()


@app.get("/api/download/decisions", tags=["Downloads"])
def download_decisions_json():
    file_path = get_writeable_filepath(os.path.join("outputs", "decisions", "inventory_decisions.json"), for_write=False)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Inventory decisions JSON file not found.")
    return FileResponse(file_path, filename="inventory_decisions.json", media_type="application/json")



@app.post("/api/reset", tags=["Pipeline"])
def reset_system_state():
    """Resets all in-memory approval overrides, clears product cache, and removes generated output files."""
    global _APPROVAL_RECORDS, _PRODUCTS_CACHE
    _APPROVAL_RECORDS.clear()
    _PRODUCTS_CACHE.clear()
    
    root_dir = os.path.join(os.path.dirname(__file__), "..")
    files_to_remove = [
        os.path.join(root_dir, "handoff", "inventory_decision_input.json"),
        os.path.join(root_dir, "outputs", "decisions", "inventory_decisions.json"),
        os.path.join(root_dir, "outputs", "predictions", "customer_forecasts.csv"),
        os.path.join(root_dir, "data", "processed", "customer_input.csv")
    ]
    
    for fpath in files_to_remove:
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
            except Exception as e:
                logger.warning(f"Could not remove file during reset '{fpath}': {e}")
                
    logger.info("System state reset performed. Product cache cleared.")
    return {
        "status": "success",
        "message": "System state reset. Ready for CSV upload.",
        "products": []
    }


# Serve static React frontend build if present (Railway / Docker / Monolith deployment)
frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))
if os.path.exists(frontend_dist):
    assets_dir = os.path.join(frontend_dist, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="static_assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_frontend_spa(full_path: str):
        api_prefixes = ("api/", "health", "products", "decision", "analyze", "alternative-products", "approval", "approvals", "demo", "docs", "openapi.json")
        if any(full_path == p or full_path.startswith(p) for p in api_prefixes):
            raise HTTPException(status_code=404, detail="API endpoint not found.")
        
        target_file = os.path.join(frontend_dist, full_path)
        if os.path.exists(target_file) and os.path.isfile(target_file):
            return FileResponse(target_file)
            
        index_file = os.path.join(frontend_dist, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        raise HTTPException(status_code=404, detail="Page not found")



