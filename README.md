# BFWAI AI Inventory Decision Agent

An end-to-end, data-driven AI Inventory Decision Agent for supply chain management. The system processes raw customer inventory datasets, trains customer-specific demand forecasting models, retrieves external Gemini + NewsAPI market intelligence, executes a deterministic decision engine, and presents interactive decision support via a modern web dashboard.

---

## 💡 Project Overview

The **BFWAI AI Inventory Decision Agent** is an enterprise-grade decision support platform designed to eliminate stockouts, prevent excess inventory, and optimize reorder decisions across dynamic SKU catalogs.

### System Architecture Overview

```text
                                RAW CUSTOMER DATASET
                      (data/raw/customer_inventory_upload.csv)
                                         │
                                         ▼
                             1. Data Validation & Adapter
                     (src/data_validation.py & customer_data_adapter.py)
                                         │
                                         ▼
                        2. Customer-Specific ML Forecasting
                            (src/customer_forecasting.py)
                             └─► customer_forecasts.csv
                                         │
                                         ▼
                     3. External Market Intelligence (Gemini + NewsAPI)
                           (src/market_intelligence.py)
                                         │
                                         ▼
                           4. Person 1 -> Person 2 Handoff
                           (src/product_intelligence.py)
                            └─► handoff/inventory_decision_input.json
                                         │
                                         ▼
                    5. Person 2 Deterministic Decision Engine
                        (src/inventory_decision_engine.py)
                           └─► outputs/decisions/inventory_decisions.json
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
          FastAPI Backend (port 8000)               Vite React Frontend (port 3000)
            (backend/main.py)                        (frontend/)
```

### Key Principles & Capabilities

1. **100% Data-Driven & Customer-Adaptable**:
   - Works dynamically with any customer raw dataset (varying product count, row count, date ranges, categories, and brands).
   - Dynamically discovers products and categories without hardcoded SKU IDs or fixed labels.
   - Handles missing optional columns safely with logical defaults without fabricating fake customer data.

2. **Customer-Specific ML Forecasting**:
   - Trains demand forecasting models exclusively on customer historical sales data.
   - Implements documented fallback forecasting (recent average daily demand) for products with sparse historical records (< 14 days).
   - Generates 7-day forecast (`forecast_7d`), 30-day forecast (`forecast_30d`), and demand trend indicators (`increasing`, `stable`, `decreasing`).

3. **External Market Intelligence (Gemini + NewsAPI)**:
   - Queries NewsAPI for relevant market news and performs batch sentiment analysis using Google Gemini API.
   - External market sentiment is analyzed independently without requiring customer-uploaded sentiment data.
   - Defaults cleanly to neutral signals if external news or API keys are unavailable.
   - **Gemini is strictly used for external sentiment—it does NOT make final inventory decisions.**

4. **Deterministic Rule-Based Decision Engine**:
   - Computes standard supply chain metrics per SKU:
     - $\text{available\_inventory} = \text{current\_inventory} - \text{reserved\_inventory} + \text{incoming\_inventory}$
     - $\text{average\_daily\_demand} = \frac{\text{forecast\_7d}}{7.0}$
     - $\text{inventory\_coverage\_days} = \frac{\text{available\_inventory}}{\text{average\_daily\_demand}}$
     - $\text{lead\_time\_demand} = \text{average\_daily\_demand} \times \text{lead\_time\_days}$
   - Determines deterministic decisions: `INCREASE`, `MAINTAIN`, or `REDUCE`.
   - Generates `recommended_order_quantity` adhering to supplier Minimum Order Quantities (MOQ).
   - Decisions are calculated strictly from data without artificially forced distribution balances.

5. **Human-in-the-Loop Oversight & Web Dashboard**:
   - Interactive React dashboard provides executive KPI overviews, stockout risk alerts, AI explanations, alternative product recommendations for capital re-allocation, and human approval workflows (`APPROVE`, `MODIFY`, `REJECT`).

---

## 🛠️ Technologies Used

### Backend & Machine Learning
- **Python 3.10+**: Core programming language for data pipelines and APIs.
- **FastAPI**: Modern, high-performance RESTful API framework for decision delivery and analysis.
- **Uvicorn**: High-speed ASGI production and development server.
- **Pydantic (v2)**: Robust data validation, settings management, and type contracts.
- **Scikit-Learn**: Machine learning demand forecasting algorithms.
- **Pandas & NumPy**: High-performance data manipulation, time-series analysis, and numerical processing.
- **Google GenAI SDK (Gemini)**: Batch market sentiment and external market signal analysis.
- **NewsAPI Client / REST**: Live news and market trend retrieval.
- **Pytest**: Comprehensive automated unit and integration testing suite.

### Frontend & User Interface
- **React 18**: Component-driven UI library for building interactive user dashboards.
- **Vite 6**: Fast frontend development environment and asset bundler.
- **Lucide React**: Modern iconography for supply chain KPIs and status indicators.
- **Custom CSS Design System**: Dark-themed, responsive interface with glassmorphism, KPI metric cards, filtering, and decision detail drawers.

---

## ⚙️ Setup & Installation Steps

### Prerequisites
Ensure the following tools are installed on your system:
- **Python**: Version `3.10` or higher (`python --version`)
- **Node.js**: Version `18.0.0` or higher (`node --version`)
- **npm**: Version `9.0.0` or higher (`npm --version`)

### 1. Clone or Open the Repository
```bash
cd BFWAI-main
```

### 2. Set Up Python Virtual Environment & Dependencies
Create and activate a virtual environment (recommended):

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

Install the backend Python dependencies:
```bash
pip install fastapi uvicorn pydantic pandas numpy scikit-learn pytest python-dotenv google-genai
```

### 3. Set Up Frontend Dependencies
Navigate to the `frontend/` directory and install the required npm packages:

```bash
cd frontend
npm install
cd ..
```

### 4. Configure Environment Variables (Optional)
Create a `.env` file in the root directory based on `.env.example`:

```bash
cp .env.example .env
```

Optionally set your API keys for live market sentiment analysis:
```env
# Google Gemini API Key (get from https://aistudio.google.com/app/apikey)
GEMINI_API_KEY=your_gemini_api_key_here

# NewsAPI.org API Key (get from https://newsapi.org/)
NEWS_API_KEY=your_news_api_key_here
```
> **Note**: Both keys are **optional**. If keys are not provided, the pipeline gracefully defaults to neutral external market sentiment signals.

---

## 🚀 How to Run the Project

### Option A: Run Complete End-to-End Pipeline & Tests
Executes data validation, customer ML forecasting, market intelligence, handoff generation, deterministic decision evaluation, and the full test suite:

```bash
python run_customer_pipeline.py
```

### Option B: Run Fullstack Web Application (Backend + Frontend)

#### Step 1: Start FastAPI Backend Server
Launch the REST API on port `8000`:

```bash
python -m uvicorn backend.main:app --reload --port 8000
```
- **API Health Check**: `http://localhost:8000/health`
- **Interactive Swagger Documentation**: `http://localhost:8000/docs`
- **Products & Decisions Endpoint**: `http://localhost:8000/products`

#### Step 2: Start Frontend React Dashboard
In a separate terminal, launch the Vite development server on port `3000`:

```bash
cd frontend
npm run dev
```
- **Web Application Interface**: **`http://localhost:3000`**

### Option C: Run Automated Tests
Run all 78 automated unit and integration tests:

```bash
pytest -v
```

---

## 📂 Output Artifacts

| Output File Path | Description |
| :--- | :--- |
| `data/processed/customer_input.csv` | Standardized and cleaned customer dataset. |
| `outputs/predictions/customer_forecasts.csv` | Product-level demand forecasts (7-day, 30-day, trend, and fallback indicators). |
| `handoff/inventory_decision_input.json` | Official Person 1 -> Person 2 handoff JSON (1 record per unique customer product). |
| `outputs/decisions/inventory_decisions.json` | Final deterministic inventory decisions, confidence scores, stockout risks, MOQ order quantities, and reasons. |
| `outputs/reports/customer_forecasting_metrics.csv` | Evaluation metrics (MAE, RMSE, MAPE) for the customer-trained model. |

---

## 📁 Project Directory Structure

```text
BFWAI/
├── data/
│   ├── raw/                        # Uploaded customer raw inventory CSVs
│   ├── processed/                  # Cleaned customer input datasets
│   └── external/                   # Cached external news feeds
├── src/
│   ├── customer_schema.py          # Data schema definitions & validation
│   ├── customer_data_adapter.py    # Raw CSV adapter & column cleaner
│   ├── data_validation.py          # Dynamic dataset quality validation
│   ├── customer_forecasting.py     # Customer ML forecasting & fallbacks
│   ├── market_intelligence.py      # NewsAPI + Gemini market signal analysis
│   ├── product_intelligence.py     # Person 1 handoff JSON builder
│   └── inventory_decision_engine.py # Person 2 deterministic decision engine
├── backend/
│   ├── main.py                     # FastAPI web application
│   ├── schemas.py                  # Pydantic data models
│   ├── decision_engine.py          # Decision engine scoring & rules
│   ├── inventory_analysis.py       # Supply chain metric calculations
│   ├── explanation_agent.py        # Executive narrative generator
│   └── alternative_agent.py        # Capital re-allocation agent
├── frontend/                       # Vite + React web interface
│   ├── src/                        # React UI components & styling
│   ├── package.json                # Frontend dependencies & scripts
│   └── vite.config.js              # Vite server & proxy configuration
├── handoff/
│   └── inventory_decision_input.json # Person 1 -> Person 2 handoff JSON
├── outputs/
│   ├── predictions/
│   │   └── customer_forecasts.csv  # ML forecasts
│   └── decisions/
│       └── inventory_decisions.json # Final deterministic decisions
├── tests/                          # 78 automated Pytest test cases
├── run_customer_pipeline.py        # End-to-end pipeline execution script
└── README.md                       # Project documentation
```
