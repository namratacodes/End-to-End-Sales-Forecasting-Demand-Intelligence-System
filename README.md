# Sales Forecasting & Demand Intelligence System

End-to-end sales forecasting and demand intelligence project built on the Superstore Sales dataset (2015–2018). This project covers exploratory analysis, time series decomposition, multi model forecasting, category/region-level forecasting, anomaly detection, product demand segmentation, an interactive dashboard, and an executive business report.

**Live Dashboard:** https://end-to-end-sales-forecasting-demand-intelligence-system-dnbwqn.streamlit.app/

---

## Problem Statement

Retail businesses need to answer one core question every month: *how much of each product will we sell next, and will we have enough stock to meet that demand?* This project builds a working system to predict future demand, detect unusual sales patterns, segment products by demand behavior, and present it all through a dashboard a business manager could actually use.

---

## Repository Structure

```
SalesForecasting_NamrataSingh/
├── analysis.ipynb          # Full analysis notebook (Tasks 1–6)
├── train.csv               # Superstore Sales dataset
├── app.py                  # Streamlit dashboard (Task 7)
├── requirements.txt        # Python dependencies
├── summary.docx            # Executive business report (Task 8)
├── charts/                 # Exported chart images from the notebook
└── README.md
```

---

## What's Inside

### 1. Data Exploration (Task 1)
Loaded and cleaned the Superstore dataset, parsed dates (verified as `DD/MM/YYYY`), engineered time features (year, month, week, quarter, season), and aggregated sales to daily, weekly, and monthly granularity. Key findings:
- Technology, Furniture, and Office Supplies contribute roughly equal revenue shares
- East region shows the most consistent year-over-year growth; South is the most volatile
- Shipping speed does not meaningfully vary by region
- Sales are strongly seasonal ; September, November, and December are consistently the strongest months; January and February the weakest

### 2. Time Series Decomposition (Task 2)
Decomposed monthly sales into trend, seasonal, and residual components (multiplicative model). Verified stationarity using the Augmented Dickey Fuller test, with and without differencing.

### 3. Forecasting Models (Task 3)
Built and compared three forecasting approaches on a 3 month holdout (Oct–Dec 2018):

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| SARIMA | 19,244 | 19,950 | 20.53% |
| Prophet | 21,542 | 22,056 | 23.12% |
| **XGBoost (recommended)** | **17,721** | **19,906** | **18.01%** |

XGBoost was selected for production use based on lowest error across all three metrics, though all three models under-predicted the November 2018 sales spike , a noted limitation.

### 4. Category & Region Forecasting (Task 4)
Repeated the XGBoost model across Furniture, Technology, Office Supplies, West, and East. Furniture showed the strongest projected growth; Technology was the only segment forecast to decline.

### 5. Anomaly Detection (Task 5)
Applied both Isolation Forest and Z-score methods to weekly sales data. The two methods agreed on the most extreme spikes (e.g., the September 2015 back to school surge) while diverging on borderline cases — a documented comparison of method sensitivity.

### 6. Product Demand Segmentation (Task 6)
Clustered 17 product sub-categories into 4 demand segments using K Means (features: total sales, growth rate, volatility, average order value), visualized via PCA. Segments range from "High Volume, Established Demand" to "Declining, High Value" (Machines).

### 7. Interactive Dashboard (Task 7)
A 4-page Streamlit app (`app.py`):
- **Sales Overview** : KPIs, yearly/monthly trends, region and category breakdowns with filters
- **Forecast Explorer** : select category/region and forecast horizon (1–3 months), view XGBoost forecast with model accuracy metrics
- **Anomaly Report** : weekly sales with flagged anomalies and a detail table
- **Product Segments** : PCA cluster visualization and segment assignment table

### 8. Executive Business Report (Task 8)
A 2 page, non technical business report (`summary.docx`) covering the executive summary, key findings, 3 month forecast in plain language, top anomalies, segmentation strategy, three data backed recommendations, and a stated risk/limitation.

---

## Tech Stack

- **Analysis:** Python, Pandas, NumPy, Statsmodels (SARIMA, decomposition, ADF test), Prophet, XGBoost, Scikit-learn (Isolation Forest, K-Means, PCA)
- **Visualization:** Matplotlib, Plotly
- **Dashboard:** Streamlit
- **Report:** Word (docx)

---

## Running Locally

**1. Clone the repo and install dependencies:**

**2. Run the dashboard:**
```bash
streamlit run app.py
```
Requires `train.csv` in the same folder (already included in this repo). The app opens at `http://localhost:8501`.

**3. Explore the full analysis:**
Open `analysis.ipynb` in Jupyter or Google Colab to see all 8 tasks, including model comparisons, decomposition plots, and clustering.

---

## Key Limitations

- The dataset covers only 4 years (~3.75 seasonal cycles), which limits SARIMA's seasonal parameter reliability and causes all three forecasting models to under predict extreme holiday season peaks
- Forecasts should be treated as a conservative floor during known peak months (Nov–Dec), not an exact prediction
- MAPE as an evaluation metric becomes unstable when actual sales values are small (documented in the East region anomaly analysis)

---

## Author

Namrata Singh : Data Science Internship Project, Weeks 3–4
