import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

st.set_page_config(page_title="Sales Intelligence", layout="wide", initial_sidebar_state="expanded")

ACCENT = "#2454A6"
ACCENT_DARK = "#173461"
BG = "#F5F7FA"
CARD = "#FFFFFF"
TEXT_MUTED = "#6B7280"
PALETTE = ["#2454A6", "#3E92CC", "#7AB8E8", "#F2A65A", "#D9534F", "#5CB85C"]
CHART_CONFIG = {"displayModeBar": False, "responsive": True}

st.markdown(f"""
<style>
    .stApp {{ background-color: {BG}; }}
    html, body, [class*="css"] {{ font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif; }}
    section[data-testid="stSidebar"] {{ background-color: {ACCENT_DARK}; }}
    section[data-testid="stSidebar"] * {{ color: #E8EEF7 !important; }}
    h1, h2, h3,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stHeading"] {{ color: {ACCENT_DARK} !important; font-weight: 600; }}
    [data-testid="stCaptionContainer"] {{ color: {TEXT_MUTED} !important; }}
    .kpi-card {{
        background-color: {CARD};
        border-radius: 10px;
        padding: 18px 22px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border-left: 4px solid {ACCENT};
    }}
    .kpi-label {{ font-size: 13px; color: {TEXT_MUTED}; text-transform: uppercase; letter-spacing: 0.4px; margin-bottom: 4px; }}
    .kpi-value {{ font-size: 26px; font-weight: 700; color: {ACCENT_DARK}; }}
    .kpi-sub {{ font-size: 12px; color: {TEXT_MUTED}; margin-top: 2px; }}
    .section-divider {{ border-top: 1px solid #E2E5EA; margin: 28px 0 18px 0; }}
</style>
""", unsafe_allow_html=True)


def kpi_card(label, value, sub=""):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def style_fig(fig, title="", height=420):
    # tickformat=",.0f" forces plain comma-separated numbers (e.g. 500,000)
    # instead of Plotly's default SI-suffix axis labels, which can render
    # visually truncated (just "1, 2, 3, 4") on some browser/DPI combinations.
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, font=dict(size=16, color=ACCENT_DARK)),
        font=dict(family="Segoe UI, Arial", size=13, color="#2B2B2B"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        transition_duration=0,
    )
    fig.update_yaxes(tickformat=",.0f", showgrid=True, gridcolor="#EEF1F5")
    fig.update_xaxes(showgrid=False)
    return fig


@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d/%m/%Y")
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Quarter"] = df["Order Date"].dt.quarter
    return df


def get_season(month):
    return 0 if month in [12, 1, 2] else 1 if month in [3, 4, 5] else 2 if month in [6, 7, 8] else 3


@st.cache_data
def monthly_series(df, filter_col=None, filter_val=None):
    data = df if filter_col is None else df[df[filter_col] == filter_val]
    monthly = data.groupby(data["Order Date"].dt.to_period("M"))["Sales"].sum().reset_index()
    monthly.columns = ["YearMonth", "Sales"]
    monthly["YearMonth"] = monthly["YearMonth"].dt.to_timestamp()
    monthly = monthly.set_index("YearMonth")
    full_range = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
    monthly = monthly.reindex(full_range, fill_value=0)
    monthly.index.name = "YearMonth"
    return monthly


def build_features(monthly):
    m = monthly.copy()
    m["lag1"] = m["Sales"].shift(1)
    m["lag2"] = m["Sales"].shift(2)
    m["lag3"] = m["Sales"].shift(3)
    m["rolling_mean_3"] = m["Sales"].shift(1).rolling(3).mean()
    m["Month"] = m.index.month
    m["Quarter"] = m.index.quarter
    m["Season"] = m["Month"].apply(get_season)
    return m.dropna()

FEATURES = ["lag1", "lag2", "lag3", "rolling_mean_3", "Month", "Quarter", "Season"]

@st.cache_data
def evaluate_model(monthly):
    ml = build_features(monthly)
    X, y = ml[FEATURES], ml["Sales"]
    X_train, X_test = X.iloc[:-3], X.iloc[-3:]
    y_train, y_test = y.iloc[:-3], y.iloc[-3:]
    model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return float(mae), float(rmse), y_test, preds


@st.cache_data
def forecast_ahead(monthly, horizon):
    ml = build_features(monthly)
    X, y = ml[FEATURES], ml["Sales"]
    model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X, y)

    history = monthly["Sales"].tolist()
    last_date = monthly.index.max()
    future_dates, future_preds = [], []

    for step in range(horizon):
        next_date = last_date + pd.DateOffset(months=step + 1)
        lag1, lag2, lag3 = history[-1], history[-2], history[-3]
        roll3 = float(np.mean(history[-3:]))
        row = pd.DataFrame([{
            "lag1": lag1, "lag2": lag2, "lag3": lag3, "rolling_mean_3": roll3,
            "Month": next_date.month, "Quarter": next_date.quarter, "Season": get_season(next_date.month),
        }])[FEATURES]
        pred = float(model.predict(row)[0])
        future_dates.append(next_date)
        future_preds.append(pred)
        history.append(pred)

    return pd.Series(future_preds, index=pd.DatetimeIndex(future_dates))


@st.cache_data
def weekly_series(df):
    weekly = df.groupby(df["Order Date"].dt.to_period("W"))["Sales"].sum().reset_index()
    weekly.columns = ["Week", "Sales"]
    weekly["Week"] = weekly["Week"].dt.start_time
    weekly = weekly.sort_values("Week").reset_index(drop=True)
    return weekly


@st.cache_data
def detect_anomalies(weekly):
    w = weekly.copy()
    iso = IsolationForest(contamination=0.05, random_state=42)
    w["anomaly"] = iso.fit_predict(w[["Sales"]])
    return w[w["anomaly"] == -1].sort_values("Sales", ascending=False)


@st.cache_data
def build_clusters(df):
    subcat = df.groupby("Sub-Category").agg(
        total_sales=("Sales", "sum"), avg_order_value=("Sales", "mean")
    ).reset_index()

    yearly = df.groupby(["Sub-Category", "Year"])["Sales"].sum().reset_index()
    pivot = yearly.pivot(index="Sub-Category", columns="Year", values="Sales")
    y0, y1 = pivot.columns.min(), pivot.columns.max()
    pivot["growth_rate"] = (pivot[y1] - pivot[y0]) / pivot[y0] * 100

    monthly = df.groupby(["Sub-Category", df["Order Date"].dt.to_period("M")])["Sales"].sum().reset_index()
    vol = monthly.groupby("Sub-Category")["Sales"].std().reset_index()
    vol.columns = ["Sub-Category", "volatility"]

    data = subcat.merge(pivot[["growth_rate"]], on="Sub-Category").merge(vol, on="Sub-Category")

    cols = ["total_sales", "avg_order_value", "growth_rate", "volatility"]
    X_scaled = StandardScaler().fit_transform(data[cols])
    data["Cluster"] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X_scaled)

    coords = PCA(n_components=2).fit_transform(X_scaled)
    data["PCA1"], data["PCA2"] = coords[:, 0], coords[:, 1]
    return data


CLUSTER_LABELS = {
    0: "High Volume, Established",
    1: "High-Value Outlier",
    2: "Stable / Mixed Demand",
    3: "Declining, High-Value",
}


def page_overview(df):
    st.title("Sales Overview")
    st.caption("Four-year performance summary across categories and regions")

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Total Sales", f"${df['Sales'].sum():,.0f}")
    with c2: kpi_card("Total Orders", f"{df['Order ID'].nunique():,}")
    with c3: kpi_card("Avg Order Value", f"${df['Sales'].mean():,.2f}")
    with c4: kpi_card("Years Covered", f"{df['Year'].min()}\u2013{df['Year'].max()}")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.4])
    with col1:
        yearly = df.groupby("Year")["Sales"].sum().reset_index()
        yearly["Year"] = yearly["Year"].astype(str)
        fig = go.Figure(go.Bar(x=yearly["Year"], y=yearly["Sales"], marker_color=ACCENT))
        st.plotly_chart(style_fig(fig, "Total Sales by Year"), use_container_width=True, theme=None, config=CHART_CONFIG)

    with col2:
        monthly = df.groupby(df["Order Date"].dt.to_period("M"))["Sales"].sum().reset_index()
        monthly["Order Date"] = monthly["Order Date"].dt.to_timestamp()
        fig = go.Figure(go.Scatter(x=monthly["Order Date"], y=monthly["Sales"],
                                    mode="lines+markers", line=dict(color=ACCENT, width=2),
                                    marker=dict(size=6)))
        st.plotly_chart(style_fig(fig, "Monthly Sales Trend"), use_container_width=True, theme=None, config=CHART_CONFIG)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    fc1, fc2 = st.columns(2)
    regions = fc1.multiselect("Region", sorted(df["Region"].unique()), default=sorted(df["Region"].unique()))
    categories = fc2.multiselect("Category", sorted(df["Category"].unique()), default=sorted(df["Category"].unique()))
    filtered = df[df["Region"].isin(regions) & df["Category"].isin(categories)]

    col3, col4 = st.columns(2)
    with col3:
        by_region = filtered.groupby("Region")["Sales"].sum().reset_index().sort_values("Sales", ascending=False)
        fig = go.Figure(go.Bar(x=by_region["Region"], y=by_region["Sales"],
                                marker_color=PALETTE[:len(by_region)]))
        st.plotly_chart(style_fig(fig, "Sales by Region"), use_container_width=True, theme=None, config=CHART_CONFIG)
    with col4:
        by_cat = filtered.groupby("Category")["Sales"].sum().reset_index().sort_values("Sales", ascending=False)
        fig = go.Figure(go.Bar(x=by_cat["Category"], y=by_cat["Sales"],
                                marker_color=PALETTE[:len(by_cat)]))
        st.plotly_chart(style_fig(fig, "Sales by Category"), use_container_width=True, theme=None, config=CHART_CONFIG)


def page_forecast(df):
    st.title("Forecast Explorer")
    st.caption("XGBoost-based forecast, selectable by category or region")

    c1, c2, c3 = st.columns([1, 1, 1])
    level = c1.selectbox("Forecast Level", ["Overall", "Category", "Region"])

    filter_col, filter_val = None, None
    if level == "Category":
        filter_col = "Category"
        filter_val = c2.selectbox("Select Category", sorted(df["Category"].unique()))
    elif level == "Region":
        filter_col = "Region"
        filter_val = c2.selectbox("Select Region", sorted(df["Region"].unique()))

    horizon = c3.slider("Forecast Horizon (months ahead)", 1, 3, 3)

    monthly = monthly_series(df, filter_col, filter_val)
    mae, rmse, y_test, test_preds = evaluate_model(monthly)
    future = forecast_ahead(monthly, horizon)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly.index, y=monthly["Sales"], name="Historical",
                              mode="lines", line=dict(color=ACCENT, width=2)))
    fig.add_trace(go.Scatter(x=future.index, y=future.values, name="Forecast",
                              mode="lines+markers", line=dict(color=PALETTE[3], dash="dash", width=2),
                              marker=dict(size=8)))
    st.plotly_chart(style_fig(fig, f"Sales Forecast \u2014 {filter_val or 'Overall'}"),
                     use_container_width=True, theme=None, config=CHART_CONFIG)

    m1, m2, m3 = st.columns(3)
    with m1: kpi_card("Model MAE", f"${mae:,.0f}", "on last 3 known months")
    with m2: kpi_card("Model RMSE", f"${rmse:,.0f}", "on last 3 known months")
    with m3: kpi_card("Forecast Horizon", f"{horizon} month(s)", future.index[-1].strftime("%b %Y"))

    st.dataframe(
        future.rename("Forecasted Sales").reset_index().rename(columns={"index": "Month"}),
        use_container_width=True, hide_index=True
    )


def page_anomalies(df):
    st.title("Anomaly Report")
    st.caption("Weeks with unusually high or low sales, flagged via Isolation Forest")

    weekly = weekly_series(df)
    anomalies = detect_anomalies(weekly)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly["Week"], y=weekly["Sales"], name="Weekly Sales",
                              mode="lines", line=dict(color=ACCENT, width=1.5)))
    fig.add_trace(go.Scatter(x=anomalies["Week"], y=anomalies["Sales"], name="Anomaly",
                              mode="markers", marker=dict(color=PALETTE[4], size=11)))
    st.plotly_chart(style_fig(fig, "Weekly Sales with Flagged Anomalies"),
                     use_container_width=True, theme=None, config=CHART_CONFIG)

    c1, c2 = st.columns(2)
    with c1: kpi_card("Anomalies Detected", f"{len(anomalies)}", f"out of {len(weekly)} weeks")
    with c2: kpi_card("Largest Spike", f"${anomalies['Sales'].max():,.0f}")

    st.dataframe(
        anomalies[["Week", "Sales"]].rename(columns={"Sales": "Sales ($)"}),
        use_container_width=True, hide_index=True
    )


def page_segments(df):
    st.title("Product Demand Segments")
    st.caption("Sub-categories grouped by volume, growth, and volatility")

    clusters = build_clusters(df)
    clusters["Segment"] = clusters["Cluster"].map(CLUSTER_LABELS)

    fig = go.Figure()
    for i, seg in enumerate(clusters["Segment"].unique()):
        subset = clusters[clusters["Segment"] == seg]
        fig.add_trace(go.Scatter(
            x=subset["PCA1"], y=subset["PCA2"], mode="markers+text", name=seg,
            text=subset["Sub-Category"], textposition="top center",
            marker=dict(size=14, color=PALETTE[i % len(PALETTE)]),
        ))
    st.plotly_chart(style_fig(fig, "Sub-Category Clusters (PCA-reduced)", height=520),
                     use_container_width=True, theme=None, config=CHART_CONFIG)

    st.dataframe(
        clusters[["Sub-Category", "Segment", "total_sales", "growth_rate", "volatility"]]
        .rename(columns={"total_sales": "Total Sales", "growth_rate": "Growth %", "volatility": "Volatility"})
        .sort_values("Segment"),
        use_container_width=True, hide_index=True
    )


st.sidebar.markdown("<h2 style='color:#E8EEF7;'>Sales Intelligence</h2>", unsafe_allow_html=True)
page = st.sidebar.radio("Navigate", ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Segments"])

df = load_data()

if page == "Sales Overview":
    page_overview(df)
elif page == "Forecast Explorer":
    page_forecast(df)
elif page == "Anomaly Report":
    page_anomalies(df)
elif page == "Product Segments":
    page_segments(df)