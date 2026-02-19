
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Optional (for FRED series). If not installed, app will still run with Yahoo-only mode.
try:
    from pandas_datareader import data as pdr
    PDR_OK = True
except Exception:
    PDR_OK = False

try:
    import yfinance as yf
    YF_OK = True
except Exception:
    YF_OK = False


# =========================
# Config
# =========================
st.set_page_config(page_title="Market Raw + Risk Dashboard", layout="wide")

DEFAULT_LOOKBACK_DAYS = 730  # 2y baseline for percentiles
DEFAULT_DISPLAY_DAYS = 365

# Yahoo tickers (generally reliable)
YF_TICKERS = {
    "NASDAQ (^IXIC)": "^IXIC",
    "S&P 500 (^GSPC)": "^GSPC",
    "Russell 2000 (^RUT)": "^RUT",
    "VIX (^VIX)": "^VIX",
    "US 10Y (proxy ^TNX)": "^TNX",  # often yield*10
    "DXY (DX-Y.NYB)": "DX-Y.NYB",
    "WTI Oil (CL=F)": "CL=F",
    "Gold (GC=F)": "GC=F",
    "Copper (HG=F)": "HG=F",
    "Bitcoin (BTC-USD)": "BTC-USD",
}

# FRED series (best for rates/liquidity/credit)
# NOTE: Many FRED calls work without an API key, but some environments require it.
FRED_SERIES = {
    "US10Y (DGS10)": "DGS10",
    "US2Y (DGS2)": "DGS2",
    "HY Spread (BAMLH0A0HYM2)": "BAMLH0A0HYM2",   # ICE BofA US High Yield OAS
    "IG Spread (BAMLC0A0CM)": "BAMLC0A0CM",       # ICE BofA US Corporate OAS (investment grade)
    "TGA (WTREGEN)": "WTREGEN",                   # Treasury General Account
    "ON RRP (RRPONTSYD)": "RRPONTSYD",            # Overnight Reverse Repurchase Agreements: Total Securities Sold by the Fed
    "Bank Reserves (WRESBAL)": "WRESBAL",         # Reserve Balances with Federal Reserve Banks
    "Breakeven 10Y (T10YIE)": "T10YIE",            # 10-Year Breakeven Inflation Rate
}

# Default risk weights (sum doesn't need to be 1; we normalize)
DEFAULT_WEIGHTS = {
    "VIX": 0.20,
    "Rates_10Y": 0.20,
    "Credit_HY": 0.15,
    "Equity_Trend": 0.15,
    "USD_Shock": 0.10,
    "Oil_Shock": 0.10,
    "Liquidity": 0.10,
}

# =========================
# Helpers
# =========================
def _to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

@st.cache_data(ttl=60*60)
def fetch_yahoo_close(ticker: str, start: str) -> pd.Series:
    if not YF_OK:
        raise RuntimeError("yfinance is not installed. Run: pip install yfinance")
    df = yf.download(ticker, start=start, progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError(f"No data from Yahoo for ticker: {ticker}")
    s = df["Close"].dropna()
    s.index = pd.to_datetime(s.index)
    return s

@st.cache_data(ttl=60*60)
def fetch_fred_series(series_id: str, start: str, api_key: str | None = None) -> pd.Series:
    if not PDR_OK:
        raise RuntimeError("pandas_datareader is not installed. Run: pip install pandas_datareader")
    if api_key:
        os.environ["FRED_API_KEY"] = api_key
    df = pdr.DataReader(series_id, "fred", start)
    if df is None or df.empty:
        raise RuntimeError(f"No data from FRED for series: {series_id}")
    s = df[series_id].dropna()
    s.index = pd.to_datetime(s.index)
    return s

def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    """
    Percentile rank of each point vs previous window points, scaled 0..100.
    Uses a simple method: rank within rolling window.
    """
    s = series.copy().dropna()
    vals = s.values

    out = np.full(len(s), np.nan, dtype=float)
    for i in range(len(s)):
        j0 = max(0, i - window + 1)
        w = vals[j0:i+1]
        if len(w) < max(20, window // 10):  # require some history
            continue
        out[i] = 100.0 * (np.sum(w <= vals[i]) - 1) / max(1, len(w) - 1)
    return pd.Series(out, index=s.index)

def ma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def pct_change_bp(series_pct: pd.Series, days: int) -> pd.Series:
    # series_pct in % units (e.g., 4.5). change in bp over given days.
    return (series_pct - series_pct.shift(days)) * 100.0

def align_series(series_dict: dict[str, pd.Series]) -> pd.DataFrame:
    idx = None
    for s in series_dict.values():
        if idx is None:
            idx = s.index
        else:
            idx = idx.union(s.index)
    df = pd.DataFrame(index=idx).sort_index()
    for k, s in series_dict.items():
        df[k] = s
    return df.sort_index()

def safe_last(series: pd.Series):
    s = series.dropna()
    return None if s.empty else float(s.iloc[-1])

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def normalize_weights(w: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(v)) for v in w.values())
    if total <= 0:
        return {k: 0.0 for k in w}
    return {k: max(0.0, float(v)) / total for k, v in w.items()}

# =========================
# UI: Controls
# =========================
st.title("Market Dashboard: Raw Indicators + Risk Index (0 = Buy-leaning, 100 = Sell-leaning)")

with st.expander("데이터 소스 설정 / 주의", expanded=False):
    st.write(
        "- 기본은 Yahoo Finance(yfinance) + (가능하면) FRED(pandas_datareader)를 같이 씁니다.\n"
        "- FRED는 금리/유동성/신용 스프레드가 더 정확합니다.\n"
        "- 어떤 지표는 무료 데이터 지연/결측이 있을 수 있습니다."
    )

col1, col2, col3, col4 = st.columns(4)
with col1:
    display_days = st.selectbox("그래프 표시 기간(일)", [90, 180, 365, 730, 1825], index=2)
with col2:
    lookback_days = st.selectbox("Risk 퍼센타일 기준 기간(일)", [180, 365, 730, 1825], index=2)
with col3:
    ma_window = st.selectbox("추세 MA(일)", [50, 100, 200], index=2)
with col4:
    smooth_days = st.selectbox("Risk 스무딩(일)", [1, 3, 5, 10], index=1)

start = (date.today() - timedelta(days=int(lookback_days) + 10)).isoformat()
start_display = (date.today() - timedelta(days=int(display_days) + 10)).isoformat()

# Optional FRED key
fred_key = st.text_input("FRED API Key (있으면 입력, 없으면 비워도 됨)", value="", type="password")

# Weights
st.subheader("Risk Index 가중치 (합성 위험지수 구성)")
wcol = st.columns(7)
weights = {}
for i, (k, v) in enumerate(DEFAULT_WEIGHTS.items()):
    with wcol[i]:
        weights[k] = st.slider(k, 0.0, 0.5, float(v), 0.01)
weights = normalize_weights(weights)

# =========================
# Fetch data
# =========================
errors = []
series = {}

# Yahoo basics
needed_yf = {
    "NASDAQ": YF_TICKERS["NASDAQ (^IXIC)"],
    "SPX": YF_TICKERS["S&P 500 (^GSPC)"],
    "RUT": YF_TICKERS["Russell 2000 (^RUT)"],
    "VIX": YF_TICKERS["VIX (^VIX)"],
    "TNX": YF_TICKERS["US 10Y (proxy ^TNX)"],
    "DXY": YF_TICKERS["DXY (DX-Y.NYB)"],
    "OIL": YF_TICKERS["WTI Oil (CL=F)"],
    "GOLD": YF_TICKERS["Gold (GC=F)"],
    "COPPER": YF_TICKERS["Copper (HG=F)"],
    "BTC": YF_TICKERS["Bitcoin (BTC-USD)"],
}

if not YF_OK:
    st.error("yfinance 미설치. 로컬에서: pip install yfinance")
else:
    for key, tk in needed_yf.items():
        try:
            series[key] = fetch_yahoo_close(tk, start)
        except Exception as e:
            errors.append(str(e))

# FRED series (optional but recommended)
fred = {}
if PDR_OK:
    for name, sid in FRED_SERIES.items():
        try:
            fred[name] = fetch_fred_series(sid, start, api_key=fred_key.strip() or None)
        except Exception as e:
            # Do not hard fail; we'll continue with what we have.
            pass

# Prefer FRED DGS10 over TNX if available
if "US10Y (DGS10)" in fred:
    # Convert to % (already in %)
    series["US10Y_%"] = fred["US10Y (DGS10)"]
else:
    # Heuristic conversion: ^TNX often = yield*10
    if "TNX" in series:
        tnx = series["TNX"].copy()
if tnx is not None and not tnx.empty:
    tnx_clean = tnx.dropna()
    if not tnx_clean.empty and tnx_clean.median() > 20:
        series["US10Y_%"] = tnx_clean / 10.0
    else:
        series["US10Y_%"] = tnx_clean
else:
    # 데이터 없으면 10년물 제외하고 진행
    pass

# Credit spreads
if "HY Spread (BAMLH0A0HYM2)" in fred:
    series["HY_OAS"] = fred["HY Spread (BAMLH0A0HYM2)"]
if "IG Spread (BAMLC0A0CM)" in fred:
    series["IG_OAS"] = fred["IG Spread (BAMLC0A0CM)"]

# Liquidity proxies
if "TGA (WTREGEN)" in fred:
    series["TGA"] = fred["TGA (WTREGEN)"]
if "ON RRP (RRPONTSYD)" in fred:
    series["RRP"] = fred["ON RRP (RRPONTSYD)"]
if "Bank Reserves (WRESBAL)" in fred:
    series["RESERVES"] = fred["Bank Reserves (WRESBAL)"]

if errors:
    with st.expander("데이터 로딩 경고(일부 누락 가능)", expanded=False):
        for e in errors[:10]:
            st.write("-", e)

df = align_series(series).sort_index()

# =========================
# Build Risk Components (0..100)
# =========================
# Percentile window in trading days (~252/yr). We'll approximate:
win = int(int(lookback_days) * 0.7)  # daily series includes weekends missing; rough mapping
win = max(120, min(win, 1200))

risk_components = {}

# VIX: higher => higher risk
if "VIX" in df:
    vix_pct = rolling_percentile(df["VIX"], win)
    risk_components["VIX"] = vix_pct

# Equity trend: use NASDAQ vs MA and percentile of ratio
if "NASDAQ" in df:
    ratio = df["NASDAQ"] / ma(df["NASDAQ"], int(ma_window))
    ratio_pct = rolling_percentile(ratio, win)
    # When ratio is high (above MA), risk tends to be lower => invert
    risk_components["Equity_Trend"] = 100.0 - ratio_pct

# USD shock: use DXY 5d change percentile (higher change => higher risk)
if "DXY" in df:
    dxy_5d = df["DXY"].pct_change(5) * 100.0
    dxy_shock_pct = rolling_percentile(dxy_5d, win)
    risk_components["USD_Shock"] = dxy_shock_pct

# Oil shock: 5d change percentile (spike => risk up)
if "OIL" in df:
    oil_5d = df["OIL"].pct_change(5) * 100.0
    oil_shock_pct = rolling_percentile(oil_5d, win)
    risk_components["Oil_Shock"] = oil_shock_pct

# Rates_10Y: combine level + spike (bp over 5d)
if "US10Y_%" in df:
    y = df["US10Y_%"]
    y_level_pct = rolling_percentile(y, win)
    y_spike_bp = pct_change_bp(y, 5)
    y_spike_pct = rolling_percentile(y_spike_bp, win)
    # weighted inside this component: emphasize spike
    risk_components["Rates_10Y"] = 0.4 * y_level_pct + 0.6 * y_spike_pct

# Credit: HY spread percentile (higher => higher risk)
if "HY_OAS" in df:
    hy_pct = rolling_percentile(df["HY_OAS"], win)
    risk_components["Credit_HY"] = hy_pct
else:
    # fallback: if IG exists, use it lightly
    if "IG_OAS" in df:
        ig_pct = rolling_percentile(df["IG_OAS"], win)
        risk_components["Credit_HY"] = ig_pct  # name kept for weighting compatibility

# Liquidity: TGA up = risk up, RRP down = risk up, Reserves down = risk up
liq_parts = []
if "TGA" in df:
    tga_pct = rolling_percentile(df["TGA"].diff(5), win)  # 5d change; up => risk up
    liq_parts.append(tga_pct)
if "RRP" in df:
    rrp_pct = rolling_percentile(-df["RRP"].diff(5), win)  # falling RRP => risk up (invert diff)
    liq_parts.append(rrp_pct)
if "RESERVES" in df:
    res_pct = rolling_percentile(-df["RESERVES"].diff(5), win)  # falling reserves => risk up
    liq_parts.append(res_pct)

if liq_parts:
    risk_components["Liquidity"] = sum(liq_parts) / len(liq_parts)

risk_df = pd.DataFrame(risk_components).dropna(how="all")

# Weighted Risk Index
w = weights
common = [k for k in w.keys() if k in risk_df.columns]
if not common:
    st.error("Risk Index를 계산할 핵심 지표가 부족합니다. (VIX/NASDAQ/US10Y 등이 로드되지 않음)")
    st.stop()

risk_index = sum(w[k] * risk_df[k] for k in common)
risk_index.name = "RISK"

# Smoothing
risk_smooth = risk_index.rolling(int(smooth_days)).mean()

# Clip
risk_smooth = risk_smooth.clip(0, 100)

# Latest snapshot
latest_date = risk_smooth.dropna().index.max()
latest_risk = float(risk_smooth.dropna().iloc[-1]) if latest_date is not None else np.nan

# =========================
# Tabs
# =========================
tab_raw, tab_risk = st.tabs(["① Raw Dashboard (지표)", "② Risk Dashboard (0~100)"])

# ---- RAW ----
with tab_raw:
    st.subheader("원본 지표 (선형 그래프)")

    # selector
    left, right = st.columns([1, 2])
    with left:
        options = list(YF_TICKERS.keys())
        # add available FRED series as options too
        fred_opts = [f"FRED: {k}" for k in FRED_SERIES.keys() if k in fred]
        selected = st.multiselect(
            "보고 싶은 지표 선택",
            options=options + fred_opts,
            default=["NASDAQ (^IXIC)", "VIX (^VIX)", "US 10Y (proxy ^TNX)", "DXY (DX-Y.NYB)", "WTI Oil (CL=F)"]
        )

    # build series list to plot
    plot_series = {}
    for name in selected:
        if name in YF_TICKERS:
            key_map = {
                "NASDAQ (^IXIC)": "NASDAQ",
                "S&P 500 (^GSPC)": "SPX",
                "Russell 2000 (^RUT)": "RUT",
                "VIX (^VIX)": "VIX",
                "US 10Y (proxy ^TNX)": "US10Y_%",
                "DXY (DX-Y.NYB)": "DXY",
                "WTI Oil (CL=F)": "OIL",
                "Gold (GC=F)": "GOLD",
                "Copper (HG=F)": "COPPER",
                "Bitcoin (BTC-USD)": "BTC",
            }
            k = key_map.get(name)
            if k and k in df:
                plot_series[name] = df[k]
        elif name.startswith("FRED: "):
            fred_name = name.replace("FRED: ", "")
            if fred_name in fred:
                plot_series[name] = fred[fred_name]

    if not plot_series:
        st.info("선택된 지표 데이터가 없습니다.")
    else:
        plot_df = align_series(plot_series)
        plot_df = plot_df[plot_df.index >= pd.to_datetime(start_display)]

        fig = go.Figure()
        for c in plot_df.columns:
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[c], mode="lines", name=c))
        fig.update_layout(height=560, margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.caption("Tip: 금리/유동성/스프레드가 필요하면 FRED 데이터가 더 정확합니다.")

# ---- RISK ----
with tab_risk:
    st.subheader("합성 Risk Index (0 = 매수 성향, 100 = 매도 성향)")

    c1, c2, c3 = st.columns([1.2, 1.8, 1.0])

    with c1:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=latest_risk,
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [
                    {"range": [0, 30]},
                    {"range": [30, 60]},
                    {"range": [60, 80]},
                    {"range": [80, 100]},
                ],
            },
            title={"text": "RISK (Smoothed)"}
        ))
        gauge.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(gauge, use_container_width=True)

        # Simple label
        label = "중립"
        if latest_risk <= 30:
            label = "매수 성향"
        elif latest_risk <= 60:
            label = "중립/분할"
        elif latest_risk <= 80:
            label = "경계(리스크 높음)"
        else:
            label = "매도 성향"

        st.metric("오늘 판단", label, f"{latest_risk:.1f} / 100")

    with c2:
        # Risk line
        rd = pd.DataFrame({"RISK": risk_smooth})
        rd = rd[rd.index >= pd.to_datetime(start_display)]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=rd.index, y=rd["RISK"], mode="lines", name="RISK"))
        fig2.add_hline(y=30, line_dash="dash")
        fig2.add_hline(y=60, line_dash="dash")
        fig2.add_hline(y=80, line_dash="dash")
        fig2.update_layout(height=360, margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(fig2, use_container_width=True)

        st.caption("기본 구간: 0~30 매수 / 30~60 중립 / 60~80 경계 / 80~100 매도 성향")

    with c3:
        # Contribution table (latest)
        st.write("**구성요소(오늘)**")
        last_row = risk_df.loc[:latest_date].dropna().tail(1)
        if not last_row.empty:
            comp = last_row.iloc[0].copy()
            # contributions
            contrib = {k: (weights.get(k, 0.0) * comp.get(k, np.nan)) for k in common}
            contrib_s = pd.Series(contrib).sort_values(ascending=False)
            comp_s = comp[common].sort_values(ascending=False)

            out = pd.DataFrame({
                "Risk(0-100)": comp_s.round(1),
                "Weight": pd.Series({k: weights[k] for k in common}),
                "Contribution": contrib_s.round(2),
            }).sort_values("Contribution", ascending=False)

            st.dataframe(out, use_container_width=True, height=340)
        else:
            st.info("구성요소를 계산할 데이터가 부족합니다.")

    st.divider()
    st.subheader("최근 15일 스냅샷")
    snap = pd.DataFrame({"RISK": risk_smooth}).join(risk_df[common], how="left")
    snap = snap.dropna().tail(15)
    st.dataframe(snap.round(2), use_container_width=True)

    st.caption(
        "주의: Risk Index는 '설명 가능한 룰(퍼센타일+추세+급등락)'로 만든 합성지표입니다. "
        "데이터 결측/지연이 있을 수 있고, 투자성과를 보장하지 않습니다."
    )
