import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Stock Regime Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent / "data"
SECTOR_FILES = {
    "Bank": DATA_DIR / "bank_panel.csv",
    "Securities": DATA_DIR / "securities_panel.csv",
}
BAD_REGIMES = {"Stock Bear", "Stock Divergence"}
REGIME_COLORS = {
    "Stock Bull": "rgba(22, 163, 74, 0.16)",
    "Stock Sideway": "rgba(100, 116, 139, 0.11)",
    "Stock Bear": "rgba(220, 38, 38, 0.16)",
    "Stock Divergence": "rgba(124, 58, 237, 0.16)",
}

TEXT = {
    "vi": {
        "app_title": "Stock Regime Dashboard",
        "subtitle": "Dashboard Streamlit cho regime cổ phiếu ngân hàng và chứng khoán, dữ liệu full 2015–2026.",
        "language": "Ngôn ngữ",
        "sector": "Ngành",
        "ticker": "Mã cổ phiếu",
        "range": "Khoảng thời gian chart",
        "full": "Toàn bộ 2015–nay",
        "3y": "3 năm gần nhất",
        "1y": "1 năm gần nhất",
        "custom": "Tùy chọn",
        "start": "Từ ngày",
        "end": "Đến ngày",
        "overview": "Tổng quan",
        "chart": "Chart regime",
        "current_validation": "Kiểm định trạng thái hiện tại",
        "trend_capture": "Trend capture",
        "warning": "Dự báo trạng thái xấu",
        "methodology": "Quy trình",
        "deploy": "Deploy",
        "latest": "Snapshot mới nhất",
        "current_state_auc": "AUC trạng thái hiện tại",
        "future_warning_auc": "AUC cảnh báo 20 phiên",
        "trend_capture_rate": "Tỷ lệ bắt up-trend",
        "bad_capture_rate": "Tỷ lệ bắt bad-trend",
        "chart_note": "Chart chỉ render mã đang chọn để tránh tải nặng. Vùng nền thể hiện Stock Regime.",
        "current_desc": "Phần này kiểm tra model có xác định đúng trạng thái tại ngày T hay không, không phải dự báo tương lai.",
        "trend_desc": "Phần này kiểm tra model có bắt được các nhịp tăng/xấu khách quan hay không: có xuất hiện regime đúng trong nhịp, xuất hiện sớm không, và độ phủ bao nhiêu.",
        "warning_desc": "Phần này kiểm tra risk score tại ngày T có cảnh báo được việc 20 phiên tới cổ phiếu rơi vào Bear/Divergence hay không.",
    },
    "en": {
        "app_title": "Stock Regime Dashboard",
        "subtitle": "Streamlit dashboard for bank and securities stock regimes, full 2015–2026 data.",
        "language": "Language",
        "sector": "Sector",
        "ticker": "Ticker",
        "range": "Chart range",
        "full": "Full 2015–current",
        "3y": "Last 3 years",
        "1y": "Last 1 year",
        "custom": "Custom",
        "start": "Start date",
        "end": "End date",
        "overview": "Overview",
        "chart": "Regime chart",
        "current_validation": "Current-state validation",
        "trend_capture": "Trend capture",
        "warning": "Future bad-state warning",
        "methodology": "Process",
        "deploy": "Deploy",
        "latest": "Latest snapshot",
        "current_state_auc": "Current-state AUC",
        "future_warning_auc": "20D warning AUC",
        "trend_capture_rate": "Up-trend capture rate",
        "bad_capture_rate": "Bad-trend capture rate",
        "chart_note": "Only the selected ticker is rendered to keep the app responsive. Background shading shows Stock Regime.",
        "current_desc": "This section validates whether the model identifies the current state at date T. It is not a future-return test.",
        "trend_desc": "This section checks whether the model captures objective up-trends/bad-trends: whether the right regime appears, appears early, and covers the trend.",
        "warning_desc": "This section tests whether risk score at date T warns that the stock will enter Bear/Divergence within the next 20 sessions.",
    },
}

# ============================================================
# Styling
# ============================================================
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 3rem;}
    div[data-testid="stMetric"] {background: #ffffff; border: 1px solid #e5e7eb; padding: 14px 16px; border-radius: 16px; box-shadow: 0 4px 16px rgba(15,23,42,.05);}    
    .small-note {font-size: 0.86rem; color: #64748b; line-height: 1.45;}
    .section-card {border:1px solid #e5e7eb; border-radius:16px; padding:16px; background:#fff; box-shadow:0 4px 16px rgba(15,23,42,.04);}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Utilities
# ============================================================
def fmt_pct(x, digits=1):
    if pd.isna(x):
        return "—"
    return f"{x*100:.{digits}f}%"


def fmt_num(x, digits=3):
    if pd.isna(x):
        return "—"
    return f"{x:.{digits}f}"


def risk_col(df):
    for c in ["stock_regime_risk_score", "score_stock_momentum_risk", "risk_score_unified", "stock_risk_score", "stock_risk_score_smooth"]:
        if c in df.columns:
            return c
    return None


@st.cache_data(show_spinner=False)
def load_sector(sector: str) -> pd.DataFrame:
    path = SECTOR_FILES[sector]
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # Streamlit Cloud may install a recent pandas version where errors="ignore"
    # is no longer accepted by pd.to_numeric. Keep text/categorical columns intact
    # and coerce only true numeric-like columns.
    text_cols = {
        "sector", "ticker", "stock_regime", "stock_regime_risk_zone",
        "sector_regime", "sector_context", "market_regime", "market_context",
        "dashboard_decision"
    }
    for c in df.columns:
        if c not in text_cols and c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "drawdown60" not in df.columns and "drawdown_60" in df.columns:
        df["drawdown60"] = df["drawdown_60"]
    if "stock_regime_risk_score" not in df.columns and "score_stock_momentum_risk" in df.columns:
        df["stock_regime_risk_score"] = df["score_stock_momentum_risk"]
    if "stock_risk_score" not in df.columns and "stock_risk_score_smooth" in df.columns:
        df["stock_risk_score"] = df["stock_risk_score_smooth"]
    if "sector_risk_score" not in df.columns and "sector_risk_score_smooth" in df.columns:
        df["sector_risk_score"] = df["sector_risk_score_smooth"]
    df = add_objective_states(df)
    return df


@st.cache_data(show_spinner=False)
def load_all_summary() -> pd.DataFrame:
    rows = []
    for sector in SECTOR_FILES:
        df = load_sector(sector)
        cv = current_state_validation(df)
        auc_row = cv[cv["test"] == "Risk Score vs Objective Bad State"]
        wt = warning_by_ticker(df)
        ts = trend_capture_summary(df)
        up = ts[ts["trend_type"].str.contains("up-trend", na=False)] if not ts.empty else pd.DataFrame()
        bad = ts[ts["trend_type"].str.contains("bad-trend", na=False)] if not ts.empty else pd.DataFrame()
        rows.append({
            "sector": sector,
            "tickers": df["ticker"].nunique(),
            "data_start": df["date"].min(),
            "data_end": df["date"].max(),
            "current_state_auc": auc_row["roc_auc"].iloc[0] if len(auc_row) else np.nan,
            "current_state_ap": auc_row["average_precision"].iloc[0] if len(auc_row) else np.nan,
            "future_warning_auc_mean": wt["roc_auc"].mean() if len(wt) else np.nan,
            "future_warning_ap_lift_mean": wt["ap_lift"].mean() if len(wt) else np.nan,
            "up_trend_capture_rate": up["capture_rate"].iloc[0] if len(up) else np.nan,
            "up_trend_early_capture_rate": up["early_capture_rate"].iloc[0] if len(up) else np.nan,
            "bad_trend_capture_rate": bad["capture_rate"].iloc[0] if len(bad) else np.nan,
            "bad_trend_early_capture_rate": bad["early_capture_rate"].iloc[0] if len(bad) else np.nan,
        })
    return pd.DataFrame(rows)


def add_objective_states(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["ret20", "ret60", "above_ma20", "above_ma60", "drawdown60", "close"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["objective_up_state"] = (
        (df["ret20"] > 0)
        & (df["ret60"] > 0)
        & (df["above_ma20"] == 1)
        & (df["above_ma60"] == 1)
        & (df["drawdown60"] > -0.12)
    ).astype(int)
    df["objective_down_state"] = (
        (df["ret20"] < 0)
        & (df["ret60"] < 0)
        & (df["above_ma20"] == 0)
        & (df["above_ma60"] == 0)
    ).astype(int)
    df["objective_bad_state"] = (
        (df["ret20"] < -0.03)
        | (df["ret60"] < -0.06)
        | (df["drawdown60"] < -0.15)
        | ((df["above_ma20"] == 0) & (df["above_ma60"] == 0))
    ).astype(int)
    df["model_bull_state"] = (df["stock_regime"] == "Stock Bull").astype(int)
    df["model_bad_state"] = df["stock_regime"].isin(BAD_REGIMES).astype(int)
    rc = risk_col(df)
    df["risk_score_unified"] = pd.to_numeric(df[rc], errors="coerce") if rc else np.nan
    return df


def safe_div(a, b):
    return np.nan if b == 0 else a / b


def binary_metrics(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(int)
    tp = int(((y == 1) & (p == 1)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    f1 = safe_div(2 * precision * recall, precision + recall) if not (pd.isna(precision) or pd.isna(recall) or precision + recall == 0) else np.nan
    balanced_accuracy = np.nanmean([recall, specificity])
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else np.nan
    po = safe_div(tp + tn, len(y))
    pe = safe_div((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn), len(y) ** 2)
    kappa = safe_div(po - pe, 1 - pe) if pe != 1 else np.nan
    return {
        "n": len(y), "positive_rate": float(np.mean(y)), "precision": precision, "recall": recall,
        "specificity": specificity, "f1": f1, "balanced_accuracy": balanced_accuracy, "mcc": mcc,
        "cohens_kappa": kappa, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def roc_auc(y, s):
    d = pd.DataFrame({"y": y, "s": s}).dropna()
    if d.empty or d["y"].nunique() < 2:
        return np.nan
    ranks = d["s"].rank(method="average")
    npos = int((d["y"] == 1).sum())
    nneg = int((d["y"] == 0).sum())
    return float((ranks[d["y"] == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def avg_precision(y, s):
    d = pd.DataFrame({"y": y, "s": s}).dropna().sort_values("s", ascending=False)
    if d.empty or d["y"].sum() == 0:
        return np.nan
    d["tp"] = d["y"].cumsum()
    d["rank"] = np.arange(1, len(d) + 1)
    d["precision"] = d["tp"] / d["rank"]
    return float(d.loc[d["y"] == 1, "precision"].sum() / d["y"].sum())


@st.cache_data(show_spinner=False)
def current_state_validation(df):
    rows = []
    for test, y, p in [
        ("Bull vs Objective Up State", "objective_up_state", "model_bull_state"),
        ("Bear/Divergence vs Objective Bad State", "objective_bad_state", "model_bad_state"),
    ]:
        m = binary_metrics(df[y], df[p])
        m["test"] = test
        rows.append(m)
    ap = avg_precision(df["objective_bad_state"], df["risk_score_unified"])
    base = df["objective_bad_state"].mean()
    rows.append({
        "test": "Risk Score vs Objective Bad State",
        "n": int(df["risk_score_unified"].notna().sum()),
        "positive_rate": base,
        "roc_auc": roc_auc(df["objective_bad_state"], df["risk_score_unified"]),
        "average_precision": ap,
        "ap_lift": ap / base if base else np.nan,
    })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def regime_daily_summary(df):
    return df.groupby(["stock_regime"]).agg(
        observations=("ticker", "size"),
        avg_ret20=("ret20", "mean"),
        avg_ret60=("ret60", "mean"),
        pct_objective_up=("objective_up_state", "mean"),
        pct_objective_bad=("objective_bad_state", "mean"),
        pct_above_ma20=("above_ma20", "mean"),
        pct_above_ma60=("above_ma60", "mean"),
        avg_drawdown60=("drawdown60", "mean"),
        avg_risk_score=("risk_score_unified", "mean"),
    ).reset_index()


@st.cache_data(show_spinner=False)
def objective_matrix(df):
    cls = np.where(
        df["objective_bad_state"] == 1, "Objective Bad",
        np.where(df["objective_up_state"] == 1, "Objective Up",
                 np.where(df["objective_down_state"] == 1, "Objective Down", "Objective Neutral"))
    )
    tmp = df[["stock_regime"]].copy()
    tmp["objective_class"] = cls
    return pd.crosstab(tmp["stock_regime"], tmp["objective_class"], normalize="index")


@st.cache_data(show_spinner=False)
def regime_episodes(df, min_days=5):
    rows = []
    for ticker, g in df.sort_values(["ticker", "date"]).groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)
        eid = (g["stock_regime"] != g["stock_regime"].shift()).cumsum()
        for _, e in g.groupby(eid):
            if len(e) < min_days:
                continue
            c = e["close"].astype(float).to_numpy()
            if len(c) < 2 or not np.isfinite(c[0]) or c[0] <= 0:
                continue
            rel = c / c[0] - 1
            slope = np.polyfit(np.arange(len(c)), np.log(np.maximum(c, 1e-12)), 1)[0]
            regime = e["stock_regime"].iloc[0]
            rows.append({
                "ticker": ticker,
                "stock_regime": regime,
                "start_date": e["date"].iloc[0],
                "end_date": e["date"].iloc[-1],
                "duration_days": len(e),
                "episode_return": c[-1] / c[0] - 1,
                "mfe_max_runup": np.nanmax(rel),
                "mae_max_drawdown": np.nanmin(rel),
                "trend_slope_daily": slope,
                "objective_up_purity": e["objective_up_state"].mean(),
                "objective_bad_purity": e["objective_bad_state"].mean(),
                "continuation_hit": (c[-1] > c[0]) if regime == "Stock Bull" else (c[-1] < c[0]) if regime == "Stock Bear" else np.nan,
            })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def episode_summary(df, min_days=5):
    eps = regime_episodes(df, min_days=min_days)
    if eps.empty:
        return eps
    rows = []
    for regime, g in eps.groupby("stock_regime"):
        w = g["duration_days"].astype(float)
        rows.append({
            "stock_regime": regime,
            "episode_count": len(g),
            "median_duration_days": g["duration_days"].median(),
            "median_episode_return": g["episode_return"].median(),
            "duration_weighted_return": np.average(g["episode_return"], weights=w) if w.sum() else np.nan,
            "continuation_hit_rate": g["continuation_hit"].dropna().mean() if g["continuation_hit"].notna().any() else np.nan,
            "median_max_runup": g["mfe_max_runup"].median(),
            "median_max_drawdown": g["mae_max_drawdown"].median(),
            "median_objective_up_purity": g["objective_up_purity"].median(),
            "median_objective_bad_purity": g["objective_bad_purity"].median(),
            "positive_slope_rate": (g["trend_slope_daily"] > 0).mean(),
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def trend_capture_events(df, min_trend=10):
    rows = []
    for ticker, g in df.sort_values(["ticker", "date"]).groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)
        tests = [
            ("objective_up_state", "Objective up-trend captured by Stock Bull", lambda x: x["stock_regime"].eq("Stock Bull")),
            ("objective_bad_state", "Objective bad-trend captured by Bear/Divergence", lambda x: x["stock_regime"].isin(BAD_REGIMES)),
        ]
        for objective_col, label, model_func in tests:
            active = g[objective_col].astype(int)
            rid = (active.ne(active.shift()) | (active == 0)).cumsum()
            for _, e in g[active == 1].groupby(rid[active == 1]):
                if len(e) < min_trend:
                    continue
                c = e["close"].astype(float).to_numpy()
                if len(c) < 2 or not np.isfinite(c[0]) or c[0] <= 0:
                    continue
                cond = model_func(e).to_numpy().astype(bool)
                has_capture = bool(cond.any())
                lag = int(np.argmax(cond)) if has_capture else np.nan
                captured_return = 0.0
                if has_capture and cond.sum() >= 2:
                    cc = e.loc[cond, "close"].astype(float).to_numpy()
                    captured_return = cc[-1] / cc[0] - 1
                rows.append({
                    "ticker": ticker,
                    "trend_type": label,
                    "start_date": e["date"].iloc[0],
                    "end_date": e["date"].iloc[-1],
                    "duration_days": len(e),
                    "total_return": c[-1] / c[0] - 1,
                    "captured_return_inside_model_state": captured_return,
                    "capture_success": int(has_capture),
                    "early_capture": int(has_capture and lag <= max(3, int(len(e) * 0.2))),
                    "lag_days": lag,
                    "coverage_ratio": cond.mean(),
                })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def trend_capture_summary(df, min_trend=10):
    tr = trend_capture_events(df, min_trend=min_trend)
    if tr.empty:
        return tr
    return tr.groupby("trend_type").agg(
        trend_count=("ticker", "size"),
        median_duration_days=("duration_days", "median"),
        median_total_return=("total_return", "median"),
        capture_rate=("capture_success", "mean"),
        early_capture_rate=("early_capture", "mean"),
        median_lag_days=("lag_days", "median"),
        median_coverage_ratio=("coverage_ratio", "median"),
        median_captured_return=("captured_return_inside_model_state", "median"),
    ).reset_index()


@st.cache_data(show_spinner=False)
def warning_by_ticker(df):
    d = df[(df["model_bad_state"] == 0) & df["future_bad_within20"].notna() & df["risk_score_unified"].notna()].copy()
    if d.empty:
        return pd.DataFrame()
    d["future_bad_within20"] = d["future_bad_within20"].astype(int)
    rows = []
    for ticker, g in d.groupby("ticker"):
        if len(g) > 50 and g["future_bad_within20"].nunique() > 1:
            ap = avg_precision(g["future_bad_within20"], g["risk_score_unified"])
            base = g["future_bad_within20"].mean()
            rows.append({
                "ticker": ticker,
                "n": len(g),
                "roc_auc": roc_auc(g["future_bad_within20"], g["risk_score_unified"]),
                "average_precision": ap,
                "base_rate": base,
                "ap_lift": ap / base if base else np.nan,
            })
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False) if rows else pd.DataFrame()


@st.cache_data(show_spinner=False)
def warning_by_year(df):
    d = df[(df["model_bad_state"] == 0) & df["future_bad_within20"].notna() & df["risk_score_unified"].notna()].copy()
    if d.empty:
        return pd.DataFrame()
    d["future_bad_within20"] = d["future_bad_within20"].astype(int)
    d["year"] = d["date"].dt.year
    rows = []
    for year, g in d.groupby("year"):
        if len(g) > 100 and g["future_bad_within20"].nunique() > 1:
            ap = avg_precision(g["future_bad_within20"], g["risk_score_unified"])
            base = g["future_bad_within20"].mean()
            rows.append({"year": int(year), "n": len(g), "roc_auc": roc_auc(g["future_bad_within20"], g["risk_score_unified"]), "average_precision": ap, "base_rate": base, "ap_lift": ap / base if base else np.nan})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def risk_decile_calibration(df):
    d = df[(df["model_bad_state"] == 0) & df["future_bad_within20"].notna() & df["risk_score_unified"].notna()].copy()
    if d.empty:
        return pd.DataFrame()
    d["future_bad_within20"] = d["future_bad_within20"].astype(int)
    try:
        d["risk_decile"] = pd.qcut(d["risk_score_unified"], 10, labels=False, duplicates="drop") + 1
    except Exception:
        return pd.DataFrame()
    return d.groupby("risk_decile").agg(n=("ticker", "size"), avg_risk_score=("risk_score_unified", "mean"), future_bad_rate=("future_bad_within20", "mean")).reset_index()


def latest_snapshot(df):
    idx = df.groupby("ticker")["date"].idxmax()
    return df.loc[idx].sort_values("risk_score_unified")


def filter_date_range(df, mode, start=None, end=None):
    if df.empty:
        return df
    max_date = df["date"].max()
    if mode == "1Y":
        min_date = max_date - pd.DateOffset(years=1)
    elif mode == "3Y":
        min_date = max_date - pd.DateOffset(years=3)
    elif mode == "Custom" and start is not None and end is not None:
        return df[(df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))]
    else:
        min_date = df["date"].min()
    return df[(df["date"] >= min_date) & (df["date"] <= max_date)]


def make_chart(tdf):
    tdf = tdf.sort_values("date").copy()
    fig = go.Figure()
    if all(c in tdf.columns for c in ["open", "high", "low", "close"]):
        fig.add_trace(go.Candlestick(
            x=tdf["date"], open=tdf["open"], high=tdf["high"], low=tdf["low"], close=tdf["close"],
            name="OHLC", increasing_line_color="#16a34a", decreasing_line_color="#dc2626"
        ))
    else:
        fig.add_trace(go.Scatter(x=tdf["date"], y=tdf["close"], mode="lines", name="Close"))
    for ma in ["ma20", "ma60", "ma120", "ma200"]:
        if ma in tdf.columns and tdf[ma].notna().any():
            fig.add_trace(go.Scatter(x=tdf["date"], y=tdf[ma], mode="lines", name=ma.upper(), line=dict(width=1.4)))
    if "stock_regime" in tdf.columns:
        episode = (tdf["stock_regime"] != tdf["stock_regime"].shift()).cumsum()
        shapes = []
        for _, e in tdf.groupby(episode):
            regime = e["stock_regime"].iloc[0]
            color = REGIME_COLORS.get(regime, "rgba(148,163,184,0.08)")
            shapes.append(dict(type="rect", xref="x", yref="paper", x0=e["date"].iloc[0], x1=e["date"].iloc[-1], y0=0, y1=1, fillcolor=color, line=dict(width=0), layer="below"))
        fig.update_layout(shapes=shapes)
    fig.update_layout(
        height=680,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    return fig


def show_download(df, filename, label="Download CSV"):
    st.download_button(label, df.to_csv(index=False).encode("utf-8-sig"), file_name=filename, mime="text/csv")

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    lang_label = st.radio("Ngôn ngữ / Language", ["Tiếng Việt", "English"], horizontal=True)
    lang = "vi" if lang_label == "Tiếng Việt" else "en"
    t = TEXT[lang]
    st.divider()
    sector = st.selectbox(t["sector"], list(SECTOR_FILES.keys()))
    df = load_sector(sector)
    tickers = sorted(df["ticker"].dropna().unique())
    ticker = st.selectbox(t["ticker"], tickers, index=0)
    range_label = st.selectbox(t["range"], [t["3y"], t["1y"], t["full"], t["custom"]])
    mode = {t["3y"]: "3Y", t["1y"]: "1Y", t["full"]: "Full", t["custom"]: "Custom"}[range_label]
    start = end = None
    if mode == "Custom":
        start = st.date_input(t["start"], value=df["date"].min().date(), min_value=df["date"].min().date(), max_value=df["date"].max().date())
        end = st.date_input(t["end"], value=df["date"].max().date(), min_value=df["date"].min().date(), max_value=df["date"].max().date())
    st.divider()
    st.caption(f"Data: {df['date'].min().date()} → {df['date'].max().date()} | {df['ticker'].nunique()} tickers")

# ============================================================
# Header
# ============================================================
st.title(TEXT[lang]["app_title"])
st.caption(TEXT[lang]["subtitle"])
summary = load_all_summary()

# ============================================================
# Tabs
# ============================================================
tabs = st.tabs([t["overview"], t["chart"], t["current_validation"], t["trend_capture"], t["warning"], t["methodology"], t["deploy"]])

with tabs[0]:
    st.subheader(t["overview"])
    selected_summary = summary[summary["sector"] == sector].iloc[0]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tickers", int(selected_summary["tickers"]))
    c2.metric(t["current_state_auc"], fmt_num(selected_summary["current_state_auc"], 3))
    c3.metric(t["future_warning_auc"], fmt_num(selected_summary["future_warning_auc_mean"], 3))
    c4.metric(t["trend_capture_rate"], fmt_pct(selected_summary["up_trend_capture_rate"]))
    c5.metric(t["bad_capture_rate"], fmt_pct(selected_summary["bad_trend_capture_rate"]))

    st.markdown("### " + t["latest"])
    latest = latest_snapshot(df)
    cols_show = ["date", "ticker", "close", "stock_regime", "stock_regime_risk_zone", "risk_score_unified", "ret20", "ret60", "rs_vs_sector_20", "rs_vs_sector_60", "sector_regime", "sector_context", "market_regime", "market_context"]
    cols_show = [c for c in cols_show if c in latest.columns]
    st.dataframe(latest[cols_show], use_container_width=True, height=420)
    show_download(latest[cols_show], f"{sector.lower()}_latest_snapshot.csv")

    st.markdown("### Cross-sector summary")
    st.dataframe(summary, use_container_width=True)

with tabs[1]:
    st.subheader(f"{ticker} — {t['chart']}")
    st.caption(t["chart_note"])
    tdf = df[df["ticker"] == ticker].copy()
    tdf = filter_date_range(tdf, mode, start, end)
    st.plotly_chart(make_chart(tdf), use_container_width=True)
    latest_t = df[df["ticker"] == ticker].sort_values("date").tail(1).iloc[0]
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Close", f"{latest_t['close']:,.0f}")
    c2.metric("Stock Regime", str(latest_t.get("stock_regime", "—")))
    c3.metric("Risk Score", fmt_num(latest_t.get("risk_score_unified"), 3))
    c4.metric("20D Return", fmt_pct(latest_t.get("ret20")))
    c5.metric("60D Return", fmt_pct(latest_t.get("ret60")))
    c6.metric("Market", str(latest_t.get("market_context", "—")))

with tabs[2]:
    st.subheader(t["current_validation"])
    st.caption(t["current_desc"])
    cv = current_state_validation(df)
    st.markdown("#### Classification metrics")
    st.dataframe(cv, use_container_width=True)
    show_download(cv, f"{sector.lower()}_current_state_validation.csv")

    st.markdown("#### Regime daily profile")
    daily = regime_daily_summary(df)
    st.dataframe(daily, use_container_width=True)
    fig = px.bar(daily, x="stock_regime", y=["pct_objective_up", "pct_objective_bad"], barmode="group", title="Objective state share by regime")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Objective-state matrix")
    mat = objective_matrix(df)
    st.dataframe(mat.style.format("{:.1%}"), use_container_width=True)
    fig2 = px.imshow(mat, text_auto=".0%", aspect="auto", title="Regime vs objective state matrix")
    st.plotly_chart(fig2, use_container_width=True)

with tabs[3]:
    st.subheader(t["trend_capture"])
    st.caption(t["trend_desc"])
    min_trend = st.slider("Minimum objective trend length", min_value=5, max_value=30, value=10, step=1)
    min_episode = st.slider("Minimum regime episode length", min_value=3, max_value=30, value=5, step=1)
    ts = trend_capture_summary(df, min_trend=min_trend)
    es = episode_summary(df, min_days=min_episode)
    st.markdown("#### Objective trend capture summary")
    st.dataframe(ts, use_container_width=True)
    if not ts.empty:
        fig = px.bar(ts, x="trend_type", y=["capture_rate", "early_capture_rate", "median_coverage_ratio"], barmode="group", title="Trend capture metrics")
        st.plotly_chart(fig, use_container_width=True)
    show_download(ts, f"{sector.lower()}_trend_capture_summary.csv")

    st.markdown("#### Regime episode path quality")
    st.dataframe(es, use_container_width=True)
    show_download(es, f"{sector.lower()}_episode_quality_summary.csv")

    with st.expander("Show raw trend-capture events"):
        events = trend_capture_events(df, min_trend=min_trend)
        st.dataframe(events, use_container_width=True, height=450)
        show_download(events, f"{sector.lower()}_trend_capture_events.csv")

with tabs[4]:
    st.subheader(t["warning"])
    st.caption(t["warning_desc"])
    wt = warning_by_ticker(df)
    wy = warning_by_year(df)
    cal = risk_decile_calibration(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean ticker AUC", fmt_num(wt["roc_auc"].mean() if len(wt) else np.nan, 3))
    c2.metric("Mean ticker AP Lift", fmt_num(wt["ap_lift"].mean() if len(wt) else np.nan, 2))
    c3.metric("Weighted yearly AUC", fmt_num(np.average(wy["roc_auc"], weights=wy["n"]) if len(wy) else np.nan, 3))
    c4.metric("Years tested", len(wy))

    st.markdown("#### By ticker")
    st.dataframe(wt, use_container_width=True)
    fig = px.bar(wt, x="ticker", y="roc_auc", title="Future bad-state AUC by ticker") if len(wt) else None
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    show_download(wt, f"{sector.lower()}_future_warning_by_ticker.csv")

    st.markdown("#### Walk-forward by year")
    st.dataframe(wy, use_container_width=True)
    if len(wy):
        fig2 = px.line(wy, x="year", y=["roc_auc", "average_precision", "base_rate"], markers=True, title="Walk-forward warning quality by year")
        st.plotly_chart(fig2, use_container_width=True)
    show_download(wy, f"{sector.lower()}_future_warning_by_year.csv")

    st.markdown("#### Risk decile calibration")
    st.dataframe(cal, use_container_width=True)
    if len(cal):
        fig3 = px.line(cal, x="risk_decile", y=["avg_risk_score", "future_bad_rate"], markers=True, title="Calibration by risk decile")
        st.plotly_chart(fig3, use_container_width=True)

with tabs[5]:
    st.subheader(t["methodology"])
    st.markdown(
        """
### 1. Data
- Input: daily OHLCV, returns, moving averages, drawdown, sector-relative strength and regime scores.
- Universe: Bank and Securities stocks.
- Window: full 2015–2026 data included in the app package.

### 2. Regime identification
The model assigns each stock to one of four states:

- **Stock Bull**: positive trend, above key moving averages, low drawdown, supportive relative strength.
- **Stock Sideway**: neutral/flat state.
- **Stock Bear**: weak trend, under moving averages, deeper drawdown.
- **Stock Divergence**: stock may still look okay in the short term, but sector/breadth/risk context is weakening.

### 3. Current-state validation
This is not a trading backtest. It checks whether the label at date T matches transparent objective states at date T:

- Objective Up State: `ret20 > 0`, `ret60 > 0`, above MA20/MA60, drawdown not deep.
- Objective Bad State: negative returns, deep drawdown, or below MA20/MA60.

Metrics include precision, recall, F1, balanced accuracy, MCC, Cohen's Kappa and current-state AUC.

### 4. Trend capture validation
This test is designed for visual regime review. It identifies objective up-trend and bad-trend episodes, then checks:

- whether the correct regime appears inside the objective trend,
- whether it appears early,
- how much of the trend is covered by the regime,
- and how clean the regime path is through max run-up/max drawdown and slope.

### 5. Future bad-state warning
This is the forecast layer. It tests whether the risk score at date T warns that the stock will enter **Bear/Divergence** within the next 20 sessions.

Metrics include ROC AUC, Average Precision, AP Lift, walk-forward by year and risk-decile calibration.
        """
    )

with tabs[6]:
    st.subheader(t["deploy"])
    st.markdown(
        """
### Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Deploy on Streamlit Community Cloud
1. Create a GitHub repository.
2. Upload this project folder: `app.py`, `requirements.txt`, `.streamlit/config.toml`, and `data/`.
3. Go to Streamlit Community Cloud.
4. Choose **New app**.
5. Select your repo, branch, and `app.py`.
6. Click **Deploy**.

### Notes
- Data is stored as CSV to avoid Excel loading time on every app start.
- Computations are cached with `st.cache_data`.
- The chart renders only one selected ticker at a time to keep the app responsive.
        """
    )
