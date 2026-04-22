"""
Motor de Valoración de Empresas
================================
DCF, ratios de calidad y múltiplos comparables en tiempo real.
Stack: Streamlit + yfinance + plotly + numpy

Para ejecutar:
    pip install streamlit yfinance plotly numpy pandas
    streamlit run valuation_app.py
"""

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import json
import traceback

# ─────────────────────────────────────────────
# CONFIG & THEME
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Motor de Valoración",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom dark theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@400;500;600;700&display=swap');

    .stApp {
        background-color: #0a0f0d;
        font-family: 'DM Sans', sans-serif;
    }

    .main-title {
        font-family: 'DM Sans', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        color: #e8ede9;
        text-align: center;
        margin-bottom: 0;
    }

    .main-title span {
        color: #34d399;
    }

    .subtitle {
        text-align: center;
        color: #6b7c71;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #111916 0%, #0d1410 100%);
        border: 1px solid #1a2f22;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }

    .metric-label {
        color: #6b7c71;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
    }

    .metric-value {
        color: #e8ede9;
        font-size: 1.5rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }

    .metric-value.green { color: #34d399; }
    .metric-value.red { color: #f87171; }
    .metric-value.amber { color: #fbbf24; }

    .section-header {
        color: #e8ede9;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #1a2f22;
    }

    .log-entry {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        padding: 0.25rem 0;
        color: #8fa897;
    }

    .log-entry.success { color: #34d399; }
    .log-entry.warn { color: #fbbf24; }
    .log-entry.error { color: #f87171; }
    .log-entry.data { color: #60a5fa; }

    .badge {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        font-size: 0.7rem;
        font-weight: 600;
    }

    .badge-green { background: #065f46; color: #34d399; }
    .badge-red { background: #7f1d1d; color: #f87171; }
    .badge-amber { background: #78350f; color: #fbbf24; }

    .audit-valid { color: #34d399; font-weight: 600; }
    .audit-invalid { color: #f87171; font-weight: 600; }

    div[data-testid="stExpander"] {
        background: #111916;
        border: 1px solid #1a2f22;
        border-radius: 12px;
    }

    .stSlider > div > div { color: #34d399; }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #111916 0%, #0d1410 100%);
        border: 1px solid #1a2f22;
        border-radius: 12px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOGGING SYSTEM
# ─────────────────────────────────────────────
def init_logs():
    if "logs" not in st.session_state:
        st.session_state.logs = []

def log(msg, level="info"):
    """Add a log entry: levels = info, success, warn, error, data, calc"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append({
        "time": timestamp,
        "msg": msg,
        "level": level,
    })

LEVEL_ICONS = {
    "info": "◆",
    "success": "✓",
    "warn": "⚠",
    "error": "✗",
    "data": "→",
    "calc": "∑",
}


# ─────────────────────────────────────────────
# DATA LOADER (yfinance — datos reales de API)
# ─────────────────────────────────────────────
@st.cache_data(ttl=900, show_spinner=False)
def load_financial_data(ticker: str):
    """
    Carga datos financieros directamente de Yahoo Finance API.
    Devuelve datos estructurados + logs de trazabilidad.
    """
    logs = []
    warnings = []
    estimated_fields = []

    logs.append(("info", f"Iniciando análisis para {ticker.upper()}"))
    logs.append(("info", "Conectando a Yahoo Finance API (yfinance)..."))

    try:
        stock = yf.Ticker(ticker)

        # ── Intentar obtener info con diagnóstico ──
        try:
            info = stock.info
        except Exception as e_info:
            logs.append(("error", f"Error obteniendo info básica: {str(e_info)}"))
            info = {}

        # ── Validar que tenemos datos ──
        # yfinance a veces devuelve dict vacío o sin precio
        price_keys = ["currentPrice", "regularMarketPrice", "previousClose", "open"]
        price = None
        for pk in price_keys:
            if info.get(pk) is not None:
                price = info[pk]
                break

        if price is None:
            # Intentar obtener precio del historial como fallback
            logs.append(("warn", "No se encontró precio en info, intentando historial..."))
            try:
                hist = stock.history(period="5d")
                if not hist.empty:
                    price = float(hist["Close"].iloc[-1])
                    logs.append(("success", f"Precio obtenido del historial: {price}"))
                else:
                    logs.append(("error", f"No se encontraron datos para '{ticker}'. Posibles causas:"))
                    logs.append(("error", "  1. El ticker no existe o está mal escrito"))
                    logs.append(("error", "  2. Yahoo Finance está bloqueando las peticiones"))
                    logs.append(("error", "  3. Problema de conexión de red"))
                    logs.append(("error", f"  Intenta: pip install --upgrade yfinance"))
                    return None, logs, warnings, estimated_fields
            except Exception as e_hist:
                logs.append(("error", f"Error también en historial: {str(e_hist)}"))
                logs.append(("error", "Yahoo Finance puede estar bloqueando la IP del servidor."))
                logs.append(("error", "Si estás en Streamlit Cloud, prueba a ejecutar localmente."))
                return None, logs, warnings, estimated_fields

        if not info or len(info) < 5:
            # Tenemos precio del historial pero no info completa
            logs.append(("warn", "Info básica limitada — intentando cargar estados financieros directamente..."))

        # ── Precio y datos básicos ──
        price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
        market_cap = info.get("marketCap", 0)
        enterprise_value = info.get("enterpriseValue", 0)
        beta = info.get("beta")
        shares_outstanding = info.get("sharesOutstanding", 0)
        currency = info.get("financialCurrency", "USD")
        sector = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")
        company_name = info.get("longName") or info.get("shortName", ticker.upper())

        logs.append(("success", f"Datos básicos recibidos: {company_name}"))
        logs.append(("data", f"Precio: {price} {currency} | Market Cap: {market_cap/1e6:,.0f}M | Beta: {beta}"))
        logs.append(("data", f"Divisa de reporte: {currency}"))

        # ── Beta validation ──
        if beta is None or beta == 0:
            beta = 1.0
            estimated_fields.append(("Beta", 1.0, "No disponible en Yahoo Finance"))
            logs.append(("warn", "Beta no encontrado → usando default conservador: 1.0"))
        elif beta == 1.0:
            logs.append(("warn", f"Beta = 1.0 exacto — podría ser un default, verificar manualmente"))

        # ── Financial Statements ──
        income_stmt = stock.financials  # anual
        balance_sheet = stock.balance_sheet
        cashflow = stock.cashflow

        # Quarterly for TTM
        income_q = stock.quarterly_financials
        cashflow_q = stock.quarterly_cashflow

        logs.append(("success", f"Estados financieros recibidos: {len(income_stmt.columns)} años"))

        # ── Extraer datos anuales (últimos 5 años) ──
        years_data = []
        columns = sorted(income_stmt.columns, reverse=False)[-5:]  # últimos 5

        for col in columns:
            year = col.year
            revenue = _safe_get(income_stmt, col, ["Total Revenue", "Revenue"])
            net_income = _safe_get(income_stmt, col, ["Net Income", "Net Income Common Stockholders"])
            operating_income = _safe_get(income_stmt, col, ["Operating Income", "EBIT"])
            ebitda = _safe_get(income_stmt, col, ["EBITDA", "Normalized EBITDA"])

            total_assets = _safe_get(balance_sheet, col, ["Total Assets"])
            total_liab = _safe_get(balance_sheet, col, ["Total Liabilities Net Minority Interest", "Total Liab"])
            total_equity = _safe_get(balance_sheet, col, ["Total Stockholders Equity", "Stockholders Equity", "Total Equity Gross Minority Interest"])
            total_debt = _safe_get(balance_sheet, col, ["Total Debt", "Long Term Debt"])
            cash = _safe_get(balance_sheet, col, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"])
            invested_capital = _safe_get(balance_sheet, col, ["Invested Capital"])

            fcf = _safe_get(cashflow, col, ["Free Cash Flow"])
            if fcf is None:
                op_cf = _safe_get(cashflow, col, ["Operating Cash Flow", "Total Cash From Operating Activities"])
                capex = _safe_get(cashflow, col, ["Capital Expenditure", "Capital Expenditures"])
                if op_cf is not None and capex is not None:
                    fcf = op_cf + capex  # capex is negative in yfinance

            years_data.append({
                "year": year,
                "revenue": revenue,
                "net_income": net_income,
                "operating_income": operating_income,
                "ebitda": ebitda,
                "total_assets": total_assets,
                "total_liabilities": total_liab,
                "total_equity": total_equity,
                "total_debt": total_debt,
                "cash": cash,
                "invested_capital": invested_capital,
                "fcf": fcf,
            })

            logs.append(("data", f"Año {year}: Revenue={_fmt(revenue)} | NI={_fmt(net_income)} | FCF={_fmt(fcf)}"))

        # ── TTM (trailing twelve months) ──
        ttm_revenue = _ttm_sum(income_q, ["Total Revenue", "Revenue"])
        ttm_net_income = _ttm_sum(income_q, ["Net Income", "Net Income Common Stockholders"])
        ttm_fcf = _ttm_sum(cashflow_q, ["Free Cash Flow"])
        if ttm_fcf is None:
            ttm_opcf = _ttm_sum(cashflow_q, ["Operating Cash Flow", "Total Cash From Operating Activities"])
            ttm_capex = _ttm_sum(cashflow_q, ["Capital Expenditure", "Capital Expenditures"])
            if ttm_opcf is not None and ttm_capex is not None:
                ttm_fcf = ttm_opcf + ttm_capex

        logs.append(("data", f"TTM: Revenue={_fmt(ttm_revenue)} | NI={_fmt(ttm_net_income)} | FCF={_fmt(ttm_fcf)}"))

        # ── Latest balance sheet data for DCF ──
        latest_bs_col = sorted(balance_sheet.columns, reverse=False)[-1]
        latest_debt = _safe_get(balance_sheet, latest_bs_col, ["Total Debt", "Long Term Debt"]) or 0
        latest_cash = _safe_get(balance_sheet, latest_bs_col, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"]) or 0
        net_debt = latest_debt - latest_cash

        logs.append(("calc", f"Deuda Neta = Deuda {latest_debt/1e6:,.0f}M − Caja {latest_cash/1e6:,.0f}M = {net_debt/1e6:,.0f}M"))

        # ── WACC components ──
        risk_free_rate = 0.043  # 10Y US Treasury approx
        equity_risk_premium = 0.055
        cost_of_debt = info.get("debtToEquity")
        if cost_of_debt and cost_of_debt > 0:
            # Estimate cost of debt from interest expense
            interest_exp = _safe_get(income_stmt, columns[-1], ["Interest Expense"])
            if interest_exp and latest_debt > 0:
                cost_of_debt = abs(interest_exp) / latest_debt
                logs.append(("calc", f"Coste de deuda estimado: {cost_of_debt:.1%} (Intereses/Deuda)"))
            else:
                cost_of_debt = 0.05
                estimated_fields.append(("Coste de deuda", "5.0%", "No se pudo calcular"))
        else:
            cost_of_debt = 0.05
            estimated_fields.append(("Coste de deuda", "5.0%", "No disponible"))

        tax_rate = info.get("effectiveTaxRate") or _estimate_tax_rate(income_stmt, columns[-1])
        if tax_rate is None:
            tax_rate = 0.25
            estimated_fields.append(("Tasa impositiva", "25%", "Default conservador"))
        logs.append(("data", f"Tasa impositiva efectiva: {tax_rate:.1%}"))

        # ── Sector multiples (from info) ──
        pe_ratio = info.get("trailingPE") or info.get("forwardPE")
        ev_ebitda = info.get("enterpriseToEbitda")
        ps_ratio = info.get("priceToSalesTrailing12Months")

        # Sector averages (estimates by sector)
        sector_multiples = _get_sector_multiples(sector)

        # ── Outlier detection ──
        outliers = _detect_outliers(years_data)
        for o in outliers:
            warnings.append(o)
            logs.append(("warn", f"Variación inusual: {o['metric']} entre {o['year_from']}→{o['year_to']} ({o['change']:+.0%})"))

        # ── Balance audit ──
        audit_results = _audit_balance(years_data)
        for a in audit_results:
            if a["discrepancy"] > 0.01:
                logs.append(("warn", f"Discrepancia balance {a['year']}: {a['discrepancy']:.2%}"))
            else:
                logs.append(("success", f"Balance {a['year']}: OK ({a['discrepancy']:.4%})"))

        logs.append(("success", "Análisis completado. Datos listos para el motor DCF."))

        result = {
            "ticker": ticker.upper(),
            "company_name": company_name,
            "sector": sector,
            "industry": industry,
            "currency": currency,
            "price": price,
            "market_cap": market_cap,
            "enterprise_value": enterprise_value,
            "beta": beta,
            "shares_outstanding": shares_outstanding,
            "years_data": years_data,
            "ttm_revenue": ttm_revenue,
            "ttm_net_income": ttm_net_income,
            "ttm_fcf": ttm_fcf,
            "total_debt": latest_debt,
            "cash": latest_cash,
            "net_debt": net_debt,
            "risk_free_rate": risk_free_rate,
            "equity_risk_premium": equity_risk_premium,
            "cost_of_debt": cost_of_debt,
            "tax_rate": tax_rate,
            "pe_ratio": pe_ratio,
            "ev_ebitda": ev_ebitda,
            "ps_ratio": ps_ratio,
            "sector_multiples": sector_multiples,
            "audit": audit_results,
        }

        return result, logs, warnings, estimated_fields

    except Exception as e:
        tb = traceback.format_exc()
        logs.append(("error", f"Error cargando datos: {str(e)}"))
        logs.append(("error", f"Traceback:\n{tb}"))
        return None, logs, warnings, estimated_fields


def _safe_get(df, col, field_names):
    """Safely get a value from a DataFrame trying multiple field names."""
    if df is None or df.empty:
        return None
    for name in field_names:
        if name in df.index:
            val = df.loc[name, col]
            if pd.notna(val):
                return float(val)
    return None


def _ttm_sum(df, field_names):
    """Sum last 4 quarters for TTM."""
    if df is None or df.empty:
        return None
    for name in field_names:
        if name in df.index:
            vals = df.loc[name].head(4).dropna()
            if len(vals) >= 4:
                return float(vals.sum())
    return None


def _estimate_tax_rate(income_stmt, col):
    """Estimate effective tax rate from income statement."""
    pretax = _safe_get(income_stmt, col, ["Pretax Income", "Income Before Tax"])
    tax = _safe_get(income_stmt, col, ["Tax Provision", "Income Tax Expense"])
    if pretax and tax and pretax != 0:
        return abs(tax / pretax)
    return None


def _fmt(val):
    """Format a number for logging."""
    if val is None:
        return "N/A"
    if abs(val) >= 1e9:
        return f"${val/1e9:,.1f}B"
    if abs(val) >= 1e6:
        return f"${val/1e6:,.0f}M"
    return f"${val:,.0f}"


def _detect_outliers(years_data):
    """Detect unusual variations in margins and FCF growth."""
    outliers = []
    for i in range(1, len(years_data)):
        curr = years_data[i]
        prev = years_data[i - 1]

        # Net margin variation
        if curr["revenue"] and prev["revenue"] and curr["net_income"] and prev["net_income"]:
            margin_curr = curr["net_income"] / curr["revenue"]
            margin_prev = prev["net_income"] / prev["revenue"]
            if margin_prev != 0:
                change = (margin_curr - margin_prev) / abs(margin_prev)
                if abs(change) > 0.5:
                    outliers.append({
                        "metric": "Margen de Beneficio Neto",
                        "year_from": prev["year"],
                        "year_to": curr["year"],
                        "val_from": margin_prev,
                        "val_to": margin_curr,
                        "change": change,
                    })

        # FCF growth variation
        if curr["fcf"] and prev["fcf"] and prev["fcf"] != 0:
            fcf_change = (curr["fcf"] - prev["fcf"]) / abs(prev["fcf"])
            if abs(fcf_change) > 0.5:
                outliers.append({
                    "metric": "Crecimiento FCF",
                    "year_from": prev["year"],
                    "year_to": curr["year"],
                    "val_from": prev["fcf"],
                    "val_to": curr["fcf"],
                    "change": fcf_change,
                })

    return outliers


def _audit_balance(years_data):
    """Validate Assets = Liabilities + Equity for each year."""
    results = []
    for yd in years_data:
        assets = yd.get("total_assets")
        liab = yd.get("total_liabilities")
        equity = yd.get("total_equity")
        if assets and liab is not None and equity is not None:
            lp = liab + equity
            disc = abs(assets - lp) / assets if assets != 0 else 0
            results.append({
                "year": yd["year"],
                "assets": assets,
                "liab_plus_equity": lp,
                "discrepancy": disc,
                "valid": disc <= 0.01,
            })
    return results


def _get_sector_multiples(sector):
    """Return approximate sector average multiples."""
    defaults = {
        "Technology": {"pe": 30, "ev_ebitda": 22, "ps": 7},
        "Information Technology": {"pe": 30, "ev_ebitda": 22, "ps": 7},
        "Healthcare": {"pe": 22, "ev_ebitda": 15, "ps": 4},
        "Financial Services": {"pe": 14, "ev_ebitda": 10, "ps": 3},
        "Financials": {"pe": 14, "ev_ebitda": 10, "ps": 3},
        "Consumer Cyclical": {"pe": 20, "ev_ebitda": 13, "ps": 2},
        "Consumer Discretionary": {"pe": 20, "ev_ebitda": 13, "ps": 2},
        "Industrials": {"pe": 22, "ev_ebitda": 14, "ps": 2.5},
        "Energy": {"pe": 12, "ev_ebitda": 7, "ps": 1.5},
        "Communication Services": {"pe": 18, "ev_ebitda": 12, "ps": 3},
        "Consumer Staples": {"pe": 24, "ev_ebitda": 16, "ps": 2},
        "Consumer Defensive": {"pe": 24, "ev_ebitda": 16, "ps": 2},
        "Utilities": {"pe": 18, "ev_ebitda": 12, "ps": 2},
        "Real Estate": {"pe": 35, "ev_ebitda": 20, "ps": 8},
        "Basic Materials": {"pe": 15, "ev_ebitda": 9, "ps": 2},
    }
    return defaults.get(sector, {"pe": 20, "ev_ebitda": 14, "ps": 3})


# ─────────────────────────────────────────────
# FINANCIAL ENGINE (cálculos puros en Python)
# ─────────────────────────────────────────────
def calculate_wacc(beta, risk_free_rate, equity_risk_premium, cost_of_debt, tax_rate,
                   market_cap, total_debt):
    """Calculate Weighted Average Cost of Capital."""
    cost_of_equity = risk_free_rate + beta * equity_risk_premium
    total_capital = market_cap + total_debt
    if total_capital == 0:
        return cost_of_equity

    weight_equity = market_cap / total_capital
    weight_debt = total_debt / total_capital
    wacc = (weight_equity * cost_of_equity) + (weight_debt * cost_of_debt * (1 - tax_rate))
    return wacc


def run_dcf(fcf_base, growth_rate, wacc, terminal_growth, total_debt, cash,
            shares_outstanding, projection_years=5):
    """
    Discounted Cash Flow valuation.
    Returns intrinsic value per share and breakdown.
    """
    if fcf_base is None or fcf_base <= 0 or shares_outstanding <= 0:
        return None

    # Project FCFs
    projected_fcfs = []
    pv_fcfs = []
    for year in range(1, projection_years + 1):
        fcf = fcf_base * (1 + growth_rate) ** year
        pv = fcf / (1 + wacc) ** year
        projected_fcfs.append(fcf)
        pv_fcfs.append(pv)

    pv_fcfs_total = sum(pv_fcfs)

    # Terminal Value (Gordon Growth Model)
    terminal_fcf = projected_fcfs[-1] * (1 + terminal_growth)
    terminal_value = terminal_fcf / (wacc - terminal_growth)
    pv_terminal = terminal_value / (1 + wacc) ** projection_years

    # Enterprise Value
    enterprise_value = pv_fcfs_total + pv_terminal

    # Equity Value = EV - Net Debt
    net_debt = total_debt - cash
    equity_value = enterprise_value - net_debt

    # Price per share
    intrinsic_value = equity_value / shares_outstanding

    return {
        "projected_fcfs": projected_fcfs,
        "pv_fcfs": pv_fcfs,
        "pv_fcfs_total": pv_fcfs_total,
        "terminal_value": terminal_value,
        "pv_terminal": pv_terminal,
        "enterprise_value": enterprise_value,
        "net_debt": net_debt,
        "equity_value": equity_value,
        "intrinsic_value": max(intrinsic_value, 0),
    }


def calculate_ratios(data):
    """Calculate quality ratios from the latest year data."""
    latest = data["years_data"][-1] if data["years_data"] else {}
    ratios = {}

    # ROIC
    if latest.get("operating_income") and latest.get("invested_capital"):
        nopat = latest["operating_income"] * (1 - data["tax_rate"])
        ratios["roic"] = nopat / latest["invested_capital"] if latest["invested_capital"] != 0 else None
    elif latest.get("operating_income") and latest.get("total_assets"):
        nopat = latest["operating_income"] * (1 - data["tax_rate"])
        ratios["roic"] = nopat / latest["total_assets"]
    else:
        ratios["roic"] = None

    # Operating margin
    if latest.get("operating_income") and latest.get("revenue"):
        ratios["operating_margin"] = latest["operating_income"] / latest["revenue"]
    else:
        ratios["operating_margin"] = None

    # Solvency ratio (equity / assets)
    if latest.get("total_equity") and latest.get("total_assets"):
        ratios["solvency"] = latest["total_equity"] / latest["total_assets"]
    else:
        ratios["solvency"] = None

    return ratios


def rate_metric(value, thresholds):
    """Rate a metric: (bad_threshold, good_threshold) → label + color."""
    if value is None:
        return "N/A", "amber"
    bad, good = thresholds
    if value >= good:
        return "Excelente", "green"
    elif value >= bad:
        return "Aceptable", "amber"
    else:
        return "Bajo", "red"


# ─────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────
PLOT_BG = "rgba(0,0,0,0)"
PAPER_BG = "#111916"
GRID_COLOR = "#1a2f22"
TEXT_COLOR = "#8fa897"
GREEN = "#34d399"
BLUE = "#60a5fa"
EMERALD_DARK = "#065f46"
BLUE_DARK = "#1e3a5f"


def make_revenue_chart(years_data):
    """Bar chart: Revenue vs Net Income over 5 years."""
    years = [d["year"] for d in years_data]
    revenues = [d["revenue"] / 1e9 if d["revenue"] else 0 for d in years_data]
    net_incomes = [d["net_income"] / 1e9 if d["net_income"] else 0 for d in years_data]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=years, y=revenues, name="Ingresos",
        marker_color=GREEN, marker_line_width=0,
    ))
    fig.add_trace(go.Bar(
        x=years, y=net_incomes, name="Beneficio Neto",
        marker_color=BLUE, marker_line_width=0,
    ))
    fig.update_layout(
        title=dict(text="Ingresos vs Beneficio Neto", font=dict(color="#e8ede9", size=16)),
        barmode="group",
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        font=dict(color=TEXT_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TEXT_COLOR)),
        yaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TEXT_COLOR),
                   tickprefix="$", ticksuffix="B"),
        legend=dict(font=dict(color=TEXT_COLOR)),
        margin=dict(l=60, r=20, t=50, b=40),
        height=350,
    )
    return fig


def make_dcf_chart(dcf_result):
    """Bar chart: Projected FCFs and Terminal Value PV."""
    if dcf_result is None:
        return None

    years = [f"Año {i+1}" for i in range(len(dcf_result["projected_fcfs"]))]
    projected = [f / 1e9 for f in dcf_result["projected_fcfs"]]
    pv = [f / 1e9 for f in dcf_result["pv_fcfs"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=years, y=projected, name="FCF Proyectado",
        marker_color=GREEN, marker_line_width=0,
    ))
    fig.add_trace(go.Bar(
        x=years, y=pv, name="Valor Presente",
        marker_color=EMERALD_DARK, marker_line_width=0,
    ))
    fig.update_layout(
        title=dict(text="Desglose del DCF — Proyección a 5 años",
                   font=dict(color="#e8ede9", size=16)),
        barmode="group",
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        font=dict(color=TEXT_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, tickprefix="$", ticksuffix="B"),
        legend=dict(font=dict(color=TEXT_COLOR)),
        margin=dict(l=60, r=20, t=50, b=40),
        height=350,
    )
    return fig


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
def main():
    init_logs()

    # ── Header ──
    st.markdown('<p class="main-title">Valoración de <span>Empresas</span></p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">DCF, ratios de calidad y múltiplos comparables en tiempo real · Datos de Yahoo Finance API</p>', unsafe_allow_html=True)

    # ── Ticker input ──
    col_input, col_btn = st.columns([4, 1])
    with col_input:
        ticker = st.text_input("Ticker", placeholder="Introduce un ticker (ej: AAPL)", label_visibility="collapsed")
    with col_btn:
        analyze = st.button("🔍 Analizar", use_container_width=True, type="primary")

    # Quick ticker buttons
    quick_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "SAP", "META"]
    cols = st.columns(len(quick_tickers))
    for i, t in enumerate(quick_tickers):
        with cols[i]:
            if st.button(t, key=f"quick_{t}", use_container_width=True):
                ticker = t
                analyze = True

    if not ticker or not analyze:
        st.info("Introduce un ticker y pulsa 'Analizar' para comenzar.")
        return

    # ── Load data ──
    with st.spinner(f"Cargando datos para {ticker.upper()} desde Yahoo Finance..."):
        data, data_logs, warnings, estimated_fields = load_financial_data(ticker)

    # Merge logs into session
    st.session_state.logs = [{"time": l[1] if isinstance(l, tuple) else "", "msg": l[1] if isinstance(l, tuple) else l, "level": l[0] if isinstance(l, tuple) else "info"} for l in data_logs]

    if data is None:
        st.error(f"No se pudieron cargar datos para '{ticker}'.")

        # Show logs so user can see what happened
        with st.expander(f"🖥 Logs del Agente · {len(data_logs)} eventos — Ver diagnóstico", expanded=True):
            for entry in data_logs:
                level, msg = entry
                icon = LEVEL_ICONS.get(level, "◆")
                css_class = level
                st.markdown(f'<div class="log-entry {css_class}">{icon} {msg}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🔧 Posibles soluciones")
        st.markdown("""
1. **Actualizar yfinance**: `pip install --upgrade yfinance`
2. **Verificar el ticker**: Asegúrate de que sea válido (ej: AAPL, MSFT, SAP)
3. **Ejecutar localmente**: Streamlit Cloud a veces bloquea peticiones a Yahoo Finance. Ejecuta `streamlit run valuation_app.py` en tu ordenador.
4. **Esperar y reintentar**: Yahoo Finance puede limitar peticiones temporalmente.
        """)
        st.caption(f"yfinance versión: {yf.__version__}")
        return

    # ── Agent Logs ──
    with st.expander(f"🖥 Logs del Agente · {len(data_logs)} eventos", expanded=False):
        for entry in data_logs:
            level, msg = entry
            icon = LEVEL_ICONS.get(level, "◆")
            css_class = level
            st.markdown(f'<div class="log-entry {css_class}">{icon} {msg}</div>', unsafe_allow_html=True)

    # ── Summary Header ──
    st.markdown(f'<p class="section-header">{data["company_name"]} · {data["ticker"]}</p>', unsafe_allow_html=True)
    st.caption(f'{data["sector"]} · {data["industry"]} · Divisa: {data["currency"]}')

    # ── WACC calculation ──
    base_wacc = calculate_wacc(
        data["beta"], data["risk_free_rate"], data["equity_risk_premium"],
        data["cost_of_debt"], data["tax_rate"], data["market_cap"], data["total_debt"]
    )

    # ── FCF growth estimation ──
    fcf_values = [d["fcf"] for d in data["years_data"] if d["fcf"] and d["fcf"] > 0]
    if len(fcf_values) >= 2:
        base_growth = (fcf_values[-1] / fcf_values[0]) ** (1 / (len(fcf_values) - 1)) - 1
        base_growth = max(min(base_growth, 0.40), -0.10)  # cap between -10% and 40%
    else:
        base_growth = 0.08

    # ── Sidebar: DCF Sliders ──
    st.sidebar.markdown("## ⚙️ Ajustes del Modelo DCF")

    # Scenario selector
    st.sidebar.markdown("### Escenario")
    scenario_cols = st.sidebar.columns(3)
    scenario = st.session_state.get("scenario", "base")

    with scenario_cols[0]:
        if st.button("🐂 Bull", use_container_width=True):
            scenario = "bull"
            st.session_state.scenario = "bull"
    with scenario_cols[1]:
        if st.button("— Base", use_container_width=True):
            scenario = "base"
            st.session_state.scenario = "base"
    with scenario_cols[2]:
        if st.button("🐻 Bear", use_container_width=True):
            scenario = "bear"
            st.session_state.scenario = "bear"

    # Apply scenario adjustments
    if scenario == "bull":
        adj_growth = base_growth * 1.20
        adj_wacc = base_wacc - 0.01
        st.sidebar.info("📈 Bull: FCF Growth +20%, WACC -1%")
    elif scenario == "bear":
        adj_growth = base_growth * 0.80
        adj_wacc = base_wacc + 0.02
        st.sidebar.warning("📉 Bear: FCF Growth -20%, WACC +2%")
    else:
        adj_growth = base_growth
        adj_wacc = base_wacc

    growth_rate = st.sidebar.slider(
        "Tasa de Crecimiento (FCF)",
        min_value=-0.10, max_value=0.40,
        value=float(round(adj_growth, 3)),
        step=0.005, format="%.1f%%",
    )
    wacc = st.sidebar.slider(
        "WACC (Tasa de Descuento)",
        min_value=0.04, max_value=0.20,
        value=float(round(adj_wacc, 3)),
        step=0.002, format="%.1f%%",
    )
    terminal_growth = st.sidebar.slider(
        "Crecimiento Terminal",
        min_value=0.01, max_value=0.04,
        value=0.025, step=0.001, format="%.1f%%",
    )

    # Validate WACC > terminal growth
    if wacc <= terminal_growth:
        st.sidebar.error("⚠️ WACC debe ser mayor que el crecimiento terminal")
        return

    # ── Run DCF ──
    fcf_base = data["ttm_fcf"] or (data["years_data"][-1]["fcf"] if data["years_data"] else None)
    dcf = run_dcf(
        fcf_base=fcf_base,
        growth_rate=growth_rate,
        wacc=wacc,
        terminal_growth=terminal_growth,
        total_debt=data["total_debt"],
        cash=data["cash"],
        shares_outstanding=data["shares_outstanding"],
    )

    # ── Price vs Intrinsic Value ──
    col_price, col_iv = st.columns(2)
    with col_price:
        st.metric("Precio Actual", f"${data['price']:,.2f}", delta=None)
    with col_iv:
        if dcf:
            iv = dcf["intrinsic_value"]
            upside = (iv - data["price"]) / data["price"]
            color = "green" if upside > 0 else "red"
            st.metric("Valor Intrínseco", f"${iv:,.2f}", delta=f"{upside:+.1%}")
        else:
            st.metric("Valor Intrínseco", "N/A")

    # ── Key metrics row ──
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Market Cap", f"${data['market_cap']/1e9:,.1f}B")
    with m2:
        st.metric("EV", f"${data['enterprise_value']/1e9:,.1f}B")
    with m3:
        st.metric("Beta", f"{data['beta']:.2f}")
    with m4:
        if fcf_base:
            st.metric("FCF (TTM)", f"${fcf_base/1e9:,.2f}B")
        else:
            st.metric("FCF (TTM)", "N/A")

    st.divider()

    # ── Charts ──
    chart_col, dcf_col = st.columns([3, 2])

    with chart_col:
        # Revenue chart
        fig_rev = make_revenue_chart(data["years_data"])
        st.plotly_chart(fig_rev, use_container_width=True)

        # DCF chart
        if dcf:
            fig_dcf = make_dcf_chart(dcf)
            if fig_dcf:
                st.plotly_chart(fig_dcf, use_container_width=True)

    with dcf_col:
        # DCF Breakdown
        if dcf:
            st.markdown('<p class="section-header">Desglose DCF</p>', unsafe_allow_html=True)
            st.metric("PV de FCFs", f"${dcf['pv_fcfs_total']/1e9:,.2f}B")
            st.metric("Valor Terminal", f"${dcf['terminal_value']/1e9:,.2f}B")
            st.metric("PV Terminal", f"${dcf['pv_terminal']/1e9:,.2f}B")
            st.metric("Enterprise Value", f"${dcf['enterprise_value']/1e9:,.2f}B")

            st.markdown("---")
            st.markdown("**Puente EV → Equity**")
            st.caption(f"Deuda Total: ${data['total_debt']/1e6:,.0f}M")
            st.caption(f"Caja: ${data['cash']/1e6:,.0f}M")
            st.caption(f"Deuda Neta: ${dcf['net_debt']/1e6:,.0f}M")
            st.metric("Equity Value", f"${dcf['equity_value']/1e9:,.2f}B")

        # Quality ratios
        st.markdown('<p class="section-header">Ratios de Calidad</p>', unsafe_allow_html=True)
        ratios = calculate_ratios(data)

        roic_label, roic_color = rate_metric(ratios["roic"], (0.08, 0.15))
        margin_label, margin_color = rate_metric(ratios["operating_margin"], (0.10, 0.20))
        solv_label, solv_color = rate_metric(ratios["solvency"], (0.20, 0.35))

        if ratios["roic"] is not None:
            st.metric("ROIC", f"{ratios['roic']:.1%}", delta=roic_label)
        if ratios["operating_margin"] is not None:
            st.metric("Margen Operativo", f"{ratios['operating_margin']:.1%}", delta=margin_label)
        if ratios["solvency"] is not None:
            st.metric("Ratio de Solvencia", f"{ratios['solvency']:.1%}", delta=solv_label)

    st.divider()

    # ── Relative Multiples ──
    st.markdown('<p class="section-header">Múltiplos Relativos</p>', unsafe_allow_html=True)
    sect = data["sector_multiples"]
    mc1, mc2, mc3 = st.columns(3)

    with mc1:
        if data["pe_ratio"]:
            diff = (data["pe_ratio"] - sect["pe"]) / sect["pe"]
            st.metric("P/E", f"{data['pe_ratio']:.1f}x",
                      delta=f"Sector: {sect['pe']:.0f}x ({diff:+.0%})")
        else:
            st.metric("P/E", "N/A")
    with mc2:
        if data["ev_ebitda"]:
            diff = (data["ev_ebitda"] - sect["ev_ebitda"]) / sect["ev_ebitda"]
            st.metric("EV/EBITDA", f"{data['ev_ebitda']:.1f}x",
                      delta=f"Sector: {sect['ev_ebitda']:.0f}x ({diff:+.0%})")
        else:
            st.metric("EV/EBITDA", "N/A")
    with mc3:
        if data["ps_ratio"]:
            diff = (data["ps_ratio"] - sect["ps"]) / sect["ps"]
            st.metric("P/S", f"{data['ps_ratio']:.1f}x",
                      delta=f"Sector: {sect['ps']:.0f}x ({diff:+.0%})")
        else:
            st.metric("P/S", "N/A")

    st.divider()

    # ── Audit Panel ──
    st.markdown('<p class="section-header">Panel de Auditoría</p>', unsafe_allow_html=True)

    # Balance validation
    if data["audit"]:
        audit_df = pd.DataFrame(data["audit"])
        audit_df["assets_fmt"] = audit_df["assets"].apply(lambda x: f"${x/1e9:,.2f}B")
        audit_df["lpe_fmt"] = audit_df["liab_plus_equity"].apply(lambda x: f"${x/1e9:,.2f}B")
        audit_df["disc_fmt"] = audit_df["discrepancy"].apply(lambda x: f"{x:.2%}")
        audit_df["status"] = audit_df["valid"].apply(lambda x: "✅ Válido" if x else "⚠️ Revisar")

        st.dataframe(
            audit_df[["year", "assets_fmt", "lpe_fmt", "disc_fmt", "status"]].rename(columns={
                "year": "Año",
                "assets_fmt": "Activos",
                "lpe_fmt": "Pasivos + Patrimonio",
                "disc_fmt": "Discrepancia",
                "status": "Estado",
            }),
            use_container_width=True,
            hide_index=True,
        )

    # Estimated fields
    if estimated_fields:
        st.markdown("**⚠️ Datos estimados por defecto:**")
        for field, val, reason in estimated_fields:
            st.warning(f"**{field}**: {val} — {reason}")

    # Outliers
    if warnings:
        st.markdown(f"**🔺 Variaciones Inusuales Detectadas** ({len(warnings)} avisos)")
        for w in warnings:
            st.error(
                f"**{w['metric']}** entre {w['year_from']} y {w['year_to']}: "
                f"{w['val_from']:.1%} → {w['val_to']:.1%} ({w['change']:+.0%})"
                if "margin" in w["metric"].lower() or "margen" in w["metric"].lower()
                else f"**{w['metric']}** entre {w['year_from']} y {w['year_to']}: ({w['change']:+.0%})"
            )

    # Data source
    st.caption("📊 Datos obtenidos de **Yahoo Finance API** (yfinance) · Datos estructurados via API, no extraídos por LLM")


if __name__ == "__main__":
    main()
