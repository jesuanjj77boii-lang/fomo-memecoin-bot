import math
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

DB_PATH = os.getenv("FOMO_DB_PATH", "fomo_paper.db")
API_URL = "https://api.coingecko.com/api/v3"

DEFAULT_COINS = [
    ("dogecoin", "DOGE"),
    ("shiba-inu", "SHIB"),
    ("pepe", "PEPE"),
    ("bonk", "BONK"),
    ("dogwifcoin", "WIF"),
    ("floki", "FLOKI"),
]

@dataclass
class Settings:
    starting_cash: float = 1000.0
    risk_per_trade: float = 0.05
    max_position_pct: float = 0.20
    entry_score: float = 68.0
    take_profit: float = 0.12
    stop_loss: float = 0.07
    max_hold_minutes: int = 180
    trailing_stop: float = 0.05


def db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("CREATE TABLE IF NOT EXISTS account (id INTEGER PRIMARY KEY, cash REAL, realized_pnl REAL, updated_at TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS positions (coin_id TEXT PRIMARY KEY, symbol TEXT, qty REAL, entry_price REAL, entry_time TEXT, highest_price REAL, score REAL)")
    conn.execute("CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY AUTOINCREMENT, coin_id TEXT, symbol TEXT, side TEXT, qty REAL, price REAL, pnl REAL, score REAL, reason TEXT, timestamp TEXT)")
    row = conn.execute("SELECT cash FROM account WHERE id=1").fetchone()
    if row is None:
        conn.execute("INSERT INTO account(id,cash,realized_pnl,updated_at) VALUES(1,?,?,?)", (1000.0, 0.0, datetime.now(timezone.utc).isoformat()))
        conn.commit()
    return conn


def reset_db(starting_cash: float):
    conn = db()
    conn.execute("DELETE FROM positions")
    conn.execute("DELETE FROM trades")
    conn.execute("DELETE FROM account")
    conn.execute("INSERT INTO account(id,cash,realized_pnl,updated_at) VALUES(1,?,?,?)", (starting_cash, 0.0, datetime.now(timezone.utc).isoformat()))
    conn.commit()


def get_account(conn):
    row = conn.execute("SELECT cash, realized_pnl FROM account WHERE id=1").fetchone()
    return {"cash": row[0], "realized_pnl": row[1]}


def update_cash(conn, cash, realized_pnl):
    conn.execute("UPDATE account SET cash=?, realized_pnl=?, updated_at=? WHERE id=1", (cash, realized_pnl, datetime.now(timezone.utc).isoformat()))
    conn.commit()


def get_positions(conn) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM positions", conn)


def get_trades(conn) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM trades ORDER BY id DESC", conn)


def upsert_position(conn, coin_id, symbol, qty, entry_price, entry_time, highest_price, score):
    conn.execute("""INSERT INTO positions VALUES(?,?,?,?,?,?,?)
                    ON CONFLICT(coin_id) DO UPDATE SET qty=excluded.qty, entry_price=excluded.entry_price,
                    entry_time=excluded.entry_time, highest_price=excluded.highest_price, score=excluded.score""",
                 (coin_id, symbol, qty, entry_price, entry_time, highest_price, score))
    conn.commit()


def delete_position(conn, coin_id):
    conn.execute("DELETE FROM positions WHERE coin_id=?", (coin_id,))
    conn.commit()


def add_trade(conn, coin_id, symbol, side, qty, price, pnl, score, reason):
    conn.execute("INSERT INTO trades(coin_id,symbol,side,qty,price,pnl,score,reason,timestamp) VALUES(?,?,?,?,?,?,?,?,?)",
                 (coin_id, symbol, side, qty, price, pnl, score, reason, datetime.now(timezone.utc).isoformat()))
    conn.commit()


def fetch_markets() -> pd.DataFrame:
    ids = ",".join(x[0] for x in DEFAULT_COINS)
    try:
        r = requests.get(
            f"{API_URL}/coins/markets",
            params={"vs_currency": "usd", "ids": ids, "order": "market_cap_desc", "per_page": 100, "page": 1, "sparkline": "true", "price_change_percentage": "1h,24h,7d"},
            timeout=12,
            headers={"accept": "application/json", "user-agent": "FOMO-Paper-Bot/1.0"},
        )
        r.raise_for_status()
        data = r.json()
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame()


def synthetic_series(price: float, chg24: float, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = 48
    target = price / max(1e-9, (1 + chg24 / 100.0))
    base = np.linspace(target, price, n)
    noise = rng.normal(0, max(price * 0.004, 1e-12), n)
    s = np.maximum(base + np.cumsum(noise), 1e-12)
    s[-1] = price
    return s


def score_coin(row: pd.Series) -> Tuple[float, Dict[str, float]]:
    p1 = float(row.get("price_change_percentage_1h_in_currency", row.get("price_change_percentage_1h", 0)) or 0)
    p24 = float(row.get("price_change_percentage_24h_in_currency", row.get("price_change_percentage_24h", 0)) or 0)
    p7 = float(row.get("price_change_percentage_7d_in_currency", row.get("price_change_percentage_7d", 0)) or 0)
    vol = float(row.get("total_volume", 0) or 0)
    mc = float(row.get("market_cap", 0) or 0)
    cap_ratio = math.log10(max(vol, 1) / max(mc, 1))

    momentum = np.clip((p1 + 0.25 * p24 + 0.05 * p7) * 4.0 + 40, 0, 100)
    volume = np.clip(55 + cap_ratio * 18, 0, 100)
    acceleration = np.clip(50 + (p1 - p24 / 24.0) * 8, 0, 100)
    trend = np.clip(55 + p24 * 1.5 + p7 * 0.3, 0, 100)
    liquidity = np.clip(20 + math.log10(max(vol, 1)) * 8, 0, 100)

    raw = 0.30 * momentum + 0.22 * volume + 0.20 * acceleration + 0.18 * trend + 0.10 * liquidity
    # Prevent the score from rewarding extreme negative momentum.
    if p1 < -2:
        raw -= min(25, abs(p1) * 2.5)
    raw = float(np.clip(raw, 0, 100))
    return raw, {"Momentum": momentum, "Volume/Liquidity": volume, "Acceleration": acceleration, "Trend": trend, "Liquidity": liquidity}


def prepare_market(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rows = []
    for _, r in df.iterrows():
        s, parts = score_coin(r)
        price = float(r.get("current_price", 0) or 0)
        chg = float(r.get("price_change_percentage_24h", 0) or 0)
        series = synthetic_series(price, chg, seed=hash(r.get("id", "x")) % 10000 + 1)
        ret = pd.Series(series).pct_change().dropna()
        vol = float(ret.std() * math.sqrt(48))
        rr = dict(r)
        rr.update({"fomo_score": s, "volatility": vol, "score_label": "AVOID" if s < 40 else "WATCH" if s < 60 else "STRONG SETUP" if s < 80 else "EXTREME FOMO", "series": series, **{f"score_{k}": v for k, v in parts.items()}})
        rows.append(rr)
    out = pd.DataFrame(rows)
    return out.sort_values("fomo_score", ascending=False)


def mark_to_market(positions: pd.DataFrame, market: pd.DataFrame):
    if positions.empty:
        return 0.0, positions.copy()
    m = market.set_index("id")
    pos = positions.copy()
    values = []
    unreal = []
    for _, p in pos.iterrows():
        price = float(m.loc[p.coin_id, "current_price"]) if p.coin_id in m.index else p.entry_price
        values.append(p.qty * price)
        unreal.append(p.qty * (price - p.entry_price))
    pos["market_value"] = values
    pos["unrealized_pnl"] = unreal
    return float(pos.market_value.sum()), pos


def simulate_once(conn, market: pd.DataFrame, settings: Settings):
    acct = get_account(conn)
    positions = get_positions(conn)
    now = datetime.now(timezone.utc)
    pos_by_id = {r.coin_id: r for _, r in positions.iterrows()}

    # Manage existing positions first.
    for coin_id, p in list(pos_by_id.items()):
        rows = market[market.id == coin_id]
        if rows.empty:
            continue
        row = rows.iloc[0]
        price = float(row.current_price)
        highest = max(float(p.highest_price), price)
        entry = float(p.entry_price)
        age_min = (now - datetime.fromisoformat(p.entry_time)).total_seconds() / 60.0
        pnl_pct = price / entry - 1.0
        trail_hit = highest > entry * (1 + settings.take_profit * 0.5) and price <= highest * (1 - settings.trailing_stop)
        reason = None
        if pnl_pct >= settings.take_profit:
            reason = "take_profit"
        elif pnl_pct <= -settings.stop_loss:
            reason = "stop_loss"
        elif age_min >= settings.max_hold_minutes:
            reason = "max_hold"
        elif trail_hit:
            reason = "trailing_stop"
        elif float(row.fomo_score) < settings.entry_score - 15:
            reason = "signal_cooldown"
        if reason:
            gross = float(p.qty) * price
            cost = float(p.qty) * entry
            pnl = gross - cost
            acct["cash"] += gross
            acct["realized_pnl"] += pnl
            add_trade(conn, coin_id, p.symbol, "SELL", float(p.qty), price, pnl, float(row.fomo_score), reason)
            delete_position(conn, coin_id)
        else:
            upsert_position(conn, coin_id, p.symbol, float(p.qty), entry, p.entry_time, highest, float(row.fomo_score))

    # One new entry per scan, using the strongest candidate.
    positions = get_positions(conn)
    held = set(positions.coin_id.tolist()) if not positions.empty else set()
    candidates = market[(market.fomo_score >= settings.entry_score) & (~market.id.isin(held))]
    acct = get_account(conn)
    if not candidates.empty and acct["cash"] > 5:
        row = candidates.iloc[0]
        risk_budget = acct["cash"] * settings.risk_per_trade
        max_pos = acct["cash"] * settings.max_position_pct
        notional = min(risk_budget, max_pos)
        if notional >= 5 and float(row.current_price) > 0:
            qty = notional / float(row.current_price)
            acct["cash"] -= notional
            upsert_position(conn, row.id, row.symbol.upper(), qty, float(row.current_price), now.isoformat(), float(row.current_price), float(row.fomo_score))
            add_trade(conn, row.id, row.symbol.upper(), "BUY", qty, float(row.current_price), 0.0, float(row.fomo_score), "fomo_entry")

    update_cash(conn, acct["cash"], acct["realized_pnl"])


def pnl_color(v):
    return f"${v:,.2f}"


st.set_page_config(page_title="FOMO Memecoin Paper Bot", page_icon="🚀", layout="wide")

conn = db()
if "settings" not in st.session_state:
    st.session_state.settings = Settings()
settings = st.session_state.settings

st.title("🚀 FOMO Memecoin Paper-Trading Bot")
st.caption("Simulation only — no exchange keys, orders, or real-money trading are used.")

with st.sidebar:
    st.header("Bot settings")
    settings.starting_cash = st.number_input("Starting cash", min_value=100.0, value=float(settings.starting_cash), step=100.0)
    settings.entry_score = st.slider("Entry FOMO score", 40.0, 90.0, float(settings.entry_score), 1.0)
    settings.risk_per_trade = st.slider("Risk budget / trade", 0.01, 0.20, float(settings.risk_per_trade), 0.01)
    settings.max_position_pct = st.slider("Max position %", 0.05, 0.50, float(settings.max_position_pct), 0.01)
    settings.take_profit = st.slider("Take profit", 0.03, 0.50, float(settings.take_profit), 0.01)
    settings.stop_loss = st.slider("Stop loss", 0.02, 0.30, float(settings.stop_loss), 0.01)
    settings.max_hold_minutes = st.number_input("Max hold (minutes)", min_value=15, max_value=1440, value=int(settings.max_hold_minutes), step=15)
    settings.trailing_stop = st.slider("Trailing stop", 0.02, 0.20, float(settings.trailing_stop), 0.01)
    auto = st.toggle("Auto-scan every refresh", value=False)
    refresh = st.button("🔄 Refresh market")
    reset = st.button("🧹 Reset paper account")
    st.divider()
    st.write("Data source: CoinGecko public API")
    st.write("Strategy: momentum + volume/liquidity + acceleration + trend")

if reset:
    reset_db(settings.starting_cash)
    st.rerun()

market_raw = fetch_markets()
market = prepare_market(market_raw)

if market.empty:
    st.error("Could not reach the public market-data API right now. The app is still safe to run, but it needs live data to score tokens.")
    st.stop()

if auto:
    simulate_once(conn, market, settings)
elif refresh:
    simulate_once(conn, market, settings)

acct = get_account(conn)
positions = get_positions(conn)
market_value, positions_view = mark_to_market(positions, market)
equity = acct["cash"] + market_value

trades = get_trades(conn)
wins = int((trades[trades.side == "SELL"].pnl > 0).sum()) if not trades.empty else 0
losses = int((trades[trades.side == "SELL"].pnl < 0).sum()) if not trades.empty else 0
closed = wins + losses
win_rate = 100 * wins / closed if closed else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Equity", pnl_color(equity))
c2.metric("Cash", pnl_color(acct["cash"]))
c3.metric("Realized P&L", pnl_color(acct["realized_pnl"]))
c4.metric("Open positions", len(positions))
c5.metric("Win rate", f"{win_rate:.1f}%")

st.subheader("FOMO radar")
radar_cols = ["symbol", "current_price", "price_change_percentage_1h", "price_change_percentage_24h", "price_change_percentage_7d", "total_volume", "market_cap", "fomo_score", "score_label"]
display = market[radar_cols].copy()
display.columns = ["Ticker", "Price", "1h %", "24h %", "7d %", "Volume", "Market Cap", "FOMO Score", "Signal"]
display["Price"] = display["Price"].map(lambda x: f"${x:,.8f}" if x < 1 else f"${x:,.4f}")
display["Volume"] = display["Volume"].map(lambda x: f"${x/1e6:,.1f}M")
display["Market Cap"] = display["Market Cap"].map(lambda x: f"${x/1e6:,.1f}M")
st.dataframe(display, use_container_width=True, hide_index=True)

left, right = st.columns([1.5, 1])
with left:
    st.subheader("Selected token")
    selected = st.selectbox("Token", market["symbol"].str.upper().tolist(), index=0)
    row = market[market.symbol.str.upper() == selected].iloc[0]
    fig = go.Figure(go.Candlestick(
        x=list(range(len(row.series))),
        open=row.series * 0.998,
        high=row.series * 1.004,
        low=row.series * 0.996,
        close=row.series,
    ))
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Recent samples", yaxis_title="USD")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Score breakdown")
    labels = ["Momentum", "Volume/Liquidity", "Acceleration", "Trend", "Liquidity"]
    vals = [row[f"score_{x}"] for x in labels]
    fig2 = go.Figure(go.Bar(x=vals, y=labels, orientation="h", text=[f"{v:.0f}" for v in vals], textposition="auto"))
    fig2.update_xaxes(range=[0, 100])
    fig2.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Score")
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Open positions")
if positions_view.empty:
    st.info("No open paper positions. Press Refresh or turn on Auto-scan to run the strategy.")
else:
    pv = positions_view[["symbol", "qty", "entry_price", "market_value", "unrealized_pnl", "score", "entry_time"]].copy()
    pv.columns = ["Ticker", "Qty", "Entry", "Market Value", "Unrealized P&L", "Current Score", "Entry Time"]
    st.dataframe(pv, use_container_width=True, hide_index=True)

st.subheader("Trade log")
if trades.empty:
    st.info("No trades yet.")
else:
    tl = trades[["timestamp", "symbol", "side", "price", "qty", "pnl", "score", "reason"]].copy()
    tl.columns = ["Time", "Ticker", "Side", "Price", "Qty", "P&L", "Score", "Reason"]
    st.dataframe(tl, use_container_width=True, hide_index=True)

st.divider()
st.caption("This project deliberately does not connect to exchanges or place live orders. The FOMO score is a heuristic, not financial advice or a prediction engine.")
