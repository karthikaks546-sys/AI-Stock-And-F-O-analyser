"""
stockapp.py — AI Stock & F&O Analyzer (Final Version)
======================================================
pip install streamlit yfinance pandas plotly
streamlit run stockapp.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import traceback
import time

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Stock & F&O Analyzer",
    page_icon="📈",
    layout="wide"
)

# Session state — locks TP/SL once a signal fires so it stays fixed
# across auto-refreshes until TP or SL is actually reached
if 'locked_trade' not in st.session_state:
    st.session_state.locked_trade  = None
if 'locked_action' not in st.session_state:
    st.session_state.locked_action = 'none'
if 'locked_ticker' not in st.session_state:
    st.session_state.locked_ticker = ''

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")

mode = st.sidebar.radio("Select Mode", ["📊 Stock Analysis", "📉 F&O / Index Analysis"])

FO_SYMBOLS = {
    "NIFTY 50"     : "^NSEI",
    "BANK NIFTY"   : "^NSEBANK",
    "NIFTY IT"     : "^CNXIT",
    "SENSEX"       : "^BSESN",
    "NIFTY MIDCAP" : "^NSEMDCP50",
    "RELIANCE"     : "RELIANCE.NS",
    "TCS"          : "TCS.NS",
    "INFOSYS"      : "INFY.NS",
    "HDFC BANK"    : "HDFCBANK.NS",
    "ICICI BANK"   : "ICICIBANK.NS",
    "WIPRO"        : "WIPRO.NS",
    "BAJAJ FINANCE": "BAJFINANCE.NS",
    "SBI"          : "SBIN.NS",
    "AXIS BANK"    : "AXISBANK.NS",
    "MARUTI"       : "MARUTI.NS",
}

if "Stock" in mode:
    ticker = st.sidebar.text_input(
        "Enter NSE Ticker", value="INFY.NS",
        placeholder="e.g. INFY.NS, TCS.NS, RELIANCE.NS"
    )
    is_fo = False
else:
    selected = st.sidebar.selectbox("Select Index / F&O Symbol", list(FO_SYMBOLS.keys()))
    ticker   = FO_SYMBOLS[selected]
    is_fo    = True
    st.sidebar.info(f"Ticker: `{ticker}`")

timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["1d — Daily", "1h — Hourly", "15m — 15 Min", "5m — 5 Min"]
)
TF_MAP = {
    "1d — Daily"  : ("1d",  "6mo"),
    "1h — Hourly" : ("1h",  "60d"),
    "15m — 15 Min": ("15m", "20d"),
    "5m — 5 Min"  : ("5m",  "5d"),
}
interval, period = TF_MAP[timeframe]

auto_refresh = st.sidebar.checkbox("🔄 Auto refresh (5s)", value=False)
analyze_btn  = st.sidebar.button("🚀 Analyze", use_container_width=True)
st.sidebar.divider()
st.sidebar.caption("⚠️ For learning only. Not financial advice.")

# Title changes based on mode
if "Stock" in mode:
    st.title("📊 AI Stock Analyzer")
    st.caption("RSI · MA · MACD · VWAP · ATR · Fibonacci · BOS · CHoCH · OB · FVG · Candlestick Patterns · Volume Spike")
else:
    st.title("📉 AI F&O / Index Analyzer")
    st.caption("RSI · MA · MACD · VWAP · ATR · Fibonacci · BOS · CHoCH · OB · FVG · Candlestick · Volume Spike · Options")
st.divider()


# ═════════════════════════════════════════════════════════════════════════════
#  DATA
# ═════════════════════════════════════════════════════════════════════════════

def load_data(ticker, interval, period):
    """Download and clean OHLCV data."""
    try:
        raw = yf.download(ticker, period=period, interval=interval,
                          progress=False, auto_adjust=True)
    except Exception as e:
        return None, str(e)

    if raw is None or raw.empty:
        return None, "No data returned."

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw.columns = [str(c).strip().title() for c in raw.columns]
    needed = ['Open', 'High', 'Low', 'Close', 'Volume']
    miss   = [c for c in needed if c not in raw.columns]
    if miss:
        return None, f"Missing columns: {miss}"

    df = raw[needed].copy()
    for c in needed:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
    df['Volume'] = df['Volume'].fillna(0)
    return df.reset_index(drop=True), None


# ═════════════════════════════════════════════════════════════════════════════
#  INDICATORS
# ═════════════════════════════════════════════════════════════════════════════

def add_indicators(df):
    """MA20, MA50, RSI, MACD, VWAP, Bollinger Bands, Volume MA."""

    # ── MA ────────────────────────────────────────────────────────────────────
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()

    # ── RSI ───────────────────────────────────────────────────────────────────
    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # ── MACD ─────────────────────────────────────────────────────────────────
    # EMA12 - EMA26 = MACD line
    # Signal = 9-period EMA of MACD
    # Histogram = MACD - Signal
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD']        = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist']   = df['MACD'] - df['MACD_Signal']

    # ── VWAP ─────────────────────────────────────────────────────────────────
    # Typical Price × Volume / Cumulative Volume
    tp           = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP']   = (tp * df['Volume']).cumsum() / (df['Volume'].cumsum() + 1e-10)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    std           = df['Close'].rolling(20).std()
    df['BB_Upper']= df['MA20'] + 2 * std
    df['BB_Lower']= df['MA20'] - 2 * std

    # ── Volume MA ─────────────────────────────────────────────────────────────
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()

    return df


# ═════════════════════════════════════════════════════════════════════════════
#  SUPPORT & RESISTANCE
# ═════════════════════════════════════════════════════════════════════════════

def calc_support_resistance(df):
    support    = float(df['Low'].rolling(20).min().iloc[-1])
    resistance = float(df['High'].rolling(20).max().iloc[-1])
    return support, resistance


# ═════════════════════════════════════════════════════════════════════════════
#  FIBONACCI RETRACEMENT
#  Draws levels between last significant swing high and swing low.
#  Key levels: 23.6%, 38.2%, 50%, 61.8%, 78.6%
# ═════════════════════════════════════════════════════════════════════════════

def calc_fibonacci(df):
    """Return Fibonacci retracement price levels."""
    lookback = min(60, len(df))
    recent   = df.tail(lookback)
    high     = float(recent['High'].max())
    low      = float(recent['Low'].min())
    diff     = high - low
    levels   = {
        "0.0%"  : high,
        "23.6%" : high - 0.236 * diff,
        "38.2%" : high - 0.382 * diff,
        "50.0%" : high - 0.500 * diff,
        "61.8%" : high - 0.618 * diff,
        "78.6%" : high - 0.786 * diff,
        "100%"  : low,
    }
    return levels


# ═════════════════════════════════════════════════════════════════════════════
#  SMART MONEY — Swing Points, BOS, CHoCH, Order Blocks, FVG
# ═════════════════════════════════════════════════════════════════════════════

def detect_swings(df, lookback=5):
    n  = len(df)
    sh = [False] * n
    sl = [False] * n
    h  = df['High'].tolist()
    l  = df['Low'].tolist()
    for i in range(lookback, n - lookback):
        if h[i] == max(h[i-lookback : i+lookback+1]): sh[i] = True
        if l[i] == min(l[i-lookback : i+lookback+1]): sl[i] = True
    df['Swing_High'] = sh
    df['Swing_Low']  = sl
    return df


def detect_trend_smc(df):
    sh = df.loc[df['Swing_High'], 'High'].values
    sl = df.loc[df['Swing_Low'],  'Low'].values
    if len(sh) >= 2 and len(sl) >= 2:
        if sh[-1] > sh[-2] and sl[-1] > sl[-2]: return "Uptrend 📈"
        if sh[-1] < sh[-2] and sl[-1] < sl[-2]: return "Downtrend 📉"
    return "Sideways ↔"


def detect_bos_choch(df):
    n  = len(df)
    c  = df['Close'].tolist()
    h  = df['High'].tolist()
    l  = df['Low'].tolist()
    bb = [False]*n; bear = [False]*n
    cb = [False]*n; cbe  = [False]*n
    sp = [i for i in range(n) if df['Swing_High'].iloc[i]]
    lp = [i for i in range(n) if df['Swing_Low'].iloc[i]]
    for i in range(20, n):
        ps = [p for p in sp if p < i]
        pl = [p for p in lp if p < i]
        if len(ps)<2 or len(pl)<2: continue
        lsh=h[ps[-1]]; lsl=l[pl[-1]]
        up  = h[ps[-1]]>h[ps[-2]] and l[pl[-1]]>l[pl[-2]]
        dn  = h[ps[-1]]<h[ps[-2]] and l[pl[-1]]<l[pl[-2]]
        if c[i]>lsh:
            if dn: cb[i]=True
            else:  bb[i]=True
        if c[i]<lsl:
            if up: cbe[i]=True
            else:  bear[i]=True
    df['BOS_Bull']=bb; df['BOS_Bear']=bear
    df['CHoCH_Bull']=cb; df['CHoCH_Bear']=cbe
    return df


def detect_order_blocks(df):
    n  = len(df)
    o  = df['Open'].tolist()
    c  = df['Close'].tolist()
    h  = df['High'].tolist()
    l  = df['Low'].tolist()
    bh=[None]*n; bl=[None]*n; sh=[None]*n; sl=[None]*n
    for i in range(5, n):
        if df['BOS_Bull'].iloc[i] or df['CHoCH_Bull'].iloc[i]:
            for j in range(i-1, max(i-15,0), -1):
                if c[j]<o[j]: bh[i]=h[j]; bl[i]=l[j]; break
        if df['BOS_Bear'].iloc[i] or df['CHoCH_Bear'].iloc[i]:
            for j in range(i-1, max(i-15,0), -1):
                if c[j]>o[j]: sh[i]=h[j]; sl[i]=l[j]; break
    df['OB_Bull_H']=bh; df['OB_Bull_L']=bl
    df['OB_Bear_H']=sh; df['OB_Bear_L']=sl
    return df


def detect_fvg(df):
    """
    Fair Value Gap (FVG):
    Bullish FVG = candle[i-2] High < candle[i] Low  → gap up, price may return
    Bearish FVG = candle[i-2] Low  > candle[i] High → gap down
    """
    n = len(df)
    fvg_bull_h=[None]*n; fvg_bull_l=[None]*n
    fvg_bear_h=[None]*n; fvg_bear_l=[None]*n
    h = df['High'].tolist()
    l = df['Low'].tolist()
    for i in range(2, n):
        if l[i] > h[i-2]:           # bullish gap
            fvg_bull_h[i] = l[i]
            fvg_bull_l[i] = h[i-2]
        if h[i] < l[i-2]:           # bearish gap
            fvg_bear_h[i] = l[i-2]
            fvg_bear_l[i] = h[i]
    df['FVG_Bull_H']=fvg_bull_h; df['FVG_Bull_L']=fvg_bull_l
    df['FVG_Bear_H']=fvg_bear_h; df['FVG_Bear_L']=fvg_bear_l
    return df



# ═════════════════════════════════════════════════════════════════════════════
#  HIGH IMPACT ACCURACY BOOSTERS
# ═════════════════════════════════════════════════════════════════════════════

def detect_candle_patterns(df):
    """
    Candlestick Pattern Recognition
    ────────────────────────────────
    Detects the most reliable 1-2 candle reversal patterns:

    BULLISH patterns (confirm buy entry):
    - Hammer        : Small body at top, long lower wick (2×+ body), little upper wick
                      → Sellers pushed price down but buyers rejected it — reversal signal
    - Bullish Engulfing : Current green candle body fully wraps previous red candle
                          → Buyers completely overpowered sellers — strong reversal
    - Morning Doji  : Doji candle (open ≈ close) — indecision after a downmove

    BEARISH patterns (confirm sell entry):
    - Shooting Star  : Small body at bottom, long upper wick (2×+ body)
                       → Buyers pushed price up but sellers rejected it — reversal signal
    - Bearish Engulfing : Current red candle body fully wraps previous green candle
    - Evening Doji  : Doji after an upmove — indecision = possible reversal down
    """
    n      = len(df)
    o      = df['Open'].tolist()
    h      = df['High'].tolist()
    l      = df['Low'].tolist()
    c      = df['Close'].tolist()

    bull_pattern = ['' ] * n
    bear_pattern = ['' ] * n

    for i in range(1, n):
        body       = abs(c[i] - o[i])
        upper_wick = h[i] - max(c[i], o[i])
        lower_wick = min(c[i], o[i]) - l[i]
        total_range= h[i] - l[i]

        if total_range < 1e-9:
            continue

        prev_body  = abs(c[i-1] - o[i-1])
        is_green   = c[i] > o[i]
        is_red     = c[i] < o[i]
        prev_green = c[i-1] > o[i-1]
        prev_red   = c[i-1] < o[i-1]

        # ── BULLISH ──────────────────────────────────────────────────────────
        # Hammer: body in upper 1/3, lower wick >= 2× body, tiny upper wick
        if (lower_wick >= 2 * body and
                upper_wick <= 0.3 * body and
                body > 0 and
                min(c[i], o[i]) > l[i] + 0.6 * total_range):
            bull_pattern[i] = 'Hammer'

        # Bullish Engulfing: green candle body wraps previous red candle body
        elif (is_green and prev_red and
              o[i] <= c[i-1] and c[i] >= o[i-1] and
              body > prev_body * 0.8):
            bull_pattern[i] = 'Bull Engulfing'

        # Doji (open ≈ close, small body relative to range)
        elif body <= total_range * 0.1:
            bull_pattern[i] = 'Doji'   # neutral — caller decides context

        # ── BEARISH ──────────────────────────────────────────────────────────
        # Shooting Star: body in lower 1/3, upper wick >= 2× body
        if (upper_wick >= 2 * body and
                lower_wick <= 0.3 * body and
                body > 0 and
                max(c[i], o[i]) < h[i] - 0.6 * total_range):
            bear_pattern[i] = 'Shooting Star'

        # Bearish Engulfing
        elif (is_red and prev_green and
              o[i] >= c[i-1] and c[i] <= o[i-1] and
              body > prev_body * 0.8):
            bear_pattern[i] = 'Bear Engulfing'

        elif body <= total_range * 0.1 and bear_pattern[i] == '':
            bear_pattern[i] = 'Doji'

    df['Bull_Pattern'] = bull_pattern
    df['Bear_Pattern'] = bear_pattern
    return df


def detect_volume_spike(df):
    """
    Volume Spike Filter
    ────────────────────
    A signal is only worth taking if volume confirms it.
    Volume Spike = current volume > 1.5× the 20-period average volume.

    Why this works:
    - High volume = institutions/smart money is participating
    - Low volume moves are often fakeouts that reverse quickly
    - Volume spike on a bullish candle = real buying pressure
    - Volume spike on a bearish candle = real selling pressure

    Adds column: Vol_Spike (True/False)
    """
    df['Vol_Spike'] = df['Volume'] > (df['Vol_MA20'] * 1.5)
    return df


def calc_atr(df, period=14):
    """
    ATR (Average True Range) — Dynamic Stop Loss
    ─────────────────────────────────────────────
    True Range = max of:
      - High - Low
      - |High - Previous Close|
      - |Low  - Previous Close|

    ATR = rolling average of True Range over N periods

    Why ATR SL is better than fixed %:
    - Volatile stocks (TATAMOTORS) need wider SL
    - Stable stocks (INFY) need tighter SL
    - ATR automatically adapts to current volatility
    - SL = Entry - 1.5 × ATR (buy) or Entry + 1.5 × ATR (sell)
    """
    high  = df['High']
    low   = df['Low']
    close = df['Close']

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()

    true_range  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR']   = true_range.rolling(period).mean()
    return df


# ═════════════════════════════════════════════════════════════════════════════
#  SIGNAL GENERATION
#  Layer 1 — RSI + MA (your original base, always required)
#  Layer 2 — MACD confirmation
#  Layer 3 — VWAP position
#  Layer 4 — Support/Resistance zone
#  Layer 5 — BOS/CHoCH structure
#  Layer 6 — Order Block nearby
#  Strength = count of extra confirmations (0–5)
# ═════════════════════════════════════════════════════════════════════════════

def generate_signals(df, support, resistance):
    """
    Signal Layers (8 total now):
    Layer 1 — RSI + MA         (required base)
    Layer 2 — MACD             (momentum confirmation)
    Layer 3 — VWAP             (institutional level)
    Layer 4 — Support/Resistance zone
    Layer 5 — BOS/CHoCH structure
    Layer 6 — Order Block nearby
    Layer 7 — Candlestick Pattern  ← NEW (Hammer/Engulfing confirms entry)
    Layer 8 — Volume Spike         ← NEW (only trade with real volume)

    STRONG  = 5+ confirmations (high confidence)
    MODERATE= 3-4 confirmations
    WEAK    = 1-2 confirmations (shown on chart but decision engine needs more)
    """
    n   = len(df)
    c   = df['Close'].tolist()
    ma  = df['MA20'].tolist()
    rsi = df['RSI'].tolist()
    mac = df['MACD'].tolist()
    sig = df['MACD_Signal'].tolist()
    vwp = df['VWAP'].tolist()

    buy=[False]*n; sell=[False]*n; strength=[''] * n

    for i in range(26, n):
        r,cl,m = rsi[i], c[i], ma[i]
        if pd.isna(r) or pd.isna(m): continue

        # Layer 1 — base (required)
        base_buy  = (r < 40) and (cl < m)
        base_sell = (r > 60) and (cl > m)

        # Layer 2 — MACD
        macd_buy  = mac[i] > sig[i]
        macd_sell = mac[i] < sig[i]

        # Layer 3 — VWAP
        vwap_buy  = cl < vwp[i]
        vwap_sell = cl > vwp[i]

        # Layer 4 — S/R zone
        sr_buy    = cl <= support * 1.03
        sr_sell   = cl >= resistance * 0.97

        # Layer 5 — structure
        s = max(0, i-5)
        bull_struct = (any(df['BOS_Bull'].iloc[s:i+1]) or
                       any(df['CHoCH_Bull'].iloc[s:i+1]))
        bear_struct = (any(df['BOS_Bear'].iloc[s:i+1]) or
                       any(df['CHoCH_Bear'].iloc[s:i+1]))

        # Layer 6 — OB
        ob_start = max(0, i-10)
        ob_bull = any(v is not None for v in df['OB_Bull_H'].iloc[ob_start:i+1])
        ob_bear = any(v is not None for v in df['OB_Bear_H'].iloc[ob_start:i+1])

        # Layer 7 — Candlestick Pattern (NEW)
        # Check last 2 candles for a bullish/bearish pattern
        pat_window = df['Bull_Pattern'].iloc[max(0,i-2):i+1]
        bear_pat_w = df['Bear_Pattern'].iloc[max(0,i-2):i+1]
        candle_bull = any(p in ('Hammer', 'Bull Engulfing')
                          for p in pat_window)
        candle_bear = any(p in ('Shooting Star', 'Bear Engulfing')
                          for p in bear_pat_w)

        # Layer 8 — Volume Spike (NEW)
        vol_spike = bool(df['Vol_Spike'].iloc[i])

        if base_buy:
            conf = sum([macd_buy, vwap_buy, sr_buy, bull_struct,
                        ob_bull, candle_bull, vol_spike])
            buy[i] = True
            strength[i] = ('STRONG'   if conf >= 5 else
                           'MODERATE' if conf >= 3 else 'WEAK')

        if base_sell:
            conf = sum([macd_sell, vwap_sell, sr_sell, bear_struct,
                        ob_bear, candle_bear, vol_spike])
            sell[i] = True
            strength[i] = ('STRONG'   if conf >= 5 else
                           'MODERATE' if conf >= 3 else 'WEAK')

    df['Buy_Signal']  = buy
    df['Sell_Signal'] = sell
    df['Sig_Strength']= strength
    return df


def get_trade_setup(df, support, resistance):
    """
    Dynamic SL using ATR (NEW) + swing points (existing).
    ────────────────────────────────────────────────────────
    Old method: Fixed % SL (e.g. -2%) — same for INFY and TATAMOTORS
    New method: ATR × 1.5 below/above entry

    ATR adapts to the stock's current volatility:
    - Volatile stock (ATR = 50): SL = Entry - 75 (wider, avoids noise)
    - Stable stock  (ATR = 10): SL = Entry - 15 (tighter, less risk)

    We use MAX(ATR-based SL, swing-point SL) so SL is always beyond
    a real structure level AND accounts for volatility.
    """
    price = float(df['Close'].iloc[-1])
    atr   = float(df['ATR'].iloc[-1]) if pd.notna(df['ATR'].iloc[-1]) else price * 0.01

    # Swing-based SL (structural)
    sl_df = df.loc[df['Swing_Low'],  'Low'].tail(3)
    sh_df = df.loc[df['Swing_High'], 'High'].tail(3)
    swing_sl_buy  = float(sl_df.min()) if not sl_df.empty else price * 0.97
    swing_sl_sell = float(sh_df.max()) if not sh_df.empty else price * 1.03

    # ATR-based SL (volatility-adjusted)
    atr_sl_buy  = price - 1.5 * atr
    atr_sl_sell = price + 1.5 * atr

    # Use the wider of the two (more protection)
    sl_buy  = min(swing_sl_buy,  atr_sl_buy)
    sl_sell = max(swing_sl_sell, atr_sl_sell)

    # Hard floor: SL must always be on the correct side
    sl_buy  = min(sl_buy,  price * 0.995)
    sl_sell = max(sl_sell, price * 1.005)

    risk_buy  = max(price - sl_buy,  price * 0.005)
    risk_sell = max(sl_sell - price, price * 0.005)

    return {
        'price'   : price,
        'tp_buy'  : round(price + 2 * risk_buy,  2),
        'sl_buy'  : round(sl_buy,                2),
        'tp_sell' : round(price - 2 * risk_sell, 2),
        'sl_sell' : round(sl_sell,               2),
        'atr'     : round(atr, 2),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  FINAL DECISION ENGINE
#  Combines ALL signals into one clear verdict
# ═════════════════════════════════════════════════════════════════════════════

def final_decision(df, support, resistance, smc_trend):
    """
    Scoring system — each factor adds +1 (bullish) or -1 (bearish):
      RSI < 40          → +1  |  RSI > 60  → -1
      Price < MA20      → +1  |  Price > MA20 → -1  (reversal context)
      MACD > Signal     → +1  |  MACD < Signal → -1
      Price < VWAP      → +1  |  Price > VWAP → -1
      Near Support      → +1  |  Near Resistance → -1
      Recent Buy Signal → +1  |  Recent Sell Signal → -1
      Uptrend (SMC)     → +1  |  Downtrend → -1

    Score >= 4  → STRONG BUY
    Score 2–3   → MODERATE BUY
    Score -1–1  → NEUTRAL / WAIT
    Score -2–-3 → MODERATE SELL
    Score <= -4 → STRONG SELL
    """
    price = float(df['Close'].iloc[-1])
    ma20  = float(df['MA20'].iloc[-1])
    rsi   = float(df['RSI'].iloc[-1])
    macd  = float(df['MACD'].iloc[-1])
    msig  = float(df['MACD_Signal'].iloc[-1])
    vwap  = float(df['VWAP'].iloc[-1])

    score = 0
    reasons = []

    # RSI
    if rsi < 40:
        score += 1; reasons.append(f"✅ RSI {rsi:.1f} — Oversold (Bullish)")
    elif rsi > 60:
        score -= 1; reasons.append(f"❌ RSI {rsi:.1f} — Overbought (Bearish)")
    else:
        reasons.append(f"⚪ RSI {rsi:.1f} — Neutral")

    # Price vs MA20
    if price < ma20:
        score += 1; reasons.append("✅ Price below MA20 — Possible reversal up")
    else:
        score -= 1; reasons.append("❌ Price above MA20 — Possible reversal down")

    # MACD
    if macd > msig:
        score += 1; reasons.append("✅ MACD above Signal — Bullish momentum")
    else:
        score -= 1; reasons.append("❌ MACD below Signal — Bearish momentum")

    # VWAP
    if price < vwap:
        score += 1; reasons.append("✅ Price below VWAP — Institutional buy zone")
    else:
        score -= 1; reasons.append("❌ Price above VWAP — Institutional sell zone")

    # Support / Resistance
    if price <= support * 1.03:
        score += 1; reasons.append(f"✅ Near Support ₹{support:.2f} — Buy zone")
    elif price >= resistance * 0.97:
        score -= 1; reasons.append(f"❌ Near Resistance ₹{resistance:.2f} — Sell zone")
    else:
        reasons.append("⚪ Middle zone — No S/R edge")

    # Recent signal
    recent_buy  = df['Buy_Signal'].iloc[-5:].any()
    recent_sell = df['Sell_Signal'].iloc[-5:].any()
    if recent_buy:
        score += 1; reasons.append("✅ Recent Buy Signal detected")
    elif recent_sell:
        score -= 1; reasons.append("❌ Recent Sell Signal detected")

    # SMC trend
    if "Up" in smc_trend:
        score += 1; reasons.append("✅ SMC Trend: Uptrend (HH + HL)")
    elif "Down" in smc_trend:
        score -= 1; reasons.append("❌ SMC Trend: Downtrend (LH + LL)")
    else:
        reasons.append("⚪ SMC Trend: Sideways")

    # Candlestick Pattern (NEW — Layer 7)
    last_bull_pat = df['Bull_Pattern'].iloc[-3:].tolist()
    last_bear_pat = df['Bear_Pattern'].iloc[-3:].tolist()
    strong_bull_candle = any(p in ('Hammer', 'Bull Engulfing') for p in last_bull_pat)
    strong_bear_candle = any(p in ('Shooting Star', 'Bear Engulfing') for p in last_bear_pat)
    latest_bull_pat = next((p for p in reversed(last_bull_pat) if p not in ('', 'Doji')), '')
    latest_bear_pat = next((p for p in reversed(last_bear_pat) if p not in ('', 'Doji')), '')

    if strong_bull_candle:
        score += 1
        reasons.append(f"✅ Bullish Candle Pattern: {latest_bull_pat} — Buyers confirmed")
    elif strong_bear_candle:
        score -= 1
        reasons.append(f"❌ Bearish Candle Pattern: {latest_bear_pat} — Sellers confirmed")
    else:
        any_doji = 'Doji' in last_bull_pat or 'Doji' in last_bear_pat
        reasons.append(f"⚪ Candle Pattern: {'Doji (indecision)' if any_doji else 'No clear pattern'}")

    # Volume Spike (NEW — Layer 8)
    vol_spike_recent = df['Vol_Spike'].iloc[-3:].any()
    if vol_spike_recent:
        score += 1; reasons.append("✅ Volume Spike — Strong institutional participation")
    else:
        reasons.append("⚪ No volume spike — Move may lack conviction")

    # Verdict — score now out of 9 (was 7, now 9 with 2 new factors)
    if score >= 5:
        verdict = "🟢 STRONG BUY"
        action  = "buy"
    elif score >= 3:
        verdict = "🟡 MODERATE BUY — Wait for 1 more confirmation"
        action  = "buy"
    elif score <= -5:
        verdict = "🔴 STRONG SELL"
        action  = "sell"
    elif score <= -3:
        verdict = "🟠 MODERATE SELL — Wait for 1 more confirmation"
        action  = "sell"
    else:
        verdict = "⚪ NEUTRAL — No clear signal. Stay out."
        action  = "none"

    return verdict, action, score, reasons


# ═════════════════════════════════════════════════════════════════════════════
#  PLOTLY LIVE CHART
#  Clean, simple, professional — like Zerodha/Groww
#  Panel 1: Candlesticks + VWAP + MA lines + Buy/Sell markers + TP/SL zones
#  Panel 2: MACD histogram + lines
#  Panel 3: RSI with overbought/oversold
# ═════════════════════════════════════════════════════════════════════════════

def build_chart(df, trade, action):
    """
    Clean Groww/Zerodha-style chart — 3 panels, nothing extra.

    Panel 1 (60%) — Candlesticks only + VWAP (cyan) + MA20 (orange)
                     + Support/Resistance dotted lines
                     + TP green zone / SL red zone (only when signal exists)
                     + BUY (green ▲) / SELL (red ▼) markers — STRONG only
    Panel 2 (20%) — MACD histogram (green/red bars) + MACD line + Signal line
    Panel 3 (20%) — RSI line + 70/30 zones

    Rules followed for clarity:
    - No text annotations on candles (too messy)
    - No Fibonacci on chart by default (too many lines)
    - No MA50 on main chart (already have MA20 + VWAP, more is noise)
    - No Bollinger Bands (separate analysis, not needed on price chart)
    - Only STRONG signals shown as markers (filter noise)
    - TP/SL zone only shown when there is an active signal
    - All labels on right axis only (like Groww)
    - Hover shows OHLC + all values together
    """
    n = len(df)
    xs = list(range(n))

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.60, 0.20, 0.20],
        vertical_spacing=0.0,
    )

    # ══════════════════════════════════════════════════════════════════
    # PANEL 1 — CANDLESTICK CHART
    # ══════════════════════════════════════════════════════════════════

    # Green candle = close > open  |  Red candle = close < open
    fig.add_trace(go.Candlestick(
        x=xs,
        open=df['Open'], high=df['High'],
        low=df['Low'],   close=df['Close'],
        increasing=dict(line=dict(color='#26a69a', width=1),
                        fillcolor='#26a69a'),
        decreasing=dict(line=dict(color='#ef5350', width=1),
                        fillcolor='#ef5350'),
        name='Price',
        showlegend=False,
        hoverinfo='x+y',
    ), row=1, col=1)

    # MA20 — thin orange dashed line
    fig.add_trace(go.Scatter(
        x=xs, y=df['MA20'].tolist(),
        line=dict(color='#f59e0b', width=1.5, dash='dash'),
        name='MA20',
        hoverinfo='skip',
    ), row=1, col=1)

    # VWAP — thin cyan line (institutional reference)
    fig.add_trace(go.Scatter(
        x=xs, y=df['VWAP'].tolist(),
        line=dict(color='#22d3ee', width=1.5),
        name='VWAP',
        hoverinfo='skip',
    ), row=1, col=1)

    # ── TP / SL zones — ONLY shown when buy or sell signal is active ────
    # ── TP/SL ZONES ────────────────────────────────────────────────────────
    # Zone spans FULL chart width so price always moves INTO the colored band.
    # Green band = profit zone (price must reach TP to close trade)
    # Red band   = risk zone  (price hitting SL = exit immediately)
    # Horizontal dashed lines + right-side labels like TradingView/Zerodha
    # Zones only visible when action is buy or sell (hidden on WAIT)
    if action == 'buy':
        entry = trade['price']
        tp    = trade['tp_buy']
        sl    = trade['sl_buy']

        # Full-width green band: Entry → TP (price must enter here to take profit)
        fig.add_shape(type='rect',
            x0=0, x1=n+40, y0=entry, y1=tp,
            fillcolor='rgba(16,185,129,0.10)',
            line=dict(width=0), row=1, col=1)
        # Full-width red band: SL → Entry (price entering here = stop loss hit)
        fig.add_shape(type='rect',
            x0=0, x1=n+40, y0=sl, y1=entry,
            fillcolor='rgba(239,68,68,0.10)',
            line=dict(width=0), row=1, col=1)

        # Dashed lines + right-side labels for TP, Entry, SL
        for y_val, color, label in [
            (tp,    '#10b981', f' TP  ₹{tp:,.0f}'),
            (entry, '#fbbf24', f' Entry ₹{entry:,.0f}'),
            (sl,    '#ef4444', f' SL  ₹{sl:,.0f}'),
        ]:
            fig.add_shape(type='line',
                x0=0, x1=n+40, y0=y_val, y1=y_val,
                line=dict(color=color, width=2.0, dash='dash'),
                row=1, col=1)
            fig.add_annotation(
                x=n+41, y=y_val, text=label,
                showarrow=False, xanchor='left',
                font=dict(color=color, size=11, family='monospace'),
                xref='x', yref='y', row=1, col=1)

    elif action == 'sell':
        entry = trade['price']
        tp    = trade['tp_sell']
        sl    = trade['sl_sell']

        # Full-width green band: TP → Entry (price drops into here = profit)
        fig.add_shape(type='rect',
            x0=0, x1=n+40, y0=tp, y1=entry,
            fillcolor='rgba(16,185,129,0.10)',
            line=dict(width=0), row=1, col=1)
        # Full-width red band: Entry → SL (price rises into here = stop hit)
        fig.add_shape(type='rect',
            x0=0, x1=n+40, y0=entry, y1=sl,
            fillcolor='rgba(239,68,68,0.10)',
            line=dict(width=0), row=1, col=1)

        for y_val, color, label in [
            (sl,    '#ef4444', f' SL  ₹{sl:,.0f}'),
            (entry, '#fbbf24', f' Entry ₹{entry:,.0f}'),
            (tp,    '#10b981', f' TP  ₹{tp:,.0f}'),
        ]:
            fig.add_shape(type='line',
                x0=0, x1=n+40, y0=y_val, y1=y_val,
                line=dict(color=color, width=2.0, dash='dash'),
                row=1, col=1)
            fig.add_annotation(
                x=n+41, y=y_val, text=label,
                showarrow=False, xanchor='left',
                font=dict(color=color, size=11, family='monospace'),
                xref='x', yref='y', row=1, col=1)

    # ── STRONG Buy signals only — clean triangles ──────────────────────
    buy_idx  = [i for i in xs if df['Buy_Signal'].iloc[i]
                and df['Sig_Strength'].iloc[i] == 'STRONG']
    sell_idx = [i for i in xs if df['Sell_Signal'].iloc[i]
                and df['Sig_Strength'].iloc[i] == 'STRONG']

    if buy_idx:
        fig.add_trace(go.Scatter(
            x=buy_idx,
            y=[float(df['Low'].iloc[i]) * 0.988 for i in buy_idx],
            mode='markers',
            marker=dict(symbol='triangle-up', size=16,
                        color='#00e676',
                        line=dict(width=1, color='#004d2e')),
            name='BUY',
            showlegend=True,
        ), row=1, col=1)

    if sell_idx:
        fig.add_trace(go.Scatter(
            x=sell_idx,
            y=[float(df['High'].iloc[i]) * 1.012 for i in sell_idx],
            mode='markers',
            marker=dict(symbol='triangle-down', size=16,
                        color='#ff5252',
                        line=dict(width=1, color='#4d0000')),
            name='SELL',
            showlegend=True,
        ), row=1, col=1)

    # ══════════════════════════════════════════════════════════════════
    # PANEL 2 — MACD
    # ══════════════════════════════════════════════════════════════════

    hist_vals  = df['MACD_Hist'].tolist()
    bar_colors = ['#26a69a' if v >= 0 else '#ef5350' for v in hist_vals]

    fig.add_trace(go.Bar(
        x=xs, y=hist_vals,
        marker_color=bar_colors,
        opacity=0.8,
        showlegend=False,
        name='Histogram',
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=xs, y=df['MACD'].tolist(),
        line=dict(color='#60a5fa', width=1.2),
        name='MACD',
        showlegend=False,
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=xs, y=df['MACD_Signal'].tolist(),
        line=dict(color='#fb923c', width=1.2),
        name='Signal',
        showlegend=False,
    ), row=2, col=1)

    # Zero line
    fig.add_shape(type='line',
        x0=0, x1=n-1, y0=0, y1=0,
        line=dict(color='#334155', width=0.8),
        row=2, col=1)

    # ══════════════════════════════════════════════════════════════════
    # PANEL 3 — RSI
    # ══════════════════════════════════════════════════════════════════

    rsi_vals = df['RSI'].tolist()

    # Overbought shading
    fig.add_shape(type='rect',
        x0=0, x1=n-1, y0=70, y1=100,
        fillcolor='rgba(239,68,68,0.07)', line=dict(width=0),
        row=3, col=1)
    # Oversold shading
    fig.add_shape(type='rect',
        x0=0, x1=n-1, y0=0, y1=30,
        fillcolor='rgba(16,185,129,0.07)', line=dict(width=0),
        row=3, col=1)

    fig.add_trace(go.Scatter(
        x=xs, y=rsi_vals,
        line=dict(color='#c084fc', width=1.5),
        showlegend=False,
        name='RSI',
    ), row=3, col=1)

    # 70 / 30 / 50 reference lines
    for level, color in [(70, '#ef4444'), (50, '#334155'), (30, '#10b981')]:
        fig.add_shape(type='line',
            x0=0, x1=n-1, y0=level, y1=level,
            line=dict(color=color, width=0.8,
                      dash='dash' if level in (70,30) else 'solid'),
            row=3, col=1)

    # ══════════════════════════════════════════════════════════════════
    # LAYOUT — clean dark theme like Groww/Zerodha
    # ══════════════════════════════════════════════════════════════════

    fig.update_layout(
        height=680,
        paper_bgcolor='#0b1120',
        plot_bgcolor='#0b1120',
        font=dict(color='#94a3b8', family='Inter, sans-serif', size=11),
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='#1e293b',
            font_size=11,
            font_color='white',
        ),
        legend=dict(
            orientation='h',
            x=0, y=1.01,
            bgcolor='rgba(0,0,0,0)',
            font=dict(size=11),
        ),
        margin=dict(l=0, r=20, t=10, b=0),
        bargap=0.1,
    )

    # Shared x-axis style
    axis_style = dict(
        showgrid=True,
        gridcolor='#1e293b',
        gridwidth=0.5,
        zeroline=False,
        showline=False,
        tickfont=dict(size=10),
    )

    fig.update_xaxes(**axis_style)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=1)
    fig.update_xaxes(showticklabels=True,  row=3, col=1)
    # Extend x range so TP/SL labels at n+5 are visible (right of last candle)
    fig.update_xaxes(range=[-1, n + 55], row=1, col=1)
    fig.update_xaxes(range=[-1, n + 55], row=2, col=1)
    fig.update_xaxes(range=[-1, n + 55], row=3, col=1)

    fig.update_yaxes(**axis_style, side='right')
    fig.update_yaxes(range=[10, 90], row=3, col=1,
                     tickvals=[30, 50, 70])

    # Auto-scale panel 1 y-axis to actual price range
    # This fixes the "price squished at top" problem (e.g. 79k-80k showing in 0-80k scale)
    price_min = float(df['Low'].min())
    price_max = float(df['High'].max())
    pad = (price_max - price_min) * 0.15
    y_min = price_min - pad
    y_max = price_max + pad
    if action == 'buy':
        y_min = min(y_min, trade['sl_buy']  * 0.998)
        y_max = max(y_max, trade['tp_buy']  * 1.002)
    elif action == 'sell':
        y_min = min(y_min, trade['tp_sell'] * 0.998)
        y_max = max(y_max, trade['sl_sell'] * 1.002)
    fig.update_yaxes(range=[y_min, y_max], row=1, col=1)

    return fig


# ═════════════════════════════════════════════════════════════════════════════
#  F&O SUGGESTION — only shown in F&O mode
# ═════════════════════════════════════════════════════════════════════════════

def show_fo_suggestion(action, strength, price, score):
    st.subheader("📉 F&O / Options Suggestion")
    strike = round(price / 50) * 50

    if action == 'buy' and 'STRONG' in strength:
        st.success(
            f"🟢 **BUY CALL OPTION (CE)**\n\n"
            f"ATM Strike: **₹{strike} CE**  ← Best entry\n"
            f"OTM Strike: **₹{strike+50} CE**  ← Higher risk, cheaper premium\n\n"
            f"Enter when price crosses above VWAP with volume. "
            f"Exit at TP or if price closes below SL."
        )
    elif action == 'buy':
        st.warning(
            f"🟡 **Moderate CE Signal** — ATM: ₹{strike} CE\n\n"
            f"Wait for RSI to drop below 35 or MACD cross for stronger confirmation."
        )
    elif action == 'sell' and 'STRONG' in strength:
        st.error(
            f"🔴 **BUY PUT OPTION (PE)**\n\n"
            f"ATM Strike: **₹{strike} PE**  ← Best entry\n"
            f"OTM Strike: **₹{strike-50} PE**  ← Higher risk, cheaper premium\n\n"
            f"Enter when price breaks below VWAP. "
            f"Exit at TP or if price closes above SL."
        )
    elif action == 'sell':
        st.warning(
            f"🟠 **Moderate PE Signal** — ATM: ₹{strike} PE\n\n"
            f"Wait for MACD cross or RSI above 65 for stronger confirmation."
        )
    else:
        st.info(
            "⚪ **No options signal right now.**\n\n"
            "Score is neutral. Wait for RSI to reach extremes "
            "(<35 or >65) with MACD confirmation before entering options."
        )


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN — runs when Analyze button is clicked or auto-refresh fires
# ═════════════════════════════════════════════════════════════════════════════

def run_analysis():
    if not ticker.strip():
        st.warning("⚠️ Please enter a ticker symbol.")
        return

    try:
        with st.spinner(f"Fetching {interval} data for {ticker.upper()}..."):
            df, err = load_data(ticker.strip(), interval, period)

        if df is None:
            st.error(f"❌ {err}")
            st.info("💡 Use .NS suffix: INFY.NS, TCS.NS, RELIANCE.NS")
            return

        if len(df) < 30:
            st.error(f"❌ Only {len(df)} candles. Try Daily or Hourly timeframe.")
            return

        # Run all calculations
        df = add_indicators(df)
        df = detect_swings(df)
        df = detect_bos_choch(df)
        df = detect_order_blocks(df)
        df = detect_fvg(df)
        df = detect_candle_patterns(df)   # NEW — Hammer, Engulfing etc
        df = detect_volume_spike(df)      # NEW — 1.5× volume filter
        df = calc_atr(df)                 # NEW — dynamic SL calculation
        support, resistance = calc_support_resistance(df)
        df = generate_signals(df, support, resistance)
        fib = calc_fibonacci(df)

        # Drop warmup rows
        df = df[df['MA20'].notna() & df['RSI'].notna() &
                df['MACD'].notna()].reset_index(drop=True)

        if len(df) < 10:
            st.error("❌ Not enough valid data after computing indicators.")
            return

        trade     = get_trade_setup(df, support, resistance)
        smc_trend = detect_trend_smc(df)

        # Final Decision
        verdict, action, score, reasons = final_decision(
            df, support, resistance, smc_trend
        )

        # Latest values
        price     = float(df['Close'].iloc[-1])
        ma20      = float(df['MA20'].iloc[-1])
        rsi_val   = float(df['RSI'].iloc[-1])
        macd_val  = float(df['MACD'].iloc[-1])
        vwap_val  = float(df['VWAP'].iloc[-1])
        atr_val   = float(df['ATR'].iloc[-1]) if pd.notna(df['ATR'].iloc[-1]) else 0
        vol       = float(df['Volume'].iloc[-1])
        avg_vol   = float(df['Vol_MA20'].iloc[-1]) if pd.notna(df['Vol_MA20'].iloc[-1]) else 0
        op        = float(df['Open'].iloc[-1])
        cl        = float(df['Close'].iloc[-1])
        vol_spike = bool(df['Vol_Spike'].iloc[-1])
        last_bull_candle = df['Bull_Pattern'].iloc[-1]
        last_bear_candle = df['Bear_Pattern'].iloc[-1]

        # Last signal for chart
        last_strength = ''
        for i in range(len(df)-1, max(len(df)-6,0), -1):
            if df['Buy_Signal'].iloc[i]:
                last_strength = df['Sig_Strength'].iloc[i]; break
            if df['Sell_Signal'].iloc[i]:
                last_strength = df['Sig_Strength'].iloc[i]; break

        # ── DISPLAY ───────────────────────────────────────────────────────────

        # ── 1. Metric strip ───────────────────────────────────────────────────
        # Two rows of 4 metrics each — full values always visible
        st.subheader(f"**{ticker.upper()}** · {timeframe}")
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("💰 Price", f"₹{price:,.2f}")
        rc2.metric("📊 MA20",  f"₹{ma20:,.2f}")
        rc3.metric("💧 VWAP",  f"₹{vwap_val:,.2f}")
        rc4.metric("📏 ATR",   f"₹{atr_val:,.2f}")
        rsi_icon     = "🔴" if rsi_val > 60 else ("🟢" if rsi_val < 40 else "🟡")
        macd_icon    = "▲" if macd_val > 0 else "▼"
        candle_label = (last_bull_candle if last_bull_candle else
                        last_bear_candle if last_bear_candle else "—")
        rc5, rc6, rc7, rc8 = st.columns(4)
        rc5.metric("📉 RSI",    f"{rsi_icon} {rsi_val:.1f}")
        rc6.metric("📈 MACD",   f"{macd_icon} {macd_val:.2f}")
        rc7.metric("📐 Trend",  smc_trend)
        rc8.metric("🕯️ Candle", candle_label)

        st.divider()

        # ── 2. CHART — shown first so user sees it immediately ────────────────
        st.subheader("📈 Live Chart")
        vol_tag = "🔥 Volume Spike!" if vol_spike else "📊 Normal Volume"
        pat_tag = f"🕯️ {last_bull_candle}" if last_bull_candle else                   f"🕯️ {last_bear_candle}" if last_bear_candle else "🕯️ No Pattern"
        st.caption(
            f"🟢 ▲ = Strong Buy  ·  🔴 ▼ = Strong Sell  ·  "
            f"Green zone = Profit  ·  Red zone = Risk  ·  "
            f"{vol_tag}  ·  {pat_tag}"
        )
        # Lock TP/SL when a new signal fires for this ticker
        # If signal is WAIT but we had a previous signal on same ticker,
        # keep showing the locked TP/SL zones (constant until TP/SL is hit)
        current_price = float(df['Close'].iloc[-1])

        if action in ('buy', 'sell'):
            # New signal — lock it
            st.session_state.locked_trade  = trade.copy()
            st.session_state.locked_action = action
            st.session_state.locked_ticker = ticker
            chart_trade  = trade
            chart_action = action
        elif (st.session_state.locked_trade is not None and
              st.session_state.locked_ticker == ticker):
            # No new signal — check if price hit TP or SL
            lt = st.session_state.locked_trade
            la = st.session_state.locked_action
            tp_hit = (la == 'buy'  and current_price >= lt['tp_buy'])  or                      (la == 'sell' and current_price <= lt['tp_sell'])
            sl_hit = (la == 'buy'  and current_price <= lt['sl_buy'])  or                      (la == 'sell' and current_price >= lt['sl_sell'])

            if tp_hit or sl_hit:
                # TP or SL reached — clear the lock
                hit_label = "✅ TP Hit!" if tp_hit else "❌ SL Hit!"
                st.toast(f"{hit_label} Trade closed.", icon="🔔")
                st.session_state.locked_trade  = None
                st.session_state.locked_action = 'none'
                chart_trade  = trade
                chart_action = 'none'
            else:
                # Still active — keep showing locked zones
                chart_trade  = lt
                chart_action = la
                st.caption(
                    f"🔒 **Locked trade active** — "
                    f"Entry ₹{lt['price']:,.0f}  |  "
                    f"TP ₹{lt.get('tp_buy' if la=='buy' else 'tp_sell', 0):,.0f}  |  "
                    f"SL ₹{lt.get('sl_buy' if la=='buy' else 'sl_sell', 0):,.0f}"
                )
        else:
            chart_trade  = trade
            chart_action = 'none'

        fig = build_chart(df, chart_trade, chart_action)
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # ── 3. FINAL DECISION — one clear box, no clutter ────────────────────
        st.subheader("🎯 Final Decision")

        # Big colored decision box
        passed = sum(1 for r in reasons if r.startswith("✅"))
        total  = len(reasons)

        if score >= 4:
            st.success(f"## 🟢 BUY  —  {passed}/{total} checks passed")
        elif score >= 2:
            st.warning(f"## 🟡 BUY  —  {passed}/{total} checks passed  (wait for 1 more)")
        elif score <= -4:
            st.error(f"## 🔴 SELL  —  {passed}/{total} checks passed")
        elif score <= -2:
            st.warning(f"## 🟠 SELL  —  {passed}/{total} checks passed  (wait for 1 more)")
        else:
            st.info(f"## ⚪ WAIT  —  {passed}/{total} checks passed  (not enough signal)")

        # Confidence bar — show PASSED checks out of 7, not raw score
        # e.g. score=2 means 2 bullish - 0 bearish = net +2, but 5 checks passed
        passed = sum(1 for r in reasons if r.startswith("✅"))
        total  = len(reasons)
        st.caption(f"Confirmations Passed: {passed} / {total}")
        st.progress(passed / total if total > 0 else 0)

        # Reasons in a clean table — not a wall of text
        st.caption("**Why this decision:**")
        cols = st.columns(2)
        for idx, reason in enumerate(reasons):
            cols[idx % 2].caption(reason)

        st.divider()

        # ── 4. TRADE SETUP — always matches decision ──────────────────────────
        st.subheader("🛒 Trade Setup")
        if action == 'buy':
            t1, t2, t3 = st.columns(3)
            t1.metric("🟡 Entry",      f"₹{trade['price']:.2f}")
            t2.metric("✅ Target (TP)", f"₹{trade['tp_buy']:.2f}",
                      delta=f"+₹{trade['tp_buy']-trade['price']:.2f}")
            t3.metric("❌ Stop Loss",   f"₹{trade['sl_buy']:.2f}",
                      delta=f"-₹{trade['price']-trade['sl_buy']:.2f}",
                      delta_color="inverse")
            st.success("Risk/Reward = 2 : 1  |  Exit if price closes below Stop Loss")

        elif action == 'sell':
            t1, t2, t3 = st.columns(3)
            t1.metric("🟡 Entry",      f"₹{trade['price']:.2f}")
            t2.metric("✅ Target (TP)", f"₹{trade['tp_sell']:.2f}",
                      delta=f"-₹{trade['price']-trade['tp_sell']:.2f}")
            t3.metric("❌ Stop Loss",   f"₹{trade['sl_sell']:.2f}",
                      delta=f"+₹{trade['sl_sell']-trade['price']:.2f}",
                      delta_color="inverse")
            st.error("Risk/Reward = 2 : 1  |  Exit if price closes above Stop Loss")

        else:
            st.info("⚪ No trade setup right now. Score must reach ≥ 2 (buy) or ≤ -2 (sell).")

        # F&O suggestion — ONLY in F&O mode
        if is_fo:
            st.divider()
            show_fo_suggestion(action, last_strength, price, score)

        st.divider()
        st.caption("⚠️ Disclaimer: For learning only. Not financial advice.")

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.info("💡 Use .NS suffix: INFY.NS, TCS.NS, RELIANCE.NS")
        with st.expander("🔧 Error details"):
            st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────

if analyze_btn:
    run_analysis()
elif auto_refresh:
    run_analysis()
    time.sleep(5)
    st.rerun()
else:
    st.info(
        "👈 **Select your symbol in the sidebar and click Analyze**\n\n"
        "**Stock Mode** → Full stock analysis only (no F&O clutter)\n\n"
        "**F&O Mode** → Index/options analysis with CE/PE suggestions\n\n"
        "Indicators: RSI · MA20 · MA50 · MACD · VWAP · Bollinger · "
        "Fibonacci · BOS · CHoCH · Order Blocks · FVG"
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Indicators", "14")
    c2.metric("Signal Layers",    "8")
    c3.metric("Timeframes",       "4")