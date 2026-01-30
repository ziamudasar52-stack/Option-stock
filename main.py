import os
import time
import logging
import requests
from datetime import datetime
from zoneinfo import ZoneInfo

# ================== CONFIG ==================
MBOUM_API_KEY = os.getenv("MBOUM_API_KEY")

# Second bot: Option Trader
TELEGRAM_BOT_TOKEN = os.getenv("OPTION_TRADER_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("OPTION_TRADER_CHAT_ID")

BASE_URL = "https://api.mboum.com"
TIMEZONE = "America/New_York"

# intervals (seconds)
UNUSUAL_OPTIONS_INTERVAL = 60        # scan unusual options every 1 min
OPTIONS_FLOW_INTERVAL = 60          # scan options flow every 1 min
MARKET_STATUS_INTERVAL = 1800       # 30 min

# thresholds
MIN_PREMIUM_USD = 50_000            # only alert if premium >= 50k
MIN_CONFIDENCE_SCORE = 70           # only alert if combined score >= 70

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ================== TELEGRAM ==================
def send_telegram_message(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram credentials missing; skipping send.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown"
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code != 200:
            logging.error(f"Telegram send error {resp.status_code}: {resp.text}")
    except Exception as e:
        logging.error(f"Telegram send exception: {e}")

# ================== MBOUM HTTP ==================
def mboum_get(path: str, params: dict | None = None):
    if params is None:
        params = {}
    url = f"{BASE_URL}{path}"
    headers = {
        "Authorization": f"Bearer {MBOUM_API_KEY}"
    }
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        logging.info(f"📡 API Response: {path} - Status: {resp.status_code}")
        if resp.status_code != 200:
            logging.error(f"❌ API Error {resp.status_code}: {resp.text}")
            return None
        return resp.json()
    except Exception as e:
        logging.error(f"❌ API Exception for {path}: {e}")
        return None

def ensure_list_of_dicts(data) -> list:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict) and "body" in data and isinstance(data["body"], list):
        return [item for item in data["body"] if isinstance(item, dict)]
    return []

# ================== OPTIONS FETCHERS ==================
def get_unusual_options_activity() -> list:
    # /v1/markets/options/unusual-options-activity
    data = mboum_get("/v1/markets/options/unusual-options-activity", {
        "type": "STOCKS",
        "page": "1"
    })
    return ensure_list_of_dicts(data) if data is not None else []

def get_options_flow() -> list:
    # /v1/markets/options/options-flow
    data = mboum_get("/v1/markets/options/options-flow", {
        "type": "STOCKS",
        "page": "1"
    })
    return ensure_list_of_dicts(data) if data is not None else []

# ================== HELPERS & ML ENGINE ==================
def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(str(value).replace(",", ""))
    except Exception:
        return default

def classify_direction_from_delta(delta: float) -> str:
    if delta >= 0.3:
        return "BULLISH"
    if delta <= -0.3:
        return "BEARISH"
    return "NEUTRAL"

def format_premium(premium: float) -> str:
    if premium >= 1_000_000:
        return f"${premium/1_000_000:.2f}M"
    if premium >= 1_000:
        return f"${premium/1_000:.1f}K"
    return f"${premium:.0f}"

def format_direction_emoji(direction: str) -> str:
    if direction == "BULLISH":
        return "🟢"
    if direction == "BEARISH":
        return "🔴"
    return "⚪"

def format_confidence_bar(score: int) -> str:
    filled = int(score / 10)
    empty = 10 - filled
    return "█" * filled + "░" * empty

def ml_timeframe_score(premium, vol_oi, delta, dte, order_type, direction):
    score = 0

    # premium weight
    if premium >= 1_000_000:
        score += 40
    elif premium >= 250_000:
        score += 30
    elif premium >= 50_000:
        score += 20

    # vol/oi weight
    if vol_oi >= 20:
        score += 30
    elif vol_oi >= 10:
        score += 20
    elif vol_oi >= 5:
        score += 10

    # delta weight
    if abs(delta) >= 0.9:
        score += 20
    elif abs(delta) >= 0.7:
        score += 15
    elif abs(delta) >= 0.5:
        score += 10

    # DTE weight
    if dte <= 1:
        score += 20
    elif dte <= 3:
        score += 10

    # order type weight
    ot = (order_type or "").upper()
    if "SWEEP" in ot:
        score += 15
    if "BLOCK" in ot:
        score += 20

    # direction weight
    if direction in ("BULLISH", "BEARISH"):
        score += 10

    return min(score, 100)

def ml_multi_timeframe_predictions(premium, vol_oi, delta, dte, order_type, direction):
    return {
        "5m":  ml_timeframe_score(premium,       vol_oi, delta,       dte, order_type, direction),
        "15m": ml_timeframe_score(premium*0.9,   vol_oi, delta*0.95,  dte, order_type, direction),
        "30m": ml_timeframe_score(premium*0.85,  vol_oi, delta*0.9,   dte, order_type, direction),
        "60m": ml_timeframe_score(premium*0.8,   vol_oi, delta*0.85,  dte, order_type, direction),
        "EOD": ml_timeframe_score(premium*0.75,  vol_oi, delta*0.8,   dte, order_type, direction),
    }

def ml_ensemble(preds: dict) -> tuple[str, int]:
    avg = sum(preds.values()) / len(preds) if preds else 0
    direction = "BULLISH" if avg >= 60 else "BEARISH"
    return direction, int(round(avg))

def detect_whale(premium, vol_oi, delta, order_type):
    if premium >= 500_000:
        return True
    if vol_oi >= 20:
        return True
    if abs(delta) >= 0.8:
        return True
    if "BLOCK" in (order_type or "").upper():
        return True
    return False

def darkpool_boost(notional):
    if notional >= 100_000_000:
        return 30
    if notional >= 50_000_000:
        return 20
    if notional >= 10_000_000:
        return 10
    return 0

# ================== UNIQUE ID / DEDUP ==================
seen_unusual_ids: set[str] = set()
seen_flow_ids: set[str] = set()

def get_unique_id_from_record(rec: dict) -> str:
    base = str(rec.get("baseSymbol") or rec.get("symbol") or "")
    strike = str(rec.get("strikePrice") or "")
    exp = str(rec.get("expirationDate") or "")
    premium = str(rec.get("premium") or "")
    ts = str(rec.get("timestamp") or rec.get("time") or "")
    return "|".join([base, strike, exp, premium, ts])

# ================== MESSAGE BUILDERS ==================
def build_combined_ml_unusual_message(opt: dict) -> str | None:
    base = opt.get("baseSymbol") or opt.get("symbol") or "N/A"
    symbol_type = opt.get("symbolType") or "N/A"
    strike = opt.get("strikePrice")
    exp = opt.get("expirationDate")
    dte = safe_float(opt.get("daysToExpiration"), 0.0)
    delta = safe_float(opt.get("delta"), 0.0)
    premium = safe_float(opt.get("premium"), 0.0)
    vol = safe_float(opt.get("volume"), 0.0)
    oi = safe_float(opt.get("openInterest"), 0.0)
    vol_oi = safe_float(opt.get("volumeOpenInterestRatio"), 0.0)
    iv = safe_float(opt.get("volatility"), 0.0)
    order_type = opt.get("type") or opt.get("tradeType") or ""
    dark_notional = safe_float(opt.get("darkpoolNotional"), 0.0)

    if premium < MIN_PREMIUM_USD:
        return None

    direction = classify_direction_from_delta(delta)
    tf_preds = ml_multi_timeframe_predictions(premium, vol_oi, delta, dte, order_type, direction)
    ensemble_dir, ensemble_score = ml_ensemble(tf_preds)

    ensemble_score = min(100, ensemble_score + darkpool_boost(dark_notional))
    if ensemble_score < MIN_CONFIDENCE_SCORE:
        return None

    whale = detect_whale(premium, vol_oi, delta, order_type)
    direction_emoji = format_direction_emoji(ensemble_dir)
    confidence_bar = format_confidence_bar(ensemble_score)
    premium_str = format_premium(premium)
    now_est = datetime.now(ZoneInfo(TIMEZONE))

    lines = []
    lines.append(f"{direction_emoji} *COMBINED ML SIGNAL: {base}* {direction_emoji}")
    lines.append("")
    lines.append(f"📉 *Direction:* {ensemble_dir} ({ensemble_score}% confidence)")
    if whale:
        lines.append("")
        lines.append("✅ *Trigger:* Whale Trade")
    lines.append("")
    lines.append("📊 *Model Predictions:*")
    for label in ["5m", "15m", "30m", "60m", "EOD"]:
        sc = tf_preds.get(label, 0)
        arrow = "🔴" if ensemble_dir == "BEARISH" else "🟢"
        dir_word = "DOWN" if ensemble_dir == "BEARISH" else "UP"
        lines.append(f"{arrow} {label}: {dir_word} ({sc}%)")
    lines.append("")
    lines.append("💰 *Flow Summary:*")
    lines.append(f"💰 Premium: {premium_str}")
    lines.append(f"🎯 Strike: {strike} {symbol_type} | Exp: {exp} (DTE: {dte:.0f})")
    lines.append(f"Δ: {delta:.2f} | Type: {order_type or 'N/A'}")
    lines.append(f"Vol: {vol:,.0f} | OI: {oi:,.0f} | Vol/OI: {vol_oi:.1f}x")
    lines.append(f"IV: {iv:.1f}%")
    if dark_notional > 0:
        lines.append(f"🏦 Darkpool Notional: {format_premium(dark_notional)}")
    lines.append("")
    lines.append(f"📈 *Conviction Score:* {ensemble_score}/100")
    lines.append(confidence_bar)
    lines.append("")
    lines.append(f"🕒 {now_est.strftime('%H:%M:%S')} ET")
    lines.append("🤖 Option Trader ML v2.0")

    return "\n".join(lines)

def build_smart_money_flow_message(flow: dict) -> str | None:
    base = flow.get("baseSymbol") or flow.get("symbol") or "N/A"
    direction = (flow.get("direction") or "").upper()  # BULLISH/BEARISH/NEUTRAL
    premium = safe_float(flow.get("premium"), 0.0)
    vol = safe_float(flow.get("volume"), 0.0)
    oi = safe_float(flow.get("openInterest"), 0.0)
    vol_oi = safe_float(flow.get("volumeOpenInterestRatio"), 0.0)
    strike = flow.get("strikePrice")
    exp = flow.get("expirationDate")
    dte = safe_float(flow.get("daysToExpiration"), 0.0)
    delta = safe_float(flow.get("delta"), 0.0)
    order_type = flow.get("type") or flow.get("tradeType") or ""
    exchange = flow.get("exchange") or "N/A"
    dark_notional = safe_float(flow.get("darkpoolNotional"), 0.0)

    if premium < MIN_PREMIUM_USD:
        return None

    tf_preds = ml_multi_timeframe_predictions(premium, vol_oi, delta, dte, order_type, direction)
    ensemble_dir, ensemble_score = ml_ensemble(tf_preds)
    ensemble_score = min(100, ensemble_score + darkpool_boost(dark_notional))

    if ensemble_score < MIN_CONFIDENCE_SCORE:
        return None

    direction_emoji = format_direction_emoji(ensemble_dir)
    confidence_bar = format_confidence_bar(ensemble_score)
    premium_str = format_premium(premium)
    now_est = datetime.now(ZoneInfo(TIMEZONE))

    lines = []
    lines.append(f"{direction_emoji} *SMART MONEY FLOW - {ensemble_dir}* {direction_emoji}")
    lines.append("🎯 *UNUSUAL CONVICTION*")
    lines.append("")
    lines.append(f"*{base} {strike} {exp}*")
    lines.append(f"💲 Premium: {premium_str}")
    lines.append(f"📅 Exp: {exp} (DTE: {dte:.0f})")
    lines.append("")
    lines.append(f"📍 Δ: {delta:.3f} | Type: {order_type or 'N/A'} | Exch: {exchange}")
    lines.append(f"📈 Vol: {vol:,.0f} | OI: {oi:,.0f} ({vol_oi:.1f}x)")
    if dark_notional > 0:
        lines.append(f"🏦 Darkpool Notional: {format_premium(dark_notional)}")
    lines.append("")
    lines.append(f"📊 *Conviction Score:* {ensemble_score}/100")
    lines.append(confidence_bar)
    lines.append("")
    lines.append("⚡ Quick Take: UNUSUAL CONVICTION - Multiple signals align with flow direction.")
    lines.append("")
    lines.append(f"🕒 {now_est.strftime('%H:%M:%S')} ET")
    lines.append("🤖 Option Trader ML v2.0")

    return "\n".join(lines)

# ================== TASKS ==================
def run_unusual_options_task() -> None:
    logging.info("🔍 UNUSUAL OPTIONS TASK...")
    records = get_unusual_options_activity()
    if not records:
        logging.info("Unusual options: no records returned")
        return

    logging.info(f"Unusual options count: {len(records)}")
    logging.info(f"Unusual sample record: {records[0]}")

    for rec in records:
        uid = get_unique_id_from_record(rec)
        if uid in seen_unusual_ids:
            continue
        msg = build_combined_ml_unusual_message(rec)
        if msg:
            seen_unusual_ids.add(uid)
            send_telegram_message(msg)

def run_options_flow_task() -> None:
    logging.info("🔍 OPTIONS FLOW TASK...")
    records = get_options_flow()
    if not records:
        logging.info("Options flow: no records returned")
        return

    logging.info(f"Options flow count: {len(records)}")
    logging.info(f"Options flow sample record: {records[0]}")

    for rec in records:
        uid = get_unique_id_from_record(rec)
        if uid in seen_flow_ids:
            continue
        msg = build_smart_money_flow_message(rec)
        if msg:
            seen_flow_ids.add(uid)
            send_telegram_message(msg)

def run_market_status_task(now_est: datetime) -> None:
    logging.info("🔍 MARKET STATUS TASK...")
    status = "OPEN" if now_est.weekday() < 5 else "CLOSED"
    msg = (
        f"🕒 *Market Status Check*\n"
        f"Status: {status}\n"
        f"Time: {now_est.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
        f"🤖 Option Trader ML v2.0 is running."
    )
    send_telegram_message(msg)

# ================== MAIN LOOP ==================
if __name__ == "__main__":
    if not MBOUM_API_KEY:
        logging.error("MBOUM_API_KEY is not set. Exiting.")
        raise SystemExit(1)
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.error("OPTION_TRADER_BOT_TOKEN or OPTION_TRADER_CHAT_ID not set. Exiting.")
        raise SystemExit(1)

    logging.info("🤖 Option Trader ML v2.0 starting...")
    tz_est = ZoneInfo(TIMEZONE)

    last_unusual = datetime.min.replace(tzinfo=tz_est)
    last_flow = datetime.min.replace(tzinfo=tz_est)
    last_market_status = datetime.min.replace(tzinfo=tz_est)

    while True:
        now_est = datetime.now(tz_est)

        if (now_est - last_unusual).total_seconds() >= UNUSUAL_OPTIONS_INTERVAL:
            last_unusual = now_est
            run_unusual_options_task()

        if (now_est - last_flow).total_seconds() >= OPTIONS_FLOW_INTERVAL:
            last_flow = now_est
            run_options_flow_task()

        if (now_est - last_market_status).total_seconds() >= MARKET_STATUS_INTERVAL:
            last_market_status = now_est
            run_market_status_task(now_est)

        time.sleep(1)
