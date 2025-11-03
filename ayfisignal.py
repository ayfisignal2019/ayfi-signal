#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OMEGA v25 - OMNISCIENT ENGINE (FINAL VERSION)
- ALL 5 ULTRA UPGRADES + 3 OMNISCIENT UPGRADES
- STRUCTURE 100% PRESERVED - NO LINE DELETED
- Volatility Regime | Dynamic Stop/Target | Signal Clustering
"""
import os, sys, time, math, asyncio, json, logging, random, signal, sqlite3, subprocess
from datetime import datetime, timedelta
from collections import defaultdict, deque
from logging.handlers import RotatingFileHandler
from functools import partial

# Networking & async
try:
    import aiohttp
except Exception:
    aiohttp = None
import requests
import yfinance as yf

# Data science & indicators
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from cachetools import TTLCache
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ML - XGBoost + CatBoost Ensemble
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False
    from sklearn.ensemble import RandomForestClassifier
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False
from sklearn.preprocessing import StandardScaler
import joblib

# Optionals
try:
    import websockets
except Exception:
    websockets = None
try:
    import torch, torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# dotenv
from dotenv import load_dotenv
load_dotenv(".env")

# ---------------- Paths & Logging ----------------
BASE_DIR = os.getenv("BASE_DIR", os.path.join(os.getcwd(), "omega_v25"))
os.makedirs(BASE_DIR, exist_ok=True)
LOG_FILE = os.path.join(BASE_DIR, "omega_v25.log")
DB_FILE = os.path.join(BASE_DIR, "omega_v25.db")
MODEL_FILE = os.path.join(BASE_DIR, "omega_v25_online_model.pkl")
CATMODEL_FILE = os.path.join(BASE_DIR, "omega_v25_cat_model.pkl")
SIGNALS_FILE = os.path.join(BASE_DIR, "signals_history.json")

logger = logging.getLogger("omega_v25")
logger.setLevel(logging.INFO)
fh = RotatingFileHandler(LOG_FILE, maxBytes=80*1024*1024, backupCount=10)
fmt = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
fh.setFormatter(fmt)
logger.addHandler(fh)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.info("OMEGA v25 OMNISCIENT booting...")

# ---------------- Smart Env / Keys ----------------
import os
import itertools
import random

def read_keys(env):
    """Read keys from environment and return a list of non-empty stripped values."""
    v = os.getenv(env, "").strip()
    return [x.strip() for x in v.split(",") if x.strip()]

# ---------------- API Keys ----------------
# Rotating iterators to avoid daily limits
ALPHA_KEYS = itertools.cycle(read_keys("ALPHA_VANTAGE_KEY"))
TWELVE_KEYS = itertools.cycle(read_keys("TWELVEDATA_KEY"))
FINNHUB_KEYS = itertools.cycle(read_keys("FINNHUB_KEY"))
STOCKDATA_KEYS = itertools.cycle(read_keys("STOCKDATA_KEY"))
EXCHANGERATES_KEYS = itertools.cycle(read_keys("EXCHANGERATES_KEY"))
FOREXRATE_KEYS = itertools.cycle(read_keys("FOREXRATE_KEY"))
ABSTRACT_KEYS = itertools.cycle(read_keys("ABSTRACTAPI_KEY"))
FINAGE_KEYS = itertools.cycle(read_keys("FINAGE_KEY"))

# ---------------- Public / Free Reliable Sources ----------------
# 8 sources, live, free, no key needed
PUBLIC_SOURCES = [
    "https://api.freecurrencyapi.com/v1/latest",   # FreeCurrencyAPI – FX
    "https://api.coingecko.com/api/v3/simple/price",  # CoinGecko – Crypto
    "https://query1.finance.yahoo.com/v7/finance/quote", # Yahoo Finance – Stocks, Crypto
    "https://api.exchangerate.host/latest",        # ExchangeRatesAPI – FX
    "https://api.frankfurter.app/latest",          # ECB FX rates
    "https://api.binance.com/api/v3/ticker/price", # Binance – Crypto
    "https://api.metals-api.com/v1/latest",        # Metals (XAU, XAG)
    "https://api.coinpaprika.com/v1/tickers"       # CoinPaprika – Crypto
]

# ---------------- Notifications ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
GMAIL_USER = os.getenv("GMAIL_USER", "").strip()
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "").strip()

# ---------------- Engine Parameters ----------------
CYCLE_TIME = int(os.getenv("CYCLE_TIME", "60"))
ROTATION_SIZE = int(os.getenv("ROTATION_SIZE", "40"))
TOP_N = int(os.getenv("TOP_N", "6"))
MIN_CONFIDENCE_BASE = float(os.getenv("MIN_CONFIDENCE", "0.65"))
ACCOUNT_BALANCE = float(os.getenv("ACCOUNT_BALANCE", "10000"))
ENABLE_PREFETCH = os.getenv("ENABLE_PREFETCH", "true").lower() in ("1", "true", "yes")
ENABLE_WEBSOCKETS = os.getenv("ENABLE_WEBSOCKET", "true").lower() in ("1", "true", "yes")
PREFETCH_INTERVAL = int(os.getenv("PREFETCH_INTERVAL", "1800"))
SELF_UPDATE_REPO = os.getenv("GIT_REPO", "").strip()

# ---------------- Helper Functions ----------------
def get_next_key(source):
    """
    Get the next available API key from a rotating iterator.
    If no key available, returns None.
    """
    try:
        return next(source)
    except StopIteration:
        return None

def get_all_public_sources(shuffle=True):
    """
    Return the list of public sources for fallback.
    Can shuffle to avoid hammering the same source first.
    """
    sources = PUBLIC_SOURCES.copy()
    if shuffle:
        random.shuffle(sources)
    return sources

def pick_best_source(sources):
    """
    Returns the first responsive source from a list.
    This can be used for fallback logic.
    """
    for src in sources:
        try:
            # In actual implementation, you would test connectivity or status here
            # For now, we assume they are live
            return src
        except Exception:
            continue
    return None
# ---------------- Universe ----------------
CURRENCIES = os.getenv("CURRENCIES", "USD,EUR,GBP,JPY,AUD,CAD,CHF,NZD,SEK,NOK,DKK,TRY,MXN,SGD,HKD,ZAR").split(",")
CURRENCIES = [c.strip() for c in CURRENCIES if c.strip()]

def generate_dynamic_universe():
    pairs = []
    for a in CURRENCIES:
        for b in CURRENCIES:
            if a == b: continue
            pairs.append(f"{a}/{b}")
    return pairs

ALL_PAIRS = generate_dynamic_universe()

DEFAULT_ALL_ASSETS = [
    "XAU/USD", "XAG/USD", "WTI/USD", "BRENT/USD", "SPX500/USD", "NAS100/USD", "DJ30/USD",
    "BTC/USD", "ETH/USD"
]
ALL_ASSETS = os.getenv("ALL_ASSETS", ",".join(DEFAULT_ALL_ASSETS)).split(",")
ALL_ASSETS = [c.strip() for c in ALL_ASSETS if c.strip()]

UNIVERSE = list(dict.fromkeys(ALL_PAIRS + ALL_ASSETS))

# ---------------- Persistence: SQLite ----------------
def init_db():
    con = sqlite3.connect(DB_FILE, check_same_thread=False)
    cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT, direction TEXT, entry REAL, target REAL, stop REAL,
        confidence REAL, leverage REAL, suggested_capital REAL, lots REAL,
        ts TEXT, valid_from TEXT, valid_to TEXT, sources TEXT, outcome INTEGER DEFAULT 0
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS provider_stats (
        provider TEXT PRIMARY KEY, requests INTEGER, failures INTEGER, last_seen TEXT
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS signal_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT, signal_id INTEGER, pnl REAL, closed_ts TEXT
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS model_performance (
        model TEXT, correct INTEGER, total INTEGER, last_update TEXT
    )""")
    con.commit()
    return con

DB = init_db()

# ---------------- Smart Caches & Key Management ----------------
import time
import asyncio
from collections import defaultdict, deque
from cachetools import TTLCache

# Assumptions: ALPHA_KEYS, TWELVE_KEYS, FINNHUB_KEYS, etc. already defined as iterators
# OHLC_CACHE and PRICE_CACHE already exist

# ---------------- Dynamic TTL ----------------
def dynamic_ttl(source_type="public"):
    """Return TTL in seconds depending on source type"""
    if source_type == "public":
        return 5  # very fresh for public sources
    elif source_type == "private":
        return 900  # longer for personal API
    else:
        return 60  # default

PRICE_CACHE = TTLCache(maxsize=40000, ttl=dynamic_ttl("private"))
OHLC_CACHE = TTLCache(maxsize=20000, ttl=dynamic_ttl("private"))

HIST = defaultdict(float)
SIGNALS_HISTORY = deque(maxlen=10000)

# ---------------- Model Performance ----------------
MODEL_PERF = defaultdict(lambda: {"correct": 0, "total": 0, "last_update": time.time()})

def update_model_perf(model, correct, weight=1.0):
    perf = MODEL_PERF[model]
    perf["total"] += weight
    if correct:
        perf["correct"] += weight
    perf["last_update"] = time.time()
    # Optionally persist to DB if DB exists
    try:
        cur = DB.cursor()
        now = datetime.utcnow().isoformat()
        cur.execute("INSERT OR REPLACE INTO model_performance VALUES (?,?,?,?)",
                    (model, perf["correct"], perf["total"], now))
        DB.commit()
    except:
        pass

def get_model_weight(model):
    perf = MODEL_PERF[model]
    if perf["total"] == 0:
        return 0.5
    return perf["correct"] / perf["total"]

# ---------------- Key Pools & Smart Rotation ----------------
KEY_POOLS = {
    "alpha": ALPHA_KEYS, "twelve": TWELVE_KEYS, "finnhub": FINNHUB_KEYS,
    "stockdata": STOCKDATA_KEYS, "exchangerates": EXCHANGERATES_KEYS,
    "forexrate": FOREXRATE_KEYS, "abstract": ABSTRACT_KEYS, "finage": FINAGE_KEYS
}
KEY_LIMITS = {k: int(os.getenv(f"LIMIT_{k.upper()}", "20")) for k in KEY_POOLS.keys()}
KEY_COUNTER = {p: {k: 0 for k in (KEY_POOLS.get(p) or [])} for p in KEY_POOLS}
KEY_PAUSED = {p: {} for p in KEY_POOLS}

async def pick_key(provider):
    """
    Smart key picker:
    - Skips paused keys
    - Rotates keys
    - Avoids hitting daily limit
    - Returns key or None if no key available
    """
    keys = list(KEY_POOLS.get(provider) or [])
    if not keys:
        return None
    now = time.time()
    avail = [k for k in keys if KEY_PAUSED[provider].get(k, 0) < now]
    if not avail:
        # reset all if exhausted
        for k in keys:
            KEY_COUNTER[provider][k] = 0
            KEY_PAUSED[provider][k] = 0
        avail = keys
    # pick key with lowest usage ratio
    best = min(avail, key=lambda k: KEY_COUNTER[provider].get(k, 0)/max(1, KEY_LIMITS.get(provider, 20)))
    KEY_COUNTER[provider][best] += 1
    # pause if limit reached
    if KEY_COUNTER[provider][best] >= KEY_LIMITS.get(provider, 20):
        KEY_PAUSED[provider][best] = now + 60  # pause for 1 min before reuse
    return best

# ---------------- Periodic Counter Reset ----------------
async def reset_counters_task():
    while True:
        await asyncio.sleep(60)
        for p in KEY_COUNTER:
            for k in KEY_COUNTER[p]:
                # decay counters slightly for fairness
                KEY_COUNTER[p][k] = max(0, KEY_COUNTER[p][k] - 1)

# ---------------- Smart Cache Update ----------------
def smart_cache_update(cache, key, value, source_type="private"):
    """
    Update TTLCache with dynamic TTL depending on source type
    """
    ttl = dynamic_ttl(source_type)
    if isinstance(cache, TTLCache):
        cache.ttl = ttl
        cache[key] = value

# ---------------- Smart Provider Wrappers (REST, Parallel & Fallback) ----------------
import asyncio
import random
import aiohttp
import time
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

# فرض: KEY_POOLS, KEY_LIMITS, KEY_COUNTER, KEY_PAUSED, pick_key, PUBLIC_SOURCES, OHLC_CACHE, logger
# قبلاً در بخش Smart Keys و Caches تعریف شده‌اند

async def fetch_from_public(session, pair, sources):
    """
    Read from multiple public sources in parallel.
    Returns list of valid price dicts or empty list.
    """
    tasks = [fetch_public_source(session, pair, src) for src in sources]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    valid_results = [r for r in results if isinstance(r, dict) and "price" in r and r["price"] is not None]
    return valid_results

async def fetch_public_source(session, pair, source):
    """
    Read price from a single public source.
    """
    try:
        url = source
        if "{pair}" in url:
            url = url.format(pair=pair.replace("/", ""))
        async with session.get(url, timeout=10) as resp:
            if resp.status == 200:
                data = await resp.json()
                price = extract_price(data, pair)
                return {"source": source, "price": price}
    except Exception:
        return None

def extract_price(data, pair):
    """
    Extract price from public source JSON response.
    Customize per API structure.
    """
    try:
        if "market_data" in data:
            return data["market_data"]["current_price"]["usd"]
        elif "price" in data:
            return float(data["price"])
        elif pair in data:
            return float(data[pair]["usd"])
    except Exception:
        return None
    return None

async def fetch_from_private(pair, keys_iterators, count=1):
    """
    Read price from personal API keys (fallback).
    'count' specifies how many keys to try.
    """
    results = []
    for _ in range(count):
        for key_iter in keys_iterators:
            try:
                key = next(key_iter)
                price = await fetch_private_api(pair, key)
                if price is not None:
                    results.append({"key": key, "price": price})
                    break
            except Exception:
                continue
        if results:
            break
    return results

async def fetch_private_api(pair, key):
    """
    Placeholder for real API call to Alpha, TwelveData, Finnhub, etc.
    Replace with actual REST request and parsing.
    """
    await asyncio.sleep(0.05)  # simulate network
    return random.uniform(100, 200)  # replace with real API data

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6))
async def prov_alpha(session, pair):
    """
    Smart AlphaVantage wrapper.
    """
    async with session:
        public_sources = random.sample(PUBLIC_SOURCES, min(4, len(PUBLIC_SOURCES)))
        public_data = await fetch_from_public(session, pair, public_sources)
        if public_data:
            private_data = await fetch_from_private(pair, [ALPHA_KEYS], count=1)
        else:
            private_data = await fetch_from_private(pair, [ALPHA_KEYS], count=2)

        final_price = None
        if public_data:
            final_price = sum(d["price"] for d in public_data) / len(public_data)
        if private_data:
            final_price = private_data[0]["price"] if final_price is None else (final_price + private_data[0]["price"]) / 2

        if final_price is not None:
            OHLC_CACHE[f"ohlc:{pair}:alpha"] = final_price
        return final_price, 0.1

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6))
async def prov_twelve(session, pair):
    """
    Smart TwelveData wrapper.
    """
    async with session:
        public_sources = random.sample(PUBLIC_SOURCES, min(4, len(PUBLIC_SOURCES)))
        public_data = await fetch_from_public(session, pair, public_sources)
        if public_data:
            private_data = await fetch_from_private(pair, [TWELVE_KEYS], count=1)
        else:
            private_data = await fetch_from_private(pair, [TWELVE_KEYS], count=2)

        final_price = None
        if public_data:
            final_price = sum(d["price"] for d in public_data) / len(public_data)
        if private_data:
            final_price = private_data[0]["price"] if final_price is None else (final_price + private_data[0]["price"]) / 2

        if final_price is not None:
            OHLC_CACHE[f"ohlc:{pair}:twelve"] = final_price
        return final_price, 0.1

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6))
async def prov_finnhub(session, pair):
    """
    Smart Finnhub wrapper.
    """
    async with session:
        public_sources = random.sample(PUBLIC_SOURCES, min(4, len(PUBLIC_SOURCES)))
        public_data = await fetch_from_public(session, pair, public_sources)
        if public_data:
            private_data = await fetch_from_private(pair, [FINNHUB_KEYS], count=1)
        else:
            private_data = await fetch_from_private(pair, [FINNHUB_KEYS], count=2)

        final_price = None
        if public_data:
            final_price = sum(d["price"] for d in public_data) / len(public_data)
        if private_data:
            final_price = private_data[0]["price"] if final_price is None else (final_price + private_data[0]["price"]) / 2

        if final_price is not None:
            OHLC_CACHE[f"ohlc:{pair}:finnhub"] = final_price
        return final_price, 0.1

async def prov_exchangerate_host(session, pair):
    """
    Public-only FX wrapper (no key needed).
    """
    try:
        a, b = pair.split("/")
        url = f"https://api.exchangerate.host/latest?base={a}&symbols={b}"
        async with session.get(url, timeout=6) as r:
            j = await r.json()
            rate = j.get("rates", {}).get(b)
            if rate:
                OHLC_CACHE[f"ohlc:{pair}:exchangerate_host"] = rate
                return float(rate), 0
    except Exception:
        pass
    return None, 0

async def prov_yf_generic(session, symbol):
    """
    Yahoo Finance wrapper (fallback / public).
    """
    try:
        cand = [symbol.replace("/", "") + "=X", symbol.replace("/", ""), symbol]
        for c in cand:
            try:
                df = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: yf.download(c, period="2d", interval="1h", progress=False)
                )
                if df is not None and not df.empty:
                    OHLC_CACHE[f"ohlc:{symbol}:yf"] = float(df['Close'].iloc[-1])
                    return float(df['Close'].iloc[-1]), 0.1
            except Exception:
                continue
    except Exception:
        pass
    return None, 0
# ---------------- Smart Fusion & Adaptive Weighting ----------------
import asyncio
import random
import aiohttp
import numpy as np
from collections import defaultdict
from datetime import datetime

# فرض: PRICE_CACHE, PUBLIC_SOURCES, ALPHA_KEYS, TWELVE_KEYS, FINNHUB_KEYS, DB, logger
# تابع‌های prov_alpha, prov_twelve, prov_finnhub, prov_exchangerate_host, prov_yf_generic از قبل موجود هستند
# pick_key نیز از بخش Smart Keys موجود است

provider_scores = defaultdict(lambda: 1.0)
provider_latency = defaultdict(lambda: 1.0)

def update_provider_score(provider, success=True, latency=1.0):
    old = provider_scores.get(provider, 1.0)
    if success:
        provider_scores[provider] = min(3.0, old + 0.02)
    else:
        provider_scores[provider] = max(0.1, old - 0.15)
    provider_latency[provider] = 0.7 * provider_latency[provider] + 0.3 * latency

def robust_weighted_median(values, weights):
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    arr = np.array(values, dtype=float)
    w = np.array(weights, dtype=float)
    if arr.size == 0:
        return None
    mean = np.mean(arr)
    std = np.std(arr) + 1e-12
    z = np.abs((arr - mean) / std)
    mask = z < 1.5
    clean_arr = arr[mask]
    clean_w = w[mask]
    if len(clean_arr) == 0:
        return float(np.median(arr))
    idx = np.argsort(clean_arr)
    cumsum = np.cumsum(clean_w[idx])
    cutoff = cumsum[-1] / 2.0
    median_idx = np.searchsorted(cumsum, cutoff)
    median_idx = min(median_idx, len(clean_arr)-1)
    return float(clean_arr[idx[median_idx]])

# ---------------- Public sources fetch ----------------
async def fetch_public_source(session, pair, source):
    try:
        url = source
        if "{pair}" in url:
            url = url.format(pair=pair.replace("/", ""))
        async with session.get(url, timeout=6) as resp:
            if resp.status == 200:
                data = await resp.json()
                # مثال ساده برای استخراج قیمت
                price = None
                if "market_data" in data:
                    price = data["market_data"]["current_price"]["usd"]
                elif "price" in data:
                    price = float(data["price"])
                elif pair in data:
                    price = float(data[pair]["usd"])
                return {"source": source, "price": price} if price else None
    except Exception:
        return None

async def fetch_from_public(session, pair, count=4):
    sources = random.sample(PUBLIC_SOURCES, min(count, len(PUBLIC_SOURCES)))
    tasks = [asyncio.create_task(fetch_public_source(session, pair, s)) for s in sources]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    valid = [r for r in results if isinstance(r, dict) and "price" in r]
    return valid

# ---------------- Private API fetch ----------------
async def fetch_private(pair, keys_iterators, count=1):
    results = []
    for _ in range(count):
        for key_iter in keys_iterators:
            try:
                key = next(key_iter)
                if key_iter is ALPHA_KEYS:
                    price, _ = await prov_alpha(None, pair)
                elif key_iter is TWELVE_KEYS:
                    price, _ = await prov_twelve(None, pair)
                elif key_iter is FINNHUB_KEYS:
                    price, _ = await prov_finnhub(None, pair)
                else:
                    price = None
                if price is not None:
                    results.append({"key": key, "price": price})
                    break
            except Exception:
                continue
        if results:
            break
    return results

# ---------------- Main Smart get_price ----------------
async def get_price_smart(pair, timeout=6):
    key_cache = f"price:{pair}"
    if key_cache in PRICE_CACHE:
        return PRICE_CACHE[key_cache]

    results = {}
    latencies = {}
    try:
        async with aiohttp.ClientSession() as session:
            # 1. حداقل 2–4 منبع عمومی
            public_data = await fetch_from_public(session, pair, count=4)

            # 2. اگر داده معتبر داشت، یک کلید شخصی برای تطبیق
            if public_data:
                private_data = await fetch_private(pair, [ALPHA_KEYS, TWELVE_KEYS, FINNHUB_KEYS], count=1)
            else:
                # 3. اگر منابع عمومی داده ندادند، حداقل 2 کلید شخصی fallback
                private_data = await fetch_private(pair, [ALPHA_KEYS, TWELVE_KEYS, FINNHUB_KEYS], count=2)

            # ترکیب داده‌ها
            all_prices = [d["price"] for d in public_data] if public_data else []
            all_prices += [d["price"] for d in private_data] if private_data else []

            # weight بر اساس provider_scores و latency
            for d in public_data or []:
                results[d["source"]] = float(d["price"])
                latencies[d["source"]] = 0.5
                update_provider_score(d["source"], success=True, latency=0.5)
            for d in private_data or []:
                results[d["key"]] = float(d["price"])
                latencies[d["key"]] = 0.1
                update_provider_score(d["key"], success=True, latency=0.1)

            base_weights = [provider_scores.get(p, 1.0) for p in results.keys()]
            time_weights = [1.0 / (max(0.1, latencies.get(p, 1.0)) + 0.1) for p in results.keys()]
            final_weights = [bw * tw for bw, tw in zip(base_weights, time_weights)]

            fused_price = robust_weighted_median(list(results.values()), final_weights)

            if fused_price is None or not np.isfinite(fused_price):
                fused_price = float(np.median(list(results.values())))

            PRICE_CACHE[key_cache] = (float(fused_price), results)

            # update DB provider stats
            try:
                cur = DB.cursor()
                now = datetime.utcnow().isoformat()
                for p in results:
                    cur.execute(
                        "INSERT OR IGNORE INTO provider_stats(provider,requests,failures,last_seen) VALUES (?,?,?,?)",
                        (p, 0, 0, now)
                    )
                    cur.execute(
                        "UPDATE provider_stats SET requests=requests+1, last_seen=? WHERE provider=?",
                        (now, p)
                    )
                DB.commit()
            except Exception:
                logger.debug("db provider update fail", exc_info=True)

            return float(fused_price), results

    except Exception as e:
        logger.debug("get_price_smart fail for %s: %s", pair, e)
        PRICE_CACHE[key_cache] = (None, {})
        return None, {}

# ---------------- Smart OHLC Prefetch ----------------
import asyncio
import random
import aiohttp
import pandas as pd
import yfinance as yf
from itertools import islice

# Assumptions: OHLC_CACHE, UNIVERSE, HIST, ROTATION_SIZE, PREFETCH_INTERVAL are defined
# Assumptions: PUBLIC_SOURCES, ALPHA_KEYS, TWELVE_KEYS, FINNHUB_KEYS, etc. are defined in Smart Keys

async def fetch_from_public(session, pair, sources):
    """
    Fetch data concurrently from public sources.
    Returns valid data or None.
    """
    tasks = [fetch_public_source(session, pair, src) for src in sources]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    valid_results = [r for r in results if isinstance(r, dict) and "price" in r]
    return valid_results

async def fetch_public_source(session, pair, source):
    """
    Fetch data from a single public source.
    """
    try:
        url = source
        if "{pair}" in url:
            url = url.format(pair=pair.replace("/", ""))
        async with session.get(url, timeout=10) as resp:
            if resp.status == 200:
                data = await resp.json()
                price = extract_price(data, pair)
                return {"source": source, "price": price}
    except Exception:
        return None

def extract_price(data, pair):
    """
    Helper to extract price from JSON data from public sources.
    Adjust logic per API structure.
    """
    try:
        if "market_data" in data:
            return data["market_data"]["current_price"]["usd"]
        elif "price" in data:
            return float(data["price"])
        elif pair in data:
            return float(data[pair]["usd"])
    except Exception:
        return None
    return None

async def fetch_from_private(pair, keys_iterators, count=1):
    """
    Fetch data from private API keys. 
    `count` is the number of fallback keys to use.
    """
    results = []
    for _ in range(count):
        for key_iter in keys_iterators:
            try:
                key = next(key_iter)
                price = await fetch_private_api(pair, key)
                if price is not None:
                    results.append({"key": key, "price": price})
                    break
            except Exception:
                continue
        if results:
            break
    return results

async def fetch_private_api(pair, key):
    """
    Placeholder for fetching from a private API (Twelvedata, Alpha, Finnhub, etc.).
    Implement per API requirements.
    """
    await asyncio.sleep(0.1)
    return random.uniform(100, 200)  # Replace with actual API call

async def async_prefetch(session, pair):
    """
    Smart OHLC fetching for a single symbol.
    """
    try:
        # 1. Fetch at least 2-4 public sources concurrently
        public_sources = random.sample(PUBLIC_SOURCES, min(4, len(PUBLIC_SOURCES)))
        public_data = await fetch_from_public(session, pair, public_sources)

        # 2. If valid data found from public sources, use 1 private key to match
        if public_data:
            private_data = await fetch_from_private(pair, [ALPHA_KEYS, TWELVE_KEYS, FINNHUB_KEYS], count=1)
        else:
            # 3. If public sources fail, fallback to at least 2 private keys
            private_data = await fetch_from_private(pair, [ALPHA_KEYS, TWELVE_KEYS, FINNHUB_KEYS], count=2)

        # 4. Combine results and store in cache
        final_price = None
        if public_data:
            final_price = sum([d["price"] for d in public_data]) / len(public_data)
        if private_data:
            final_price = private_data[0]["price"] if final_price is None else (final_price + private_data[0]["price"]) / 2

        if final_price is not None:
            OHLC_CACHE[f"ohlc:{pair}:smart"] = final_price
            return final_price

    except Exception as e:
        logger.debug("Prefetch failed for %s: %s", pair, e)
    return None

async def prefetch_loop():
    logger.info("Smart Prefetch loop started")
    while True:
        try:
            pool = list(dict.fromkeys(list(HIST.keys()) + UNIVERSE))
            sample_size = min(len(pool), max(ROTATION_SIZE * 3, 120))
            sample = random.sample(pool, sample_size) if pool else []
            logger.info("Prefetching %d symbols", len(sample))

            if aiohttp:
                async with aiohttp.ClientSession() as session:
                    tasks = [asyncio.create_task(async_prefetch(session, p)) for p in sample]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    errors = sum(1 for r in results if isinstance(r, Exception))
                    if errors:
                        logger.warning("Prefetch had %d errors in this batch", errors)
            else:
                for p in sample:
                    await async_prefetch(None, p)

        except Exception as e:
            logger.exception("Prefetch loop error: %s", e)
        await asyncio.sleep(PREFETCH_INTERVAL)

# ---------------- Feature extraction (NEW + 4H CONFIRMATION) ----------------
scaler = StandardScaler()
online_model_xgb = None
online_model_cat = None

def load_online_model():
    global online_model_xgb, online_model_cat
    if os.path.exists(MODEL_FILE):
        try:
            online_model_xgb = joblib.load(MODEL_FILE)
            logger.info("XGBoost model loaded.")
        except:
            logger.exception("Failed to load XGBoost, creating new")
            online_model_xgb = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=1) if XGBOOST_AVAILABLE else RandomForestClassifier(n_estimators=300, random_state=42)
    else:
        online_model_xgb = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=1) if XGBOOST_AVAILABLE else RandomForestClassifier(n_estimators=300, random_state=42)

    if CATBOOST_AVAILABLE:
        if os.path.exists(CATMODEL_FILE):
            try:
                online_model_cat = joblib.load(CATMODEL_FILE)
                logger.info("CatBoost model loaded.")
            except:
                online_model_cat = CatBoostClassifier(iterations=300, depth=6, verbose=0, random_state=42)
        else:
            online_model_cat = CatBoostClassifier(iterations=300, depth=6, verbose=0, random_state=42)
    else:
        online_model_cat = None

load_online_model()

FEATURE_HISTORY = []
LABEL_HISTORY = []

def extract_features_from_df(df):
    try:
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume'] if 'Volume' in df.columns else pd.Series([0]*len(df))
        
        atr = AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
        rsi = RSIIndicator(close, window=14).rsi().iloc[-1] / 100.0
        ret1 = float(close.pct_change(1).iloc[-1]) if len(close) > 1 else 0.0
        ret5 = float(close.pct_change(5).iloc[-1]) if len(close) > 5 else 0.0
        vol = float(volume.iloc[-1])
        last = float(close.iloc[-1])
        
        hour = datetime.utcnow().hour
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
        
        atr_1h = atr
        atr_4h = AverageTrueRange(high, low, close, window=56).average_true_range().iloc[-1] if len(df) >= 56 else atr
        atr_ratio = atr_1h / (atr_4h + 1e-8)
        
        vol_mean = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else vol
        vol_z = (vol - vol_mean) / (volume.rolling(20).std().iloc[-1] + 1e-8) if len(volume) >= 20 else 0.0
        
        return np.array([
            last, atr, rsi, ret1, ret5, vol,
            hour_sin, hour_cos, atr_ratio, vol_z
        ], dtype=np.float32), vol_z, atr_1h, atr_4h
    except Exception as e:
        logger.debug("Feature extraction failed: %s", e)
        return None, 0.0, 0.0, 0.0

def online_predict(feat):
    global online_model_xgb, online_model_cat, scaler
    try:
        if online_model_xgb is None or feat is None:
            return None
        Xs = scaler.transform([feat])
        p_xgb = online_model_xgb.predict_proba(Xs)[0]
        if online_model_cat is not None:
            p_cat = online_model_cat.predict_proba(Xs)[0]
            p = (p_xgb + p_cat) / 2
        else:
            p = p_xgb
        return p
    except Exception as e:
        logger.debug("Prediction failed: %s", e)
        return None

def online_retrain():
    global FEATURE_HISTORY, LABEL_HISTORY, online_model_xgb, online_model_cat, scaler
    try:
        if len(FEATURE_HISTORY) < 500 or len(LABEL_HISTORY) < 500:
            return False
        X = np.array(FEATURE_HISTORY[-2000:])
        y = np.array(LABEL_HISTORY[-2000:])
        Xs = scaler.fit_transform(X)
        online_model_xgb.fit(Xs, y)
        joblib.dump(online_model_xgb, MODEL_FILE)
        if online_model_cat is not None:
            online_model_cat.fit(Xs, y)
            joblib.dump(online_model_cat, CATMODEL_FILE)
        logger.info("Models retrained using %d samples", len(y))
        return True
    except Exception as e:
        logger.exception("online retrain failed: %s", e)
        return False

if TORCH_AVAILABLE:
    class ShortLSTM(nn.Module):
        def __init__(self, input_size=10, hidden=64, nlayers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden, nlayers, batch_first=True)
            self.fc = nn.Linear(hidden, 3)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])
    try:
        LSTM_MODEL = ShortLSTM()
        if os.path.exists(MODEL_FILE + ".lstm"):
            LSTM_MODEL.load_state_dict(torch.load(MODEL_FILE + ".lstm"))
    except Exception:
        LSTM_MODEL = None
else:
    LSTM_MODEL = None

# ---------------- UPGRADE 1-5 + OMNISCIENT 1-3 ----------------
CORRELATION_PAIRS = {
    "EUR/USD": "DXY", "GBP/USD": "DXY", "USD/JPY": "DXY",
    "AUD/USD": "DXY", "NZD/USD": "DXY",
    "BTC/USD": "SPX500/USD", "ETH/USD": "BTC/USD"
}

HIGH_IMPACT_NEWS = ["NFP", "CPI", "FOMC", "GDP", "Retail Sales", "Unemployment"]

async def has_high_impact_news(symbol, hours=2):
    try:
        now = datetime.utcnow()
        start = now - timedelta(hours=1)
        end = now + timedelta(hours=hours)
        url = f"https://api.twelvedata.com/news?symbol={symbol.replace('/', '')}&apikey={TWELVE_KEYS[0] if TWELVE_KEYS else ''}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=5) as r:
                if r.status != 200: return False, 0.0
                data = await r.json()
                sentiment_sum = 0.0
                count = 0
                for item in data.get("news", []):
                    title = item.get("title", "").upper()
                    if any(news in title for news in HIGH_IMPACT_NEWS):
                        return True, 0.0
                    if FINNHUB_KEYS:
                        try:
                            sent_url = f"https://finnhub.io/api/v1/news-sentiment?symbol={symbol.replace('/', '')}&token={FINNHUB_KEYS[0]}"
                            async with session.get(sent_url, timeout=5) as sr:
                                if sr.status == 200:
                                    sdata = await sr.json()
                                    sentiment_sum += sdata.get("sentiment", {}).get("bearishPercent", 50) - sdata.get("sentiment", {}).get("bullishPercent", 50)
                                    count += 1
                        except:
                            pass
                avg_sentiment = sentiment_sum / max(1, count)
                return False, avg_sentiment
        return False, 0.0
    except:
        return False, 0.0

async def check_correlation_filter(symbol, direction, current_price):
    try:
        corr_pair = CORRELATION_PAIRS.get(symbol)
        if not corr_pair: return False
        price_corr, _ = await get_price(corr_pair)
        if price_corr is None: return False
        old_price = HIST.get(corr_pair, price_corr)
        change = (price_corr - old_price) / old_price if old_price > 0 else 0
        if direction == "LONG" and change > 0.0005:
            return True
        if direction == "SHORT" and change < -0.0005:
            return True
        return False
    except:
        return False

# ---------------- Analysis & Signal Generation (OMNISCIENT) ----------------
async def analyze_asset(symbol):
    try:
        fused, sources = await get_price(symbol)
        if fused is None:
            fused2 = None
            try:
                k = f"price:{symbol}"
                if k in PRICE_CACHE:
                    fused2 = PRICE_CACHE[k][0]
            except:
                fused2 = None
            if fused2 is None:
                return None
            fused = fused2
            sources = PRICE_CACHE.get(f"price:{symbol}", ({},))[1]

        df = OHLC_CACHE.get(f"ohlc:{symbol}:30d:1h") or fetch_ohlc_sync(symbol, "30d", "1h")
        if df is None or len(df) < 24:
            df = fetch_ohlc_sync(symbol, "14d", "1h")
            if df is None or len(df) < 12:
                return None

        df_4h = fetch_ohlc_sync(symbol, "30d", "4h")

        feat, vol_z, atr_1h, atr_4h = extract_features_from_df(df)
        if feat is None:
            return None
        FEATURE_HISTORY.append(feat.tolist())

        # OMNISCIENT 1: Volatility Regime Detection
        vol_regime = "HIGH" if atr_1h > 1.8 * atr_4h else "LOW"

        # 4h Confirmation
        if df_4h is not None and len(df_4h) >= 10:
            rsi_4h = RSIIndicator(df_4h['Close'], window=14).rsi().iloc[-1] / 100.0
            ret_4h = df_4h['Close'].pct_change(1).iloc[-1]
        else:
            rsi_4h = 0.5
            ret_4h = 0.0

        probs_rf = online_predict(feat)
        probs_lstm = None
        if LSTM_MODEL is not None:
            try:
                x = torch.tensor(feat.reshape(1, 1, -1), dtype=torch.float32)
                with torch.no_grad():
                    out = LSTM_MODEL(x).softmax(dim=-1).numpy()[0]
                probs_lstm = out
            except Exception:
                probs_lstm = None

        if probs_rf is None and probs_lstm is None:
            rsi = feat[2]; ret1 = feat[3]
            score_long = max(0.0, (0.5 - rsi) + max(0.0, ret1 * 5))
            score_short = max(0.0, (rsi - 0.5) + max(0.0, -ret1 * 5))
            hold = max(0.0, 1.0 - (score_long + score_short))
            raw = np.array([hold, score_long, score_short])
            probs = (raw + 1e-9) / (raw.sum() + 1e-9)
        else:
            preds = []
            weights = []
            if probs_rf is not None:
                preds.append(probs_rf)
                w_rf = get_model_weight("xgb")
                weights.append(max(0.1, w_rf))
            if probs_lstm is not None:
                preds.append(probs_lstm)
                w_lstm = get_model_weight("lstm")
                weights.append(max(0.1, w_lstm))
            if not weights:
                probs = np.array([0.5, 0.25, 0.25])
            else:
                weights = np.array(weights, dtype=float)
                weights = weights / weights.sum()
                probs = np.average(np.array(preds), axis=0, weights=weights)

        idx = int(np.argmax(probs))
        dir_map = {0: "HOLD", 1: "LONG", 2: "SHORT"}
        direction = dir_map[idx]
        confidence = float(np.max(probs))

        # Dynamic MIN_CONFIDENCE
        market_vol = 0.0
        try:
            vol_list = []
            for s in ["EUR/USD", "GBP/USD", "USD/JPY"]:
                if f"price:{s}" in PRICE_CACHE:
                    prices = [PRICE_CACHE[f"price:{s}"][0]]
                    for i in range(1, 11):
                        old_key = f"price:{s}_old_{i}"
                        if old_key in PRICE_CACHE:
                            prices.append(PRICE_CACHE[old_key][0])
                    if len(prices) > 1:
                        vol_list.append(np.std(prices[-10:]))
            market_vol = np.mean(vol_list) if vol_list else 0.0
        except:
            market_vol = 0.0
        dynamic_min_conf = MIN_CONFIDENCE_BASE + min(0.25, market_vol * 100)
        if vol_regime == "HIGH":
            dynamic_min_conf = max(dynamic_min_conf, 0.82)
        if vol_regime == "LOW":
            dynamic_min_conf = max(dynamic_min_conf, 0.72)
        if confidence < dynamic_min_conf:
            return None

        if direction == "HOLD":
            return None

        # UPGRADE 1: Spread + Volume Filter
        pip = 0.0001 if fused > 10 else 0.01
        spread_pips = 2.0 / pip
        atr = feat[1]
        if spread_pips > atr * 150:
            logger.info("Signal blocked: High spread %s", symbol)
            return None
        if abs(vol_z) < 1.0:
            logger.info("Signal blocked: Low volume %s", symbol)
            return None

        # UPGRADE 2: 4h Confirmation
        if direction == "LONG" and (rsi_4h > 0.7 or ret_4h < -0.001):
            logger.info("Signal blocked: 4h bearish %s", symbol)
            return None
        if direction == "SHORT" and (rsi_4h < 0.3 or ret_4h > 0.001):
            logger.info("Signal blocked: 4h bullish %s", symbol)
            return None

        # UPGRADE 5: News Sentiment (Multi-source fallback)
        news_block = False
        sentiment = 0.0
        news_sources = [
            ("twelvedata", TWELVE_KEYS[0] if TWELVE_KEYS else None),
            ("finnhub", FINNHUB_KEYS[0] if FINNHUB_KEYS else None),
            ("alpha", ALPHA_KEYS[0] if ALPHA_KEYS else None)
        ]
        for src, key in news_sources:
            if news_block:
                break
            try:
                if src == "twelvedata" and key:
                    url = f"https://api.twelvedata.com/news?symbol={symbol.replace('/', '')}&apikey={key}"
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, timeout=5) as r:
                            if r.status != 200:
                                continue
                            data = await r.json()
                            for item in data.get("news", []):
                                title = item.get("title", "").upper()
                                if any(news in title for news in HIGH_IMPACT_NEWS):
                                    news_block = True
                                    break
                elif src == "finnhub" and key:
                    async with aiohttp.ClientSession() as session:
                        sent_url = f"https://finnhub.io/api/v1/news-sentiment?symbol={symbol.replace('/', '')}&token={key}"
                        async with session.get(sent_url, timeout=5) as sr:
                            if sr.status == 200:
                                sdata = await sr.json()
                                sentiment += sdata.get("sentiment", {}).get("bearishPercent", 50) - sdata.get("sentiment", {}).get("bullishPercent", 50)
                elif src == "alpha" and key:
                    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol.replace('/', '')}&apikey={key}"
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, timeout=5) as r:
                            if r.status != 200:
                                continue
                            data = await r.json()
                            for item in data.get("feed", []):
                                title = item.get("title", "").upper()
                                if any(news in title for news in HIGH_IMPACT_NEWS):
                                    news_block = True
                                    break
            except Exception:
                continue

        if news_block:
            logger.info("Signal blocked: High impact news %s", symbol)
            return None
        if direction == "LONG" and sentiment < -0.3:
            logger.info("Signal blocked: Bearish news sentiment %s", symbol)
            return None
        if direction == "SHORT" and sentiment > 0.3:
            logger.info("Signal blocked: Bullish news sentiment %s", symbol)
            return None

        # Correlation Filter
        if await check_correlation_filter(symbol, direction, fused):
            logger.info("Signal blocked: Adverse correlation %s", symbol)
            return None

        # OMNISCIENT 2: Dynamic Stop/Target
        stop_factor = 1.8 - (confidence - 0.7) * 1.5
        target_factor = 3.0 + (confidence - 0.7) * 2.0
        stop_factor = max(1.2, min(2.2, stop_factor))
        target_factor = max(2.0, min(5.0, target_factor))

        entry = float(fused)
        if direction == "LONG":
            stop = entry - atr * stop_factor
            target = entry + atr * target_factor
        else:
            stop = entry + atr * stop_factor
            target = entry - atr * target_factor

        base_lev = int(confidence * 80)
        vol_lev = int(max(0, (0.5 - min(0.5, atr)) * 50))
        leverage = min(200, max(1, base_lev + vol_lev))

        lots, stop_pips = compute_position_size(
            ACCOUNT_BALANCE, entry, stop,
            risk_pct=1.0, win_rate=0.5,
            atr=atr, confidence=confidence
        )
        suggested_capital = max(10, round(ACCOUNT_BALANCE * (0.01 * confidence), 2))

        valid_from = datetime.utcnow()
        valid_minutes = int(10 + confidence * 50)
        valid_to = valid_from + timedelta(minutes=valid_minutes)

        sig = {
            "symbol": symbol,
            "direction": direction,
            "entry": round(entry, 6),
            "confidence": round(confidence, 3),
            "target": round(target, 6),
            "stop": round(stop, 6),
            "leverage": leverage,
            "suggested_capital": suggested_capital,
            "lots": lots,
            "stop_pips": stop_pips,
            "sources": sources,
            "valid_from": valid_from.isoformat(),
            "valid_to": valid_to.isoformat(),
            "ts": datetime.utcnow().isoformat()
        }

        try:
            cur = DB.cursor()
            cur.execute("""INSERT INTO signals(symbol,direction,entry,target,stop,confidence,leverage,
                        suggested_capital,lots,ts,valid_from,valid_to,sources) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (sig['symbol'], sig['direction'], sig['entry'], sig['target'], sig['stop'], sig['confidence'],
                         sig['leverage'], sig['suggested_capital'], sig['lots'], sig['ts'], sig['valid_from'], sig['valid_to'],
                         json.dumps(sig['sources'])))
            DB.commit()
            sig['id'] = cur.lastrowid
        except Exception:
            logger.debug("db insert fail", exc_info=True)

        SIGNALS_HISTORY.append(sig)
        HIST[symbol] = float(HIST.get(symbol, 0.0) + confidence * 0.01)
        return sig
    except Exception:
        logger.exception("analyze_asset failed for %s", symbol)
        return None

# ---------------- Notification ----------------
async def send_telegram(msg, max_retries=3, backoff=2.0):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug("telegram not configured")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    for attempt in range(1, max_retries + 1):
        try:
            if aiohttp:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, timeout=10) as resp:
                        if resp.status == 200:
                            return True
                        else:
                            text = await resp.text()
                            logger.warning("Telegram HTTP error: %s %s", resp.status, text)
            else:
                loop = asyncio.get_running_loop()
                resp = await loop.run_in_executor(None, lambda: requests.post(url, json=payload, timeout=10))
                if resp.status_code == 200:
                    return True
                else:
                    logger.warning("Telegram HTTP error: %s %s", resp.status_code, resp.text)
        except Exception:
            logger.exception("telegram send attempt failed")
        await asyncio.sleep(backoff ** attempt)
    try:
        from telegram import Bot
        bot = Bot(token=TELEGRAM_TOKEN)
        bot.send_message(chat_id=int(TELEGRAM_CHAT_ID), text=msg, parse_mode="Markdown")
        return True
    except Exception:
        logger.exception("telegram send final fallback failed")
        return False

def send_email(subject, body):
    if not GMAIL_USER or not GMAIL_APP_PASSWORD:
        logger.debug("gmail not configured")
        return False
    try:
        import smtplib
        from email.mime.text import MIMEText
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = GMAIL_USER
        msg['To'] = GMAIL_USER
        s = smtplib.SMTP_SSL('smtp.gmail.com', 465, timeout=60)
        s.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        s.send_message(msg)
        s.quit()
        return True
    except Exception:
        logger.exception("gmail send failed")
        return False

async def async_send_email(subject, body, max_retries=2):
    for attempt in range(max_retries):
        ok = await asyncio.get_running_loop().run_in_executor(None, send_email, subject, body)
        if ok:
            return True
        await asyncio.sleep(2 ** attempt)
    return False

def format_signal_message(sig):
    msg = (
        f"*OMEGA v25 OMNISCIENT*\n"
        f"`{sig['symbol']}` | *{sig['direction']}*\n\n"
        f"*Entry:* `{sig['entry']}`\n"
        f"*Target:* `{sig['target']}`\n"
        f"*Stop:* `{sig['stop']}`\n"
        f"*Leverage:* `{sig['leverage']}x`\n"
        f"*Suggested capital:* `{sig['suggested_capital']}`\n"
        f"*Position (lots):* `{sig['lots']}`\n"
        f"*Confidence:* `{sig['confidence']}`\n"
        f"*Valid from:* `{sig['valid_from']}`\n"
        f"*Valid to:* `{sig['valid_to']}`\n"
        f"*Sources:* `{','.join(sig.get('sources',{}).keys())}`\n"
        f"\n*ID:* `{sig.get('id','-')}`"
    )
    return msg

async def notify_signal(sig):
    msg = format_signal_message(sig)
    await send_telegram(msg)
    await async_send_email(f"OMEGA v25 SIGNAL {sig['symbol']}", msg)

# ---------------- Selection & Rotation (OMNISCIENT 3: Signal Clustering) ----------------
async def score_asset(asset):
    try:
        fused, _ = await get_price(asset)
        if fused is None: return 0.0
        df = OHLC_CACHE.get(f"ohlc:{asset}:10d:1h") or fetch_ohlc_sync(asset, "10d", "1h")
        if df is None or len(df) < 10: return 0.0
        atr = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range().iloc[-1]
        vol = float(df['Volume'].mean()) if 'Volume' in df.columns else 0.0
        hist = HIST.get(asset, 0.0)
        return float((atr * 1000.0) * 0.6 + math.log1p(vol) * 0.3 + hist * 0.1)
    except Exception:
        return 0.0

def decide_batch_size(min_bs=20, max_bs=40):
    u = max(1, len(UNIVERSE))
    bs = min(max_bs, max(min_bs, int(u * 0.05)))
    bs = max(min_bs, min(max_bs, bs))
    bs = int(bs * (0.9 + 0.2 * random.random()))
    return bs

async def select_rotation(all_assets, rotation_size=None):
    if rotation_size is None:
        rotation_size = decide_batch_size(20, 40)
    candidates = list(HIST.keys()) + list(all_assets)
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        return []
    pool_size = min(len(candidates), max(rotation_size * 3, rotation_size * 2))
    pool = random.sample(candidates, pool_size)
    tasks = [asyncio.create_task(score_asset(s)) for s in pool]
    scores = await asyncio.gather(*tasks, return_exceptions=True)
    scores_clean = [s if isinstance(s, (int, float)) else 0.0 for s in scores]
    pairs_scores = list(zip(pool, scores_clean))
    pairs_scores.sort(key=lambda x: x[1], reverse=True)
    selected = [p for p, _ in pairs_scores[:rotation_size]]
    return selected
# ---------------- Intelligent & Parallel WebSocket Fusion ----------------
import asyncio
import json
import random
import time
import websockets
import aiohttp
from itertools import islice

# Assumptions:
# - PRICE_CACHE, ENABLE_WEBSOCKETS, PUBLIC_SOURCES, ALPHA_KEYS, TWELVE_KEYS, FINNHUB_KEYS, HIST, UNIVERSE, ROTATION_SIZE, PREFETCH_INTERVAL, logger are already defined
# - rotate_key function exists for key rotation to avoid daily limits

# ---------------- Helper: Fetch from public sources ----------------
async def fetch_from_public(session, pair, sources, min_success=2):
    tasks = []
    for src in sources:
        tasks.append(fetch_public_source(session, pair, src))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    valid_results = [r for r in results if isinstance(r, (int, float))]
    if len(valid_results) >= min_success:
        return valid_results
    return valid_results

async def fetch_public_source(session, pair, source):
    try:
        url = source
        if "{pair}" in url:
            url = url.format(pair=pair.replace("/", ""))
        async with session.get(url, timeout=10) as resp:
            if resp.status == 200:
                data = await resp.json()
                price = extract_public_price(data, pair)
                return price
    except Exception:
        return None

def extract_public_price(data, pair):
    try:
        if "market_data" in data:
            return data["market_data"]["current_price"]["usd"]
        elif "price" in data:
            return float(data["price"])
        elif pair in data:
            return float(data[pair]["usd"])
    except Exception:
        return None
    return None

# ---------------- Helper: Fetch from private sources ----------------
async def fetch_from_private(pair, keys_iterators, count=1):
    results = []
    for _ in range(count):
        for key_iter in keys_iterators:
            try:
                key = next(key_iter)
                price = await fetch_private_api(pair, key)
                if price is not None:
                    results.append({"key": key, "price": price})
                    break
            except Exception:
                continue
        if results:
            break
    return results

async def fetch_private_api(pair, key):
    """
    Fetch price using private keys (TwelveData, Alpha Vantage, Finnhub).
    Tries sources in order and fallback automatically.
    """
    symbol_td = pair.replace("/", "")
    symbol_alpha = pair.replace("/", "")
    symbol_finn = pair.replace("/", "")

    async with aiohttp.ClientSession() as session:
        # --- TwelveData ---
        if TWELVE_KEYS:
            try:
                td_key = key if key in TWELVE_KEYS else TWELVE_KEYS[0]
                url_td = f"https://api.twelvedata.com/price?symbol={symbol_td}&apikey={td_key}"
                async with session.get(url_td, timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        price = float(data.get("price"))
                        if price:
                            return price
            except Exception:
                pass

        # --- Alpha Vantage ---
        if ALPHA_KEYS:
            try:
                alpha_key = key if key in ALPHA_KEYS else ALPHA_KEYS[0]
                url_alpha = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={symbol_alpha[:3]}&to_currency={symbol_alpha[3:]}&apikey={alpha_key}"
                async with session.get(url_alpha, timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        price_str = data.get("Realtime Currency Exchange Rate", {}).get("5. Exchange Rate")
                        if price_str:
                            return float(price_str)
            except Exception:
                pass

        # --- Finnhub ---
        if FINNHUB_KEYS:
            try:
                fin_key = key if key in FINNHUB_KEYS else FINNHUB_KEYS[0]
                url_finn = f"https://finnhub.io/api/v1/quote?symbol={symbol_finn}&token={fin_key}"
                async with session.get(url_finn, timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        price = data.get("c")
                        if price:
                            return float(price)
            except Exception:
                pass

    return None

# ---------------- Intelligent per-pair prefetch ----------------
async def fetch_pair_smart(session, pair):
    try:
        public_sources = random.sample(PUBLIC_SOURCES, min(4, len(PUBLIC_SOURCES)))
        public_data = await fetch_from_public(session, pair, public_sources)

        if public_data:
            private_data = await fetch_from_private(pair, [ALPHA_KEYS, TWELVE_KEYS, FINNHUB_KEYS], count=1)
        else:
            private_data = await fetch_from_private(pair, [ALPHA_KEYS, TWELVE_KEYS, FINNHUB_KEYS], count=2)

        final_price = None
        if public_data:
            final_price = sum(public_data) / len(public_data)
        if private_data:
            final_price = private_data[0]["price"] if final_price is None else (final_price + private_data[0]["price"]) / 2

        if final_price is not None:
            PRICE_CACHE[f"price:{pair}"] = (float(final_price), {"smart_ws": float(final_price)})
            return final_price

    except Exception as e:
        logger.debug("Smart WS prefetch failed for %s: %s", pair, e)
    return None

# ---------------- Intelligent reconnect ----------------
async def reconnect_loop(uri, max_backoff=60):
    backoff = 1
    while True:
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=20) as ws:
                backoff = 1
                yield ws
        except Exception as e:
            logger.warning("WS reconnect failed (%s). backing off %ds", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(max_backoff, backoff * 2)

# ---------------- Main TwelveData WS ----------------
async def twelvedata_ws(sub_pairs, keys, usage_counter, limit_per_key):
    if not ENABLE_WEBSOCKETS or websockets is None:
        logger.info("TwelveData WS disabled or not available")
        return
    while True:
        key = rotate_key(keys, usage_counter, limit_per_key)
        uri = f"wss://ws.twelvedata.com/v1/quotes?apikey={key}"
        try:
            async for ws in reconnect_loop(uri):
                try:
                    for s in sub_pairs:
                        try:
                            await ws.send(json.dumps({"action": "subscribe", "params": {"symbols": s.replace('/', '')}}))
                        except Exception:
                            logger.debug("Twelvedata subscribe failed for %s", s)

                    async for msg in ws:
                        try:
                            data = json.loads(msg)
                            if isinstance(data, dict):
                                sym = data.get("symbol")
                                price = data.get("price") or data.get("close")
                                if sym and price:
                                    pair = f"{sym[:3]}/{sym[3:]}"
                                    PRICE_CACHE[f"price:{pair}"] = (float(price), {"twelvedata_ws": float(price)})
                        except Exception:
                            async with aiohttp.ClientSession() as session:
                                await fetch_pair_smart(session, s)
                except Exception as e:
                    logger.exception("Twelvedata WS error: %s", e)
                    await asyncio.sleep(1)
        except Exception as e:
            logger.warning("TwelveData WS reconnect loop exception: %s", e)
            await asyncio.sleep(1)

# ---------------- Main Finnhub WS ----------------
async def finnhub_ws(sub_pairs, keys, usage_counter, limit_per_key):
    if not ENABLE_WEBSOCKETS or websockets is None:
        logger.info("Finnhub WS disabled or not available")
        return
    while True:
        key = rotate_key(keys, usage_counter, limit_per_key)
        uri = f"wss://ws.finnhub.io?token={key}"
        try:
            async for ws in reconnect_loop(uri):
                try:
                    for s in sub_pairs:
                        try:
                            await ws.send(json.dumps({"type": "subscribe", "symbol": s.replace('/', '_')}))
                        except Exception:
                            logger.debug("Finnhub subscribe failed for %s", s)

                    async for msg in ws:
                        try:
                            data = json.loads(msg)
                            if "data" in data:
                                for d in data["data"]:
                                    sym = d.get("s") or d.get("symbol")
                                    price = d.get("p") or d.get("price")
                                    if sym and price:
                                        pair = sym.replace("_", "/").replace(":", "/")[:7]
                                        PRICE_CACHE[f"price:{pair}"] = (float(price), {"finnhub_ws": float(price)})
                        except Exception:
                            async with aiohttp.ClientSession() as session:
                                await fetch_pair_smart(session, s)
                except Exception as e:
                    logger.exception("Finnhub WS error: %s", e)
                    await asyncio.sleep(1)
        except Exception as e:
            logger.warning("Finnhub WS reconnect loop exception: %s", e)
            await asyncio.sleep(1)

# ---------------- Self-update scaffold ----------------
async def self_update_loop():
    if not SELF_UPDATE_REPO:
        logger.info("Self-update disabled")
        return
    logger.info("Self-update enabled")
    while True:
        try:
            res = subprocess.run(["git", "pull"], cwd=os.getcwd(), capture_output=True, text=True, timeout=120)
            if res.returncode == 0 and ("Already up to date" not in res.stdout):
                logger.info("Code updated via git; restarting")
                os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception:
            logger.exception("self-update error")
        await asyncio.sleep(int(os.getenv("SELF_UPDATE_INTERVAL", "3600")))


# ---------------- Engine main loop (OMNISCIENT 3) ----------------
async def engine_loop_all():
    asyncio.create_task(reset_counters_task())
    if ENABLE_PREFETCH:
        asyncio.create_task(prefetch_loop())
    if ENABLE_WEBSOCKETS:
        sample_pairs = random.sample(UNIVERSE, min(len(UNIVERSE), 80))
        if TWELVE_KEYS and websockets:
            asyncio.create_task(twelvedata_ws(sample_pairs, TWELVE_KEYS, USAGE_COUNTER_TWELVE, LIMIT_PER_KEY_TWELVE))
        if FINNHUB_KEYS and websockets:
            asyncio.create_task(finnhub_ws(sample_pairs, FINNHUB_KEYS, USAGE_COUNTER_FINNHUB, LIMIT_PER_KEY_FINNHUB))
    if SELF_UPDATE_REPO:
        asyncio.create_task(self_update_loop())

    logger.info("Engine main loop started; total assets=%d", len(UNIVERSE))
    retrain_counter = 0
    while True:
        try:
            rotation = await select_rotation(UNIVERSE)
            if not rotation:
                logger.info("No assets selected this cycle.")
                await asyncio.sleep(CYCLE_TIME)
                continue

            # OMNISCIENT 3: Signal Clustering
            recent_signals = [
                s for s in SIGNALS_HISTORY
                if (datetime.utcnow() - datetime.fromisoformat(s['ts'])).seconds < 3600
            ]
            symbol_count = defaultdict(int)
            for s in recent_signals:
                symbol_count[s['symbol']] += 1

            tasks = []
            for a in rotation:
                if symbol_count.get(a, 0) >= 2:
                    continue
                tasks.append(asyncio.create_task(analyze_asset(a)))

            results = await asyncio.gather(*tasks, return_exceptions=True)
            signals = [r for r in results if isinstance(r, dict) and r]
            signals = sorted(signals, key=lambda x: x['confidence'], reverse=True)[:TOP_N]

            for s in signals:
                await notify_signal(s)
                logger.info("Signal emitted: %s %s conf=%.3f", s['symbol'], s['direction'], s['confidence'])

            retrain_counter += 1
            if retrain_counter >= 200:
                online_retrain()
                retrain_counter = 0
                logger.info("Triggered scheduled online retrain")

        except Exception:
            logger.exception("Engine loop error")
        await asyncio.sleep(CYCLE_TIME)
# ---------------- Graceful shutdown ----------------
from datetime import datetime

def shutdown(sig, frame):
    logger.info("Shutdown requested")
    try:
        # Save signals
        with open(SIGNALS_FILE, "w") as f:
            json.dump(list(SIGNALS_HISTORY), f, default=str)
        # Close database
        DB.commit()
        DB.close()

        # Message with UTC shutdown time
        msg = f"*OMEGA v25 OMNISCIENT*\nEngine is now INACTIVE!\nTime: {datetime.utcnow().isoformat()} UTC"

        try:
            if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
                url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
                requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}, timeout=10)
        except Exception:
            logger.exception("Telegram shutdown message failed")

        try:
            if GMAIL_USER and GMAIL_APP_PASSWORD:
                send_email("OMEGA v25 Engine INACTIVE", msg)
        except Exception:
            logger.exception("Gmail shutdown message failed")

    except Exception:
        logger.exception("Error during shutdown")
    finally:
        sys.exit(0)

signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

# ---------------- Background tasks ----------------
async def prefetch_data(interval=3600):
    while True:
        logger.info("Prefetching market data (background)...")
        try:
            await prefetch_loop()
        except Exception as e:
            logger.error("Prefetch error: %s", e)
        await asyncio.sleep(interval)

async def self_learning_update(interval=1800):
    while True:
        try:
            logger.info("Self-learning update (background)...")
            online_retrain()
        except Exception as e:
            logger.error("Self-learning error: %s", e)
        await asyncio.sleep(interval)

async def check_for_updates(interval=3600):
    if not SELF_UPDATE_REPO:
        logger.info("Auto-update disabled")
        return
    while True:
        try:
            logger.info("Checking for Omega engine updates...")
            res = subprocess.run(["git", "pull"], cwd=os.getcwd(), capture_output=True, text=True, timeout=120)
            if res.returncode == 0 and ("Already up to date" not in res.stdout):
                logger.info("Code updated via git; restarting")
                sys.exit(0)
        except Exception:
            logger.exception("Auto-update failed")
        await asyncio.sleep(interval)

# ---------------- Startup Notification ----------------
async def notify_startup():
    msg = f"*OMEGA v25 OMNISCIENT*\nEngine is now ACTIVE!\nTime: {datetime.utcnow().isoformat()} UTC"
    try:
        if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}, timeout=10)
    except Exception:
        logger.exception("Telegram startup message failed")
    try:
        if GMAIL_USER and GMAIL_APP_PASSWORD:
            send_email("OMEGA v25 Engine ACTIVE", msg)
    except Exception:
        logger.exception("Gmail startup message failed")

# ---------------- Main Runner ----------------
async def main_runner_all():
    await notify_startup()
    tasks = [engine_loop_all()]
    if ENABLE_PREFETCH:
        tasks.append(prefetch_data(interval=PREFETCH_INTERVAL))
    tasks.append(self_learning_update(interval=1800))
    if SELF_UPDATE_REPO:
        tasks.append(check_for_updates(interval=int(os.getenv("SELF_UPDATE_INTERVAL", "3600"))))
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    try:
        asyncio.run(main_runner_all())
    except Exception:
        logger.exception("Fatal error in main_runner_all")

