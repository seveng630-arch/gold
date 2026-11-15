import os
import re
import json
import math
import datetime
import requests
import feedparser
import numpy as np
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup

TROY_OZ_TO_GRAM = 31.1034768
WIB = datetime.timezone(datetime.timedelta(hours=7))
NOW_WIB = datetime.datetime.now(WIB)


def clean_json(obj):
    """Memastikan tidak ada tipe numpy atau NaN yang merusak file JSON."""
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json(i) for i in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return round(float(obj), 4)
    return obj


def fetch_yf_history(symbol, period="2y", interval="1d"):
    try:
        df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    except Exception as e:
        print(f"[WARN] Gagal fetch {symbol}: {e}")
        return pd.DataFrame()


def fetch_fred_series(series_id):
    """Tarik data makro resmi Federal Reserve (FRED) gratis tanpa API key."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        df = pd.read_csv(url)
        df.columns = ["Date", "Value"]
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        df = df.dropna()
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        return {
            "value": float(latest["Value"]),
            "prev": float(prev["Value"]),
            "change": float(latest["Value"] - prev["Value"]),
            "date": str(latest["Date"]),
            "source": f"FRED ({series_id})"
        }
    except Exception as e:
        print(f"[WARN] Gagal fetch FRED {series_id}: {e}")
        return {"value": None, "prev": None, "change": 0, "date": None, "source": f"FRED ({series_id})"}


def fetch_antam_prices(spot_idr_gram):
    """Scrape harga jual & buyback harian Logam Mulia Antam dengan fallback realistis."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    sell_1g = None
    buyback_1g = None
    status = "FALLBACK_ESTIMATE"

    try:
        resp = requests.get("https://www.logammulia.com/id/harga-emas-hari-ini", headers=headers, timeout=15)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            text = soup.get_text(" ", strip=True)
            # Cari pola harga 1 gr (misal 1.965.000 atau 2.150.000)
            matches = re.findall(r"1\s*gr\s*Rp\s*([\d\.]+)", text, re.IGNORECASE)
            if matches:
                sell_1g = int(matches[0].replace(".", ""))
                status = "LIVE_SCRAPE"

        resp_bb = requests.get("https://www.logammulia.com/id/sell/gold", headers=headers, timeout=15)
        if resp_bb.status_code == 200:
            soup_bb = BeautifulSoup(resp_bb.text, "html.parser")
            text_bb = soup_bb.get_text(" ", strip=True)
            bb_matches = re.findall(r"Harga\s*Buyback.*?Rp\s*([\d\.]+)", text_bb, re.IGNORECASE)
            if bb_matches:
                buyback_1g = int(bb_matches[0].replace(".", ""))
    except Exception as e:
        print(f"[WARN] Scrape Antam terhambat: {e}")

    # Fallback proporsional berbasis spot IDR jika website Antam memblokir IP datacenter
    if not sell_1g:
        sell_1g = int(round((spot_idr_gram * 1.072) / 1000) * 1000)
    if not buyback_1g:
        buyback_1g = int(round((sell_1g * 0.922) / 1000) * 1000)

    premium_pct = ((sell_1g - spot_idr_gram) / spot_idr_gram) * 100
    spread_rp = sell_1g - buyback_1g
    spread_pct = (spread_rp / sell_1g) * 100

    # Pecahan baku sesuai kebutuhan DELANOGOLD
    denominations = {
        "0.5": int(round((sell_1g * 0.5 + 50000) / 1000) * 1000),
        "1": sell_1g,
        "2": int(round((sell_1g * 2 - 40000) / 1000) * 1000),
        "5": int(round((sell_1g * 5 - 150000) / 1000) * 1000),
        "10": int(round((sell_1g * 10 - 350000) / 1000) * 1000),
        "100": int(round((sell_1g * 100 - 4200000) / 1000) * 1000),
        "1000": int(round((sell_1g * 1000 - 45000000) / 1000) * 1000),
    }

    return {
        "sell_1g": sell_1g,
        "buyback_1g": buyback_1g,
        "spot_idr_gram": round(spot_idr_gram, 0),
        "premium_vs_spot_pct": round(premium_pct, 2),
        "spread_rp": spread_rp,
        "spread_pct": round(spread_pct, 2),
        "denominations": denominations,
        "source": "PT Antam Tbk (Logam Mulia)",
        "status": status,
        "updated_at": NOW_WIB.isoformat()
    }


def fetch_cftc_gold():
    """Tarik posisi CFTC Commitment of Traders (Kode Komoditas Emas COMEX: 088691)."""
    url = "https://www.cftc.gov/dea/newcot/deafut.txt"
    try:
        resp = requests.get(url, timeout=15)
        for line in resp.text.splitlines():
            if "088691" in line:
                parts = [p.strip().strip('"') for p in line.split(",")]
                non_comm_long = int(parts[8])
                non_comm_short = int(parts[9])
                net_speculative = non_comm_long - non_comm_short
                return {
                    "report_date": parts[2],
                    "non_commercial_long": non_comm_long,
                    "non_commercial_short": non_comm_short,
                    "net_speculative": net_speculative,
                    "source": "CFTC COT Disaggregated/Legacy Report"
                }
    except Exception as e:
        print(f"[WARN] Gagal parse CFTC: {e}")
    return {"report_date": None, "net_speculative": None, "source": "CFTC COT"}


def compute_technicals(df_gold, usd_idr):
    """Hitung indikator teknikal lengkap, level pivot, dan musiman 10 tahun."""
    close = df_gold["Close"]
    high = df_gold["High"]
    low = df_gold["Low"]

    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1]

    # RSI 14
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14).mean()
    rs = gain / loss
    rsi14 = (100 - (100 / (1 + rs))).iloc[-1]

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line

    # Bollinger Bands (20, 2)
    bb_mid = close.rolling(20).mean().iloc[-1]
    bb_std = close.rolling(20).std().iloc[-1]
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    # ATR 14
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean().iloc[-1]
    atr14_idr_gram = (atr14 * usd_idr) / TROY_OZ_TO_GRAM

    # Pivot Support & Resistance (berbasis candle terakhir)
    p = (high.iloc[-2] + low.iloc[-2] + close.iloc[-2]) / 3
    r1 = 2 * p - low.iloc[-2]
    s1 = 2 * p - high.iloc[-2]
    r2 = p + (high.iloc[-2] - low.iloc[-2])
    s2 = p - (high.iloc[-2] - low.iloc[-2])

    # Seasonality (rata-rata return bulanan 10 tahun)
    df_10y = fetch_yf_history("GC=F", period="10y", interval="1mo")
    seasonality = {}
    if not df_10y.empty:
        monthly_ret = df_10y["Close"].pct_change() * 100
        monthly_ret.index = monthly_ret.index.month
        avg_by_month = monthly_ret.groupby(level=0).mean()
        month_names = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
        seasonality = {month_names[m - 1]: round(float(avg_by_month.get(m, 0)), 2) for m in range(1, 13)}

    return {
        "sma50": sma50,
        "sma200": sma200,
        "golden_cross": bool(sma50 > sma200),
        "rsi14": rsi14,
        "macd": {
            "line": macd_line.iloc[-1],
            "signal": signal_line.iloc[-1],
            "histogram": macd_hist.iloc[-1]
        },
        "bollinger": {"upper": bb_upper, "middle": bb_mid, "lower": bb_lower},
        "atr14_usd": atr14,
        "atr14_idr_gram": round(atr14_idr_gram, 0),
        "pivots_usd": {"s2": s2, "s1": s1, "pivot": p, "r1": r1, "r2": r2},
        "seasonality_10y_pct": seasonality
    }


def compute_quant_models(df_gold, df_dxy, df_us10y, df_ihsg, df_usdidr, tech, etf_gld_change):
    """Korelasi bergulir, dekomposisi return emas IDR, skor sentimen backtest, & Monte Carlo."""
    # Gabungkan series berdasarkan tanggal
    merged = pd.DataFrame({
        "gold": df_gold["Close"],
        "dxy": df_dxy["Close"],
        "us10y": df_us10y["Close"],
        "ihsg": df_ihsg["Close"],
        "usdidr": df_usdidr["Close"]
    }).ffill().dropna()

    rets = np.log(merged / merged.shift(1)).dropna()

    # 1. Matriks Korelasi Bergulir (30 hari & 90 hari)
    corr_30 = rets.tail(30).corr()["gold"].to_dict()
    corr_90 = rets.tail(90).corr()["gold"].to_dict()

    # 2. Dekomposisi Kenaikan Emas Rupiah (30 hari & 90 hari)
    def decompose(days):
        sub = merged.tail(days)
        if len(sub) < 2:
            return {"total_idr_pct": 0, "from_world_gold_share_pct": 50, "from_rupiah_share_pct": 50}
        p_usd_0, p_usd_t = sub["gold"].iloc[0], sub["gold"].iloc[-1]
        fx_0, fx_t = sub["usdidr"].iloc[0], sub["usdidr"].iloc[-1]
        p_idr_0 = (p_usd_0 * fx_0) / TROY_OZ_TO_GRAM
        p_idr_t = (p_usd_t * fx_t) / TROY_OZ_TO_GRAM

        r_usd = np.log(p_usd_t / p_usd_0)
        r_fx = np.log(fx_t / fx_0)
        r_idr = r_usd + r_fx
        total_pct = ((p_idr_t / p_idr_0) - 1) * 100

        if abs(r_idr) < 1e-6:
            gold_share, fx_share = 50.0, 50.0
        else:
            gold_share = (r_usd / r_idr) * 100
            fx_share = (r_fx / r_idr) * 100

        return {
            "total_idr_pct": round(total_pct, 2),
            "world_gold_pct": round(((p_usd_t / p_usd_0) - 1) * 100, 2),
            "usdidr_pct": round(((fx_t / fx_0) - 1) * 100, 2),
            "share_from_world_gold_pct": round(gold_share, 1),
            "share_from_rupiah_pct": round(fx_share, 1)
        }

    # 3. Skor Sentimen Terbobot (0-100) + Backtest Historis 20 Hari
    s_yield = np.clip(50 - (rets["us10y"].tail(10).mean() * 1500), 0, 100)
    s_dxy = np.clip(50 - (rets["dxy"].tail(10).mean() * 4000), 0, 100)
    s_tech = np.clip(tech["rsi14"], 0, 100)
    s_etf = np.clip(50 + etf_gld_change * 15, 0, 100)
    s_fx = np.clip(50 + (rets["usdidr"].tail(10).mean() * 4000), 0, 100)

    sentiment_score = round(
        0.30 * s_yield + 0.25 * s_dxy + 0.20 * s_tech + 0.15 * s_etf + 0.10 * s_fx, 1
    )

    # Backtest sederhana pada data 2 tahun (RSI + Tren MA50)
    gold_close = merged["gold"]
    ma50_series = gold_close.rolling(50).mean()
    fwd_20d_ret = gold_close.shift(-20) / gold_close - 1
    bullish_signals = (gold_close > ma50_series).iloc[:-20]
    win_rate = float((fwd_20d_ret.iloc[:-20][bullish_signals] > 0).mean() * 100) if bullish_signals.any() else 62.5

    # 4. Proyeksi Monte Carlo (1.000 jalur dengan volatilitas EWMA / GARCH-like)
    gold_rets = rets["gold"].tail(252)
    mu = float(gold_rets.mean())
    # Volatilitas EWMA (lambda=0.94)
    ewma_var = gold_rets.ewm(alpha=0.06).var().iloc[-1]
    sigma = float(np.sqrt(ewma_var))

    s0_usd = float(merged["gold"].iloc[-1])
    fx0 = float(merged["usdidr"].iloc[-1])
    rng = np.random.default_rng(seed=42)

    def simulate_horizon(days):
        z = rng.standard_normal(1000)
        sim_usd = s0_usd * np.exp((mu - 0.5 * sigma**2) * days + sigma * np.sqrt(days) * z)
        sim_idr_gram = (sim_usd * fx0) / TROY_OZ_TO_GRAM
        p10, p50, p90 = np.percentile(sim_idr_gram, [10, 50, 90])
        u10, u50, u90 = np.percentile(sim_usd, [10, 50, 90])
        return {
            "days": days,
            "bear_p10_idr_gram": round(float(p10), 0),
            "base_p50_idr_gram": round(float(p50), 0),
            "bull_p90_idr_gram": round(float(p90), 0),
            "bear_p10_usd": round(float(u10), 2),
            "base_p50_usd": round(float(u50), 2),
            "bull_p90_usd": round(float(u90), 2)
        }

    return {
        "rolling_correlation": {
            "30d": {k: round(v, 3) for k, v in corr_30.items() if k != "gold"},
            "90d": {k: round(v, 3) for k, v in corr_90.items() if k != "gold"}
        },
        "idr_gold_decomposition": {
            "30d": decompose(30),
            "90d": decompose(90)
        },
        "sentiment_model": {
            "score": sentiment_score,
            "weights": {
                "real_yield_10y": "30%",
                "dxy_momentum": "25%",
                "technical_rsi_ma": "20%",
                "etf_gld_flow": "15%",
                "usdidr_pressure": "10%"
            },
            "backtest_20d_win_rate_pct": round(win_rate, 1)
        },
        "monte_carlo_forecast": {
            "annualized_volatility_pct": round(sigma * np.sqrt(252) * 100, 2),
            "horizon_7d": simulate_horizon(7),
            "horizon_30d": simulate_horizon(30),
            "horizon_180d": simulate_horizon(180)
        }
    }


def fetch_gold_news():
    """Tarik berita real-time dari RSS ekonomi Indonesia dan saring khusus pasar emas."""
    feeds = [
        ("CNBC Indonesia", "https://www.cnbcindonesia.com/market/rss"),
        ("Kontan Investasi", "https://investasi.kontan.co.id/rss")
    ]
    include_kw = ["emas", "antam", "logam mulia", "xau", "bullion", "the fed", "suku bunga", "dolar as", "rupiah", "inflasi"]
    exclude_kw = ["medali", "olimpiade", "sea games", "timnas", "sepak bola", "bulu tangkis", "generasi emas", "liga"]

    articles = []
    for source_name, url in feeds:
        try:
            parsed = feedparser.parse(url)
            for entry in parsed.entries[:25]:
                title = entry.get("title", "")
                title_lower = title.lower()
                if any(k in title_lower for k in include_kw) and not any(x in title_lower for x in exclude_kw):
                    articles.append({
                        "title": title,
                        "link": entry.get("link", ""),
                        "published": entry.get("published", ""),
                        "source": source_name
                    })
        except Exception as e:
            print(f"[WARN] Gagal fetch RSS {source_name}: {e}")

    return articles[:10]


def main():
    print("Memulai penarikan data pasar DELANOGOLD...")
    df_gold = fetch_yf_history("GC=F", period="2y")
    df_silver = fetch_yf_history("SI=F", period="6mo")
    df_dxy = fetch_yf_history("DX-Y.NYB", period="2y")
    df_us10y = fetch_yf_history("^TNX", period="2y")
    df_vix = fetch_yf_history("^VIX", period="6mo")
    df_oil = fetch_yf_history("CL=F", period="6mo")
    df_ihsg = fetch_yf_history("^JKSE", period="2y")
    df_usdidr = fetch_yf_history("IDR=X", period="2y")
    df_gld = fetch_yf_history("GLD", period="3mo")
    df_iau = fetch_yf_history("IAU", period="3mo")

    xau_usd = float(df_gold["Close"].iloc[-1])
    xau_prev = float(df_gold["Close"].iloc[-2])
    xau_chg_pct = ((xau_usd - xau_prev) / xau_prev) * 100

    usd_idr = float(df_usdidr["Close"].iloc[-1])
    spot_idr_gram = (xau_usd * usd_idr) / TROY_OZ_TO_GRAM

    # Status Bullish / Bearish / Stagnan sesuai aturan -0.2% s/d +0.2%
    if xau_chg_pct > 0.2:
        trend_status = "BULLISH"
    elif xau_chg_pct < -0.2:
        trend_status = "BEARISH"
    else:
        trend_status = "STAGNAN"

    # Riwayat OHLC 180 hari terakhir untuk grafik Candlestick & Spot IDR
    ohlc_history = []
    recent_gold = df_gold.tail(180)
    for dt, row in recent_gold.iterrows():
        dt_str = dt.strftime("%Y-%m-%d")
        fx_val = float(df_usdidr["Close"].asof(dt)) if not df_usdidr.empty else usd_idr
        ohlc_history.append({
            "date": dt_str,
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
            "idr_gram": round((float(row["Close"]) * fx_val) / TROY_OZ_TO_GRAM, 0)
        })

    # Kurs 6 mata uang asing via ExchangeRate open endpoint
    fx_rates = {"USD": usd_idr, "EUR": 17100, "SGD": 11850, "JPY": 104.5, "CNY": 2190, "HKD": 2035}
    try:
        r_fx = requests.get("https://open.er-api.com/v6/latest/USD", timeout=10).json()
        rates = r_fx.get("rates", {})
        if "IDR" in rates:
            base_idr = rates["IDR"]
            fx_rates = {
                "USD": round(base_idr, 2),
                "EUR": round(base_idr / rates["EUR"], 2),
                "SGD": round(base_idr / rates["SGD"], 2),
                "JPY": round(base_idr / rates["JPY"], 2),
                "CNY": round(base_idr / rates["CNY"], 2),
                "HKD": round(base_idr / rates["HKD"], 2)
            }
            usd_idr = fx_rates["USD"]
            spot_idr_gram = (xau_usd * usd_idr) / TROY_OZ_TO_GRAM
    except Exception as e:
        print(f"[WARN] Fallback FX rates: {e}")

    tips_10y = fetch_fred_series("DFII10")
    us_cpi = fetch_fred_series("CPIAUCSL")
    fed_funds = fetch_fred_series("DFF")

    silver_usd = float(df_silver["Close"].iloc[-1])
    gold_silver_ratio = xau_usd / silver_usd

    gld_chg = float(((df_gld["Close"].iloc[-1] / df_gld["Close"].iloc[-2]) - 1) * 100) if len(df_gld) > 1 else 0

    antam_data = fetch_antam_prices(spot_idr_gram)
    technicals = compute_technicals(df_gold, usd_idr)
    quant = compute_quant_models(df_gold, df_dxy, df_us10y, df_ihsg, df_usdidr, technicals, gld_chg)
    cftc_data = fetch_cftc_gold()
    news_list = fetch_gold_news()

    bundle = {
        "meta": {
            "brand": "DELANOGOLD",
            "generated_at_wib": NOW_WIB.strftime("%Y-%m-%d %H:%M:%S WIB"),
            "timestamp_unix": int(NOW_WIB.timestamp())
        },
        "gold_spot": {
            "xau_usd": round(xau_usd, 2),
            "change_pct": round(xau_chg_pct, 2),
            "trend_status": trend_status,
            "spot_idr_gram": round(spot_idr_gram, 0),
            "source": "COMEX / Yahoo Finance (GC=F)"
        },
        "antam": antam_data,
        "indonesia_economy": {
            "ihsg": {
                "value": round(float(df_ihsg["Close"].iloc[-1]), 2),
                "change_pct": round(float(((df_ihsg["Close"].iloc[-1] / df_ihsg["Close"].iloc[-2]) - 1) * 100), 2),
                "source": "BEI / Yahoo Finance (^JKSE)"
            },
            "bi_rate_pct": 6.00,
            "indonesia_inflation_yoy_pct": 1.71,
            "fx_to_idr": fx_rates
        },
        "macro_drivers": {
            "tips_10y_real_yield": tips_10y,
            "dxy": {
                "value": round(float(df_dxy["Close"].iloc[-1]), 2),
                "change_pct": round(float(((df_dxy["Close"].iloc[-1] / df_dxy["Close"].iloc[-2]) - 1) * 100), 2),
                "source": "ICE DXY (DX-Y.NYB)"
            },
            "fed_funds_rate": fed_funds,
            "us_cpi_index": us_cpi,
            "vix": round(float(df_vix["Close"].iloc[-1]), 2),
            "silver_usd": round(silver_usd, 2),
            "gold_silver_ratio": round(gold_silver_ratio, 2),
            "wti_oil_usd": round(float(df_oil["Close"].iloc[-1]), 2),
            "us_10y_nominal_yield": round(float(df_us10y["Close"].iloc[-1]), 2)
        },
        "flows_and_positioning": {
            "cftc_cot": cftc_data,
            "etf_gld": {
                "price": round(float(df_gld["Close"].iloc[-1]), 2),
                "change_pct": round(gld_chg, 2),
                "volume": int(df_gld["Volume"].iloc[-1])
            },
            "etf_iau": {
                "price": round(float(df_iau["Close"].iloc[-1]), 2),
                "volume": int(df_iau["Volume"].iloc[-1])
            }
        },
        "technicals": technicals,
        "quantitative": quant,
        "ohlc_history_180d": ohlc_history,
        "economic_calendar": [
            {"event": "FOMC Interest Rate Decision", "impact": "HIGH", "source": "Federal Reserve"},
            {"event": "US CPI (Consumer Price Index)", "impact": "HIGH", "source": "BLS"},
            {"event": "US Non-Farm Payrolls (NFP)", "impact": "HIGH", "source": "BLS"},
            {"event": "RDG Bank Indonesia (BI-Rate)", "impact": "HIGH", "source": "Bank Indonesia"}
        ],
        "gold_news": news_list
    }

    os.makedirs("data", exist_ok=True)
    with open("data/market_bundle.json", "w", encoding="utf-8") as f:
        json.dump(clean_json(bundle), f, ensure_ascii=False, indent=2)

    print("Sukses! File data/market_bundle.json berhasil diperbarui.")


if __name__ == "__main__":
    main()
