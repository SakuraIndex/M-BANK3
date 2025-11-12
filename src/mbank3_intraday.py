# -*- coding: utf-8 -*-
"""
M-BANK3 intraday index snapshot (equal-weight, vs prev close, percent)

- 3 銘柄（三菱UFJ:8306.T、三井住友:8316.T、みずほ:8411.T）を等ウェイトで合成
- 前日終値比（%）を 5 分足で算出（失敗時は 15 分へフォールバック）
- 直近取引日（JST）のみ抽出し、共通 5 分グリッドに reindex + ffill
- クリップで異常値を抑制
- 出力:
    docs/outputs/mbank3_intraday.csv (ts,pct)
    docs/outputs/mbank3_intraday.png
    docs/outputs/mbank3_post_intraday.txt
    docs/outputs/mbank3_stats.json
"""

from __future__ import annotations

import os
from typing import List, Dict, Optional, Tuple
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ---------- 設定 ----------
JST_TZ = "Asia/Tokyo"
OUT_DIR = "docs/outputs"
TICKER_FILE = "docs/tickers_mbank3.txt"

CSV_PATH  = os.path.join(OUT_DIR, "mbank3_intraday.csv")
IMG_PATH  = os.path.join(OUT_DIR, "mbank3_intraday.png")
POST_PATH = os.path.join(OUT_DIR, "mbank3_post_intraday.txt")
JSON_PATH = os.path.join(OUT_DIR, "mbank3_stats.json")

# 5分優先。フォールバックは15分
PRIMARY_INTERVALS  = ["5m", "15m"]            # yfinance 側の文字列は m で OK
PCT_CLIP_LOW  = -20.0
PCT_CLIP_HIGH =  20.0

# Pandas の minute 周波数（※ '5m' は月末と解釈されるので使わない）
PD_FREQ_5  = "5min"
PD_FREQ_15 = "15min"

SESSION_START = pd.Timestamp("09:00", tz=JST_TZ).time()
SESSION_END   = pd.Timestamp("15:30", tz=JST_TZ).time()

# ---------- ユーティリティ ----------
def jst_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz=JST_TZ)

def session_bounds(day: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp.combine(day.date(), SESSION_START, tz=JST_TZ)
    end   = pd.Timestamp.combine(day.date(), SESSION_END, tz=JST_TZ)
    if end <= start:
        end += pd.Timedelta(days=1)
    return start, end

def make_grid(day: pd.Timestamp, until: Optional[pd.Timestamp], freq: str) -> pd.DatetimeIndex:
    start, end = session_bounds(day)
    if until is not None:
        end = min(end, until.floor(freq))    # ← '5min' / '15min'
    return pd.date_range(start=start, end=end, freq=freq, tz=JST_TZ)

def load_tickers(path: str) -> List[str]:
    xs: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            xs.append(s)
    if not xs:
        raise RuntimeError("No tickers in docs/tickers_mbank3.txt")
    return xs

def _to_series_1d_close(df: pd.DataFrame) -> pd.Series:
    if "Close" not in df.columns:
        raise ValueError("Close column not found")
    close = df["Close"]
    if isinstance(close, pd.Series):
        return pd.to_numeric(close, errors="coerce").dropna()
    d = close.apply(pd.to_numeric, errors="coerce")
    mask = d.notna().any(axis=0)
    d = d.loc[:, mask]
    if d.shape[1] == 0:
        raise ValueError("no numeric close column")
    s = d.iloc[:, 0] if d.shape[1] == 1 else d[d.count(axis=0).idxmax()]
    return s.dropna().astype(float)

def last_trading_day(ts_index: pd.DatetimeIndex) -> pd.Timestamp:
    idx = pd.to_datetime(ts_index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert(JST_TZ)
    return idx[-1]

def fetch_prev_close(ticker: str, day: pd.Timestamp) -> float:
    d = yf.download(ticker, period="10d", interval="1d", auto_adjust=False, progress=False, prepost=False)
    if d.empty:
        raise RuntimeError(f"prev close empty for {ticker}")
    s = _to_series_1d_close(d)
    s.index = pd.to_datetime(s.index)
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    s = s.tz_convert(JST_TZ)
    s_before = s[s.index.date < day.date()]
    return float((s_before.iloc[-1] if not s_before.empty else s.iloc[-1]))

def _try_download(ticker: str, period: str, interval: str) -> pd.Series:
    d = yf.download(
        ticker, period=period, interval=interval,
        auto_adjust=False, progress=False, prepost=False, threads=True
    )
    if d.empty:
        return pd.Series(dtype=float)
    s = _to_series_1d_close(d)
    idx = pd.to_datetime(s.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert(JST_TZ)
    return pd.Series(s.values, index=idx)

def fetch_intraday_series_smart(ticker: str) -> Tuple[pd.Series, str]:
    """yfinance の intraday を period: 3d/7d, interval: 5m/15m で順番に試す。"""
    last_err: Optional[Exception] = None
    for interval in PRIMARY_INTERVALS:           # '5m' / '15m'（※ yfinance の指定）
        for period in ["3d", "7d"]:
            try:
                s = _try_download(ticker, period, interval)
                if not s.empty:
                    return s, interval
            except Exception as e:
                last_err = e
    if last_err:
        print(f"[WARN] intraday failed for {ticker}: {last_err!r}")
    return pd.Series(dtype=float), ""

def build_equal_weight_pct(tickers: List[str]) -> Tuple[pd.Series, str]:
    indiv_pct: Dict[str, pd.Series] = {}

    # プローブ銘柄で営業日と pandas グリッド頻度を決める
    probe_s, probe_iv = pd.Series(dtype=float), ""
    for t in tickers:
        s, iv = fetch_intraday_series_smart(t)
        if not s.empty:
            probe_s, probe_iv = s, iv
            break
    if probe_s.empty:
        print("[ERROR] no intraday series available for any ticker.")
        return pd.Series(dtype=float), PD_FREQ_5

    day = last_trading_day(probe_s.index)               # JST
    grid_freq = PD_FREQ_5 if probe_iv == "5m" else PD_FREQ_15
    print(f"[INFO] target trading day (JST): {day.date()} (interval={probe_iv})")

    def _slice_day(s: pd.Series) -> pd.Series:
        x = s[(s.index.date == day.date())]
        if x.empty:
            d2 = last_trading_day(s.index)
            x = s[(s.index.date == d2.date())]
        return x

    grid = make_grid(day, until=jst_now(), freq=grid_freq)

    for t in tickers:
        try:
            s, _iv = (probe_s, probe_iv) if (probe_s is not None and t == tickers[0] and not probe_s.empty) else fetch_intraday_series_smart(t)
            s = _slice_day(s)
            if s.empty:
                print(f"[WARN] {t}: no intraday for target day, skip")
                continue
            prev = fetch_prev_close(t, day)
            pct = (s / prev - 1.0) * 100.0
            pct = pct.clip(lower=PCT_CLIP_LOW, upper=PCT_CLIP_HIGH)
            pct = pct.reindex(grid).ffill()
            indiv_pct[t] = pct.rename(t)
        except Exception as e:
            print(f"[WARN] skip {t}  # {e}")

    if not indiv_pct:
        return pd.Series(dtype=float), grid_freq

    df = pd.concat(indiv_pct.values(), axis=1)
    series = df.mean(axis=1, skipna=True).astype(float)
    series.name = "M_BANK3"
    return series, grid_freq

# ---------- 出力 ----------
def ensure_outdir():
    os.makedirs(OUT_DIR, exist_ok=True)

def save_csv(series: pd.Series, pandas_freq: str):
    ensure_outdir()
    if series is None or len(series) == 0:
        # 空でもグリッド出力（0.0 埋め）
        today = jst_now().date()
        start = pd.Timestamp.combine(today, SESSION_START, tz=JST_TZ)
        now = jst_now()
        grid = pd.date_range(start=start, end=now.floor(pandas_freq), freq=pandas_freq, tz=JST_TZ)
        out = pd.DataFrame({"ts": grid.strftime("%Y-%m-%dT%H:%M:%S%z"), "pct": 0.0})
    else:
        s = series.dropna()
        out = pd.DataFrame({
            "ts": s.index.tz_convert(JST_TZ).strftime("%Y-%m-%dT%H:%M:%S%z"),
            "pct": s.round(4).values
        })
    out.to_csv(CSV_PATH, index=False)
    print(f"[INFO] wrote CSV rows: {len(out)}")

def save_plot(series: Optional[pd.Series]):
    ensure_outdir()
    plt.close("all")
    fig, ax = plt.subplots(figsize=(14, 6), dpi=140)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    for sp in ax.spines.values():
        sp.set_color("#333")
    ax.grid(True, color="#2a2a2a", alpha=0.5, linestyle="--", linewidth=0.7)
    title = f"M-BANK3 Intraday Snapshot ({jst_now().strftime('%Y/%m/%d %H:%M JST')})"
    if series is None or len(series) == 0:
        ax.set_title(title + " (no data)", color="white")
        ax.axhline(0, color="#666", linewidth=1.0)
    else:
        ax.plot(series.index, series.values, color="#f87171", linewidth=2.0)
        ax.axhline(0, color="#666", linewidth=1.0)
        ax.set_title(title, color="white")
    ax.tick_params(colors="white")
    ax.set_xlabel("Time", color="white")
    ax.set_ylabel("Change vs Prev Close (%)", color="white")
    fig.tight_layout()
    fig.savefig(IMG_PATH, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

def save_post_and_stats(series: Optional[pd.Series]):
    ensure_outdir()
    if series is None or len(series) == 0 or series.dropna().empty:
        last = 0.0
    else:
        last = float(series.dropna().iloc[-1])
    sign = "+" if last >= 0 else ""
    txt = (
        f"▲ M-BANK3 日中スナップショット（{jst_now().strftime('%Y/%m/%d %H:%M JST')}）\n"
        f"{sign}{last:.2f}%（前日終値比）\n"
        f"※ 三菱UFJ・三井住友・みずほの等ウェイト\n"
        f"#メガバンク  #M_BANK3 #日本株\n"
    )
    with open(POST_PATH, "w", encoding="utf-8") as f:
        f.write(txt)

    js = {
        "index_key": "mbank3",
        "label": "M-BANK3",
        "pct_intraday": float(last),
        "basis": "prev_close",
        "updated_at": jst_now().isoformat(),
    }
    import json
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        f.write(json.dumps(js, ensure_ascii=False, indent=2))

# ---------- メイン ----------
def main():
    try:
        tickers = load_tickers(TICKER_FILE)
        print(f"[INFO] tickers: {', '.join(tickers)}")
        series, grid_freq = build_equal_weight_pct(tickers)
        save_csv(series, grid_freq)
        save_plot(series)
        save_post_and_stats(series)
        if series is not None and len(series) > 0:
            print("[INFO] tail:")
            print(pd.DataFrame({"ts": series.index[-5:], "pct": series[-5:]}))
        else:
            print("[INFO] series empty -> zero-filled CSV emitted")
    except Exception as e:
        print(f"[FATAL] intraday build failed: {e!r}")

if __name__ == "__main__":
    main()
