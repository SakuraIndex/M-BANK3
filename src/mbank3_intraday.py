# -*- coding: utf-8 -*-
"""
M-BANK3 intraday index snapshot (equal-weight, vs prev close, percent)

- 構成3銘柄を等ウェイトで合成（前日終値比 %）
- 5分足優先（ダメなら15分へフォールバック）
- 直近取引日（JST）のみ抽出
- 共通グリッドへ reindex + ffill で整列
- 出力先:
    docs/outputs/mbank3_intraday.csv   (ts,pct)
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
from datetime import time

# ====== 設定 ======
JST_TZ = "Asia/Tokyo"
OUT_DIR = "docs/outputs"
TICKER_FILE = "docs/tickers_mbank3.txt"

CSV_PATH  = os.path.join(OUT_DIR, "mbank3_intraday.csv")
IMG_PATH  = os.path.join(OUT_DIR, "mbank3_intraday.png")
POST_PATH = os.path.join(OUT_DIR, "mbank3_post_intraday.txt")
STAT_PATH = os.path.join(OUT_DIR, "mbank3_stats.json")

# yfinance の interval は "5m"/"15m"、pandas の freq は "5min"/"15min"
YF_INTERVALS  = ["5m", "15m"]
YF_PERIODS    = ["3d", "7d"]
PD_FREQ_5     = "5min"
PD_FREQ_15    = "15min"

PCT_CLIP_LOW  = -20.0
PCT_CLIP_HIGH =  20.0

SESSION_START = time(9, 0)
SESSION_END   = time(15, 30)

# ====== ユーティリティ ======
def jst_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz=JST_TZ)

def session_bounds(day: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp.combine(day, SESSION_START).tz_localize(JST_TZ)
    end   = pd.Timestamp.combine(day, SESSION_END).tz_localize(JST_TZ)
    if end <= start:
        end += pd.Timedelta(days=1)
    return start, end

def make_grid(day: pd.Timestamp, until: Optional[pd.Timestamp], pd_freq: str) -> pd.DatetimeIndex:
    start, end = session_bounds(day)
    if until is not None:
        end = min(end, until.floor(pd_freq))
    return pd.date_range(start=start, end=end, freq=pd_freq, tz=JST_TZ)

def load_tickers(path: str) -> List[str]:
    xs: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            xs.append(s)
    if not xs:
        raise RuntimeError(f"No tickers found in {path}")
    return xs

def _to_close_series(df: pd.DataFrame) -> pd.Series:
    """DataFrame から Close 1列の Series を返す（複数列も許容）"""
    if "Close" not in df.columns:
        raise ValueError("Close column not found")
    close = df["Close"]
    if isinstance(close, pd.Series):
        s = pd.to_numeric(close, errors="coerce")
        return s.dropna()
    # マルチカラム
    d = close.apply(pd.to_numeric, errors="coerce")
    d = d.loc[:, d.notna().any(axis=0)]
    if d.shape[1] == 0:
        raise ValueError("no numeric Close column")
    s = d.iloc[:, 0] if d.shape[1] == 1 else d[d.count(axis=0).idxmax()]
    return s.dropna().astype(float)

def last_trading_day(idx: pd.DatetimeIndex) -> pd.Timestamp:
    i = pd.to_datetime(idx)
    if i.tz is None:
        i = i.tz_localize("UTC")
    i = i.tz_convert(JST_TZ)
    return i[-1].normalize()

def fetch_prev_close(ticker: str, day: pd.Timestamp) -> float:
    d = yf.download(ticker, period="10d", interval="1d", auto_adjust=False, progress=False, prepost=False)
    if d.empty:
        raise RuntimeError(f"prev close empty for {ticker}")
    s = _to_close_series(d)
    s.index = pd.to_datetime(s.index, utc=True).tz_convert(JST_TZ)
    s_before = s[s.index.date < day.date()]
    return float((s_before.iloc[-1] if not s_before.empty else s.iloc[-1]))

def _try_download(ticker: str, period: str, interval: str) -> pd.Series:
    d = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        prepost=False,
        threads=True,
    )
    if d.empty:
        return pd.Series(dtype=float)
    s = _to_close_series(d)
    idx = pd.to_datetime(s.index, utc=True).tz_convert(JST_TZ)
    return pd.Series(s.values, index=idx)

def fetch_intraday_smart(ticker: str) -> Tuple[pd.Series, str, str]:
    """5m/3d → 5m/7d → 15m/3d → 15m/7d の順に試行。戻り値: (Series[JST], period, interval)"""
    last_err: Optional[Exception] = None
    for iv in YF_INTERVALS:
        for pr in YF_PERIODS:
            try:
                s = _try_download(ticker, pr, iv)
                if not s.empty:
                    return s, pr, iv
            except Exception as e:
                last_err = e
    if last_err:
        print(f"[WARN] all intraday attempts failed for {ticker}: {last_err!r}")
    return pd.Series(dtype=float), "", ""

def first_available_probe(tickers: List[str]) -> Tuple[str, pd.Series, str]:
    """最初に時系列が取れた銘柄をプローブにする。戻り: (ticker, series, pandas_freq)"""
    last_err: Optional[Exception] = None
    for t in tickers:
        try:
            s, _, iv = fetch_intraday_smart(t)
            if not s.empty:
                pd_freq = PD_FREQ_5 if iv == "5m" else PD_FREQ_15
                return t, s, pd_freq
        except Exception as e:
            last_err = e
            print(f"[WARN] probe failed for {t}: {e}")
    raise RuntimeError(f"no available intraday series for probe (last_err={last_err})")

def build_equal_weight_pct(tickers: List[str]) -> Tuple[pd.Series, str]:
    """等ウェイト [%] と実際に使った pandas freq を返す"""
    indiv: Dict[str, pd.Series] = {}

    probe_t, probe_s, pd_freq = first_available_probe(tickers)
    day = last_trading_day(probe_s.index)
    print(f"[INFO] target trading day (JST): {day.date()} (probe={probe_t}, pd_freq={pd_freq})")

    def slice_day(s: pd.Series) -> pd.Series:
        x = s[(s.index.date == day.date())]
        if x.empty:
            d2 = last_trading_day(s.index)
            x = s[(s.index.date == d2.date())]
        return x

    grid = make_grid(day, until=jst_now(), pd_freq=pd_freq)

    for t in tickers:
        try:
            s = probe_s if t == probe_t else fetch_intraday_smart(t)[0]
            s = slice_day(s)
            if s.empty:
                print(f"[WARN] {t}: no intraday for target day, skip")
                continue
            prev = fetch_prev_close(t, day)
            pct = (s / prev - 1.0) * 100.0
            pct = pct.clip(lower=PCT_CLIP_LOW, upper=PCT_CLIP_HIGH)
            pct = pct.reindex(grid).ffill()
            indiv[t] = pct.rename(t)
        except Exception as e:
            print(f"[WARN] skip {t}  # {e}")

    if not indiv:
        print("[ERROR] 0 series collected. Check tickers/network.")
        return pd.Series(dtype=float), pd_freq

    df = pd.concat(indiv.values(), axis=1)
    series = df.mean(axis=1, skipna=True).astype(float)
    series.name = "M_BANK3"
    return series, pd_freq

# ====== 出力 ======
def ensure_outdir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

def save_csv(series: pd.Series, path: str, pd_freq: str) -> None:
    ensure_outdir(path)
    if series is None or len(series) == 0:
        # 空でも当日グリッドを0.0で出す（サイト側の表示を安定化）
        today = jst_now().normalize()
        grid = make_grid(today, until=jst_now(), pd_freq=pd_freq)
        out = pd.DataFrame({"ts": grid.strftime("%Y-%m-%dT%H:%M:%S%z"), "pct": 0.0})
        out.to_csv(path, index=False)
        print(f"[INFO] CSV zero-filled rows: {len(out)}")
        return
    s = series.dropna()
    out = pd.DataFrame({
        "ts": s.index.tz_convert(JST_TZ).strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pct": s.round(4).values,
    })
    out.to_csv(path, index=False)
    print(f"[INFO] wrote CSV rows: {len(out)}")

def plot_png(series: Optional[pd.Series], path: str) -> None:
    ensure_outdir(path)
    plt.close("all")
    fig, ax = plt.subplots(figsize=(14, 6), dpi=140)
    fig.patch.set_facecolor("black"); ax.set_facecolor("black")
    for sp in ax.spines.values(): sp.set_color("#333")
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
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

def save_post_and_stats(series: Optional[pd.Series], post_path: str, stat_path: str) -> None:
    ensure_outdir(post_path); ensure_outdir(stat_path)
    if series is None or len(series) == 0 or series.dropna().empty:
        last = 0.0
    else:
        last = float(series.dropna().iloc[-1])
    sign = "+" if last >= 0 else ""
    txt = (
        f"▲ M-BANK3 日中スナップショット（{jst_now().strftime('%Y/%m/%d %H:%M JST')}）\n"
        f"{sign}{last:.2f}%（前日終値比）\n"
        f"※ メガバンク3社の等ウェイト\n"
        f"#メガバンク  #M_BANK3 #日本株\n"
    )
    with open(post_path, "w", encoding="utf-8") as f:
        f.write(txt)

    import json
    with open(stat_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "index_key": "mbank3",
                "label": "M-BANK3",
                "pct_intraday": round(last, 4),
                "basis": "prev_close",
                "updated_at": jst_now().isoformat(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

# ====== メイン ======
def main():
    try:
        tickers = load_tickers(TICKER_FILE)
        print(f"[INFO] tickers: {', '.join(tickers)}")

        print("[INFO] building equal-weight percent series...")
        series, pd_freq = build_equal_weight_pct(tickers)

        save_csv(series, CSV_PATH, pd_freq)
        plot_png(series, IMG_PATH)
        save_post_and_stats(series, POST_PATH, STAT_PATH)

        if series is not None and len(series) > 0:
            tail = pd.DataFrame({"ts": series.index[-5:], "pct": series[-5:]})
            print("[INFO] tail:\n", tail)
        else:
            print("[INFO] series empty → zero-filled CSV/PNG/TXT/JSON written")
    except Exception as e:
        print(f"[FATAL] intraday build failed: {e!r}")

if __name__ == "__main__":
    main()
