# -*- coding: utf-8 -*-
"""
M-BANK3 intraday index snapshot (equal-weight, vs prev close, percent)

- 3 銘柄（8306.T / 8316.T / 8411.T）を等ウェイトで合成
- 前日終値比（%）を 5 分足で算出（取得できない場合は 15 分足へ自動フォールバック）
- 直近の取引日（JST）だけを抽出
- 共通グリッドに reindex + ffill で整列
- クリップで異常値を抑制
- 出力:
    docs/outputs/mbank3_intraday.csv (ts,pct)
    docs/outputs/mbank3_intraday.png
    docs/outputs/mbank3_post_intraday.txt
    docs/outputs/mbank3_stats.json
"""

from __future__ import annotations

import json
import os
from typing import List, Dict, Optional, Tuple
from datetime import datetime, time

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ---------- 設定 ----------
JST_TZ = "Asia/Tokyo"
OUT_DIR = "docs/outputs"
TICKER_FILE = "docs/tickers_mbank3.txt"

CSV_PATH = os.path.join(OUT_DIR, "mbank3_intraday.csv")
IMG_PATH = os.path.join(OUT_DIR, "mbank3_intraday.png")
POST_PATH = os.path.join(OUT_DIR, "mbank3_post_intraday.txt")
STATS_PATH = os.path.join(OUT_DIR, "mbank3_stats.json")

# まず 5m を狙い、ダメなら 15m にフォールバック
PRIMARY_INTERVALS = ["5m", "15m"]
PRIMARY_PERIODS = ["3d", "7d"]

PCT_CLIP_LOW = -20.0
PCT_CLIP_HIGH = 20.0

SESSION_START = time(9, 0)    # 09:00 JST
SESSION_END = time(15, 30)    # 15:30 JST


# ---------- ユーティリティ ----------
def jst_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz=JST_TZ)


def load_tickers(path: str) -> List[str]:
    xs: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            xs.append(s)
    if not xs:
        raise RuntimeError("No tickers found in docs/tickers_mbank3.txt")
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
    if d.shape[1] == 1:
        s = d.iloc[:, 0]
    else:
        s = d[d.count(axis=0).idxmax()]
    return s.dropna().astype(float)


def last_trading_day(ts_index: pd.DatetimeIndex) -> datetime.date:
    idx = pd.to_datetime(ts_index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert(JST_TZ)
    return idx[-1].date()


def fetch_prev_close(ticker: str, day: datetime.date) -> float:
    d = yf.download(ticker, period="10d", interval="1d", auto_adjust=False, progress=False, prepost=False)
    if d.empty:
        raise RuntimeError(f"prev close empty for {ticker}")
    s = _to_series_1d_close(d)
    s.index = pd.to_datetime(s.index)
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    s = s.tz_convert(JST_TZ)
    s_before = s[s.index.date < day]
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
    s = _to_series_1d_close(d)
    idx = pd.to_datetime(s.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert(JST_TZ)
    return pd.Series(s.values, index=idx)


def fetch_intraday_series_smart(ticker: str) -> Tuple[pd.Series, str, str]:
    """
    5m/3d → 5m/7d → 15m/3d → 15m/7d の順に試す。
    戻り値: (Series[JST], period, interval)
    """
    last_err: Optional[Exception] = None
    for interval in PRIMARY_INTERVALS:
        for period in PRIMARY_PERIODS:
            try:
                s = _try_download(ticker, period, interval)
                if not s.empty:
                    return s, period, interval
            except Exception as e:
                last_err = e
    if last_err:
        print(f"[WARN] all intraday attempts failed for {ticker}: {last_err!r}")
    return pd.Series(dtype=float), "", ""


def session_grid(day: datetime.date, freq: str) -> pd.DatetimeIndex:
    start = pd.Timestamp.combine(pd.Timestamp(day), SESSION_START).tz_localize(JST_TZ)
    end = pd.Timestamp.combine(pd.Timestamp(day), SESSION_END).tz_localize(JST_TZ)
    now = jst_now()
    end = min(end, now.floor(freq))
    return pd.date_range(start=start, end=end, freq=freq, tz=JST_TZ)


def build_equal_weight_pct(tickers: List[str]) -> Tuple[pd.Series, str]:
    """
    等ウェイト [%] シリーズと、使った頻度（'5m' or '15m'）を返す
    """
    indiv_pct: Dict[str, pd.Series] = {}

    # probe
    probe_t, probe_s, _, probe_iv = None, None, "", ""
    for t in tickers:
        s, p, iv = fetch_intraday_series_smart(t)
        if not s.empty:
            probe_t, probe_s, probe_iv = t, s, iv
            break
    if probe_s is None:
        print("[ERROR] probe series not available")
        return pd.Series(dtype=float), "5m"

    day = last_trading_day(probe_s.index)
    grid_freq = "5m" if probe_iv == "5m" else "15m"
    grid = session_grid(day, grid_freq)
    print(f"[INFO] target trading day (JST): {day} (probe={probe_t}, interval={probe_iv})")

    def _slice_day(s: pd.Series) -> pd.Series:
        x = s[(s.index.date == day)]
        if x.empty:
            d2 = last_trading_day(s.index)
            x = s[(s.index.date == d2)]
        return x

    for t in tickers:
        try:
            s = probe_s if t == probe_t else fetch_intraday_series_smart(t)[0]
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
        print("[ERROR] 0 series collected. Check tickers/network.")
        return pd.Series(dtype=float), grid_freq

    df = pd.concat(indiv_pct.values(), axis=1)
    series = df.mean(axis=1, skipna=True).astype(float)
    series.name = "M_BANK3"
    return series, grid_freq


# ---------- 出力 ----------
def save_ts_pct_csv(series: pd.Series, path: str, grid_freq: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if series is None or len(series) == 0:
        # 空でも当日グリッドを 0.0 で出す（開発時の視認性用）
        grid = session_grid(jst_now().date(), grid_freq)
        out = pd.DataFrame({"ts": grid.strftime("%Y-%m-%dT%H:%M:%S%z"), "pct": 0.0})
        out.to_csv(path, index=False)
        print(f"[INFO] CSV written with zero-filled grid rows: {len(out)} (no data case)")
        return

    s = series.dropna()
    out = pd.DataFrame({
        "ts": s.index.tz_convert(JST_TZ).strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pct": s.round(4).values
    })
    out.to_csv(path, index=False)
    print(f"[INFO] wrote CSV rows: {len(out)}")


def plot_debug(series: Optional[pd.Series], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.close("all")
    fig, ax = plt.subplots(figsize=(14, 6), dpi=140)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    for sp in ax.spines.values():
        sp.set_color("#333333")
    ax.grid(True, color="#2a2a2a", alpha=0.5, linestyle="--", linewidth=0.7)

    title = f"M-BANK3 Intraday Snapshot ({jst_now().strftime('%Y/%m/%d %H:%M JST')})"
    if series is None or len(series) == 0:
        ax.set_title(title + "  (no data)", color="white")
        ax.axhline(0, color="#666666", linewidth=1.0)
    else:
        ax.plot(series.index, series.values, color="#f87171", linewidth=2.0)
        ax.axhline(0, color="#666666", linewidth=1.0)
        ax.set_title(title, color="white")
    ax.tick_params(colors="white")
    ax.set_xlabel("Time", color="white")
    ax.set_ylabel("Change vs Prev Close (%)", color="white")
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_post_and_stats(series: Optional[pd.Series], post_path: str, stats_path: str) -> None:
    if series is None or len(series) == 0:
        last = 0.0
    else:
        s = series.dropna()
        last = float(s.iloc[-1]) if not s.empty else 0.0

    sign = "+" if last >= 0 else ""
    text = (
        f"▲ M-BANK3 日中スナップショット（{jst_now().strftime('%Y/%m/%d %H:%M JST')}）\n"
        f"{sign}{last:.2f}%（前日終値比）\n"
        f"※ 三菱UFJ・三井住友・みずほの等ウェイト\n"
        f"#メガバンク  #M_BANK3 #日本株\n"
    )
    os.makedirs(os.path.dirname(post_path), exist_ok=True)
    with open(post_path, "w", encoding="utf-8") as f:
        f.write(text)

    stats = {
        "index_key": "mbank3",
        "label": "M-BANK3",
        "pct_intraday": float(last),
        "basis": "prev_close",
        "updated_at": jst_now().isoformat(),
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


# ---------- メイン ----------
def main():
    try:
        tickers = load_tickers(TICKER_FILE)
        print(f"[INFO] tickers: {', '.join(tickers)}")

        print("[INFO] building equal-weight percent series...")
        series, grid_freq = build_equal_weight_pct(tickers)

        save_ts_pct_csv(series, CSV_PATH, grid_freq)
        plot_debug(series, IMG_PATH)
        save_post_and_stats(series, POST_PATH, STATS_PATH)

        print("[INFO] done.")
        if series is not None and len(series) > 0:
            tail = pd.DataFrame({"ts": series.index[-5:], "pct": series[-5:]})
            print("[INFO] tail:")
            print(tail)
        else:
            print("[INFO] series empty → zero-filled grid CSV written")
    except Exception as e:
        print(f"[FATAL] intraday build failed: {e!r}")


if __name__ == "__main__":
    main()
