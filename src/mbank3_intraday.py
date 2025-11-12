# -*- coding: utf-8 -*-
"pct": s.round(4).values,
})
out.to_csv(path, index=False)




def _plot(series: Optional[pd.Series], path: str):
os.makedirs(os.path.dirname(path), exist_ok=True)
plt.close("all")
fig, ax = plt.subplots(figsize=(14, 6), dpi=140)
fig.patch.set_facecolor("black")
ax.set_facecolor("black")
for sp in ax.spines.values():
sp.set_color("#333")
ax.grid(True, color="#2a2a2a", linestyle="--", linewidth=0.7, alpha=0.5)


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




def _save_post_and_json(series: Optional[pd.Series], txt_path: str, json_path: str):
if series is None or len(series) == 0:
last = 0.0
else:
s = series.dropna()
last = float(s.iloc[-1]) if not s.empty else 0.0
sign = "+" if last >= 0 else ""
text = (
f"▲ M-BANK3 日中スナップショット（{jst_now().strftime('%Y/%m/%d %H:%M JST')}）\n"
f"{sign}{last:.2f}%（前日終値比）\n"
f"※ メガバンク3社の等ウェイト\n"
f"#メガバンク #M_BANK3 #日本株\n"
)
os.makedirs(os.path.dirname(txt_path), exist_ok=True)
with open(txt_path, "w", encoding="utf-8") as f:
f.write(text)


info = {
"index_key": "mbank3",
"label": "M-BANK3",
"pct_intraday": float(last),
"basis": "prev_close",
"updated_at": jst_now().isoformat(),
}
with open(json_path, "w", encoding="utf-8") as f:
f.write(pd.io.json.dumps(info, ensure_ascii=False, indent=2))




def main():
try:
tickers = _load_tickers(TICKER_FILE)
print(f"[INFO] tickers: {', '.join(tickers)}")
series, grid_freq = build_equal_weight_pct(tickers)
_save_csv(series, CSV_PATH, grid_freq)
_plot(series, IMG_PATH)
_save_post_and_json(series, POST_PATH, JSON_PATH)
print("[INFO] done.")
except Exception as e:
print(f"[FATAL] mbank3 intraday failed: {e!r}")




if __name__ == "__main__":
main()
