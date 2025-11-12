# M-BANK3 (Megabanks Equal-Weight Intraday)


- 三菱UFJ(8306.T)・三井住友(8316.T)・みずほ(8411.T)の **等ウェイト** 指数（前日終値比%）。
- 5分足優先（15分足へ自動フォールバック）。
- 公式サイト（最初期デザイン）向けPNG/テキスト/JSONも自動生成。


## 出力
- `docs/outputs/mbank3_intraday.csv` — ts,pct（JST, ISO8601）
- `docs/outputs/mbank3_intraday.png` — デバッグ/サイト確認用
- `docs/outputs/mbank3_post_intraday.txt` — 投稿文
- `docs/outputs/mbank3_stats.json` — サイト用メタ


## 使い方
1. `docs/tickers_mbank3.txt` を確認（8306.T, 8316.T, 8411.T）
2. Actions: **Auto intraday & publish (5min, JST weekdays)** を `Run workflow`（営業時間内）
3. サイトに自動反映させる場合は、Secrets を設定：
- `SITE_REPO` = `SakuraIndex/Sakura-Index-Site`（例）
- `SITE_TOKEN` = PAT（`repo` 権限）


※ Secrets が未設定でも、このリポ内の `docs/outputs/*` は更新されます。
