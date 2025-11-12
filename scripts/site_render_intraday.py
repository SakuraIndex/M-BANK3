# -*- coding: utf-8 -*-
"""Copy M-BANK3 intraday outputs into the Sakura-Index-Site layout (legacy design).


- expects the site repo to be checked out into ./site
- writes into: site/docs/charts/M_BANK3/
"""
from __future__ import annotations


import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
SRC = ROOT / "docs" / "outputs"
DST = SITE / "docs" / "charts" / "M_BANK3"


PNG_SRC = SRC / "mbank3_intraday.png"
TXT_SRC = SRC / "mbank3_post_intraday.txt"
JSON_SRC= SRC / "mbank3_stats.json"


PNG_DST = DST / "intrad ay.png" # site 側の命名に合わせるなら変更
TXT_DST = DST / "post_intraday.txt"
JSON_DST= DST / "stats.json"


# NOTE: 公式サイトは『初期デザイン』準拠を想定


def main():
if not SITE.exists():
print("[INFO] site repo not checked out. skip copy.")
return
DST.mkdir(parents=True, exist_ok=True)
if PNG_SRC.exists():
shutil.copy2(PNG_SRC, DST / "intraday.png")
if TXT_SRC.exists():
shutil.copy2(TXT_SRC, TXT_DST)
if JSON_SRC.exists():
shutil.copy2(JSON_SRC, JSON_DST)
print("[INFO] site assets updated.")


if __name__ == "__main__":
main()
