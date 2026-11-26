"""海外のレースの成績を集める（netkeiba の馬のページから）
JRA のオープン以上を走ったことがある馬だけ調べる（海外に行くのは、ほぼこのクラスの馬）
保存先: data/overseas_results.csv（途中で止めても、続きから再開できる）
"""
import os, re, time
import pandas as pd
import requests
from bs4 import BeautifulSoup

import features as F

OUT = "data/overseas_results.csv"
H = {"User-Agent": "Mozilla/5.0"}


def overseas_runs(horse_id):
    """その馬の海外のレースだけを返す（race_id に英字が入っているのが海外）"""
    r = requests.get(f"https://db.netkeiba.com/horse/result/{horse_id}/", headers=H, timeout=30)
    r.encoding = r.apparent_encoding
    s = BeautifulSoup(r.text, "lxml")
    rows = []
    for tr in s.select("table.db_h_race_results tr")[1:]:
        td = [x.get_text(strip=True) for x in tr.find_all("td")]
        link = [a.get("href", "") for a in tr.find_all("a") if "/race/" in a.get("href", "") and "/list/" not in a.get("href", "") and "/sum/" not in a.get("href", "")]
        rid = re.search(r"/race/(\w+)/", link[0]).group(1) if link else None
        if rid is None or rid.isdigit():
            continue
        m = re.match(r"(芝|ダ)(\d+)", td[14])
        rows.append({"race_id": rid, "horse_id": horse_id, "date": td[0].replace("/", "-"), "開催": td[1], "レース名": td[4],
                     "出走頭数": td[6], "馬番": td[8], "人気": td[10], "着順": td[11], "騎手": td[12], "斤量": td[13],
                     "コース": m.group(1) if m else None, "距離": m.group(2) if m else None, "馬場状態": td[16]})
    return rows


# 調べる馬：オープン以上（1着賞金2,400万円以上）のレースを走ったことがある馬
d = F.clean(F.load_results())
d["格"] = d.groupby("race_id")["賞金"].transform("max")
horses = sorted(d.loc[d["格"] >= 2400, "horse_id"].dropna().unique())
done = set()
if os.path.exists(OUT):
    done = set(pd.read_csv(OUT, dtype=str)["horse_id"])
if os.path.exists(OUT + ".done"):
    done |= set(open(OUT + ".done").read().split())
todo = [h for h in horses if h not in done]
print(f"{len(horses)}頭のうち、あと{len(todo)}頭を調べます")

for i, h in enumerate(todo, 1):
    try:
        rows = overseas_runs(h)
    except Exception as e:
        print("失敗", h, e); time.sleep(5); continue
    if rows:
        pd.DataFrame(rows).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
    with open(OUT + ".done", "a") as f:
        f.write(h + "\n")
    if i % 200 == 0:
        n = len(pd.read_csv(OUT)) if os.path.exists(OUT) else 0
        print(f"{i}/{len(todo)}頭  海外のレース {n}走", flush=True)
    time.sleep(0.6)
print("おわり")
