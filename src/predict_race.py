"""指定したレースを、データだけの AI（3人の平均）で予想する（全馬を表示）
使い方: python src/predict_race.py <YYYY-MM-DD> <race_id> [race_id ...]
（出馬表が出る前の「登録馬」でも予想できる。そのときは馬番が空になる）
"""
import os, re, sys
import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

import features as F
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")).read().split("def load_fukusho")[0].split("from features import make_features")[1])   # FEATURES, CATEGORIES

DATE = sys.argv[1]
RACES = sys.argv[2:]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # keiba_AI フォルダ
OUT = os.path.join(ROOT, "predictions")      # レースごとの予想
MODELS = os.path.join(ROOT, "models")        # 学習したモデル
os.makedirs(OUT, exist_ok=True); os.makedirs(MODELS, exist_ok=True)
H = {"User-Agent": "Mozilla/5.0", "Referer": "https://race.netkeiba.com/"}
find = lambda p, t: (re.search(p, t).group(1) if re.search(p, t) else None)
BABA = {"稍": "稍重", "不": "不良", "重": "重", "良": "良"}


def get(url):
    r = requests.get(url, headers=H, timeout=30)
    r.encoding = r.apparent_encoding
    return BeautifulSoup(r.text, "lxml")


def shutuba(rid):
    s = get(f"https://race.netkeiba.com/race/shutuba.html?race_id={rid}")
    cond = " ".join(s.select_one(".RaceData01").get_text(" ", strip=True).split())
    info2 = " ".join(s.select_one(".RaceData02").get_text(" ", strip=True).split())
    m = re.search(r"(芝|ダ|障)(\d+)m", cond)
    baba = find(r"馬場:(\S+)", cond)
    rows = []
    for i, tr in enumerate(s.select("table.Shutuba_Table tr.HorseList"), 1):
        info = tr.select_one(".HorseInfo")
        if info is None or "Cancel" in tr.get("class", []):
            continue
        tds = tr.find_all("td")
        link = lambda sel: (tr.select_one(sel).find("a")["href"].rstrip("/").split("/")[-1]
                            if tr.select_one(sel) and tr.select_one(sel).find("a") else None)
        trainer = tr.select_one(".Trainer").get_text(" ", strip=True)
        rows.append({"race_id": rid, "着順": None, "枠番": tds[0].get_text(strip=True), "馬番": tds[1].get_text(strip=True) or str(100 + i),
                     "馬名": info.get_text(strip=True), "性齢": tr.select_one(".Barei").get_text(strip=True),
                     "斤量": tr.select_one(".Barei").find_next_sibling("td").get_text(strip=True),
                     "騎手": tr.select_one(".Jockey").get_text(strip=True), "タイム": None, "通過": None, "上り": None,
                     "馬体重": tr.select_one(".Weight").get_text("", strip=True).replace(" ", "") if tr.select_one(".Weight") else "",
                     "調教師": ("[東]" if "美浦" in trainer else "[西]") + trainer.split()[-1], "賞金(万円)": "0", "date": DATE,
                     "レース名": s.select_one(".RaceName").get_text(strip=True), "コース": m.group(1), "距離": int(m.group(2)),
                     "回り": find(r"(左|右|直線)", cond), "天候": find(r"天候:(\S+)", cond), "馬場状態": BABA.get(baba[:1], baba) if baba else None,
                     "horse_id": link(".HorseInfo"), "jockey_id": link(".Jockey"), "trainer_id": link(".Trainer"),
                     "本賞金": float(find(r"本賞金:(\d+)", info2) or 0)})
    ent = pd.DataFrame(rows)
    ent["出走頭数"] = min(len(ent), 18)
    return ent, cond


def odds(rid):
    """単勝オッズと人気（まだ出ていなければ空）"""
    try:
        j = requests.get(f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={rid}&type=1&action=update",
                         headers=H, timeout=30).json()
        w = j["data"]["odds"]["1"]
        return {int(k): (v[0], v[2]) for k, v in w.items()}
    except Exception:
        return {}


# 1. 出馬表を取って、特徴量を作る
ents, conds = [], {}
for rid in RACES:
    e, c = shutuba(rid)
    ents.append(e); conds[rid] = c
ent = pd.concat(ents, ignore_index=True).assign(海外=0)
df = pd.concat([F.load_results(), ent], ignore_index=True)
for c in ["枠番", "馬番", "斤量", "距離", "出走頭数"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = F.add_class_history(F.add_horse_history(F.clean(df)))
today = df["race_id"].isin(RACES)
df.loc[today, "レースの格"] = df.loc[today, "本賞金"]
df["前走との格の比"] = df["レースの格"] / df["前走のレースの格"]
df = F.add_speed_history(df)
for cond_, name in [("コース", "同じコース"), ("場", "同じ競馬場"), ("距離区分", "同じ距離区分")]:
    df = F.add_condition_history(df, cond_, name)
df = F.add_person_history(df, "jockey_id", "騎手", "騎手の騎乗数")
df = F.add_person_history(df, "trainer_id", "調教師", "調教師の出走数")
df = F.add_race_comparison(df)
df = F.add_form_history(df)   # 調子のヒント
df = F.add_new_history(df)    # 季節・脚質・枠・騎手の相性・同じ舞台のヒント
today = df["race_id"].isin(RACES)   # 並べ直したので、今日のレースの印を付け直す
for c in CATEGORIES:
    df[c] = df[c].astype("category")

# 2. モデル（その日の前までのデータで学習。保存済みならそれを使う）
race = df.groupby("race_id")
target = ~(df["レース名"].str.contains("新馬", na=False) | (df["コース"] == "障")
           | ((race["年齢"].transform("max") == 2) & (df["date"].dt.month <= 9)) | (df["海外"] == 1))
past = df[target & (df["date"] < DATE)]
tr_, va = past[past["date"] < "2025-07-01"], past[past["date"] >= "2025-07-01"]
P = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 63, "min_data_in_leaf": 100,
     "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1}
models = []
for seed in range(3):
    path = os.path.join(MODELS, f"model_{DATE}_v3_{seed}.txt")
    if os.path.exists(path):
        models.append(lgb.Booster(model_file=path)); continue
    m1 = lgb.train({**P, "seed": seed}, lgb.Dataset(tr_[FEATURES], tr_["3着以内"]), 3000,
                   valid_sets=[lgb.Dataset(va[FEATURES], va["3着以内"])], callbacks=[lgb.early_stopping(50, verbose=False)])
    mm = lgb.train({**P, "seed": seed}, lgb.Dataset(past[FEATURES], past["3着以内"]), m1.best_iteration)
    mm.save_model(path); models.append(mm)

# 3. 確率を出して全馬を表示
t = df[today].copy()
t["確率"] = np.mean([mm.predict(t[FEATURES]) for mm in models], axis=0)
t["確率"] = t["確率"] / t.groupby("race_id")["確率"].transform("sum") * 3
for rid in RACES:
    r = t[t["race_id"] == rid].sort_values("確率", ascending=False)
    r.to_pickle(os.path.join(OUT, f"race_{rid}.pkl"))
    o = odds(rid)
    place = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京", "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}.get(rid[4:6], "")
    print(f"\n■ {place}{int(rid[10:])}R {r['レース名'].iloc[0]}  {conds[rid]}  {len(r)}頭")
    for i, (_, x) in enumerate(r.head(int(os.environ.get("TOP", 99))).iterrows(), 1):
        num = int(x["馬番"])
        od = f"  単勝{o[num][0]}倍 {o[num][1]}人気" if num in o else ""
        print(f"  {i:2d}位 {num if num < 100 else '-':>2} {x['馬名']}（{x['騎手']}・{x['性齢']}・{x['斤量']}）{min(x['確率'], .99) * 100:.0f}%{od}")
