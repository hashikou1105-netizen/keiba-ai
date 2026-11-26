"""海外の成績を入れる前と後で、AI の当てる力（AUC）を比べる
2023年までで勉強（2024年で小テスト）、2025年〜でテスト。乱数を変えた AI 3人の平均
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import os, sys
sys.path.insert(0, "src")   # AI本体のフォルダ（keiba_AI フォルダから実行する）
import features as F
exec(open("src/train.py").read().split("def load_fukusho")[0].split("from features import make_features")[1])   # FEATURES, CATEGORIES

P = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 63, "min_data_in_leaf": 100,
     "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1}


def run(use_overseas):
    path = F.OVERSEAS
    if not use_overseas:
        F.OVERSEAS = "data/__なし__.csv"
    df = F.make_features()
    F.OVERSEAS = path
    for c in CATEGORIES:
        df[c] = df[c].astype("category")
    # 海外を走ったことがある馬か（今回より前に）
    df["海外経験"] = df.groupby("horse_id")["海外"].cumsum() - df["海外"] > 0
    race = df.groupby("race_id")
    df = df[~(df["レース名"].str.contains("新馬", na=False) | (df["コース"] == "障")
              | ((race["年齢"].transform("max") == 2) & (df["date"].dt.month <= 9)) | (df["海外"] == 1))]
    train = df[df["date"] < "2024-01-01"]
    valid = df[(df["date"] >= "2024-01-01") & (df["date"] < "2025-01-01")]
    test = df[df["date"] >= "2025-01-01"].copy()
    preds = []
    for seed in range(3):
        m = lgb.train({**P, "seed": seed}, lgb.Dataset(train[FEATURES], train["3着以内"]), 3000,
                      valid_sets=[lgb.Dataset(valid[FEATURES], valid["3着以内"])], callbacks=[lgb.early_stopping(50, verbose=False)])
        preds.append(m.predict(test[FEATURES], num_iteration=m.best_iteration))
    test["確率"] = np.mean(preds, axis=0)
    return test[["race_id", "horse_id", "馬名", "レース名", "レースの格", "3着以内", "確率", "海外経験"]]


before, after = run(False), run(True)
key = ["race_id", "horse_id"]
x = before.merge(after[key + ["確率", "海外経験"]], on=key, suffixes=("_前", "_後"))
x["海外経験"] = x["海外経験_後"]
auc = lambda d, c: roc_auc_score(d["3着以内"], d[c])
print(f"\nテスト（2025年〜） {len(x)}頭")
print(f"  全部の馬           入れる前 {auc(x, '確率_前'):.4f} → 入れた後 {auc(x, '確率_後'):.4f}")
o = x[x["海外経験"]]
print(f"  海外を走ったことがある馬 {len(o)}頭: 入れる前 {auc(o, '確率_前'):.4f} → 入れた後 {auc(o, '確率_後'):.4f}")
g = x[x["レースの格"] >= 4000]
print(f"  重賞（1着賞金4,000万以上） {len(g)}頭: 入れる前 {auc(g, '確率_前'):.4f} → 入れた後 {auc(g, '確率_後'):.4f}")
o = o.assign(差=o["確率_後"] - o["確率_前"])
print("\n海外を走ったことがある馬で、確率が大きく変わった例")
print(o.sort_values("差")[["馬名", "レース名", "3着以内", "確率_前", "確率_後"]].round(2).iloc[np.r_[0:5, -5:0]].to_string(index=False))
