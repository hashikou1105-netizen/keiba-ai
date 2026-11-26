"""騎手の3着以内率を「騎手の人気以上率」に置き換えるテスト。もとは調子・騎手・牝馬限定・血統のヒント（近走の流れ・人気との差・休み明け何戦目・道悪の成績など）を入れる前と後で AUC を比べる
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

df = F.add_extra_history(F.add_form_history(F.make_features()))
import sys
GROUPS = {"調子": F.FORM_FEATURES, "騎手": F.EXTRA_FEATURES[:3], "牝馬": F.EXTRA_FEATURES[3:6], "血統": F.EXTRA_FEATURES[6:]}
NEW = sum(GROUPS.values(), [])
for c in CATEGORIES:
    df[c] = df[c].astype("category")
race = df.groupby("race_id")
df = df[~(df["レース名"].str.contains("新馬", na=False) | (df["コース"] == "障")
          | ((race["年齢"].transform("max") == 2) & (df["date"].dt.month <= 9)) | (df["海外"] == 1))]
r = df.groupby("race_id")["騎手の人気以上率"]
df["騎手の人気以上率のレース内順位"] = r.rank(ascending=False, method="min")
df["騎手の人気以上率のレース平均との差"] = df["騎手の人気以上率"] - r.transform("mean")
train = df[df["date"] < "2024-01-01"]
valid = df[(df["date"] >= "2024-01-01") & (df["date"] < "2025-01-01")]
test = df[df["date"] >= "2025-01-01"].copy()


def run(feats):
    preds, imp = [], 0
    for seed in range(3):
        m = lgb.train({**P, "seed": seed}, lgb.Dataset(train[feats], train["3着以内"]), 3000,
                      valid_sets=[lgb.Dataset(valid[feats], valid["3着以内"])], callbacks=[lgb.early_stopping(50, verbose=False)])
        preds.append(m.predict(test[feats], num_iteration=m.best_iteration))
        imp = imp + pd.Series(m.feature_importance("gain"), index=feats)
    return np.mean(preds, axis=0), imp


auc = lambda d, c: roc_auc_score(d["3着以内"], d[c])
g = test["レースの格"] >= 4000
OLD_J = ["騎手の3着以内率", "騎手の3着以内率のレース内順位", "騎手の3着以内率のレース平均との差"]
ALL = FEATURES + NEW
cols = {"いま": FEATURES, "全部": ALL,
        "全部・騎手置換": [f for f in ALL if f not in OLD_J] + ["騎手の人気以上率のレース内順位", "騎手の人気以上率のレース平均との差"]}
print(f"\nテスト（2025年〜） {len(test)}頭 / 重賞 {test.loc[g, 'race_id'].nunique()}レース")
for name, feats in cols.items():
    test[name], im = run(feats)
    test[name + "_n"] = test[name] / test.groupby("race_id")[name].transform("sum") * 3
    top = test.loc[test.groupby("race_id")[name].idxmax()]
    gt = top[top["レースの格"] >= 4000]
    print(f"{name:10} 全体AUC {auc(test, name):.4f}  重賞AUC {auc(test[g], name):.4f}  AI1位 {top['3着以内'].mean():.1%}  重賞AI1位 {gt['3着以内'].mean():.1%}", flush=True)

# 騎手ごと：AIの予想（レースで合計3にそろえた%）の平均と、実際の3着以内率
print("\n騎乗数の多い騎手：AIの予想% と 実際の3着以内率（2025年〜）")
jk = test.groupby("騎手").filter(lambda d: len(d) >= 400)
t = jk.groupby("騎手").agg(騎乗=("3着以内", "size"), 実際=("3着以内", "mean"),
                          いま=("いま_n", "mean"), 全部=("全部_n", "mean"), 置換=("全部・騎手置換_n", "mean"))
t = t.sort_values("実際", ascending=False).head(15)
print((t[["実際", "いま", "全部", "置換"]] * 100).round(1).assign(騎乗=t["騎乗"]).to_string())
# 人気薄（6番人気以下）でのルメール
lm = test[test["騎手"].str.contains("ルメール", na=False) & (pd.to_numeric(test["人気"], errors="coerce") >= 6)]
print(f"\nルメール騎手・6番人気以下（{len(lm)}回）実際 {lm['3着以内'].mean():.1%} / いま {lm['いま_n'].mean():.1%} / 全部 {lm['全部_n'].mean():.1%} / 置換 {lm['全部・騎手置換_n'].mean():.1%}")
