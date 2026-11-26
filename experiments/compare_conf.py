"""AI1位の確率が高いときの3着以内率を、いま／調子だけ／全部で比べる。もとは調子・騎手・牝馬限定・血統のヒント（近走の流れ・人気との差・休み明け何戦目・道悪の成績など）を入れる前と後で AUC を比べる
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


cols = {"いま": FEATURES, "調子だけ": FEATURES + F.FORM_FEATURES, "全部": FEATURES + NEW}
for name, feats in cols.items():
    test[name], _ = run(feats)
    test[name] = test[name] / test.groupby("race_id")[name].transform("sum") * 3   # レースで合計3にそろえる
    print(name, "完了", flush=True)
test[["race_id", "date", "馬番", "人気", "レースの格", "3着以内"] + list(cols)].to_pickle("experiments/results/compare_conf_preds.pkl")

def table(d, label):
    print(f"\n【{label}】AI1位の確率ごとの、AI1位の3着以内率（レース数）")
    rows = []
    for name in cols:
        top = d.loc[d.groupby("race_id")[name].idxmax()]
        r = {"": name}
        for lo in [0, 0.5, 0.6, 0.7, 0.8]:
            t = top[top[name] >= lo]
            r[f"{int(lo*100)}%以上"] = f"{t['3着以内'].mean():.1%}（{len(t)}）"
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))

table(test, "全レース 2025年〜")
table(test[test["レースの格"] >= 4000], "重賞だけ")
