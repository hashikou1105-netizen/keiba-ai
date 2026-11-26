"""調子・騎手・牝馬限定・血統のヒント（近走の流れ・人気との差・休み明け何戦目・道悪の成績など）を入れる前と後で AUC を比べる
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
cols = {"いま": FEATURES}
for k, v in GROUPS.items():
    cols[f"＋{k}"] = FEATURES + v
cols["全部"] = FEATURES + NEW
imp = None
print(f"\nテスト（2025年〜） {len(test)}頭 / 重賞 {test.loc[g, 'race_id'].nunique()}レース")
print(f"{'':8} {'全体AUC':>8} {'重賞AUC':>8} {'AI1位の3着内':>10} {'重賞AI1位':>8}")
for name, feats in cols.items():
    test[name], im = run(feats)
    if name == "全部": imp = im
    top = test.loc[test.groupby("race_id")[name].idxmax()]
    gt = top[top["レースの格"] >= 4000]
    print(f"{name:8} {auc(test, name):8.4f} {auc(test[g], name):8.4f} {top['3着以内'].mean():10.1%} {gt['3着以内'].mean():8.1%}", flush=True)
imp = (imp / imp.sum()).sort_values(ascending=False)
rank = imp.rank(ascending=False).astype(int)
print("\n新しいヒントの大事さ")
for c in NEW:
    print(f"  {c:<24} {imp[c]:.2%}  {rank[c]}位/{len(imp)}")
