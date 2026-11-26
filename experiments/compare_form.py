"""調子のヒント（近走の流れ・人気との差・休み明け何戦目・道悪の成績など）を入れる前と後で AUC を比べる
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

df = F.add_form_history(F.make_features())
NEW = F.FORM_FEATURES
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


test["前"], _ = run(FEATURES)
test["後"], imp = run(FEATURES + NEW)
auc = lambda d, c: roc_auc_score(d["3着以内"], d[c])
print(f"\nテスト（2025年〜） {len(test)}頭")
print(f"  全部の馬   入れる前 {auc(test, '前'):.4f} → 入れた後 {auc(test, '後'):.4f}")
g = test[test["レースの格"] >= 4000]
print(f"  重賞       入れる前 {auc(g, '前'):.4f} → 入れた後 {auc(g, '後'):.4f}")
w = test[test["馬場状態"].isin(["稍重", "重", "不良"])]
print(f"  道悪のレース 入れる前 {auc(w, '前'):.4f} → 入れた後 {auc(w, '後'):.4f}")
for c in ["前", "後"]:   # AI 1位が3着以内に来た割合
    top = test.loc[test.groupby("race_id")[c].idxmax()]
    print(f"  AI1位の3着以内率（{c}） {top['3着以内'].mean():.1%}")
imp = (imp / imp.sum()).sort_values(ascending=False)
print("\n新しいヒントの大事さ（全体の中の割合と順位）")
rank = imp.rank(ascending=False).astype(int)
for c in NEW:
    print(f"  {c:<24} {imp[c]:.2%}  {rank[c]}位/{len(imp)}")
