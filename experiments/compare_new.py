"""新しいヒント（季節・脚質・枠の有利さ・騎手の相性・同じ舞台）を入れる前と後で比べる
2023年までで勉強（2024年で小テスト）、2025年〜でテスト。乱数を変えた LightGBM 3つの平均（今のAIと同じ作り方）
使い方: keiba_AI フォルダで  .venv/bin/python experiments/compare_new.py
"""
import sys
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src")
import features as F
exec(open("src/train.py").read().split("def load_fukusho")[0].split("from features import make_features")[1])   # FEATURES, CATEGORIES

P = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 63, "min_data_in_leaf": 100,
     "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1}

df = F.add_new_history(F.add_form_history(F.make_features()))
for c in CATEGORIES:
    df[c] = df[c].astype("category")
race = df.groupby("race_id")
df = df[~(df["レース名"].str.contains("新馬", na=False) | (df["コース"] == "障")
          | ((race["年齢"].transform("max") == 2) & (df["date"].dt.month <= 9)) | (df["海外"] == 1))]
train = df[df["date"] < "2024-01-01"]
valid = df[(df["date"] >= "2024-01-01") & (df["date"] < "2025-01-01")]
test = df[df["date"] >= "2025-01-01"].copy()
graded = test["レースの格"] >= 4000
ALL_NEW = sum(F.NEW_GROUPS.values(), [])
FEATURES = [c for c in FEATURES if c not in ALL_NEW]   # 本番に入れた後でも「入れる前」と比べられるように


def run(feats):
    preds, imp = [], 0
    for seed in range(3):
        m = lgb.train({**P, "seed": seed}, lgb.Dataset(train[feats], train["3着以内"]), 3000,
                      valid_sets=[lgb.Dataset(valid[feats], valid["3着以内"])], callbacks=[lgb.early_stopping(50, verbose=False)])
        preds.append(m.predict(test[feats], num_iteration=m.best_iteration))
        imp = imp + pd.Series(m.feature_importance("gain"), index=feats)
    return np.mean(preds, axis=0), imp


def top1(c, mask):
    d = test[mask]
    return d.loc[d.groupby("race_id")[c].idxmax(), "3着以内"].mean()


cases = {"今のAI": FEATURES, **{f"＋{k}": FEATURES + v for k, v in F.NEW_GROUPS.items()}, "＋全部": FEATURES + ALL_NEW}
rows = []
for name, feats in cases.items():
    test[name], imp = run(feats)
    allm = test.index == test.index
    rows.append({"ヒント": name, "AUC（全部）": roc_auc_score(test["3着以内"], test[name]),
                 "AUC（重賞）": roc_auc_score(test.loc[graded, "3着以内"], test.loc[graded, name]),
                 "AI1位の3着以内率（全部）": top1(name, allm), "AI1位の3着以内率（重賞）": top1(name, graded)})
    print(rows[-1], flush=True)

res = pd.DataFrame(rows)
res.to_csv("experiments/results/compare_new.csv", index=False, encoding="utf-8-sig")
test[["race_id", "date", "馬番", "人気", "レースの格", "3着以内"] + list(cases)].to_pickle("experiments/results/compare_new_preds.pkl")
print(f"\nテスト：2025年〜 {test['race_id'].nunique()}レース（重賞 {test.loc[graded, 'race_id'].nunique()}レース）")
print(res.round(4).to_string(index=False))
imp = (imp / imp.sum()).sort_values(ascending=False)
rank = imp.rank(ascending=False).astype(int)
print("\n新しいヒントの大事さ（全部入れたとき）")
for c in ALL_NEW:
    print(f"  {c:<30} {imp[c]:.2%}  {rank[c]}位/{len(imp)}")
