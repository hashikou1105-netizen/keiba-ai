"""勉強に使う期間（何年分か）で、AIの強さが変わるか比べる
テストはいつもと同じ 2025年〜、小テストは 2024年。勉強に使うのは「○年〜2023年」
ヒント（過去の成績など）は全部のデータから作る。変えるのは「勉強に使うレース」だけ
使い方: keiba_AI フォルダで  .venv/bin/python experiments/compare_years.py
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
valid = df[(df["date"] >= "2024-01-01") & (df["date"] < "2025-01-01")]
test = df[df["date"] >= "2025-01-01"].copy()
graded = test["レースの格"] >= 4000


def run(train, weight=None):
    preds = []
    for seed in range(3):
        m = lgb.train({**P, "seed": seed}, lgb.Dataset(train[FEATURES], train["3着以内"], weight=weight), 3000,
                      valid_sets=[lgb.Dataset(valid[FEATURES], valid["3着以内"])], callbacks=[lgb.early_stopping(50, verbose=False)])
        preds.append(m.predict(test[FEATURES], num_iteration=m.best_iteration))
    return np.mean(preds, axis=0)


def top1(c, mask):
    d = test[mask]
    return d.loc[d.groupby("race_id")[c].idxmax(), "3着以内"].mean()


cases = {}
for start in [2015, 2017, 2019, 2021, 2022]:
    cases[f"{start}〜2023年（{2024 - start}年分）"] = (df[(df["date"] >= f"{start}-01-01") & (df["date"] < "2024-01-01")], None)
# 全部使うけど、新しい年ほど重くする（1年古くなるごとに 0.85 倍）
tr = df[df["date"] < "2024-01-01"]
cases["全部＋新しい年ほど重視"] = (tr, 0.85 ** (2023 - tr["date"].dt.year).values)

rows = []
for name, (tr, w) in cases.items():
    test[name] = run(tr, w)
    rows.append({"勉強に使う期間": name, "勉強の頭数": len(tr),
                 "AUC（全部）": roc_auc_score(test["3着以内"], test[name]),
                 "AUC（重賞）": roc_auc_score(test.loc[graded, "3着以内"], test.loc[graded, name]),
                 "AI1位の3着以内率（全部）": top1(name, test.index == test.index),
                 "AI1位の3着以内率（重賞）": top1(name, graded)})
    print(rows[-1], flush=True)

res = pd.DataFrame(rows)
res.to_csv("experiments/results/compare_years.csv", index=False, encoding="utf-8-sig")
test[["race_id", "date", "馬番", "人気", "レースの格", "3着以内"] + list(cases)].to_pickle("experiments/results/compare_years_preds.pkl")
print(f"\nテスト：2025年〜 {test['race_id'].nunique()}レース（重賞 {test.loc[graded, 'race_id'].nunique()}レース）")
print(res.round(4).to_string(index=False))
