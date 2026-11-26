"""今日の馬場に合わせたヒント（今日は道悪か・今日の馬場での成績・得意さ）を入れる前と後で比べる
2023年までで勉強（2024年で小テスト）、2025年〜でテスト。乱数を変えた LightGBM 3つの平均
「もしも馬場」の効き目も見る：テストの全レースの馬場を、良⇔重に入れ替えたとき、AIの順位がどれくらい変わるか
使い方: keiba_AI フォルダで  .venv/bin/python experiments/compare_baba.py
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

df = F.add_baba_parts(F.add_new_history(F.add_form_history(F.make_features())))
for c in CATEGORIES:
    df[c] = df[c].astype("category")
race = df.groupby("race_id")
df = df[~(df["レース名"].str.contains("新馬", na=False) | (df["コース"] == "障")
          | ((race["年齢"].transform("max") == 2) & (df["date"].dt.month <= 9)) | (df["海外"] == 1))]
train = df[df["date"] < "2024-01-01"]
valid = df[(df["date"] >= "2024-01-01") & (df["date"] < "2025-01-01")]
test = df[df["date"] >= "2025-01-01"].copy()
graded = test["レースの格"] >= 4000
wet = test["馬場状態"].isin(F.WET)
BASE = [c for c in FEATURES if c not in F.BABA_FEATURES]
cases = {"今のAI": BASE, "＋今日の馬場": BASE + F.BABA_FEATURES}


def norm(p, d):
    return p / pd.Series(p, index=d.index).groupby(d["race_id"]).transform("sum").to_numpy() * 3


def top1(c, mask):
    d = test[mask]
    return d.loc[d.groupby("race_id")[c].idxmax(), "3着以内"].mean()


rows, flips = [], {}
for name, feats in cases.items():
    ms = []
    for seed in range(3):
        ms.append(lgb.train({**P, "seed": seed}, lgb.Dataset(train[feats], train["3着以内"]), 3000,
                            valid_sets=[lgb.Dataset(valid[feats], valid["3着以内"])], callbacks=[lgb.early_stopping(50, verbose=False)]))
    pred = lambda d: norm(np.mean([m.predict(d[feats], num_iteration=m.best_iteration) for m in ms], axis=0), d)
    test[name] = pred(test)
    # もしも：馬場を入れ替える（良・稍重 → 重、重・不良 → 良）
    sw = test.copy()
    sw["馬場状態"] = pd.Categorical(np.where(sw["馬場状態"].isin(["重", "不良"]), "良", "重"), categories=test["馬場状態"].cat.categories)
    sw = F.add_baba_features(sw)
    sp = pred(sw)
    r1 = test.groupby("race_id")[name].rank(ascending=False)
    r2 = pd.Series(sp, index=test.index).groupby(test["race_id"]).rank(ascending=False)
    flips[name] = {"確率の平均の動き": np.abs(sp - test[name]).mean(), "順位が変わる馬の割合": (r1 != r2).mean(),
                   "1位が変わるレースの割合": (test.assign(a=r1, b=r2).groupby("race_id").apply(lambda g: g.loc[g["a"].idxmin(), "b"] != 1)).mean()}
    rows.append({"ヒント": name, "AUC（全部）": roc_auc_score(test["3着以内"], test[name]),
                 "AUC（重賞）": roc_auc_score(test.loc[graded, "3着以内"], test.loc[graded, name]),
                 "AUC（道悪のレース）": roc_auc_score(test.loc[wet, "3着以内"], test.loc[wet, name]),
                 "AI1位の3着以内率（全部）": top1(name, test.index == test.index),
                 "AI1位の3着以内率（重賞）": top1(name, graded),
                 "AI1位の3着以内率（道悪）": top1(name, wet)})
    print(rows[-1], flips[name], flush=True)

res = pd.DataFrame(rows)
res.to_csv("experiments/results/compare_baba.csv", index=False, encoding="utf-8-sig")
test[["race_id", "date", "馬番", "人気", "馬場状態", "レースの格", "3着以内"] + list(cases)].to_pickle("experiments/results/compare_baba_preds.pkl")
print(f"\nテスト：2025年〜 {test['race_id'].nunique()}レース（重賞 {test.loc[graded, 'race_id'].nunique()}、道悪 {test.loc[wet, 'race_id'].nunique()}レース）")
print(res.round(4).to_string(index=False))
print("\nもしも馬場（良⇔重を入れ替えたとき）")
print(pd.DataFrame(flips).T.round(4).to_string())
