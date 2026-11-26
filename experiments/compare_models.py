"""いろいろなモデルを同じ条件で比べる（LightGBM・ロジスティック回帰・ランダムフォレスト・XGBoost・CatBoost）
2023年までで勉強（2024年で小テスト＝止めどころを決める）、2025年〜でテスト。ヒント（特徴量）は今のAIと同じ
使い方: keiba_AI フォルダで  .venv/bin/python experiments/compare_models.py
"""
import os, sys, time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss

sys.path.insert(0, "src")   # AI本体のフォルダ
import features as F
exec(open("src/train.py").read().split("def load_fukusho")[0].split("from features import make_features")[1])   # FEATURES, CATEGORIES

SEED = 0

# ---- データの準備（今のAIと同じ） ----
df = F.add_form_history(F.make_features())
for c in CATEGORIES:
    df[c] = df[c].astype("category")
race = df.groupby("race_id")
df = df[~(df["レース名"].str.contains("新馬", na=False) | (df["コース"] == "障")
          | ((race["年齢"].transform("max") == 2) & (df["date"].dt.month <= 9)) | (df["海外"] == 1))]
train = df[df["date"] < "2024-01-01"]
valid = df[(df["date"] >= "2024-01-01") & (df["date"] < "2025-01-01")]
test = df[df["date"] >= "2025-01-01"].copy()
y_tr, y_va = train["3着以内"], valid["3着以内"]
NUM = [c for c in FEATURES if c not in CATEGORIES]
print(f"勉強 {len(train)}頭 / 小テスト {len(valid)}頭 / テスト {len(test)}頭、ヒント {len(FEATURES)}個", flush=True)


def codes(d):
    """文字のヒントを数字の番号にする（ランダムフォレスト・XGBoost用）"""
    d = d[FEATURES].copy()
    for c in CATEGORIES:
        d[c] = d[c].cat.codes.replace(-1, np.nan)
    return d


# ---- それぞれのモデル ----
def m_lightgbm():
    import lightgbm as lgb
    P = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 63, "min_data_in_leaf": 100,
         "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "seed": SEED}
    m = lgb.train(P, lgb.Dataset(train[FEATURES], y_tr), 3000,
                  valid_sets=[lgb.Dataset(valid[FEATURES], y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
    return m.predict(test[FEATURES], num_iteration=m.best_iteration)


def m_logistic():
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), NUM),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), CATEGORIES)])
    X = lambda d: d[FEATURES].astype({c: str for c in CATEGORIES})
    m = make_pipeline(prep, LogisticRegression(max_iter=2000, C=1.0))
    m.fit(X(train), y_tr)
    return m.predict_proba(X(test))[:, 1]


def m_random_forest():
    from sklearn.ensemble import RandomForestClassifier
    m = RandomForestClassifier(n_estimators=300, min_samples_leaf=100, max_features="sqrt",
                               n_jobs=-1, random_state=SEED)
    m.fit(codes(train), y_tr)
    return m.predict_proba(codes(test))[:, 1]


def m_xgboost():
    import xgboost as xgb
    m = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.03, max_depth=7, min_child_weight=20,
                          subsample=0.8, colsample_bytree=0.8, tree_method="hist", enable_categorical=True,
                          early_stopping_rounds=50, eval_metric="logloss", n_jobs=-1, random_state=SEED)
    m.fit(train[FEATURES], y_tr, eval_set=[(valid[FEATURES], y_va)], verbose=False)
    return m.predict_proba(test[FEATURES])[:, 1]


def m_catboost():
    from catboost import CatBoostClassifier
    X = lambda d: d[FEATURES].astype({c: str for c in CATEGORIES})
    m = CatBoostClassifier(iterations=3000, learning_rate=0.05, depth=7, od_type="Iter", od_wait=50,
                           cat_features=CATEGORIES, random_seed=SEED, verbose=False, thread_count=-1)
    m.fit(X(train), y_tr, eval_set=(X(valid), y_va), use_best_model=True)
    return m.predict_proba(X(test))[:, 1]


MODELS = {"LightGBM（今のAI）": m_lightgbm, "ロジスティック回帰": m_logistic,
          "ランダムフォレスト": m_random_forest, "XGBoost": m_xgboost, "CatBoost": m_catboost}

# ---- 走らせて、同じ物差しで比べる ----
rows = []
graded = test["レースの格"] >= 4000


def top1(c, mask):
    d = test[mask]
    return d.loc[d.groupby("race_id")[c].idxmax(), "3着以内"].mean()


for name, fn in MODELS.items():
    t0 = time.time()
    test[name] = fn()
    sec = time.time() - t0
    r = {"モデル": name,
         "AUC（全部）": roc_auc_score(test["3着以内"], test[name]),
         "AUC（重賞）": roc_auc_score(test.loc[graded, "3着以内"], test.loc[graded, name]),
         "AI1位の3着以内率（全部）": top1(name, test.index == test.index),
         "AI1位の3着以内率（重賞）": top1(name, graded),
         "logloss": log_loss(test["3着以内"], test[name]),
         "かかった時間（秒）": round(sec)}
    rows.append(r)
    print(f"{name:<16} AUC {r['AUC（全部）']:.4f} / 重賞 {r['AUC（重賞）']:.4f}  "
          f"AI1位 {r['AI1位の3着以内率（全部）']:.1%} / 重賞 {r['AI1位の3着以内率（重賞）']:.1%}  {sec:.0f}秒", flush=True)

res = pd.DataFrame(rows)
os.makedirs("experiments/results", exist_ok=True)
res.to_csv("experiments/results/compare_models.csv", index=False, encoding="utf-8-sig")
test[["race_id", "date", "馬番", "人気", "レースの格", "3着以内"] + list(MODELS)].to_pickle("experiments/results/compare_models_preds.pkl")
print(f"\nテスト：2025年〜 {test['race_id'].nunique()}レース（重賞 {test.loc[graded, 'race_id'].nunique()}レース）")
print(res.to_string(index=False))
