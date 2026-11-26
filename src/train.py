import glob

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features import make_features
from features import add_form_history, FORM_FEATURES
from features import add_new_history, NEW_GROUPS

# AI に渡すヒント（特徴量）
# ※ そのレースの結果（着順・タイム・上り・賞金など）は答えそのものなので、絶対に入れない
FEATURES = [
    # レースの条件
    "場", "コース", "距離", "回り", "馬場状態", "天候", "出走頭数",
    # 馬の基本情報（出馬表で分かるもの）
    "枠番", "馬番", "性別", "年齢", "斤量", "体重", "体重増減", "所属",
    # 馬の過去成績
    "過去の出走数", "前走の着順", "過去5走の平均着順", "過去の3着以内率",
    "過去5走の上り平均", "前走からの日数", "前走との距離の差",
    # レースの格と実績
    "レースの格", "前走のレースの格", "過去5走のレースの格の平均", "前走との格の比", "過去の獲得賞金",
    # 騎手・調教師の成績（前日まで）
    "騎手の3着以内率", "騎手の騎乗数", "調教師の3着以内率", "調教師の出走数",
    # 今回と同じ条件での成績
    "同じコースの出走数", "同じコースの3着以内率",
    "同じ競馬場の出走数", "同じ競馬場の3着以内率",
    "同じ距離区分の出走数", "同じ距離区分の3着以内率",
    # 同じレースの中で比べたヒント
    "前走の着順のレース内順位", "前走の着順のレース平均との差",
    "過去5走の平均着順のレース内順位", "過去5走の平均着順のレース平均との差",
    "過去5走の上り平均のレース内順位", "過去5走の上り平均のレース平均との差",
    "騎手の3着以内率のレース内順位", "騎手の3着以内率のレース平均との差",
    "同じコースの3着以内率のレース内順位", "同じコースの3着以内率のレース平均との差",
    "過去の獲得賞金のレース内順位", "過去の獲得賞金のレース平均との差",
    # タイム差とスピード指数
    "前走のタイム差", "過去5走のタイム差の平均", "過去5走の一番小さいタイム差",
    "前走のスピード指数", "過去5走のスピード指数の平均", "過去5走のスピード指数の最高",
    "過去5走のスピード指数の平均のレース内順位", "過去5走のスピード指数の平均のレース平均との差",
    "過去5走のタイム差の平均のレース内順位", "過去5走のタイム差の平均のレース平均との差",
    # 馬場差で補正したスピード指数
    "前走の補正スピード指数", "過去5走の補正スピード指数の平均", "過去5走の補正スピード指数の最高",
    "過去5走の補正スピード指数の平均のレース内順位", "過去5走の補正スピード指数の平均のレース平均との差",
    "過去5走の補正スピード指数の最高のレース内順位",
]

# 調子のヒント（2026-11-19 追加：前走の人気・近走の流れ・休み明け何戦目・道悪の成績など）
FEATURES = FEATURES + FORM_FEATURES

# 季節・脚質・枠の有利さ・騎手の相性・同じ舞台のヒント（2026-11-24 追加）
FEATURES = FEATURES + sum(NEW_GROUPS.values(), [])

# 文字のヒント（AI に「これは種類を表す文字だよ」と教える）
CATEGORIES = ["場", "コース", "回り", "馬場状態", "天候", "性別", "所属"]

# 何人の AI を作って平均するか
N_MODELS = 5


def load_fukusho():
    """複勝の払戻金を読み込む（race_id と 馬番 ごとの払戻）"""
    files = sorted(glob.glob("data/payouts_*.csv"))
    pay = pd.concat([pd.read_csv(f, dtype={"race_id": str, "馬番": str}) for f in files])
    pay = pay[pay["券種"] == "複勝"].copy()
    pay["馬番"] = pay["馬番"].astype(int)
    return pay[["race_id", "馬番", "払戻"]]


# ---- 1. データを用意する（ヒントは全レースを使って作る）
df = add_new_history(add_form_history(make_features()))
for c in CATEGORIES:
    df[c] = df[c].astype("category")

# ---- 予想するレースを決める（AI が苦手なレースは、練習にもテストにも使わない）
race = df.groupby("race_id")
new_horse = df["レース名"].str.contains("新馬", na=False)                  # 新馬戦
jump = df["コース"] == "障"                                               # 障害レース
early_2yo = (race["年齢"].transform("max") == 2) & (df["date"].dt.month <= 9)  # 2歳の早い時期（6〜9月）
overseas = df["海外"] == 1                                                # 海外のレース（過去の成績のヒントにだけ使う）
print(f"予想しないレース：新馬戦 {df.loc[new_horse, 'race_id'].nunique()} / "
      f"障害 {df.loc[jump, 'race_id'].nunique()} / 2歳の早い時期 {df.loc[early_2yo & ~new_horse, 'race_id'].nunique()}")
df = df[~(new_horse | jump | early_2yo | overseas)]

# ---- 2. 時期で3つに分ける
#   練習用（2015〜2023年）：AI が勉強する
#   確認用（2024年）      ：勉強しすぎを防ぐための小テスト
#   テスト用（2025年〜）  ：本番のテスト。AI はこのデータを一度も見ていない
train = df[df["date"] < "2024-01-01"]
valid = df[(df["date"] >= "2024-01-01") & (df["date"] < "2025-01-01")]
test = df[df["date"] >= "2025-01-01"].copy()
print(f"練習用 {len(train)} 頭 / 確認用 {len(valid)} 頭 / テスト用 {len(test)} 頭")

# ---- 3. 学習させる（乱数を変えた AI を 5 人作る）
params = {
    "objective": "binary",
    "learning_rate": 0.03,      # 学ぶ速さ：ゆっくり丁寧に
    "num_leaves": 63,           # 1本の木で分けられる数
    "min_data_in_leaf": 100,    # 1つの葉に最低100頭：少ないデータで細かく分けすぎない
    "feature_fraction": 0.8,    # 木を1本作るたびに、ヒントの80%をランダムに使う
    "bagging_fraction": 0.8,    # 木を1本作るたびに、馬の80%をランダムに使う
    "bagging_freq": 1,
    "verbose": -1,
}
models = []
for seed in range(N_MODELS):
    print(f"{seed + 1} 人目の AI を学習しています…")
    model = lgb.train(
        {**params, "seed": seed},   # 人ごとに乱数を変える（ランダムに選ぶヒントや馬が変わる）
        lgb.Dataset(train[FEATURES], train["3着以内"]),
        num_boost_round=3000,
        valid_sets=[lgb.Dataset(valid[FEATURES], valid["3着以内"])],
        callbacks=[lgb.early_stopping(50, verbose=False)],   # 小テストの成績が伸びなくなったら止める
    )
    models.append(model)

# ---- 4. テスト用データで予想させる（5人の確率の平均）
test["予想確率"] = np.mean([m.predict(test[FEATURES], num_iteration=m.best_iteration) for m in models], axis=0)

auc = roc_auc_score(test["3着以内"], test["予想確率"])
print(f"\nAUC: {auc:.3f}  （0.5 = でたらめ、1.0 = 完璧）")

# ---- 5. 「毎レース、AI の本命の複勝を100円買ったら？」を計算する
test = test.merge(load_fukusho(), on=["race_id", "馬番"], how="left")
test["払戻"] = test["払戻"].fillna(0)


def report(name, picks):
    hit = picks["3着以内"].mean()
    roi = picks["払戻"].sum() / (100 * len(picks))
    print(f"  {name:<12} 的中率 {hit:6.1%}   回収率 {roi:6.1%}   （{len(picks)} レース）")


print(f"\n毎レース1頭だけ、複勝を100円ずつ買った場合（{test['date'].min():%Y-%m-%d} 〜）")
report("AIの本命", test.loc[test.groupby("race_id")["予想確率"].idxmax()])
report("1番人気", test[test["人気"] == 1].drop_duplicates("race_id"))
report("ランダム", test.groupby("race_id").sample(1, random_state=0))

# ---- 6. 本命の確率が○%以上のときだけ買った場合（確率はレースごとに合計3にそろえる）
test["そろえた確率"] = test["予想確率"] / test.groupby("race_id")["予想確率"].transform("sum") * 3
top = test.loc[test.groupby("race_id")["そろえた確率"].idxmax()]
print("\n本命の確率が高いときだけ、複勝を買った場合")
for threshold in [0.7, 0.8, 0.9]:
    report(f"{int(threshold * 100)}%以上", top[top["そろえた確率"] >= threshold])

# ---- 7. AI がどのヒントを重視したか（5人の平均）
importance = pd.Series(np.mean([m.feature_importance("gain") for m in models], axis=0), index=FEATURES)
importance = importance.sort_values(ascending=False)
print("\nAI が重視したヒント")
for name, value in importance.items():
    print(f"  {name:<12} {value / importance.sum():6.1%}")