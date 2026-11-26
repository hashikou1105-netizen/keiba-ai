import glob
import os

import numpy as np
import pandas as pd


def load_results():
    """data/ にある年ごとの結果 CSV を全部読み込んで、1つの表にする（海外のレースもあれば足す）"""
    files = sorted(glob.glob("data/race_results_*.csv"))
    df = pd.concat([pd.read_csv(f, dtype={"race_id": str, "horse_id": str,
                                          "jockey_id": str, "trainer_id": str})
                    for f in files])
    df["海外"] = 0
    if os.path.exists(OVERSEAS):
        df = pd.concat([df, load_overseas()], ignore_index=True)
    return df


# 海外のレース：賞金が分からないので、グレードから JRA くらいの1着賞金（万円）を決める
OVERSEAS = "data/overseas_results.csv"
GRADE_PRIZE = {"GI": 20000, "GII": 6500, "GIII": 4300}
PLACE_RATE = {1: 1.0, 2: 0.4, 3: 0.25, 4: 0.15, 5: 0.1}   # 2着は1着の4割、…（JRA と同じくらい）


def load_overseas():
    """海外のレースの成績を、JRA の結果と同じ形に直す
    ・タイムや上りは比べられないので入れない（スピード指数は JRA のレースだけで作る）
    ・AI の練習や予想には使わない。過去の成績のヒントにだけ使う
    """
    o = pd.read_csv(OVERSEAS, dtype=str).drop_duplicates(["race_id", "horse_id"])
    grade = o["レース名"].str.extract(r"\((GI{1,3})\)")[0]
    o["海外の格"] = grade.map(GRADE_PRIZE).fillna(2400)
    rank = pd.to_numeric(o["着順"], errors="coerce")
    o["賞金(万円)"] = (o["海外の格"] * rank.map(PLACE_RATE).fillna(0)).round(1).astype(str)
    o["馬場状態"] = o["馬場状態"].map({"良": "良", "稍": "稍重", "重": "重", "不": "不良"})
    for c in ["距離", "出走頭数", "馬番", "斤量", "人気"]:
        o[c] = pd.to_numeric(o[c], errors="coerce")
    o["海外"] = 1
    return o.drop(columns=["開催"])


def time_to_seconds(t):
    """'1:50.2' → 110.2 秒"""
    if pd.isna(t):
        return None
    minutes, seconds = t.split(":")
    return int(minutes) * 60 + float(seconds)


def clean(df):
    """文字のデータを、計算できる数字に直す"""
    df = df.copy()

    # 出走取消（取）・競走除外（除）の馬は走っていないので、取り除く
    df = df[~df["着順"].astype(str).isin(["取", "除"])]

    # 着順：「中」（競走中止）「失」（失格）は空欄に。「3(降)」は 3 に
    df["着順"] = pd.to_numeric(df["着順"].astype(str).str.extract(r"^(\d+)")[0], errors="coerce")

    # 性齢：「牝3」→ 性別「牝」と 年齢 3
    df["性別"] = df["性齢"].str[0]
    df["年齢"] = pd.to_numeric(df["性齢"].str[1:], errors="coerce")

    # 馬体重：「494(+24)」→ 体重 494 と 増減 24（「計不」は空欄）
    weight = df["馬体重"].astype(str).str.extract(r"(\d+)\(([+-]?\d+)\)")
    df["体重"] = pd.to_numeric(weight[0], errors="coerce")
    df["体重増減"] = pd.to_numeric(weight[1], errors="coerce")

    # タイム：「1:50.2」→ 110.2 秒
    df["タイム秒"] = df["タイム"].apply(time_to_seconds)

    # 通過：「12-12-8-12」→ 最初のコーナー 12、最後のコーナー 12
    corners = df["通過"].astype(str).str.split("-")
    df["最初のコーナー"] = pd.to_numeric(corners.str[0], errors="coerce")
    df["最後のコーナー"] = pd.to_numeric(corners.str[-1], errors="coerce")

    # 調教師：「[東]小野次郎」→ 所属「東」（東・西・地・外）
    df["所属"] = df["調教師"].str.extract(r"\[(.)\]")[0]

    # 賞金：「2,331.5」→ 2331.5（空欄は 0 円）
    df["賞金"] = pd.to_numeric(df["賞金(万円)"].astype(str).str.replace(",", ""), errors="coerce").fillna(0)

    # 競馬場：race_id の5〜6桁目（01 札幌 〜 10 小倉）
    df["場"] = df["race_id"].str[4:6]

    # 距離の区分：細かすぎると経験のない馬ばかりになるので、4つにまとめる
    df["距離区分"] = pd.cut(df["距離"], bins=[0, 1400, 1800, 2200, 9999],
                         labels=["短距離", "マイル", "中距離", "長距離"]).astype(str)

    # 日付を「日付の型」にする（引き算などができるようになる）
    df["date"] = pd.to_datetime(df["date"])
    return df


def past_average(df, column, n=5):
    """馬ごとに、今回より前の直近 n 走の平均を出す"""
    previous = df.groupby("horse_id")[column].shift(1)   # 1つ前にずらす（今回の結果を使わない）
    average = previous.groupby(df["horse_id"]).rolling(n, min_periods=1).mean()
    return average.reset_index(level=0, drop=True).sort_index()


def past_best(df, column, n=5, smaller_is_better=False):
    """馬ごとに、今回より前の直近 n 走で一番良かった値を出す"""
    previous = df.groupby("horse_id")[column].shift(1)
    rolling = previous.groupby(df["horse_id"]).rolling(n, min_periods=1)
    best = rolling.min() if smaller_is_better else rolling.max()
    return best.reset_index(level=0, drop=True).sort_index()


def add_horse_history(df):
    """馬ごとの「そのレースより前」の成績から、ヒントを作る"""
    # 日付順に並べる（同じ日の中では race_id 順）
    df = df.sort_values(["date", "race_id", "馬番"]).reset_index(drop=True)

    # AI に当てさせる答え：3着以内なら 1、それ以外（競走中止なども）は 0
    df["3着以内"] = (df["着順"] <= 3).astype(int)

    # 馬ごとのグループ
    horse = df.groupby("horse_id")

    # 過去の出走数：その馬の何走目か（0 なら初出走）
    df["過去の出走数"] = horse.cumcount()

    # 前走の情報（shift(1) = 1つ前のレース）
    df["前走の着順"] = horse["着順"].shift(1)
    df["前走からの日数"] = (df["date"] - horse["date"].shift(1)).dt.days
    df["前走との距離の差"] = df["距離"] - horse["距離"].shift(1)

    # 過去5走の平均（1つずらしてから、直近5つの平均）
    df["過去5走の平均着順"] = past_average(df, "着順")
    jra = df["海外"] != 1 if "海外" in df else pd.Series(True, index=df.index)   # 上りも JRA のレースだけで数える
    df.loc[jra, "過去5走の上り平均"] = past_average(df[jra], "上り")

    # 過去の3着以内率：今回より前の「3着以内の回数 ÷ 出走数」
    past_top3 = horse["3着以内"].cumsum() - df["3着以内"]
    df["過去の3着以内率"] = past_top3 / df["過去の出走数"]   # 初出走は 0÷0 で空欄になる
    return df


def add_class_history(df):
    """レースの格（1着の賞金）と、その馬の実績からヒントを作る"""
    # レースの格：そのレースの1着の賞金（レースの前から決まっている）
    df["レースの格"] = df.groupby("race_id")["賞金"].transform("max")
    if "海外の格" in df:   # 海外は日本の馬が勝っていないこともあるので、グレードから決めた格を使う
        df["レースの格"] = df["海外の格"].fillna(df["レースの格"])

    horse = df.groupby("horse_id")
    df["前走のレースの格"] = horse["レースの格"].shift(1)
    df["過去5走のレースの格の平均"] = past_average(df, "レースの格")

    # 今回 ÷ 前走：1 より大きければ昇級（格上挑戦）、小さければ降級
    df["前走との格の比"] = df["レースの格"] / df["前走のレースの格"]

    # 過去の獲得賞金：今回より前に稼いだ賞金の合計
    df["過去の獲得賞金"] = horse["賞金"].cumsum() - df["賞金"]
    return df


def add_speed_history(df):
    """タイム差（勝ち馬から何秒遅れたか）とスピード指数のヒントを作る"""
    # タイム差：勝ち馬のタイムとの差（勝った馬は 0）
    df["勝ちタイム"] = df.groupby("race_id")["タイム秒"].transform("min")
    df["タイム差"] = df["タイム秒"] - df["勝ちタイム"]

    # 標準タイム：同じ競馬場・コース・距離・馬場状態の、それより前のレースの勝ちタイムの真ん中の値
    races = df.drop_duplicates("race_id")[["race_id", "date", "場", "コース", "距離", "馬場状態", "勝ちタイム"]]
    races = races.sort_values(["date", "race_id"])
    group = races.groupby(["場", "コース", "距離", "馬場状態"])
    races["標準タイム"] = group["勝ちタイム"].transform(lambda s: s.shift(1).expanding(min_periods=3).median())

    # 馬場差：その日・その競馬場・そのコースの全レースで、勝ち馬が標準タイムより平均どれだけ速かったか
    # （馬場が速い日はプラス、遅い日はマイナス。過去のレースの補正にだけ使う）
    races["勝ち馬の速さ"] = (races["標準タイム"] - races["勝ちタイム"]) / races["距離"] * 1000
    races["馬場差"] = races.groupby(["date", "場", "コース"])["勝ち馬の速さ"].transform("mean")
    df = df.merge(races[["race_id", "標準タイム", "馬場差"]], on="race_id", how="left")

    # スピード指数：標準タイムより、1000m あたり何秒速かったか（プラスなら速い）
    df["スピード指数"] = (df["標準タイム"] - df["タイム秒"]) / df["距離"] * 1000
    # 補正スピード指数：その日の馬場の速さを差し引いた、本当の速さ
    df["補正スピード指数"] = df["スピード指数"] - df["馬場差"]

    # 前走までの情報だけでヒントにする
    # タイムは JRA のレースしか比べられないので、海外のレースを飛ばして「JRA の直近5走」で数える
    jra = df["海外"] != 1 if "海外" in df else pd.Series(True, index=df.index)
    j = df[jra].copy()
    horse = j.groupby("horse_id")
    j["前走のタイム差"] = horse["タイム差"].shift(1)
    j["過去5走のタイム差の平均"] = past_average(j, "タイム差")
    j["過去5走の一番小さいタイム差"] = past_best(j, "タイム差", smaller_is_better=True)
    j["前走のスピード指数"] = horse["スピード指数"].shift(1)
    j["過去5走のスピード指数の平均"] = past_average(j, "スピード指数")
    j["過去5走のスピード指数の最高"] = past_best(j, "スピード指数")
    j["前走の補正スピード指数"] = horse["補正スピード指数"].shift(1)
    j["過去5走の補正スピード指数の平均"] = past_average(j, "補正スピード指数")
    j["過去5走の補正スピード指数の最高"] = past_best(j, "補正スピード指数")
    cols = [c for c in j.columns if c not in df.columns or c.startswith(("前走の", "過去5走の"))]
    for c in cols:
        df.loc[jra, c] = j[c]
    return df


def add_condition_history(df, condition, name):
    """その馬の「今回と同じ条件」での、これまでの成績からヒントを作る"""
    # 馬 × 条件 ごとのグループ（例：イクイノックス × 芝）
    group = df.groupby(["horse_id", condition])

    past_n = group.cumcount()                                  # 同じ条件での過去の出走数
    past_hit = group["3着以内"].cumsum() - df["3着以内"]          # 同じ条件での過去の3着以内の回数
    df[f"{name}の出走数"] = past_n
    df[f"{name}の3着以内率"] = past_hit / past_n                  # 経験がなければ 0÷0 で空欄
    return df


def add_person_history(df, key, name, count_name):
    """騎手・調教師の「前日まで」の成績から、ヒントを作る"""
    # 人ごと・日ごとに、出走数と3着以内の数を数える
    daily = df.groupby([key, "date"]).agg(数=("3着以内", "size"), 当たり=("3着以内", "sum")).reset_index()
    daily = daily.sort_values([key, "date"])

    # 前日までの累計（今日の分を引く）
    person = daily.groupby(key)
    past_n = person["数"].cumsum() - daily["数"]
    past_hit = person["当たり"].cumsum() - daily["当たり"]

    # 回数が少ないとブレるので、全体の平均に 20 回分寄せる
    average = df["3着以内"].mean()
    daily[f"{name}の3着以内率"] = (past_hit + average * 20) / (past_n + 20)
    daily[count_name] = past_n

    return df.merge(daily[[key, "date", f"{name}の3着以内率", count_name]], on=[key, "date"], how="left")


# レースの中で比べるヒントと、「小さいほど良い」かどうか
COMPARE = {
    "前走の着順": True,
    "過去5走の平均着順": True,
    "過去5走の上り平均": True,
    "騎手の3着以内率": False,
    "同じコースの3着以内率": False,
    "過去の獲得賞金": False,
    "過去5走のスピード指数の平均": False,
    "過去5走のタイム差の平均": True,
    "過去5走の補正スピード指数の平均": False,
    "過去5走の補正スピード指数の最高": False,
}


def add_race_comparison(df):
    """同じレースの中で、ほかの馬と比べたヒントを作る"""
    race = df.groupby("race_id")
    for column, smaller_is_better in COMPARE.items():
        # レース内順位：一番良い馬が 1 位（空欄の馬は順位なし）
        df[f"{column}のレース内順位"] = race[column].rank(ascending=smaller_is_better, method="min")
        # レース平均との差：メンバーの平均より、どれだけ大きいか小さいか
        df[f"{column}のレース平均との差"] = df[column] - race[column].transform("mean")
    return df


def make_features():
    """CSV を読み込んで、ヒントを全部作った表を返す"""
    print("データを読み込んでいます…")
    df = clean(load_results())
    print("馬の過去成績を計算しています…")
    df = add_horse_history(df)
    df = add_class_history(df)
    print("タイム差とスピード指数を計算しています…")
    df = add_speed_history(df)
    print("同じ条件での成績を計算しています…")
    df = add_condition_history(df, "コース", "同じコース")
    df = add_condition_history(df, "場", "同じ競馬場")
    df = add_condition_history(df, "距離区分", "同じ距離区分")
    print("騎手・調教師の成績を計算しています…")
    df = add_person_history(df, "jockey_id", "騎手", "騎手の騎乗数")
    df = add_person_history(df, "trainer_id", "調教師", "調教師の出走数")
    print("同じレースの中で比べています…")
    df = add_race_comparison(df)
    return df


if __name__ == "__main__":
    df = make_features()
    print(len(df), "行")

    # 例として、イクイノックスの全レースを表示する
    example = df[df["馬名"] == "イクイノックス"]
    cols = ["date", "レース名", "着順", "スピード指数", "馬場差", "補正スピード指数"]
    print(example[cols].round(2).to_string(index=False))

# ---- 調子のヒント
FORM_FEATURES = ["2走前の着順", "3走前の着順", "前走の人気", "前走の人気と着順の差", "前走の着順と過去5走平均の差",
                 "休み明け何戦目", "前走の上り順位", "前走の補正スピード指数と過去5走平均の差", "直近3走の補正スピード指数の平均",
                 "道悪の出走数", "道悪の3着以内率", "道悪と良の3着以内率の差"]


def add_form_history(df):
    """調子（最近の流れ）と、雨の日の強さのヒントを作る"""
    df = df.copy()
    df["人気"] = pd.to_numeric(df["人気"], errors="coerce")
    horse = df.groupby("horse_id")
    df["2走前の着順"] = horse["着順"].shift(2)
    df["3走前の着順"] = horse["着順"].shift(3)
    df["前走の人気"] = horse["人気"].shift(1)
    df["前走の人気と着順の差"] = df["前走の人気"] - df["前走の着順"]   # プラスなら人気より上の着順
    df["前走の着順と過去5走平均の差"] = df["前走の着順"] - df["過去5走の平均着順"]
    # 休み明け何戦目：10週（70日）以上空いたら1戦目
    rest = (df["前走からの日数"].isna() | (df["前走からの日数"] >= 70)).astype(int)
    block = rest.groupby(df["horse_id"]).cumsum()
    df["休み明け何戦目"] = df.groupby([df["horse_id"], block]).cumcount() + 1

    jra = df["海外"] != 1 if "海外" in df else pd.Series(True, index=df.index)
    j = df[jra].copy()
    j["上り順位"] = j.groupby("race_id")["上り"].rank(method="min")
    h = j.groupby("horse_id")
    j["前走の上り順位"] = h["上り順位"].shift(1)
    j["前走の補正スピード指数と過去5走平均の差"] = j["前走の補正スピード指数"] - j["過去5走の補正スピード指数の平均"]
    j["直近3走の補正スピード指数の平均"] = past_average(j, "補正スピード指数", 3)
    # 道悪の成績（同じ芝/ダートで、今回より前）
    wet = j["馬場状態"].isin(["稍重", "重", "不良"]).astype(int)
    j["_w"], j["_wt"] = wet, wet * j["3着以内"]
    j["_d"], j["_dt"] = 1 - wet, (1 - wet) * j["3着以内"]
    g = j.groupby(["horse_id", "コース"])
    for c in ["_w", "_wt", "_d", "_dt"]:
        j[c + "s"] = g[c].cumsum() - j[c]
    j["道悪の出走数"] = j["_ws"]
    j["道悪の3着以内率"] = j["_wts"] / j["_ws"].replace(0, np.nan)
    j["道悪と良の3着以内率の差"] = j["道悪の3着以内率"] - j["_dts"] / j["_ds"].replace(0, np.nan)
    for c in FORM_FEATURES[6:]:
        df.loc[jra, c] = j[c]
    return df


# ---- 騎手・牝馬限定・血統のヒント（2026-11 追加の実験）
EXTRA_FEATURES = ["騎手の人気以上率", "乗り替わり", "この騎手との過去の出走数",
                  "牝馬限定", "牝馬限定の3着以内率", "混合の3着以内率",
                  "父の同じコースの人気以上率", "父の同じ馬場の人気以上率", "父の同じ距離区分の人気以上率",
                  "母父の同じコースの人気以上率"]


def prior_mean(df, keys, value, k, base=0.0):
    """keys ごとに「前日まで」の value の平均を出す（回数が少ないときは base に k 回分寄せる）"""
    ok = df[value].notna()
    tmp = pd.DataFrame({"v": df[value].where(ok, 0), "n": ok.astype(int), "date": df["date"]})
    for c in keys:
        tmp[c] = df[c]
    daily = tmp.groupby(keys + ["date"], dropna=False)[["v", "n"]].sum().reset_index().sort_values("date")
    g = daily.groupby(keys, dropna=False)
    sv = g["v"].cumsum() - daily["v"]
    sn = g["n"].cumsum() - daily["n"]
    daily["_r"] = (sv + base * k) / (sn + k)
    out = tmp[keys + ["date"]].merge(daily[keys + ["date", "_r"]], on=keys + ["date"], how="left")
    return out["_r"].values


def add_extra_history(df):
    df = df.copy()
    df["人気"] = pd.to_numeric(df["人気"], errors="coerce")
    df = df.sort_values(["date", "race_id", "馬番"]).reset_index(drop=True)

    # 人気から見た「3着以内に来る見込み」（2023年までの平均）との差 ＝ 人気以上に走ったか
    old = df[(df["date"] < "2024-01-01") & df["人気"].notna()]
    expect = old.groupby(old["人気"].clip(upper=18))["3着以内"].mean()
    df["_exp"] = df["人気"].clip(upper=18).map(expect)
    df["_res"] = df["3着以内"] - df["_exp"]
    df.loc[df["着順"].isna() & df["人気"].isna(), "_res"] = np.nan   # これから走るレース

    # 騎手：馬の人気（＝馬の強さ）を差し引いた腕前
    df["騎手の人気以上率"] = prior_mean(df, ["jockey_id"], "_res", 300)
    horse = df.groupby("horse_id")
    df["乗り替わり"] = (horse["jockey_id"].shift(1) != df["jockey_id"]).astype(float)
    df.loc[horse.cumcount() == 0, "乗り替わり"] = np.nan
    df["この騎手との過去の出走数"] = df.groupby(["horse_id", "jockey_id"]).cumcount()

    # 牝馬限定：出走馬が全部牝馬のレース
    df["牝馬限定"] = (df.groupby("race_id")["性別"].transform(lambda s: (s == "牝").all())).astype(int)
    for flag, name in [(1, "牝馬限定"), (0, "混合")]:
        m = (df["牝馬限定"] == flag).astype(int)
        df["_m"], df["_mt"] = m, m * df["3着以内"]
        g = df.groupby("horse_id")
        n = g["_m"].cumsum() - df["_m"]
        t = g["_mt"].cumsum() - df["_mt"]
        df[f"{name}の3着以内率"] = t / n.replace(0, np.nan)

    # 血統：父・母父の子どもが、その条件で人気以上に走っているか
    ped = pd.read_csv("data/pedigree.csv", dtype=str, encoding="utf-8-sig")[["horse_id", "父_id", "母父_id"]]
    df = df.merge(ped.drop_duplicates("horse_id"), on="horse_id", how="left")
    df["_wet"] = df["馬場状態"].isin(["稍重", "重", "不良"]).astype(int)
    df["父の同じコースの人気以上率"] = prior_mean(df, ["父_id", "コース"], "_res", 100)
    df["父の同じ馬場の人気以上率"] = prior_mean(df, ["父_id", "コース", "_wet"], "_res", 100)
    df["父の同じ距離区分の人気以上率"] = prior_mean(df, ["父_id", "距離区分"], "_res", 100)
    df["母父の同じコースの人気以上率"] = prior_mean(df, ["母父_id", "コース"], "_res", 100)
    for c in [c for c in EXTRA_FEATURES if c.startswith(("父", "母父"))]:
        df.loc[df["父_id"].isna() if c.startswith("父") else df["母父_id"].isna(), c] = np.nan
    return df.drop(columns=[c for c in df.columns if c.startswith("_")])


# ---- 季節・脚質・枠・騎手の相性のヒント（2026-11-23 追加の実験）
SEASON_FEATURES = ["月", "同じ季節の出走数", "同じ季節の3着以内率", "同じ季節とほかの季節の3着以内率の差"]
STYLE_FEATURES = ["過去5走の先行度", "過去5走の先行度のレース内順位", "レース内の先行馬の数"]
GATE_FEATURES = ["この枠の位置の有利さ"]
JOCKEY_FIT_FEATURES = ["騎手のこの競馬場の成績", "騎手と調教師のコンビの成績"]
SPOT_FEATURES = ["同じ競馬場・コース・距離の出走数", "同じ競馬場・コース・距離の3着以内率"]
NEW_GROUPS = {"季節": SEASON_FEATURES, "脚質": STYLE_FEATURES, "枠の有利さ": GATE_FEATURES,
              "騎手の相性": JOCKEY_FIT_FEATURES, "同じ舞台": SPOT_FEATURES}


def add_new_history(df):
    """季節（冬に強い馬など）・脚質（前に行く馬か）・枠の有利さ・騎手の相性・まったく同じ舞台の成績"""
    df = df.copy()
    df = df.sort_values(["date", "race_id", "馬番"]).reset_index(drop=True)
    done = df["着順"].notna()   # もう結果が出ているレース

    # 季節：冬（11〜2月）・春（3〜5月）・夏（6〜8月）・秋（9〜10月）
    df["月"] = df["date"].dt.month
    df["_season"] = df["月"].map({11: 0, 12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3})
    g = df.groupby(["horse_id", "_season"])
    n_same = g.cumcount()
    hit_same = g["3着以内"].cumsum() - df["3着以内"]
    df["同じ季節の出走数"] = n_same
    df["同じ季節の3着以内率"] = hit_same / n_same.replace(0, np.nan)
    h = df.groupby("horse_id")
    n_all = h.cumcount()
    hit_all = h["3着以内"].cumsum() - df["3着以内"]
    df["同じ季節とほかの季節の3着以内率の差"] = df["同じ季節の3着以内率"] - (hit_all - hit_same) / (n_all - n_same).replace(0, np.nan)

    # 脚質：最初のコーナーで前から何割の位置にいたか（0に近いほど前に行く馬）
    df["_pos"] = (df["最初のコーナー"] - 1) / (df["出走頭数"] - 1).clip(lower=1)
    df["過去5走の先行度"] = past_average(df, "_pos")
    df["過去5走の先行度のレース内順位"] = df.groupby("race_id")["過去5走の先行度"].rank(method="min")
    df["レース内の先行馬の数"] = df.groupby("race_id")["過去5走の先行度"].transform(lambda s: (s < 0.25).sum())

    # 枠の有利さ：同じ競馬場・コース・距離で、内〜外（5段階）の馬が、頭数から見た平均よりどれだけ来ていたか
    df["_gate"] = pd.cut((df["馬番"] - 0.5) / df["出走頭数"], [0, .2, .4, .6, .8, 1], labels=False)
    df["_over"] = (df["3着以内"] - 3 / df["出走頭数"]).where(done)
    df["この枠の位置の有利さ"] = prior_mean(df, ["場", "コース", "距離", "_gate"], "_over", 200)

    # 騎手の相性：その競馬場での成績、その調教師と組んだときの成績（頭数から見た平均との差）
    df["騎手のこの競馬場の成績"] = prior_mean(df, ["jockey_id", "場"], "_over", 50)
    df["騎手と調教師のコンビの成績"] = prior_mean(df, ["jockey_id", "trainer_id"], "_over", 30)

    # まったく同じ舞台（競馬場・コース・距離）での、その馬の成績
    df["_spot"] = df["場"] + df["コース"].astype(str) + df["距離"].astype(str)
    df = add_condition_history(df, "_spot", "同じ競馬場・コース・距離")
    return df.drop(columns=[c for c in df.columns if c.startswith("_")])
