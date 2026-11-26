"""競馬AI（フル版）— 自分のMacで動かす予想アプリ
使い方: keiba_AI フォルダで  .venv/bin/streamlit run app/app.py
"""
import datetime as dt
import glob
import os
import re
import subprocess
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
PRED = os.path.join(ROOT, "predictions")
MODELS = os.path.join(ROOT, "models")
PY = sys.executable
sys.path.insert(0, SRC)
exec(open(os.path.join(SRC, "train.py")).read().split("def load_fukusho")[0].split("from features import make_features")[1])   # FEATURES, CATEGORIES

H = {"User-Agent": "Mozilla/5.0", "Referer": "https://race.netkeiba.com/"}
PLACE = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京", "06": "中山",
         "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}
SEASON = {11: "冬", 12: "冬", 1: "冬", 2: "冬", 3: "春", 4: "春", 5: "春", 6: "夏", 7: "夏", 8: "夏", 9: "秋", 10: "秋"}

st.set_page_config(page_title="競馬AI", page_icon="🏇", layout="wide")


# ---------------- データの取得 ----------------
@st.cache_data(ttl=600, show_spinner=False)
def race_list(date):
    """その日のレース一覧（race_id, 名前, 重賞かどうか）"""
    r = requests.get(f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date.replace('-', '')}",
                     headers=H, timeout=30)
    r.encoding = r.apparent_encoding
    out = []
    for li in BeautifulSoup(r.text, "lxml").select("li.RaceList_DataItem"):
        a = li.find("a")
        m = re.search(r"race_id=(\d+)", a["href"]) if a else None
        if not m:
            continue
        rid = m.group(1)
        text = " ".join(li.get_text(" ", strip=True).split())
        grade = re.search(r"Icon_GradeType(\d+)", str(li))
        g = {"1": "G1", "2": "G2", "3": "G3"}.get(grade.group(1)) if grade else None
        skip = "新馬" in text or "障" in text
        text = re.sub(r"^\d+R\s*", "", text)   # 先頭の「11R」はこちらで付けるので消す
        out.append({"race_id": rid, "名前": f"{PLACE.get(rid[4:6], '')}{int(rid[10:])}R {text[:30]}",
                    "グレード": g, "予想できない": skip})
    return out


@st.cache_data(ttl=120, show_spinner=False)
def odds(rid):
    """単勝オッズと人気（自分のMacで見るだけ。公開版には出さない）"""
    try:
        j = requests.get(f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={rid}&type=1&action=update",
                         headers=H, timeout=30).json()
        return {int(k): (float(v[0].replace(",", "")), int(v[2])) for k, v in j["data"]["odds"]["1"].items() if v[0] not in ("", "---")}
    except Exception:
        return {}


def saved_races():
    """もう予想してあるレース（predictions/ の中）"""
    out = []
    for f in sorted(glob.glob(os.path.join(PRED, "race_*.pkl")), key=os.path.getmtime, reverse=True):
        t = pd.read_pickle(f)
        rid = t["race_id"].iloc[0]
        out.append({"race_id": rid, "date": t["date"].iloc[0].strftime("%Y-%m-%d"),
                    "名前": f"{t['date'].iloc[0]:%m/%d} {PLACE.get(rid[4:6], '')}{int(rid[10:])}R {t['レース名'].iloc[0]}"})
    return out


@st.cache_resource(show_spinner=False)
def load_models(date):
    files = sorted(glob.glob(os.path.join(MODELS, f"model_{date}_v3_*.txt")))
    return [lgb.Booster(model_file=f) for f in files]


# ---------------- 計算 ----------------
def predict(t, models):
    """確率（1レースの合計が3になるように直す）と、ヒントごとの「上げた・下げた」"""
    raw = np.mean([m.predict(t[FEATURES]) for m in models], axis=0)
    prob = raw / raw.sum() * 3
    contrib = np.mean([m.predict(t[FEATURES], pred_contrib=True) for m in models], axis=0)[:, :-1]
    return np.minimum(prob, .99), pd.DataFrame(contrib, columns=FEATURES, index=t.index)


def fmt(name, v):
    """ヒントの値を読みやすくする"""
    if pd.isna(v):
        return "データなし"
    if name == "場":
        return PLACE.get(str(v), str(v))
    if name == "月":
        return f"{int(v)}月"
    if "先行度" in name and "順位" not in name:
        return f"前から{v:.0%}の位置"
    if name.endswith("率"):
        return f"{v:.0%}"
    if name.endswith("の差") and abs(v) <= 1 and "率" in name:
        return f"{v:+.0%}"
    if "有利さ" in name or name.startswith("騎手のこの") or name.startswith("騎手と調教師"):
        return f"平均より{v:+.1%}"
    if name.endswith("人気"):
        return f"{int(v)}番人気"
    if name.endswith("順位") or name.endswith("着順"):
        return f"{int(v)}位" if "順位" in name else f"{int(v)}着"
    if name.endswith("日数"):
        return f"{int(v)}日"
    if name in ("出走頭数", "レース内の先行馬の数"):
        return f"{int(v)}頭"
    if name.endswith("出走数") or name.endswith("騎乗数") or name == "休み明け何戦目":
        return f"{int(v)}回" if name != "休み明け何戦目" else f"{int(v)}戦目"
    if name == "所属":
        return {"東": "美浦（関東）", "西": "栗東（関西）"}.get(str(v), str(v))
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        return f"{v:,.1f}" if abs(v) < 1000 else f"{v:,.0f}"
    return str(v)


def tags(x):
    """馬の特徴を短いラベルにする"""
    out = []
    pos = x.get("過去5走の先行度")
    if pd.notna(pos):
        out.append("先行" if pos < .3 else "中団" if pos < .6 else "差し・追込")
    if x.get("道悪の出走数", 0) >= 2 and pd.notna(x.get("道悪の3着以内率")):
        if x["道悪の3着以内率"] >= .5:
            out.append("道悪◎")
        elif x["道悪の3着以内率"] == 0:
            out.append("道悪△")
    if x.get("同じ季節の出走数", 0) >= 3 and (x.get("同じ季節とほかの季節の3着以内率の差") or 0) >= .3:
        out.append(f"{SEASON[int(x['月'])]}に強い")
    if x.get("同じ競馬場・コース・距離の出走数", 0) >= 2 and (x.get("同じ競馬場・コース・距離の3着以内率") or 0) >= .5:
        out.append("この舞台◎")
    if (x.get("前走からの日数") or 0) >= 70:
        out.append(f"休み明け（{int(x['前走からの日数'])}日ぶり）")
    if (x.get("過去の出走数") or 0) == 0:
        out.append("⚠️データなし（海外馬・初出走など。AIは評価できない）")
    return out


# ---------------- 画面：レースを選ぶ（ページの一番上） ----------------
st.title("🏇 競馬AI")
st.caption("フル版（自分のMacだけで動く）")
box = st.expander("🏁 レースを選ぶ", expanded=True)

saved = saved_races()
mode = box.radio("レースの選び方", ["予想済みのレース", "新しく予想する"], horizontal=True)
rid = date = None

if mode == "予想済みのレース":
    if not saved:
        box.info("まだ予想したレースがありません。「新しく予想する」から始めてください。")
    else:
        pick = box.selectbox("レース", saved, format_func=lambda r: r["名前"])
        rid, date = pick["race_id"], pick["date"]
else:
    today = dt.date.today()
    next_sat = today + dt.timedelta(days=(5 - today.weekday()) % 7)
    d = box.date_input("開催日", next_sat)
    date = d.strftime("%Y-%m-%d")
    only_graded = box.toggle("重賞だけ表示", value=True)
    try:
        races = [r for r in race_list(date) if not r["予想できない"] and (r["グレード"] or not only_graded)]
    except requests.exceptions.SSLError:
        box.error("netkeibaにつながりません（SSLのエラー）。ネットワークを切り替えてから、もう一度試してください。")
        races = []
    except Exception as e:
        box.error(f"レース一覧を取れませんでした：{e}")
        races = []
    if not races:
        box.info("この日の対象レースが見つかりません。")
    else:
        pick = box.selectbox("レース", races,
                                    format_func=lambda r: (f"【{r['グレード']}】" if r["グレード"] else "") + r["名前"])
        need_train = not load_models(date)
        if need_train:
            box.caption("⏳ この日のAIはまだ学習していないので、最初は5分くらいかかります。")
        if box.button("AIで予想する", type="primary", use_container_width=True):
            with st.spinner("出馬表を取って、AIが予想しています…"):
                p = subprocess.run([PY, os.path.join(SRC, "predict_race.py"), date, pick["race_id"]],
                                   cwd=ROOT, capture_output=True, text=True)
            if p.returncode != 0:
                st.error("予想に失敗しました。")
                st.code(p.stderr[-2000:])
            else:
                load_models.clear()
                st.session_state["last"] = (pick["race_id"], date)
        # 予想したレースは、ほかの所を触っても表示したままにする
        last = st.session_state.get("last")
        if last and last[0] == pick["race_id"] and last[1] == date:
            rid = last[0]

st.caption("※ 予想は参考情報です。馬券は自己責任で。20歳未満は馬券を買えません。")

# ---------------- 画面：メイン ----------------
if not rid or not os.path.exists(os.path.join(PRED, f"race_{rid}.pkl")):
    st.write("上の「レースを選ぶ」から、レースを選んでください。")
    st.stop()

t = pd.read_pickle(os.path.join(PRED, f"race_{rid}.pkl")).reset_index(drop=True)
models = load_models(date)
if not models:
    st.error(f"{date} のAI（モデル）が見つかりません。「新しく予想する」から予想し直してください。")
    st.stop()

# 馬場状態（まだ発表前なら空）
baba = t["馬場状態"].iloc[0] if pd.notna(t["馬場状態"].iloc[0]) else "未発表"
st.header(f"{PLACE.get(rid[4:6], '')}{int(rid[10:])}R　{t['レース名'].iloc[0]}")
info = t["条件"].iloc[0] if "条件" in t else f"{t['コース'].iloc[0]}{int(t['距離'].iloc[0])}m"
st.caption(f"{t['date'].iloc[0]:%Y年%m月%d日}　{info}　{len(t)}頭")

t2 = t.copy()
prob, contrib = predict(t2, models)
t2["確率"] = prob
t2["AI順位"] = t2["確率"].rank(ascending=False, method="first").astype(int)
t2 = t2.sort_values("AI順位")

# レース全体
o = odds(rid)
top = t2["確率"].to_numpy()
gap = top[0] - top[1]
conf = "自信あり" if top[0] >= .6 and gap >= .15 else "やや自信あり" if gap >= .08 else "混戦"
pace = "ペースが速くなりそう" if t2["レース内の先行馬の数"].iloc[0] >= 4 else "ペースは落ち着きそう"
c1, c2 = st.columns(2)
c1.markdown(f"**AIの自信**：{conf}（1位と2位の差 {gap:.0%}）  \n**AI1位**：{t2['馬名'].iloc[0]}（{top[0]:.0%}）")
c2.markdown(f"**先行しそうな馬**：{int(t2['レース内の先行馬の数'].iloc[0])}頭（{pace}）  \n**馬場**：{baba}"
            + ("（雨の日の強さは、下の1頭ずつの説明で「道悪の3着以内率」を見てください）" if baba in ("稍重", "重", "不良") else ""))
if t2["レースの格"].iloc[0] >= 4000:
    st.info("重賞では、AIの%は高めに出やすいです（70%以上でも3着以内は約6割）。1割くらい割り引いて見てください。")

# 全馬の表
rows = []
for _, x in t2.iterrows():
    num = int(x["馬番"])
    rows.append({"AI順位": x["AI順位"], "馬番": num if num < 100 else None, "馬名": x["馬名"], "騎手": x["騎手"],
                 "性齢": x["性齢"], "斤量": x["斤量"], "3着以内の確率": round(x["確率"] * 100),
                 "単勝": o.get(num, (None, None))[0], "人気": o.get(num, (None, None))[1],
                 "特徴": "・".join(tags(x))})
table = pd.DataFrame(rows)
# まだ決まっていないもの（枠順の前の馬番、発売前のオッズ）は空欄にする
for c in ["馬番", "人気"]:
    table[c] = table[c].map(lambda v: "" if pd.isna(v) else str(int(v)))
table["単勝"] = table["単勝"].map(lambda v: "" if pd.isna(v) else f"{v:.1f}倍")
st.subheader("AIの予想（全馬）")
st.dataframe(table, hide_index=True, use_container_width=True, height=min(38 * len(table) + 40, 720),
             column_config={
                 "3着以内の確率": st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100),
             })
st.caption("※ AIはオッズを見ずに予想しています。オッズはnetkeibaから取得（このMacで見るだけ）。")

# 1頭ずつの詳しい説明
st.subheader("馬を選んで、AIの見方を見る")
name = st.selectbox("馬", t2["馬名"].tolist(),
                    format_func=lambda n: f"{int(t2.loc[t2['馬名'] == n, 'AI順位'].iloc[0])}位　{n}")
x = t2[t2["馬名"] == name].iloc[0]
cx = contrib.loc[x.name].sort_values()
left, right = st.columns(2)
with left:
    st.markdown(f"### {name}　{x['確率']:.0%}（AI {x['AI順位']}位）")
    st.write(" ".join(f"`{s}`" for s in tags(x)) or "特に目立つ特徴なし")
    feats = {
        "脚質（過去5走の先行度）": fmt("過去5走の先行度", x["過去5走の先行度"]),
        "最近のスピード指数（過去5走の平均・出走馬の中で）": f"{int(x['過去5走の補正スピード指数の平均のレース内順位'])}位" if pd.notna(x["過去5走の補正スピード指数の平均のレース内順位"]) else "データなし",
        "道悪の3着以内率": f"{fmt('道悪の3着以内率', x['道悪の3着以内率'])}（{0 if pd.isna(x['道悪の出走数']) else int(x['道悪の出走数'])}戦）",
        f"{SEASON[int(x['月'])]}の3着以内率": f"{fmt('同じ季節の3着以内率', x['同じ季節の3着以内率'])}（{int(x['同じ季節の出走数'])}戦）",
        "この舞台（競馬場・コース・距離）の3着以内率": f"{fmt('x率', x['同じ競馬場・コース・距離の3着以内率'])}（{int(x['同じ競馬場・コース・距離の出走数'])}戦）",
        "前走からの日数": f"{int(x['前走からの日数'])}日" if pd.notna(x["前走からの日数"]) else "初出走",
        "騎手のこの競馬場の成績": fmt("騎手のこの競馬場の成績", x["騎手のこの競馬場の成績"]),
        "この枠の位置の有利さ": fmt("この枠の位置の有利さ", x["この枠の位置の有利さ"]),
    }
    st.table(pd.DataFrame(feats.items(), columns=["特徴", "値"]).set_index("特徴"))
with right:
    st.markdown("### AIがこの確率にした理由")
    up = cx[cx > 0].sort_values(ascending=False).head(5)
    down = cx[cx < 0].head(5)
    st.markdown("**👍 確率を上げたヒント**")
    for f, v in up.items():
        st.write(f"- {f}：**{fmt(f, x[f])}**")
    st.markdown("**👎 確率を下げたヒント**")
    for f, v in down.items():
        st.write(f"- {f}：**{fmt(f, x[f])}**")
    import plotly.graph_objects as go
    chart = pd.concat([up, down]).sort_values()
    fig = go.Figure(go.Bar(x=chart.values, y=chart.index, orientation="h",
                           marker_color=["#2e7d32" if v > 0 else "#c62828" for v in chart.values]))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="影響の大きさ")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("棒が右に長いほど確率を上げ、左に長いほど下げています（LightGBMの寄与度）。")
