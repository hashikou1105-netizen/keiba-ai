import os
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

SAVE_DIR = "data"
os.makedirs(SAVE_DIR, exist_ok=True)   # 保存用フォルダを作る（すでにあれば何もしない）

# 結果の表から残す列（有料会員向けの空の列などは捨てる）
COLUMNS = ["着順", "枠番", "馬番", "馬名", "性齢", "斤量", "騎手", "タイム", "着差",
           "通過", "上り", "単勝", "人気", "馬体重", "調教師", "馬主", "賞金(万円)"]


def fetch(url):
    """ページを取ってくる。通信に失敗したら、待ってから最大5回やり直す"""
    for attempt in range(1, 6):
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            # 昔のページは EUC-JP、2026年秋ごろから UTF-8 になった。サーバーが文字コードを教えてくれればそれを使う
            if "charset" not in response.headers.get("content-type", "").lower():
                response.encoding = "EUC-JP"
            return response
        except requests.exceptions.RequestException as e:
            wait = attempt * 20    # 20秒、40秒、60秒…と、だんだん長く待つ
            print(f"  通信エラー（{attempt}回目）: {type(e).__name__}。{wait}秒後にやり直します")
            time.sleep(wait)
    raise ConnectionError("5回やり直しても通信できませんでした。ネットを確認して、もう一度実行してください")


def find(pattern, text):
    """text の中から pattern を探す。見つからなければ None（空欄）を返す"""
    m = re.search(pattern, text)
    if m is None:
        return None
    return m.group(1)


def read_table(table):
    """HTML の表を、1マスずつ読んで DataFrame にする"""
    rows = table.find_all("tr")
    header = [th.get_text(strip=True) for th in rows[0].find_all("th")]
    data = []
    for tr in rows[1:]:
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) == len(header):
            data.append(cells)
    return pd.DataFrame(data, columns=header)


def read_payouts(soup, race_id):
    """払戻金の表を読む。1行 = 1つの当たり（例：複勝 9番 160円）"""
    payouts = []
    for table in soup.find_all("table", class_="pay_table_01"):
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            kind = tr.find("th").get_text(strip=True)
            numbers = tds[0].get_text("\n", strip=True).split("\n")
            prices = tds[1].get_text("\n", strip=True).split("\n")
            for number, price in zip(numbers, prices):
                payouts.append({"race_id": race_id, "券種": kind,
                                "馬番": number, "払戻": int(price.replace(",", ""))})
    return pd.DataFrame(payouts)


def get_race(race_id):
    """1レース分の結果と払戻金を取ってきて、(結果の表, 払戻金の表) を返す"""
    response = fetch(f"https://db.netkeiba.com/race/{race_id}/")

    # 存在しないレースだと、別のレースのページに飛ばされることがある
    if race_id not in response.url:
        return None, None
    soup = BeautifulSoup(response.text, "lxml")

    # 結果の表がなければ、存在しないレース
    table = soup.find("table", class_="race_table_01")
    if table is None:
        return None, None

    # 結果の表（必要な列だけ残す）
    df = read_table(table)
    df = df[[c for c in COLUMNS if c in df.columns]].copy()
    df.insert(0, "race_id", race_id)

    # 日付
    m = re.search(r"(\d+)年(\d+)月(\d+)日", soup.title.text)
    df["date"] = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    # レースの条件（<span> の中だけを探す）
    intro = soup.find("div", class_="data_intro")
    cond = " ".join(intro.find("span").text.split())
    df["レース名"] = intro.find("h1").text.strip()
    df["コース"] = find(r"(芝|ダ|障)", cond)
    df["回り"] = find(r"(左|右|直線)", cond)
    df["距離"] = int(find(r"(\d+)m", cond))
    df["天候"] = find(r"天候 : (\S+)", cond)
    df["馬場状態"] = find(r"(?:芝|ダート) : (\S+)", cond)
    df["発走時間"] = find(r"発走 : (\d+:\d+)", cond)
    df["出走頭数"] = len(df)

    # 馬・騎手・調教師の ID
    for kind in ["horse", "jockey", "trainer"]:
        links = table.find_all("a", href=re.compile(rf"/{kind}/(result/recent/)?\w+/$"))
        df[f"{kind}_id"] = [a["href"].rstrip("/").split("/")[-1] for a in links]

    return df, read_payouts(soup, race_id)


def save(df, path):
    """CSV の末尾に書き足す（ファイルがなければ新しく作る）"""
    is_new = not os.path.exists(path)
    df.to_csv(path, mode="a", header=is_new, index=False,
              encoding="utf-8-sig" if is_new else "utf-8")


def scrape_day(year, place, kai, day, done):
    """1日分（1R〜12R）を取って保存する。見つかったレース数を返す"""
    found = 0
    for r in range(1, 13):
        race_id = f"{year}{place}{kai:02d}{day:02d}{r:02d}"

        if race_id in done:        # 取得済みなら飛ばす
            found += 1
            continue

        try:
            df, payouts = get_race(race_id)
        except ConnectionError:    # 通信できないときは、プログラムを止める
            raise
        except Exception as e:     # ページの形が違うなどの失敗は、記録して次へ
            print(race_id, "→ エラー:", e)
            found += 1
            time.sleep(1)
            continue
        time.sleep(1)              # サイトに負担をかけないよう、1秒待つ

        if df is None:
            break
        save(df, f"{SAVE_DIR}/race_results_{year}.csv")
        save(payouts, f"{SAVE_DIR}/payouts_{year}.csv")
        found += 1
    return found


def scrape_kai(year, place, kai, done):
    """1回分の開催（1日目〜最大12日目）を取る。見つかった日数を返す"""
    days = 0
    for day in range(1, 13):
        n = scrape_day(year, place, kai, day, done)
        if n == 0:
            break
        print(f"  {year}年 場{place} {kai}回 {day}日目: {n}レース")
        days += 1
    return days


def scrape_year(year):
    """1年分、全競馬場（01札幌〜10小倉）を取る"""
    path = f"{SAVE_DIR}/race_results_{year}.csv"

    # すでに保存されている race_id を読み込む（再開用）
    if os.path.exists(path):
        done = set(pd.read_csv(path, usecols=["race_id"], dtype=str)["race_id"])
    else:
        done = set()
    print(f"=== {year}年（取得済み {len(done)} レース）===")

    for place in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]:
        for kai in range(1, 10):
            if scrape_kai(year, place, kai, done) == 0:
                break


# ---- 取りたい年をここに書く（順番に取る）
YEARS = [2026, 2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016]

for year in YEARS:
    scrape_year(year)