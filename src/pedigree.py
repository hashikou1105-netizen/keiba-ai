import glob
import os
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

SAVE_PATH = "data/pedigree.csv"


def fetch(url):
    """ページを取ってくる。通信に失敗したら、待ってから最大5回やり直す"""
    for attempt in range(1, 6):
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            response.encoding = "EUC-JP"
            return response
        except requests.exceptions.RequestException as e:
            wait = attempt * 20    # 20秒、40秒、60秒…と、だんだん長く待つ
            print(f"  通信エラー（{attempt}回目）: {type(e).__name__}。{wait}秒後にやり直します")
            time.sleep(wait)
    raise ConnectionError("5回やり直しても通信できませんでした。ネットを確認して、もう一度実行してください")


def name_and_id(td):
    """血統表の1マスから、馬名と馬IDを取り出す（「Sligo Bay (愛)」→「Sligo Bay」）"""
    a = td.find("a") if td is not None else None
    if a is None:
        return None, None
    name = a.get_text(strip=True).split("\xa0")[0]
    horse_id = a["href"].rstrip("/").split("/")[-1]
    return name, horse_id


def get_pedigree(horse_id):
    """1頭分の血統（父・母・母の父）を取ってくる"""
    response = fetch(f"https://db.netkeiba.com/horse/ped/{horse_id}/")
    soup = BeautifulSoup(response.text, "lxml")
    table = soup.find("table", class_="blood_table")

    # 血統表の左の列：大きいマス（rowspan=16）が 父・母、次の列（rowspan=8）が 父の父・父の母・母の父・母の母
    gen1 = table.find_all("td", rowspan="16") if table else []
    gen2 = table.find_all("td", rowspan="8") if table else []
    sire, sire_id = name_and_id(gen1[0] if len(gen1) > 0 else None)
    dam, dam_id = name_and_id(gen1[1] if len(gen1) > 1 else None)
    damsire, damsire_id = name_and_id(gen2[2] if len(gen2) > 2 else None)

    return {"horse_id": horse_id, "父": sire, "父_id": sire_id, "母": dam, "母_id": dam_id,
            "母父": damsire, "母父_id": damsire_id}


def horses_to_fetch():
    """集める馬の一覧（最近走った馬から順番に）"""
    files = sorted(glob.glob("data/race_results_*.csv"))
    df = pd.concat([pd.read_csv(f, usecols=["horse_id", "date"], dtype=str) for f in files])
    last_run = df.groupby("horse_id")["date"].max().sort_values(ascending=False)
    return list(last_run.index)


def save(row, path):
    """CSV の末尾に1行書き足す（ファイルがなければ新しく作る）"""
    is_new = not os.path.exists(path)
    pd.DataFrame([row]).to_csv(path, mode="a", header=is_new, index=False,
                               encoding="utf-8-sig" if is_new else "utf-8")


def main():
    horses = horses_to_fetch()

    # すでに保存されている馬は飛ばす（再開用）
    if os.path.exists(SAVE_PATH):
        done = set(pd.read_csv(SAVE_PATH, usecols=["horse_id"], dtype=str)["horse_id"])
    else:
        done = set()
    todo = [h for h in horses if h not in done]
    print(f"全 {len(horses)} 頭 / 取得済み {len(done)} 頭 / 残り {len(todo)} 頭"
          f"（約 {len(todo) * 1.85 / 3600:.1f} 時間）")

    for i, horse_id in enumerate(todo, 1):
        try:
            row = get_pedigree(horse_id)
        except ConnectionError:    # 通信できないときは、プログラムを止める
            raise
        except Exception as e:     # ページの形が違うなどの失敗は、記録して次へ
            print(horse_id, "→ エラー:", e)
            time.sleep(1)
            continue
        save(row, SAVE_PATH)
        time.sleep(1)              # サイトに負担をかけないよう、1秒待つ

        if i % 100 == 0:
            print(f"  {i} / {len(todo)} 頭（最新: {row['父']} × {row['母父']}）")


if __name__ == "__main__":
    main()