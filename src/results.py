"""レース結果と払戻を netkeiba から取る
使い方: from results import result  →  finish（馬番→着順・馬名・人気）, pay（券種→[(馬番の組, 払戻)]）
"""
import re
import requests
from bs4 import BeautifulSoup

H = {"User-Agent": "Mozilla/5.0"}


def result(rid):
    r = requests.get(f"https://race.netkeiba.com/race/result.html?race_id={rid}", headers=H, timeout=30)
    r.encoding = r.apparent_encoding
    s = BeautifulSoup(r.text, "lxml")
    finish = {}
    for tr in s.select("table.RaceTable01 tr")[1:]:
        td = [x.get_text(strip=True) for x in tr.find_all("td")]
        if len(td) > 10 and td[0].isdigit():
            finish[int(td[2])] = {"着順": int(td[0]), "馬名": td[3], "人気": td[9], "単勝": td[10]}
    pay = {}
    for tr in s.select("table.Payout_Detail_Table tr"):
        kind = tr.find("th").get_text(strip=True)
        nums = [[int(x) for x in re.findall(r"\d+", ul.get_text(" "))] for ul in tr.select("td.Result ul")] \
            or [[int(dv.get_text(strip=True))] for dv in tr.select("td.Result div") if dv.get_text(strip=True)]
        yen = [int(x.replace(",", "")) for x in re.findall(r"([\d,]+)円", tr.select_one("td.Payout").get_text(" "))]
        pay[kind] = list(zip(nums, yen))
    return finish, pay


def paid(pay, kind, combo):
    """その券種・馬番の組が当たっていれば払戻（100円あたり）"""
    return sum(y for n, y in pay.get(kind, []) if sorted(n) == sorted(combo))
