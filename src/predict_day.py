"""1日の全レースを予想する（新馬戦と障害は予想しない）。各レースの上位5頭を表示
使い方: python src/predict_day.py <YYYY-MM-DD>
"""
import os, re, subprocess, sys
import requests
from bs4 import BeautifulSoup

DATE = sys.argv[1]
r = requests.get(f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={DATE.replace('-', '')}",
                 headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
r.encoding = r.apparent_encoding
races = []
for li in BeautifulSoup(r.text, "lxml").select("li.RaceList_DataItem"):
    text = " ".join(li.get_text(" ", strip=True).split())
    if "新馬" in text or "障" in text:
        continue
    races.append(re.search(r"race_id=(\d+)", li.find("a")["href"]).group(1))
print(f"{DATE}: {len(races)}レースを予想します", flush=True)
subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "predict_race.py"), DATE, *races], env={**os.environ, "TOP": "5"}, check=True)
