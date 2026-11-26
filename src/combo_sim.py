"""AI の確率でレースを何万回も走らせて、3連複の買い目が当たる確率と、オッズから見た期待値を出す
使い方: from combo_sim import Race;  r = Race(race_id);  r.check(買い目のリスト)
"""
from itertools import combinations
import numpy as np
import pandas as pd
import os
import requests

H = {"User-Agent": "Mozilla/5.0"}


class Race:
    def __init__(self, rid, n=40000, seed=0):
        t = pd.read_pickle(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "predictions", f"race_{rid}.pkl"))
        self.num = t["馬番"].astype(int).to_numpy()
        p3 = t["確率"].clip(upper=.99).to_numpy()
        rng = np.random.default_rng(seed)
        sim = lambda w: np.argsort(-(np.log(w)[None] + rng.gumbel(size=(n, len(w)))), 1)
        w = p3 / p3.sum()
        for _ in range(10):   # 3着以内の確率が AI と同じになるように強さを合わせる
            s3 = np.bincount(sim(w)[:, :3].ravel(), minlength=len(w)) / n
            w = w * (p3 / s3) ** .8; w /= w.sum()
        top3 = np.sort(self.num[sim(w)[:, :3]], axis=1)
        self.freq = pd.Series(1, index=[tuple(x) for x in top3]).groupby(level=0).sum() / n
        j = requests.get(f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={rid}&type=7&action=update",
                         headers=H, timeout=30).json()
        o = (j.get("data") or {}).get("odds", {}).get("7", {})
        self.odds = {tuple(sorted(int(k[i:i + 2]) for i in (0, 2, 4))): float(v[0].replace(",", "")) for k, v in o.items() if v[0] not in ("", "---")}

    def check(self, tickets, name=""):
        tickets = sorted({tuple(sorted(t)) for t in tickets})
        p = sum(self.freq.get(t, 0) for t in tickets)
        ev = sum(self.freq.get(t, 0) * self.odds.get(t, 0) for t in tickets) / len(tickets)
        od = sorted(self.odds.get(t, 0) for t in tickets)
        print(f"{name:<34} {len(tickets):2d}点  当たる確率{p:5.1%}  期待値{ev:5.2f}  オッズ{od[0]:.1f}〜{od[-1]:.1f}倍")
        return tickets


def formation(a, b, c):
    return [t for t in {tuple(sorted((x, y, z))) for x in a for y in b for z in c} if len(set(t)) == 3]


def box(h):
    return list(combinations(h, 3))
