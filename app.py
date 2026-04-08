#!/usr/bin/env python3
""" ╔══════════════════════════════════════════════════════════════════════╗
    ║ ⚽ PREMIER LEAGUE PREDICTOR PRO v4.1 (TURBO EDITION) ⚽              ║
    ║ ✅ V4.1: Turbo Training (n_jobs=-1, optimized stacking limits)       ║
    ║ ✅ V4.1: SyntaxError Print Fix for older Python versions             ║
    ║ ✅ V4.0: Deep Stats Integration & 68 ML features                     ║
    ╚══════════════════════════════════════════════════════════════════════╝ """
import requests
import json
import math
import os
import sys
import time
import hashlib
import pickle
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# --- المتطلبات والمكتبات ---
try:
    import pandas as pd
except ImportError:
    print("❌ Pandas is missing!")
    sys.exit(1)

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

ML_AVAILABLE = False
XGBOOST_AVAILABLE = False
try:
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.isotonic import IsotonicRegression
    ML_AVAILABLE = True
except ImportError:
    pass

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    pass

# --- الإعدادات ---
FOOTBALL_DATA_KEY = os.environ.get("FOOTBALL_DATA_KEY", "")
FOOTBALL_DATA_URL = "https://api.football-data.org/v4"
PL = "PL"
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"

WEIGHTS = {'dixon_coles': 0.25, 'elo': 0.18, 'form': 0.12, 'h2h': 0.08, 'home_advantage': 0.08, 'fatigue': 0.04, 'draw_model': 0.10, 'ml': 0.15}
ELO_INIT = 1500; ELO_K = 32; ELO_HOME = 65; FORM_N = 8
CALIBRATION_FILE = "calibration_v4.pkl"
XGB_MODEL_FILE = "football_xgboost_pipeline_v4.pkl"
TEAMS_MAP_FILE = "teams_master_map.json"
ELO_RATINGS_FILE = "teams_elo_ratings_v4.pkl"
MIN_SAMPLES_PER_CLASS = 10

RIVALRIES = {
    frozenset({'Arsenal', 'Tottenham'}): 'North London Derby',
    frozenset({'Liverpool', 'Everton'}): 'Merseyside Derby',
    frozenset({'Man United', 'Man City'}): 'Manchester Derby',
    frozenset({'Man United', 'Liverpool'}): 'Northwest Derby',
    frozenset({'Chelsea', 'Arsenal'}): 'London Derby',
}

ALIASES: Dict[str, str] = {
    'manchester united': 'Man United', 'manchester city': 'Man City', 'tottenham hotspur': 'Tottenham',
    'newcastle united': 'Newcastle', 'west ham united': 'West Ham', 'wolverhampton wanderers': 'Wolves',
    'nottingham forest': 'Nottm Forest', 'leicester city': 'Leicester', 'brighton & hove albion': 'Brighton',
    'sheffield united': 'Sheffield Utd', 'luton town': 'Luton', 'ipswich town': 'Ipswich'
}

TEAMS_MAP: Dict[str, str] = {}
ELO_RATINGS: Dict[str, float] = {}

def load_external_files():
    global ELO_RATINGS
    if Path(ELO_RATINGS_FILE).exists():
        try:
            with open(ELO_RATINGS_FILE, 'rb') as f:
                raw_elo = pickle.load(f)
            if isinstance(raw_elo, dict):
                for key, val in raw_elo.items():
                    ELO_RATINGS[key] = float(val)
        except Exception: pass

load_external_files()

# --- الأدوات المساعدة ---
def poisson_pmf(k: int, mu: float) -> float:
    return (mu ** k) * math.exp(-mu) / math.factorial(k) if mu > 0 else (1.0 if k == 0 else 0.0)

def safe_div(a: float, b: float, d: float = 0.0) -> float:
    return a / b if b else d

def norm_name(n: str) -> str:
    lo = n.lower().strip()
    if lo in ALIASES: return ALIASES[lo]
    for k, v in ALIASES.items():
        if lo.startswith(k) or k.startswith(lo): return v
    return n

def is_derby(h: str, a: str) -> Optional[str]:
    return RIVALRIES.get(frozenset({norm_name(h), norm_name(a)}))

def parse_date(s: str) -> Optional[datetime]:
    if not s: return None
    try:
        c = s.replace('Z', '')
        fmt = '%Y-%m-%dT%H:%M:%S' if 'T' in c else '%Y-%m-%d %H:%M:%S'
        return datetime.strptime(c[:19], fmt)
    except Exception: return None

# --- قارئ الإحصائيات العميقة CSV ---
def load_csv_history(csv_path: str) -> List[dict]:
    if not Path(csv_path).exists(): return []
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
        matches = []
        for idx, row in df.iterrows():
            try:
                if pd.isna(row.get('FTHG')) or pd.isna(row.get('FTAG')): continue
                h_name = norm_name(str(row['HomeTeam']))
                a_name = norm_name(str(row['AwayTeam']))
                hid = int(hashlib.md5(h_name.encode()).hexdigest()[:8], 16) % 10000
                aid = int(hashlib.md5(a_name.encode()).hexdigest()[:8], 16) % 10000
                date_str = str(row['Date'])
                fmt = '%d/%m/%y' if len(date_str.split('/')[2]) == 2 else '%d/%m/%Y'
                dt = datetime.strptime(date_str, fmt)
                
                match_data = {
                    'status': 'FINISHED',
                    'utcDate': dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    'homeTeam': {'id': hid, 'shortName': h_name, 'name': h_name},
                    'awayTeam': {'id': aid, 'shortName': a_name, 'name': a_name},
                    'score': {'fullTime': {'home': int(row['FTHG']), 'away': int(row['FTAG'])}},
                    'stats': {
                        'HST': row.get('HST', 0), 'AST': row.get('AST', 0), 
                        'HC': row.get('HC', 0), 'AC': row.get('AC', 0),     
                        'HF': row.get('HF', 0), 'AF': row.get('AF', 0),     
                        'HY': row.get('HY', 0), 'AY': row.get('AY', 0),     
                        'HR': row.get('HR', 0), 'AR': row.get('AR', 0)      
                    }
                }
                matches.append(match_data)
            except Exception: continue
        matches.sort(key=lambda x: x.get('utcDate', ''))
        return matches
    except Exception: return []

# --- كلاسات الألوان (CLI) ---
class C:
    G = '\033[92m'; Y = '\033[93m'; R = '\033[91m'; B = '\033[94m'; CN = '\033[96m'
    BD = '\033[1m'; DM = '\033[2m'; E = '\033[0m'; W = '\033[97m'
    @staticmethod
    def bold(t): return f"{C.BD}{t}{C.E}"
    @staticmethod
    def green(t): return f"{C.G}{t}{C.E}"
    @staticmethod
    def yellow(t): return f"{C.Y}{t}{C.E}"
    @staticmethod
    def red(t): return f"{C.R}{t}{C.E}"
    @staticmethod
    def cyan(t): return f"{C.CN}{t}{C.E}"
    @staticmethod
    def form_char(ch): return f"{C.G}{C.BD}W{C.E}" if ch=='W' else (f"{C.Y}{C.BD}D{C.E}" if ch=='D' else f"{C.R}{C.BD}L{C.E}")
    @staticmethod
    def form_str(s): return ' '.join(C.form_char(c) for c in s)
    @staticmethod
    def pct_bar(v, w=20, color=None):
        f = int(max(0.0, min(1.0, v)) * w); e = w - f
        return f"{color or C.G}{'█' * f}{C.E}{C.DM}{'░' * e}{C.E}"

def box(t): return f" {C.B}│{C.E} {t}"

# --- API ---
class FootballAPI:
    def __init__(self, token: str):
        self.s = requests.Session()
        self.s.headers.update({'X-Auth-Token': token, 'Accept': 'application/json'})
    def _get(self, ep: str, p: dict = None):
        try:
            r = self.s.get(f"{FOOTBALL_DATA_URL}/{ep}", params=p, timeout=15)
            return r.json() if r.status_code == 200 else None
        except: return None
    def season_year(self) -> Optional[int]:
        d = self._get(f"competitions/{PL}")
        if d and d.get('currentSeason'): return int(d['currentSeason']['startDate'][:4])
        return None
    def finished(self, season: int = None) -> List[dict]:
        d = self._get(f"competitions/{PL}/matches", {'status': 'FINISHED', 'season': season} if season else {'status': 'FINISHED'})
        return sorted(d['matches'], key=lambda x: x.get('utcDate','')) if d and 'matches' in d else []
    def upcoming(self, days: int = 14) -> List[dict]:
        t = datetime.now().strftime('%Y-%m-%d'); e = (datetime.now() + timedelta(days=days)).strftime('%Y-%m-%d')
        d = self._get(f"competitions/{PL}/matches", {'status': 'SCHEDULED,TIMED', 'dateFrom': t, 'dateTo': e})
        return sorted(d['matches'], key=lambda x: x.get('utcDate','')) if d and 'matches' in d else []

class OddsAPI:
    def __init__(self, key: str): self.key = key; self.cache = {}
    def ok(self) -> bool: return bool(self.key) and len(self.key) > 10
    def fetch(self): return {}
    def find(self, hn: str, an: str) -> Optional[dict]: return None

# --- النظام الأساسي ---
class Team:
    def __init__(self, tid: int, name: str):
        self.id = tid; self.name = name
        self.played = self.wins = self.draws = self.losses = self.gf = self.ga = self.pts = self.pos = 0
        self.h_p = self.h_w = self.h_d = self.h_gf = self.h_ga = 0
        self.a_p = self.a_w = self.a_d = self.a_gf = self.a_ga = 0
        self.results = []; self.elo = ELO_RATINGS.get(name, ELO_INIT); self.elo_hist = [self.elo]
        self.match_dates = []; self.cs = self.fts = self.consec_draws = self.win_streak = self.loss_streak = self.unbeaten = 0
        self._last_draw = False
        
        self.stats_played = self.sot_for = self.sot_against = self.corners_for = self.corners_against = self.discipline_pts = 0

    @property
    def gd(self): return self.gf - self.ga
    @property
    def avg_gf(self): return safe_div(self.gf, self.played)
    @property
    def avg_ga(self): return safe_div(self.ga, self.played)
    @property
    def h_avg_gf(self): return safe_div(self.h_gf, self.h_p)
    @property
    def h_avg_ga(self): return safe_div(self.h_ga, self.h_p)
    @property
    def a_avg_gf(self): return safe_div(self.a_gf, self.a_p)
    @property
    def a_avg_ga(self): return safe_div(self.a_ga, self.a_p)
    @property
    def h_wr(self): return safe_div(self.h_w, self.h_p, 0.45)
    @property
    def a_wr(self): return safe_div(self.a_w, self.a_p, 0.30)
    @property
    def wr(self): return safe_div(self.wins, self.played)
    @property
    def dr(self): return safe_div(self.draws, self.played)
    @property
    def h_dr(self): return safe_div(self.h_d, self.h_p)
    @property
    def a_dr(self): return safe_div(self.a_d, self.a_p)
    @property
    def cs_r(self): return safe_div(self.cs, self.played)
    @property
    def fts_r(self): return safe_div(self.fts, self.played)
    @property
    def ppg(self): return safe_div(self.pts, self.played)

    @property
    def avg_sot(self): return safe_div(self.sot_for, self.stats_played)
    @property
    def avg_corners(self): return safe_div(self.corners_for, self.stats_played)
    @property
    def avg_discipline(self): return safe_div(self.discipline_pts, self.stats_played)

    @property
    def form_score(self) -> float:
        rec = self.results[-FORM_N:]
        if not rec: return 50.0
        tot = max_t = 0.0
        for i, r in enumerate(rec):
            w = math.exp(0.3 * (i - len(rec) + 1)); pts = {'W':3,'D':1,'L':0}[r[0]]
            tot += pts * w; max_t += 3 * w
        return (tot / max_t) * 100 if max_t else 50.0

    @property
    def goal_form(self) -> float:
        rec = self.results[-FORM_N:]
        if not rec: return self.avg_gf
        return sum(r[1]*math.exp(0.2*(i-len(rec)+1)) for i,r in enumerate(rec)) / sum(math.exp(0.2*(i-len(rec)+1)) for i in range(len(rec)))

    @property
    def defense_form(self) -> float:
        rec = self.results[-FORM_N:]
        if not rec: return self.avg_ga
        return sum(r[2]*math.exp(0.2*(i-len(rec)+1)) for i,r in enumerate(rec)) / sum(math.exp(0.2*(i-len(rec)+1)) for i in range(len(rec)))

    @property
    def draw_form(self) -> float: return sum(1 for r in self.results[-FORM_N:] if r[0] == 'D') / min(len(self.results[-FORM_N:]) or 1, FORM_N)
    @property
    def form_string(self) -> str: return ''.join(r[0] for r in self.results[-6:])

    @property
    def momentum(self) -> int:
        if self.win_streak >= 5: return 90
        if self.win_streak >= 3: return 60 + self.win_streak * 5
        if self.win_streak >= 2: return 40
        if self.unbeaten >= 5: return 30
        if self.loss_streak >= 4: return -80
        if self.loss_streak >= 3: return -50
        if self.loss_streak >= 2: return -25
        return 0

    @property
    def volatility(self) -> float:
        rec = self.results[-10:]
        if len(rec) < 4: return 0.5
        goals = [r[1] + r[2] for r in rec]; mean = sum(goals) / len(goals)
        var = sum((g - mean) ** 2 for g in goals) / len(goals)
        return min(1.0, math.sqrt(var) / 2.0)

    def days_rest(self, ref: datetime = None) -> int:
        return max(0, ((ref or datetime.now()) - max(self.match_dates)).days) if self.match_dates else 7

    def matches_in(self, n: int = 14, ref: datetime = None) -> int:
        cut = (ref or datetime.now()) - timedelta(days=n)
        return sum(1 for d in self.match_dates if d >= cut)

class EloSystem:
    def update(self, h: Team, a: Team, hg: int, ag: int):
        ha = h.elo + ELO_HOME; eh = 1.0 / (1.0 + 10 ** ((a.elo - ha) / 400)); ea = 1 - eh
        ah, aa = (1.0, 0.0) if hg > ag else ((0.0, 1.0) if hg < ag else (0.5, 0.5))
        gd = abs(hg - ag); m = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11 + gd) / 8)
        kh = ELO_K * (1.5 if h.played < 5 else (0.85 if h.elo > 1600 else 1.0))
        ka = ELO_K * (1.5 if a.played < 5 else (0.85 if a.elo > 1600 else 1.0))
        h.elo += kh * m * (ah - eh); a.elo += ka * m * (aa - ea)
        h.elo_hist.append(h.elo); a.elo_hist.append(a.elo)

    def predict(self, h: Team, a: Team) -> Tuple[float, float, float]:
        ha = h.elo + ELO_HOME; eh = 1.0 / (1.0 + 10 ** ((a.elo - ha) / 400)); ea = 1 - eh
        db = max(0.18, 0.32 - abs(ha - a.elo) / 1200)
        hw = eh * (1 - db); aw = ea * (1 - db); t = hw + db + aw
        return (hw/t, db/t, aw/t)

class DixonColes:
    @staticmethod
    def prob(hg, ag, lh, la, rho=-0.13):
        b = poisson_pmf(hg, lh) * poisson_pmf(ag, la)
        tau = 1 - lh * la * rho if (hg==0 and ag==0) else (1 + lh * rho if hg==0 and ag==1 else (1 + la * rho if hg==1 and ag==0 else (1 - rho if hg==1 and ag==1 else 1.0)))
        return max(0, b * tau)

    @staticmethod
    def predict(hxg, axg, rho=-0.13, mg=10):
        hw = dr = aw = 0.0
        for i in range(mg):
            for j in range(mg):
                p = DixonColes.prob(i, j, hxg, axg, rho)
                if i > j: hw += p
                elif i == j: dr += p
                else: aw += p
        t = hw + dr + aw
        return (hw/t, dr/t, aw/t) if t > 0 else (0.4, 0.25, 0.35)

    @staticmethod
    def matrix(hxg, axg, rho=-0.13, mg=10):
        return {(i, j): DixonColes.prob(i, j, hxg, axg, rho) for i in range(mg) for j in range(mg)}

class DrawPredictor:
    @staticmethod
    def predict(h: Team, a: Team, derby: bool = False, elo_d: float = 0) -> Tuple[float, float, float]:
        boost = 0.0; ad = abs(elo_d)
        if ad < 30: boost += 0.08
        elif ad < 60: boost += 0.05
        avg_dr = (h.dr + a.dr) / 2
        if avg_dr > 0.35: boost += 0.06
        if (h.draw_form + a.draw_form) / 2 > 0.3: boost += 0.05
        if derby: boost += 0.04
        if abs(h.momentum) > 50 or abs(a.momentum) > 50: boost -= 0.03
        bd = min(0.42, 0.25 + boost); rem = 1.0 - bd
        hp, ap = (rem * 0.58, rem * 0.42) if elo_d > 0 else ((rem * 0.42, rem * 0.58) if elo_d < 0 else (rem * 0.50, rem * 0.50))
        return (hp, bd, ap)

class Fatigue:
    @staticmethod
    def score(t: Team, ref: datetime = None) -> float:
        rd = t.days_rest(ref); m14 = t.matches_in(14, ref); m30 = t.matches_in(30, ref)
        rs = {0:40,1:40,2:40,3:30,4:20,5:10}.get(rd, 0 if rd <= 7 else -5)
        return max(0.0, min(100.0, rs + (35 if m14>=5 else 0) + (25 if m30>=9 else 0)))
    @staticmethod
    def impact(t: Team, ref: datetime = None) -> float: return 1.05 - (Fatigue.score(t, ref) / 100) * 0.17
    @staticmethod
    def predict(h: Team, a: Team, ref: datetime = None) -> Tuple[float, float, float]:
        hi = Fatigue.impact(h, ref); ai = Fatigue.impact(a, ref); t = hi + ai
        if t == 0: return (0.4, 0.25, 0.35)
        hp = hi/t; ap = ai/t; d = max(0.18, 0.30 - abs(hp - ap) * 0.3)
        hp *= (1-d); ap *= (1-d); tt = hp+d+ap
        return (hp/tt, d/tt, ap/tt)

class Calibrator:
    def __init__(self): self.ok = False; self.models = {}; self.hist = []
    def add(self, probs: Tuple[float, float, float], actual: str): self.hist.append({'probs': probs, 'actual': actual})
    def calibrate(self) -> bool:
        if not ML_AVAILABLE or len(self.hist) < 30: return False
        try:
            for idx, out in enumerate(['HOME', 'DRAW', 'AWAY']):
                ps = np.array([h['probs'][idx] for h in self.hist])
                ac = np.array([1 if h['actual'] == out else 0 for h in self.hist])
                iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds='clip')
                iso.fit(ps, ac)
                self.models[out] = iso
            self.ok = True; return True
        except: return False
    def adjust(self, probs: Tuple[float, float, float]) -> Tuple[float, float, float]:
        if not self.ok: return probs
        try:
            adj = [float(self.models[o].predict([probs[i]])[0]) if o in self.models else probs[i] for i, o in enumerate(['HOME', 'DRAW', 'AWAY'])]
            t = sum(adj)
            return tuple(p / t for p in adj) if t > 0 else probs
        except: return probs
    def save(self):
        try:
            with open(CALIBRATION_FILE, 'wb') as f: pickle.dump({'hist': self.hist, 'ok': self.ok, 'models': self.models}, f)
        except: pass
    def load(self):
        try:
            if Path(CALIBRATION_FILE).exists():
                with open(CALIBRATION_FILE, 'rb') as f: d = pickle.load(f)
                self.hist = d.get('hist', []); self.ok = d.get('ok', False); self.models = d.get('models', {})
                return True
        except: pass
        return False

class DataProc:
    def __init__(self):
        self.teams = {}; self.elo = EloSystem(); self.avg_h = 1.53; self.avg_a = 1.16; self.fixes = []; self.h2h = defaultdict(list)
    def process(self, matches: List[dict], do_elo: bool = True):
        for m in sorted(matches, key=lambda m: m.get('utcDate', '')):
            if m.get('status') != 'FINISHED': continue
            hid, hn = m['homeTeam']['id'], m['homeTeam'].get('shortName', '?')
            aid, an = m['awayTeam']['id'], m['awayTeam'].get('shortName', '?')
            hg, ag = m.get('score', {}).get('fullTime', {}).get('home'), m.get('score', {}).get('fullTime', {}).get('away')
            if hg is None or ag is None: continue
            ds = m.get('utcDate', '')
            
            if hid not in self.teams: self.teams[hid] = Team(hid, hn)
            if aid not in self.teams: self.teams[aid] = Team(aid, an)
            h = self.teams[hid]; a = self.teams[aid]
            
            md = parse_date(ds)
            if md: h.match_dates.append(md); a.match_dates.append(md)
            if do_elo: self.elo.update(h, a, hg, ag)

            h.played += 1; a.played += 1
            h.gf += hg; h.ga += ag; a.gf += ag; a.ga += hg
            h.h_p += 1; h.h_gf += hg; h.h_ga += ag
            a.a_p += 1; a.a_gf += ag; a.a_ga += hg
            if ag == 0: h.cs += 1
            if hg == 0: a.cs += 1
            h.fts += 1

            stats = m.get('stats')
            if stats and not pd.isna(stats.get('HST', np.nan)):
                h.stats_played += 1; a.stats_played += 1
                h.sot_for += stats.get('HST', 0); h.sot_against += stats.get('AST', 0)
                a.sot_for += stats.get('AST', 0); a.sot_against += stats.get('HST', 0)
                h.corners_for += stats.get('HC', 0); h.corners_against += stats.get('AC', 0)
                a.corners_for += stats.get('AC', 0); a.corners_against += stats.get('HC', 0)
                h.discipline_pts += (stats.get('HF', 0) + stats.get('HY', 0)*3 + stats.get('HR', 0)*10)
                a.discipline_pts += (stats.get('AF', 0) + stats.get('AY', 0)*3 + stats.get('AR', 0)*10)

            if hg > ag:
                h.wins += 1; h.h_w += 1; a.losses += 1; h.pts += 3
                h.results.append(('W', hg, ag, ds)); a.results.append(('L', ag, hg, ds))
                h.win_streak += 1; h.loss_streak = 0; h.unbeaten += 1
                a.win_streak = 0; a.loss_streak += 1; a.unbeaten = 0
            elif hg < ag:
                a.wins += 1; a.a_w += 1; h.losses += 1; a.pts += 3
                h.results.append(('L', hg, ag, ds)); a.results.append(('W', ag, hg, ds))
                a.win_streak += 1; a.loss_streak = 0; a.unbeaten += 1
                h.win_streak = 0; h.loss_streak += 1; h.unbeaten = 0
            else:
                h.draws += 1; a.draws += 1; h.h_d += 1; a.a_d += 1; h.pts += 1; a.pts += 1
                h.results.append(('D', hg, ag, ds)); a.results.append(('D', ag, hg, ds))
                h.win_streak = 0; a.win_streak = 0; h.loss_streak = 0; a.loss_streak = 0
                h.unbeaten += 1; a.unbeaten += 1

            self.h2h[f"{min(hid,aid)}_{max(hid,aid)}"].append({'home_id': hid, 'away_id': aid, 'home_goals': hg, 'away_goals': ag, 'date': ds})
            self.fixes.append({'home_id': hid, 'away_id': aid, 'home_goals': hg, 'away_goals': ag, 'date': ds, 'home_name': hn, 'away_name': an, 'stats': stats})
        
        tm = sum(t.h_p for t in self.teams.values())
        if tm:
            self.avg_h = sum(t.h_gf for t in self.teams.values()) / tm
            self.avg_a = sum(t.a_gf for t in self.teams.values()) / tm
        for i, t in enumerate(sorted(self.teams.values(), key=lambda t: (t.pts, t.gd, t.gf), reverse=True), 1): t.pos = i

    def team_by_name(self, name: str) -> Optional[Team]:
        for t in self.teams.values():
            if t.name.lower() == name.lower().strip() or name.lower().strip() in t.name.lower(): return t
        return None

# --- ML (النسخة السريعة TURBO) ---
class MLPred:
    N_FEATURES = 68

    def __init__(self):
        self.pipeline = None; self.trained = False; self.acc = 0.0; self._external = False

    def feats(self, h: Team, a: Team, data: DataProc, md: datetime = None, derby: bool = False) -> List[float]:
        ah = max(data.avg_h, 0.5); aa = max(data.avg_a, 0.5)
        return [
            h.elo, a.elo, h.elo - a.elo, h.form_score, a.form_score, h.form_score - a.form_score, abs(h.form_score - a.form_score),
            h.h_avg_gf, a.a_avg_gf, h.goal_form, a.goal_form, h.goal_form - a.goal_form, h.h_avg_gf - a.a_avg_gf,
            h.h_avg_ga, a.a_avg_ga, h.defense_form, a.defense_form, h.defense_form - a.defense_form, h.h_avg_ga - a.a_avg_ga,
            safe_div(h.h_avg_gf, ah, 1), safe_div(a.a_avg_gf, aa, 1), safe_div(h.h_avg_ga, ah, 1), safe_div(a.a_avg_ga, aa, 1),
            h.h_wr, a.a_wr, h.wr, a.wr, h.pos, a.pos, a.pos - h.pos, h.pts, a.pts, h.ppg - a.ppg, h.gd, a.gd,
            h.cs_r, a.cs_r, h.fts_r, a.fts_r, Fatigue.score(h, md), Fatigue.score(a, md),
            h.dr, a.dr, (h.dr + a.dr) / 2, h.draw_form, a.draw_form, h.h_dr, a.a_dr, h.volatility, a.volatility,
            1.0 if derby else 0.0, abs(h.elo - a.elo) / 100,
            h.momentum / 100, a.momentum / 100, (h.momentum - a.momentum) / 100,
            h.win_streak, a.win_streak, h.loss_streak, a.loss_streak,
            h.avg_sot, a.avg_sot, (h.avg_sot - a.avg_sot),
            h.avg_corners, a.avg_corners, (h.avg_corners - a.avg_corners),
            h.avg_discipline, a.avg_discipline, (h.avg_discipline - a.avg_discipline)
        ]

    def _try_load_external(self) -> bool:
        if not Path(XGB_MODEL_FILE).exists(): return False
        try:
            with open(XGB_MODEL_FILE, 'rb') as f: loaded = pickle.load(f)
            self.pipeline = loaded; self.trained = True; self._external = True; return True
        except: return False

    def train(self, data: DataProc, fixes: List[dict] = None, force_retrain: bool = False) -> bool:
        if not ML_AVAILABLE: return False
        if not force_retrain and self._try_load_external(): return True

        fixes = fixes or data.fixes
        if len(fixes) < 40: return False

        X, y = [], []
        sim = DataProc()
        sf = sorted(fixes, key=lambda f: f.get('date', ''))
        warm = int(len(sf) * 0.30)
        
        for idx, f in enumerate(sf):
            if idx >= warm:
                ht, at = sim.teams.get(f['home_id']), sim.teams.get(f['away_id'])
                if ht and at and ht.played >= 3 and at.played >= 3:
                    try:
                        md = parse_date(f.get('date', ''))
                        ft = self.feats(ht, at, sim, md, bool(is_derby(f['home_name'], f['away_name'])))
                        X.append(ft)
                        y.append(0 if f['home_goals'] > f['away_goals'] else (1 if f['home_goals'] == f['away_goals'] else 2))
                    except: pass
            sim.process([{'status': 'FINISHED', 'homeTeam': {'id': f['home_id']}, 'awayTeam': {'id': f['away_id']}, 'score': {'fullTime': {'home': f['home_goals'], 'away': f['away_goals']}}, 'utcDate': f.get('date', ''), 'stats': f.get('stats', {})}], do_elo=True)

        if len(X) < 30: return False

        X_arr, y_arr = np.array(X, dtype=np.float64), np.array(y, dtype=np.int64)
        classes, counts = np.unique(y_arr, return_counts=True)
        if int(counts.min()) < MIN_SAMPLES_PER_CLASS: return False

        # --- السرعة القصوى TURBO MODE ---
        n_splits = min(3, max(2, int(counts.min()) // 5))

        base_rf = RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_split=5, class_weight='balanced', random_state=42, n_jobs=-1)
        base_gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        estimators = [('rf', base_rf), ('gb', base_gb)]
        
        if XGBOOST_AVAILABLE:
            freq = counts / counts.sum()
            spw = float(freq[0] / freq[1]) if len(freq) > 1 else 1.0
            estimators.append(('xgb', XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.08, scale_pos_weight=spw, random_state=42, n_jobs=-1)))

        meta_lr = LogisticRegression(max_iter=500, class_weight='balanced')
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        try:
            # إزالة التكرار المتداخل واستخدام كل الأنوية
            stacker = StackingClassifier(estimators=estimators, final_estimator=meta_lr, cv=skf, n_jobs=-1)
            full_pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler()), ('model', stacker)])
            
            cv_scores = cross_val_score(full_pipeline, X_arr, y_arr, cv=skf, scoring='balanced_accuracy', n_jobs=-1)
            self.acc = float(cv_scores.mean())
            full_pipeline.fit(X_arr, y_arr)
            self.pipeline = full_pipeline
        except:
            return False

        self.trained = True
        return True

    def predict(self, h: Team, a: Team, data: DataProc, md: datetime = None, derby: bool = False) -> Optional[Tuple[float, float, float]]:
        if not self.trained or self.pipeline is None: return None
        try:
            ft = self.feats(h, a, data, md, derby)
            probs = self.pipeline.predict_proba(np.array([ft], dtype=np.float64))[0]
            prob_dict = {cls: pr for cls, pr in zip(self.pipeline.classes_, probs)}
            return (float(prob_dict.get(0, 0.0)), float(prob_dict.get(1, 0.0)), float(prob_dict.get(2, 0.0)))
        except: return None

    def save_pipeline(self):
        if self.pipeline:
            try:
                with open(XGB_MODEL_FILE, 'wb') as f: pickle.dump(self.pipeline, f)
            except: pass

# --- محرك التوقعات ---
class Pred:
    def __init__(self):
        self.home = self.away = self.date = self.result = self.dc_recommend = self.h_form = self.a_form = self.derby_name = ""
        self.hp = self.dp = self.ap = self.hxg = self.axg = self.conf = self.dc_1x = self.dc_x2 = self.dc_12 = self.h_elo = self.a_elo = 0.0
        self.pred_sc = (0, 0); self.top_sc = []; self.models = {}; self.calibrated = self.is_derby = False

class Engine:
    def __init__(self, data: DataProc, ml: MLPred = None, cal: Calibrator = None):
        self.data = data; self.ml = ml; self.cal = cal; self.w = dict(WEIGHTS)
        if not ml or not ml.trained:
            mw = self.w.pop('ml', 0); rem = sum(self.w.values())
            if rem > 0:
                for k in self.w: self.w[k] += mw * (self.w[k] / rem)

    def predict(self, hid: int, aid: int, date: str = "") -> Optional[Pred]:
        h, a = self.data.teams.get(hid), self.data.teams.get(aid)
        if not h or not a or h.played < 2 or a.played < 2: return None
        
        p = Pred(); p.home = h.name; p.away = a.name; md = parse_date(date) or datetime.now()
        derby = is_derby(h.name, a.name); p.is_derby = bool(derby); p.derby_name = derby or ""
        p.h_elo = h.elo; p.a_elo = a.elo

        ah = max(self.data.avg_h, 0.5); aa = max(self.data.avg_a, 0.5)
        p.hxg = max(0.25, min(safe_div(h.h_avg_gf, ah, 1) * safe_div(a.a_avg_ga, ah, 1) * ah * (0.7 + 0.3 * min(safe_div(h.goal_form, max(h.avg_gf, 0.5), 1.0), 2.0)), 4.5)) * Fatigue.impact(h, md)
        p.axg = max(0.25, min(safe_div(a.a_avg_gf, aa, 1) * safe_div(h.h_avg_ga, aa, 1) * aa * (0.7 + 0.3 * min(safe_div(a.goal_form, max(a.avg_gf, 0.5), 1.0), 2.0)), 4.5)) * Fatigue.impact(a, md)
        if h.momentum > 40: p.hxg *= 1.05
        elif h.momentum < -40: p.hxg *= 0.95
        if a.momentum > 40: p.axg *= 1.05
        elif a.momentum < -40: p.axg *= 0.95

        m = {'dixon_coles': DixonColes.predict(p.hxg, p.axg), 'elo': self.data.elo.predict(h, a), 'fatigue': Fatigue.predict(h, a, md), 'draw_model': DrawPredictor.predict(h, a, p.is_derby, h.elo + ELO_HOME - a.elo)}
        if self.ml and self.ml.trained:
            mp = self.ml.predict(h, a, self.data, md, p.is_derby)
            if mp: m['ml'] = mp
        
        p.models = m
        hp = dp = ap = tw = 0.0
        for nm, probs in m.items():
            wt = self.w.get(nm, 0)
            if wt > 0: hp += probs[0]*wt; dp += probs[1]*wt; ap += probs[2]*wt; tw += wt
        if tw > 0: hp /= tw; dp /= tw; ap /= tw
        
        if self.cal and self.cal.ok: hp, dp, ap = self.cal.adjust((hp, dp, ap)); p.calibrated = True
        p.hp, p.dp, p.ap = hp, dp, ap
        p.dc_1x = hp + dp; p.dc_x2 = ap + dp; p.dc_12 = hp + ap
        
        pd_map = {'HOME': hp, 'DRAW': dp, 'AWAY': ap}
        p.result = max(pd_map, key=pd_map.get); p.conf = max(pd_map.values()) * 100
        
        mx = DixonColes.matrix(p.hxg, p.axg)
        p.top_sc = [(s[0][0], s[0][1], s[1]) for s in sorted(mx.items(), key=lambda x: x[1], reverse=True)[:6]]
        if p.top_sc: p.pred_sc = (p.top_sc[0][0], p.top_sc[0][1])
        return p

# --- التطبيق الرئيسي ---
class App:
    def __init__(self, token: str):
        self.api = FootballAPI(token); self.data = DataProc()
        self.cal = Calibrator(); self.eng = None; self.ml = None

    def init(self) -> bool:
        print("🔄 Loading historical deep stats from CSV...")
        
        # البحث عن ملف البيانات
        csv_matches = load_csv_history("E0_Master.csv")
        if not csv_matches: csv_matches = load_csv_history("E0 (1).csv")
        if not csv_matches:
            print("❌ لم يتم العثور على ملفات البيانات (E0_Master.csv أو E0 (1).csv)")
            return False

        self.raw = csv_matches
        print(f"✅ Loaded {len(self.raw)} matches!")
        print("⚙️ Processing Data...")
        self.data.process(self.raw)
        
        if ML_AVAILABLE:
            print("🧠 Training ML Pipeline (TURBO MODE)...")
            self.ml = MLPred()
            if self.ml.train(self.data):
                print(f"✅ ML trained: {self.ml.acc*100:.1f}% balanced CV acc")
                if not self.ml._external: self.ml.save_pipeline()
        
        if self.cal.load(): print("✅ Calibration loaded")
        self.eng = Engine(self.data, self.ml, self.cal)
        print("✅ Engine v4.1 ready!")
        return True

    def backtest(self):
        print("🔬 Running Backtest...")
        fin = [m for m in self.raw if m.get('status') == 'FINISHED']
        train = fin[:int(len(fin) * 0.7)]
        test = fin[int(len(fin) * 0.7):]
        
        td = DataProc(); td.process(train)
        ml = MLPred(); ml.train(td, force_retrain=True)
        eng = Engine(td, ml)
        
        cr = total = csc = 0
        for m in test[len(test)//2:]:
            hid = m.get('homeTeam', {}).get('id'); aid = m.get('awayTeam', {}).get('id')
            ahg = m.get('score', {}).get('fullTime', {}).get('home'); aag = m.get('score', {}).get('fullTime', {}).get('away')
            if not hid or not aid or ahg is None or aag is None: continue
            pr = eng.predict(hid, aid, m.get('utcDate', ''))
            if not pr: continue
            
            ahg, aag = int(ahg), int(aag)
            actual = 'HOME' if ahg>aag else ('AWAY' if ahg<aag else 'DRAW')
            total += 1
            if pr.result == actual: cr += 1
            if pr.pred_sc[0]==ahg and pr.pred_sc[1]==aag: csc += 1
            td.process([m])

        ra = cr / total * 100 if total else 0
        sa = csc / total * 100 if total else 0
        
        print("\n" + "="*40)
        print(f"📊 BACKTEST RESULTS:")
        print(f"✅ 1X2 Accuracy: {ra:.1f}% ({cr}/{total})")
        print(f"🎯 Score Accuracy: {sa:.1f}%")
        acc_str = f"{ml.acc*100:.1f}%" if ml.trained else "N/A"
        print(f"🤖 ML Balanced Acc: {acc_str}")
        print("="*40)
        return {'result_acc': ra, 'score_acc': sa, 'ml_acc': ml.acc if ml.trained else 0}

if __name__ == "__main__":
    pass
