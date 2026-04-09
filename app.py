#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║ ⚽ FOOTBALL PREDICTOR PRO v5.0 (MULTI-LEAGUE FINAL EDITION) ⚽      ║
║                                                                        ║
║ ✅ V5.0: Multi-League Support                                         ║
║ ✅ V5.0: Separate Data Files per League                               ║
║ ✅ V5.0: Separate Training/Model Files per League                     ║
║ ✅ V5.0: Backtest Bug Fixed (136% issue)                              ║
║ ✅ V4.2: Deep Stats Integration (Shots, Corners, Discipline)          ║
║ ✅ V4.2: ML features = 68 features                                    ║
╚══════════════════════════════════════════════════════════════════════╝
"""

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

try:
    import pandas as pd
except ImportError:
    print("❌ Pandas missing! pip install pandas")
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
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.isotonic import IsotonicRegression
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    ML_AVAILABLE = True
except ImportError:
    pass

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    pass

# ══════════════════════════════════════════════════════════════
# LEAGUES CONFIG LOADER
# ══════════════════════════════════════════════════════════════

LEAGUES_CONFIG_FILE = "leagues_config.json"

DEFAULT_GLOBAL_SETTINGS = {
    "elo_init": 1500,
    "elo_k": 32,
    "form_n": 8,
    "backtest_split": 0.70,
    "min_samples_per_class": 10,
    "n_features": 68
}

def load_leagues_config() -> dict:
    if not Path(LEAGUES_CONFIG_FILE).exists():
        print(f"⚠️ {LEAGUES_CONFIG_FILE} not found! Creating default...")
        create_default_config()
    try:
        with open(LEAGUES_CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return {"leagues": {}, "global_settings": DEFAULT_GLOBAL_SETTINGS}

def create_default_config():
    default = {
        "leagues": {
            "PL": {
                "name": "Premier League",
                "country": "England",
                "api_code": "PL",
                "api_url": "https://api.football-data.org/v4",
                "data_files": ["data/PL_Master.csv"],
                "model_file": "models/PL_model_v5.pkl",
                "calibration_file": "models/PL_calibration_v5.pkl",
                "elo_file": "models/PL_elo_v5.pkl",
                "teams_map_file": "config/PL_teams_map.json",
                "aliases_file": "config/PL_aliases.json",
                "rivalries_file": "config/PL_rivalries.json",
                "home_advantage": 65,
                "avg_home_goals": 1.53,
                "avg_away_goals": 1.16
            }
        },
        "global_settings": DEFAULT_GLOBAL_SETTINGS
    }
    for folder in ["data", "models", "config"]:
        Path(folder).mkdir(exist_ok=True)
    with open(LEAGUES_CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(default, f, indent=2, ensure_ascii=False)
    print(f"✅ Created {LEAGUES_CONFIG_FILE}")

_GLOBAL_CONFIG = load_leagues_config()
GLOBAL_SETTINGS = _GLOBAL_CONFIG.get("global_settings", DEFAULT_GLOBAL_SETTINGS)
LEAGUES_CONFIG = _GLOBAL_CONFIG.get("leagues", {})

ELO_INIT = GLOBAL_SETTINGS.get("elo_init", 1500)
ELO_K = GLOBAL_SETTINGS.get("elo_k", 32)
FORM_N = GLOBAL_SETTINGS.get("form_n", 8)
BACKTEST_SPLIT = GLOBAL_SETTINGS.get("backtest_split", 0.70)
MIN_SAMPLES_PER_CLASS = GLOBAL_SETTINGS.get("min_samples_per_class", 10)
N_FEATURES = GLOBAL_SETTINGS.get("n_features", 68)

WEIGHTS = {
    'dixon_coles': 0.25,
    'elo': 0.18,
    'form': 0.12,
    'h2h': 0.08,
    'home_advantage': 0.08,
    'fatigue': 0.04,
    'draw_model': 0.10,
    'ml': 0.15,
}

# ══════════════════════════════════════════════════════════════
# LEAGUE RESOURCES
# ══════════════════════════════════════════════════════════════

class LeagueResources:
    def __init__(self, league_code: str, config: dict):
        self.code = league_code
        self.config = config
        self.name = config.get("name", league_code)
        self.country = config.get("country", "")
        self.api_code = config.get("api_code", league_code)
        self.api_url = config.get("api_url", "https://api.football-data.org/v4")
        self.home_advantage = config.get("home_advantage", 65)
        self.avg_home_goals = config.get("avg_home_goals", 1.53)
        self.avg_away_goals = config.get("avg_away_goals", 1.16)
        self.data_files: List[str] = config.get("data_files", [])
        self.model_file: str = config.get("model_file", f"models/{league_code}_model_v5.pkl")
        self.calibration_file: str = config.get("calibration_file", f"models/{league_code}_calibration_v5.pkl")
        self.elo_file: str = config.get("elo_file", f"models/{league_code}_elo_v5.pkl")
        self.teams_map_file: str = config.get("teams_map_file", f"config/{league_code}_teams_map.json")
        self.aliases_file: str = config.get("aliases_file", f"config/{league_code}_aliases.json")
        self.rivalries_file: str = config.get("rivalries_file", f"config/{league_code}_rivalries.json")
        self.teams_map: Dict[str, str] = {}
        self.aliases: Dict[str, str] = {}
        self.rivalries: Dict[frozenset, str] = {}
        self.elo_ratings: Dict[str, float] = {}
        self._load_all()

    def _load_all(self):
        self._load_teams_map()
        self._load_aliases()
        self._load_rivalries()
        self._load_elo()

    def _load_teams_map(self):
        if Path(self.teams_map_file).exists():
            try:
                with open(self.teams_map_file, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                self.teams_map = self._build_safe_teams_map(raw)
            except Exception as e:
                print(f"⚠️ [{self.code}] teams_map error: {e}")

    def _load_aliases(self):
        if Path(self.aliases_file).exists():
            try:
                with open(self.aliases_file, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                self.aliases = {k.lower().strip(): v for k, v in raw.items()}
            except Exception as e:
                print(f"⚠️ [{self.code}] aliases error: {e}")
        else:
            self.aliases = self._get_default_aliases()

    def _load_rivalries(self):
        if Path(self.rivalries_file).exists():
            try:
                with open(self.rivalries_file, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                for item in raw:
                    if isinstance(item, dict):
                        teams = item.get("teams", [])
                        derby_name = item.get("name", "Derby")
                        if len(teams) == 2:
                            self.rivalries[frozenset(teams)] = derby_name
            except Exception as e:
                print(f"⚠️ [{self.code}] rivalries error: {e}")

    def _load_elo(self):
        if Path(self.elo_file).exists():
            try:
                with open(self.elo_file, 'rb') as f:
                    raw_elo = pickle.load(f)
                if isinstance(raw_elo, dict):
                    for key, val in raw_elo.items():
                        if isinstance(key, str) and isinstance(val, (int, float)):
                            self.elo_ratings[key] = float(val)
            except Exception as e:
                print(f"⚠️ [{self.code}] elo error: {e}")

    def save_elo(self, elo_data: Dict[str, float]):
        Path(self.elo_file).parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.elo_file, 'wb') as f:
                pickle.dump(elo_data, f)
        except Exception as e:
            print(f"⚠️ [{self.code}] save elo error: {e}")

    @staticmethod
    def _build_safe_teams_map(raw: dict) -> Dict[str, str]:
        safe = {}
        for k, v in raw.items():
            k_str = str(k).strip()
            if ' v ' in k_str.lower() or any(ch.isdigit() for ch in k_str) or len(k_str) > 40:
                continue
            safe[k_str.lower()] = str(v)
        return safe

    @staticmethod
    def _get_default_aliases() -> Dict[str, str]:
        return {
            'manchester united': 'Man United',
            'manchester city': 'Man City',
            'tottenham hotspur': 'Tottenham',
            'tottenham': 'Tottenham',
            'newcastle united': 'Newcastle',
            'west ham united': 'West Ham',
            'wolverhampton wanderers': 'Wolves',
            'wolverhampton': 'Wolves',
            'nottingham forest': 'Nottm Forest',
            'leicester city': 'Leicester',
            'brighton & hove albion': 'Brighton',
            'brighton': 'Brighton',
            'crystal palace': 'Crystal Palace',
            'aston villa': 'Aston Villa',
            'arsenal': 'Arsenal',
            'chelsea': 'Chelsea',
            'liverpool': 'Liverpool',
            'everton': 'Everton',
            'brentford': 'Brentford',
            'fulham': 'Fulham',
            'bournemouth': 'Bournemouth',
            'luton town': 'Luton',
            'sheffield united': 'Sheffield Utd',
            'burnley': 'Burnley',
            'ipswich town': 'Ipswich',
            'southampton': 'Southampton',
            'real madrid': 'Real Madrid',
            'fc barcelona': 'Barcelona',
            'atletico madrid': 'Atletico Madrid',
            'athletic bilbao': 'Athletic Club',
            'fc bayern münchen': 'Bayern Munich',
            'borussia dortmund': 'Dortmund',
            'bayer 04 leverkusen': 'Leverkusen',
            'inter milan': 'Inter',
            'ac milan': 'Milan',
            'juventus fc': 'Juventus',
            'paris saint-germain': 'PSG',
            'olympique de marseille': 'Marseille',
            'olympique lyonnais': 'Lyon',
        }

    def norm_name(self, name: str) -> str:
        lo = name.lower().strip()
        if lo in self.aliases:
            return self.aliases[lo]
        for k, v in self.aliases.items():
            if lo.startswith(k) or k.startswith(lo):
                return v
        if self.teams_map:
            if lo in self.teams_map:
                return self.teams_map[lo]
            lo_words = set(lo.split())
            candidates = [v for k, v in self.teams_map.items() if set(k.split()) == lo_words]
            if len(candidates) == 1:
                return candidates[0]
        return name

    def is_derby(self, home: str, away: str) -> Optional[str]:
        return self.rivalries.get(frozenset({self.norm_name(home), self.norm_name(away)}))

    def load_csv_data(self) -> List[dict]:
        all_matches = []
        for csv_path in self.data_files:
            if not Path(csv_path).exists():
                print(f"⚠️ [{self.code}] File not found: {csv_path}")
                continue
            matches = self._parse_csv(csv_path)
            all_matches.extend(matches)
            print(f"✅ [{self.code}] Loaded {len(matches)} matches from {csv_path}")
        all_matches.sort(key=lambda x: x.get('utcDate', ''))
        return all_matches

    def _parse_csv(self, csv_path: str) -> List[dict]:
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
            matches = []
            for idx, row in df.iterrows():
                try:
                    if pd.isna(row.get('FTHG')) or pd.isna(row.get('FTAG')):
                        continue
                    h_name = self.norm_name(str(row['HomeTeam']))
                    a_name = self.norm_name(str(row['AwayTeam']))
                    hid = int(hashlib.md5(f"{self.code}_{h_name}".encode()).hexdigest()[:8], 16) % 100000
                    aid = int(hashlib.md5(f"{self.code}_{a_name}".encode()).hexdigest()[:8], 16) % 100000
                    date_str = str(row['Date'])
                    try:
                        fmt = '%d/%m/%y' if len(date_str.split('/')[-1]) == 2 else '%d/%m/%Y'
                        dt = datetime.strptime(date_str, fmt)
                    except ValueError:
                        try:
                            dt = pd.to_datetime(date_str).to_pydatetime()
                        except Exception:
                            continue
                    stats = {}
                    for col in ['HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']:
                        val = row.get(col, 0)
                        stats[col] = 0 if pd.isna(val) else float(val)
                    match_data = {
                        'status': 'FINISHED',
                        'utcDate': dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
                        'homeTeam': {'id': hid, 'shortName': h_name, 'name': h_name},
                        'awayTeam': {'id': aid, 'shortName': a_name, 'name': a_name},
                        'score': {'fullTime': {'home': int(row['FTHG']), 'away': int(row['FTAG'])}},
                        'stats': stats,
                        'league': self.code
                    }
                    matches.append(match_data)
                except Exception:
                    continue
            return matches
        except Exception as e:
            print(f"❌ [{self.code}] CSV parse error {csv_path}: {e}")
            return []

# ══════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════

def poisson_pmf(k: int, mu: float) -> float:
    if mu <= 0:
        return 1.0 if k == 0 else 0.0
    return (mu ** k) * math.exp(-mu) / math.factorial(k)

def safe_div(a: float, b: float, d: float = 0.0) -> float:
    return a / b if b else d

def parse_date(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        c = s.replace('Z', '')
        fmt = '%Y-%m-%dT%H:%M:%S' if 'T' in c else '%Y-%m-%d %H:%M:%S'
        return datetime.strptime(c[:19], fmt)
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════
# COLOUR HELPERS
# ══════════════════════════════════════════════════════════════

class C:
    H  = '\033[95m'
    B  = '\033[94m'
    CN = '\033[96m'
    G  = '\033[92m'
    Y  = '\033[93m'
    R  = '\033[91m'
    BD = '\033[1m'
    DM = '\033[2m'
    E  = '\033[0m'
    W  = '\033[97m'

    @staticmethod
    def bold(t): return f"{C.BD}{t}{C.E}"
    @staticmethod
    def green(t): return f"{C.G}{t}{C.E}"
    @staticmethod
    def red(t): return f"{C.R}{t}{C.E}"
    @staticmethod
    def yellow(t): return f"{C.Y}{t}{C.E}"
    @staticmethod
    def cyan(t): return f"{C.CN}{t}{C.E}"
    @staticmethod
    def blue(t): return f"{C.B}{t}{C.E}"
    @staticmethod
    def dim(t): return f"{C.DM}{t}{C.E}"
    @staticmethod
    def magenta(t): return f"{C.H}{t}{C.E}"

    @staticmethod
    def form_char(ch):
        if ch == 'W': return f"{C.G}{C.BD}W{C.E}"
        if ch == 'D': return f"{C.Y}{C.BD}D{C.E}"
        if ch == 'L': return f"{C.R}{C.BD}L{C.E}"
        return ch

    @staticmethod
    def form_str(s): return ' '.join(C.form_char(c) for c in s)

    @staticmethod
    def pct_bar(v, w=20, color=None):
        color = color or C.G
        f = int(max(0.0, min(1.0, v)) * w)
        e = w - f
        return f"{color}{'█' * f}{C.E}{C.DM}{'░' * e}{C.E}"

def box(t):
    return f" {C.blue('│')} {t}"

# ══════════════════════════════════════════════════════════════
# TEAM
# ══════════════════════════════════════════════════════════════

class Team:
    def __init__(self, tid: int, name: str, elo_init: float = None):
        self.id = tid
        self.name = name
        self.played = 0
        self.wins = 0
        self.draws = 0
        self.losses = 0
        self.gf = 0
        self.ga = 0
        self.pts = 0
        self.pos = 0
        self.h_p = 0
        self.h_w = 0
        self.h_d = 0
        self.h_gf = 0
        self.h_ga = 0
        self.a_p = 0
        self.a_w = 0
        self.a_d = 0
        self.a_gf = 0
        self.a_ga = 0
        self.results: List[Tuple] = []
        self.elo = elo_init if elo_init is not None else ELO_INIT
        self.elo_hist = [self.elo]
        self.match_dates: List[datetime] = []
        self.cs = 0
        self.fts = 0
        self._last_draw = False
        self.consec_draws = 0
        self.win_streak = 0
        self.loss_streak = 0
        self.unbeaten = 0
        self.stats_played = 0
        self.sot_for = 0
        self.sot_against = 0
        self.corners_for = 0
        self.corners_against = 0
        self.discipline_pts = 0

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
        total = 0.0
        max_t = 0.0
        for i, r in enumerate(rec):
            w = math.exp(0.3 * (i - len(rec) + 1))
            pts = {'W': 3, 'D': 1, 'L': 0}[r[0]]
            total += pts * w
            max_t += 3 * w
        return (total / max_t) * 100 if max_t else 50.0

    @property
    def goal_form(self) -> float:
        rec = self.results[-FORM_N:]
        if not rec: return self.avg_gf
        total = 0.0
        wt = 0.0
        for i, r in enumerate(rec):
            w = math.exp(0.2 * (i - len(rec) + 1))
            total += r[1] * w
            wt += w
        return total / wt if wt else self.avg_gf

    @property
    def defense_form(self) -> float:
        rec = self.results[-FORM_N:]
        if not rec: return self.avg_ga
        total = 0.0
        wt = 0.0
        for i, r in enumerate(rec):
            w = math.exp(0.2 * (i - len(rec) + 1))
            total += r[2] * w
            wt += w
        return total / wt if wt else self.avg_ga

    @property
    def draw_form(self) -> float:
        rec = self.results[-FORM_N:]
        if not rec: return self.dr
        return sum(1 for r in rec if r[0] == 'D') / len(rec)

    @property
    def form_string(self) -> str:
        return ''.join(r[0] for r in self.results[-6:])

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
        goals = [r[1] + r[2] for r in rec]
        mean = sum(goals) / len(goals)
        var = sum((g - mean) ** 2 for g in goals) / len(goals)
        return min(1.0, math.sqrt(var) / 2.0)

    def days_rest(self, ref: datetime = None) -> int:
        if not self.match_dates: return 7
        ref = ref or datetime.now()
        return max(0, (ref - max(self.match_dates)).days)

    def matches_in(self, n: int = 14, ref: datetime = None) -> int:
        ref = ref or datetime.now()
        cut = ref - timedelta(days=n)
        return sum(1 for d in self.match_dates if d >= cut)

# ══════════════════════════════════════════════════════════════
# ELO SYSTEM
# ══════════════════════════════════════════════════════════════

class EloSystem:
    def __init__(self, home_advantage: int = 65):
        self.k = ELO_K
        self.ha = home_advantage

    def expected(self, ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400))

    def gd_mult(self, gd: int) -> float:
        gd = abs(gd)
        if gd <= 1: return 1.0
        if gd == 2: return 1.5
        return (11 + gd) / 8

    def update(self, h: Team, a: Team, hg: int, ag: int):
        ha = h.elo + self.ha
        eh = self.expected(ha, a.elo)
        ea = 1 - eh
        if hg > ag:
            ah, aa = 1.0, 0.0
        elif hg < ag:
            ah, aa = 0.0, 1.0
        else:
            ah, aa = 0.5, 0.5
        m = self.gd_mult(hg - ag)
        kh = self.k * (1.5 if h.played < 5 else (0.85 if h.elo > 1600 else 1.0))
        ka = self.k * (1.5 if a.played < 5 else (0.85 if a.elo > 1600 else 1.0))
        h.elo += kh * m * (ah - eh)
        a.elo += ka * m * (aa - ea)
        h.elo_hist.append(h.elo)
        a.elo_hist.append(a.elo)

    def predict(self, h: Team, a: Team) -> Tuple[float, float, float]:
        ha = h.elo + self.ha
        eh = self.expected(ha, a.elo)
        ea = 1 - eh
        dd = abs(ha - a.elo)
        db = max(0.18, 0.32 - dd / 1200)
        hw = eh * (1 - db)
        aw = ea * (1 - db)
        t = hw + db + aw
        return (hw / t, db / t, aw / t)

# ══════════════════════════════════════════════════════════════
# DIXON-COLES
# ══════════════════════════════════════════════════════════════

class DixonColes:
    @staticmethod
    def tau(hg, ag, lh, la, rho):
        if hg == 0 and ag == 0: return 1 - lh * la * rho
        if hg == 0 and ag == 1: return 1 + lh * rho
        if hg == 1 and ag == 0: return 1 + la * rho
        if hg == 1 and ag == 1: return 1 - rho
        return 1.0

    @staticmethod
    def prob(hg, ag, lh, la, rho=-0.13):
        b = poisson_pmf(hg, lh) * poisson_pmf(ag, la)
        return max(0, b * DixonColes.tau(hg, ag, lh, la, rho))

    @staticmethod
    def predict(hxg, axg, rho=-0.13, mg=10):
        hw = 0.0
        dr = 0.0
        aw = 0.0
        for i in range(mg):
            for j in range(mg):
                p = DixonColes.prob(i, j, hxg, axg, rho)
                if i > j:
                    hw += p
                elif i == j:
                    dr += p
                else:
                    aw += p
        t = hw + dr + aw
        return (hw / t, dr / t, aw / t) if t > 0 else (0.4, 0.25, 0.35)

    @staticmethod
    def matrix(hxg, axg, rho=-0.13, mg=10):
        return {
            (i, j): DixonColes.prob(i, j, hxg, axg, rho)
            for i in range(mg) for j in range(mg)
        }

# ══════════════════════════════════════════════════════════════
# DRAW PREDICTOR
# ══════════════════════════════════════════════════════════════

class DrawPredictor:
    @staticmethod
    def predict(h: Team, a: Team, derby: bool = False, elo_d: float = 0) -> Tuple[float, float, float]:
        boost = 0.0
        ad = abs(elo_d)
        if ad < 30:
            boost += 0.08
        elif ad < 60:
            boost += 0.05
        elif ad < 100:
            boost += 0.02
        avg_dr = (h.dr + a.dr) / 2
        if avg_dr > 0.35:
            boost += 0.06
        elif avg_dr > 0.25:
            boost += 0.03
        if (h.draw_form + a.draw_form) / 2 > 0.3:
            boost += 0.05
        if derby:
            boost += 0.04
        if (h.cs_r + a.cs_r) / 2 > 0.35:
            boost += 0.04
        if (h.fts_r + a.fts_r) / 2 > 0.25:
            boost += 0.03
        if (h.volatility + a.volatility) / 2 < 0.3:
            boost += 0.03
        if abs(h.momentum) > 50 or abs(a.momentum) > 50:
            boost -= 0.03
        bd = min(0.42, 0.25 + boost)
        rem = 1.0 - bd
        if elo_d > 0:
            hp = rem * 0.58
            ap = rem * 0.42
        elif elo_d < 0:
            hp = rem * 0.42
            ap = rem * 0.58
        else:
            hp = rem * 0.50
            ap = rem * 0.50
        return (hp, bd, ap)

# ══════════════════════════════════════════════════════════════
# FATIGUE
# ══════════════════════════════════════════════════════════════

class Fatigue:
    @staticmethod
    def score(t: Team, ref: datetime = None) -> float:
        ref = ref or datetime.now()
        rd = t.days_rest(ref)
        m14 = t.matches_in(14, ref)
        m30 = t.matches_in(30, ref)
        rest_scores = {0: 40, 1: 40, 2: 40, 3: 30, 4: 20, 5: 10}
        rs = rest_scores.get(rd, 0 if rd <= 7 else -5)
        d14 = 35 if m14 >= 5 else (25 if m14 >= 4 else (15 if m14 >= 3 else 0))
        d30 = 25 if m30 >= 9 else (15 if m30 >= 7 else 0)
        return max(0.0, min(100.0, rs + d14 + d30))

    @staticmethod
    def impact(t: Team, ref: datetime = None) -> float:
        return 1.05 - (Fatigue.score(t, ref) / 100) * 0.17

    @staticmethod
    def predict(h: Team, a: Team, ref: datetime = None) -> Tuple[float, float, float]:
        hi = Fatigue.impact(h, ref)
        ai = Fatigue.impact(a, ref)
        t = hi + ai
        if t == 0:
            return (0.4, 0.25, 0.35)
        hp = hi / t
        ap = ai / t
        d = max(0.18, 0.30 - abs(hp - ap) * 0.3)
        hp *= (1 - d)
        ap *= (1 - d)
        tt = hp + d + ap
        return (hp / tt, d / tt, ap / tt)

# ══════════════════════════════════════════════════════════════
# CALIBRATOR
# ══════════════════════════════════════════════════════════════

class Calibrator:
    def __init__(self):
        self.ok = False
        self.models: Dict[str, IsotonicRegression] = {}
        self.hist: List[dict] = []

    def add(self, probs: Tuple[float, float, float], actual: str):
        self.hist.append({'probs': probs, 'actual': actual})

    def calibrate(self) -> bool:
        if not ML_AVAILABLE or len(self.hist) < 30:
            return False
        try:
            for idx, out in enumerate(['HOME', 'DRAW', 'AWAY']):
                ps = np.array([h['probs'][idx] for h in self.hist])
                ac = np.array([1 if h['actual'] == out else 0 for h in self.hist])
                iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds='clip')
                iso.fit(ps, ac)
                self.models[out] = iso
            self.ok = True
            return True
        except Exception:
            return False

    def adjust(self, probs: Tuple[float, float, float]) -> Tuple[float, float, float]:
        if not self.ok:
            return probs
        try:
            adj = []
            for i, out in enumerate(['HOME', 'DRAW', 'AWAY']):
                if out in self.models:
                    adj.append(float(self.models[out].predict([probs[i]])[0]))
                else:
                    adj.append(probs[i])
            t = sum(adj)
            return tuple(p / t for p in adj) if t > 0 else probs
        except Exception:
            return probs

    def save(self, fn: str):
        Path(fn).parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(fn, 'wb') as f:
                pickle.dump({'hist': self.hist, 'ok': self.ok, 'models': self.models}, f)
        except Exception:
            pass

    def load(self, fn: str) -> bool:
        try:
            if Path(fn).exists():
                with open(fn, 'rb') as f:
                    d = pickle.load(f)
                self.hist = d.get('hist', [])
                self.ok = d.get('ok', False)
                self.models = d.get('models', {})
                return True
        except Exception:
            pass
        return False

# ══════════════════════════════════════════════════════════════
# DATA PROCESSOR
# ══════════════════════════════════════════════════════════════

class DataProc:
    def __init__(self, resources: LeagueResources = None):
        self.resources = resources
        self.teams: Dict[int, Team] = {}
        self.elo = EloSystem(
            home_advantage=resources.home_advantage if resources else 65
        )
        self.avg_h = resources.avg_home_goals if resources else 1.53
        self.avg_a = resources.avg_away_goals if resources else 1.16
        self.total = 0
        self.fixes: List[dict] = []
        self.h2h: Dict[str, List[dict]] = defaultdict(list)

    def _get_elo_init(self, name: str) -> float:
        if self.resources and name in self.resources.elo_ratings:
            return self.resources.elo_ratings[name]
        return ELO_INIT

    def process(self, matches: List[dict], do_elo: bool = True):
        matches.sort(key=lambda m: m.get('utcDate', ''))
        cnt = 0
        for m in matches:
            r = self._ext(m)
            if not r:
                continue
            hid, hn, aid, an, hg, ag, ds = r
            if hid not in self.teams:
                self.teams[hid] = Team(hid, hn, self._get_elo_init(hn))
            if aid not in self.teams:
                self.teams[aid] = Team(aid, an, self._get_elo_init(an))
            h = self.teams[hid]
            a = self.teams[aid]
            md = parse_date(ds)
            if md:
                h.match_dates.append(md)
                a.match_dates.append(md)
            if do_elo:
                self.elo.update(h, a, hg, ag)
            h.played += 1
            a.played += 1
            h.gf += hg
            h.ga += ag
            a.gf += ag
            a.ga += hg
            h.h_p += 1
            h.h_gf += hg
            h.h_ga += ag
            a.a_p += 1
            a.a_gf += ag
            a.a_ga += hg
            if ag == 0:
                h.cs += 1
            if hg == 0:
                a.cs += 1
            h.fts += 1
            if ag == 0 and hg > 0:
                a.fts += 1
            stats = m.get('stats')
            if stats:
                hst = stats.get('HST', 0)
                if hst is not None and not (isinstance(hst, float) and math.isnan(hst)):
                    h.stats_played += 1
                    a.stats_played += 1
                    h.sot_for += stats.get('HST', 0) or 0
                    h.sot_against += stats.get('AST', 0) or 0
                    a.sot_for += stats.get('AST', 0) or 0
                    a.sot_against += stats.get('HST', 0) or 0
                    h.corners_for += stats.get('HC', 0) or 0
                    h.corners_against += stats.get('AC', 0) or 0
                    a.corners_for += stats.get('AC', 0) or 0
                    a.corners_against += stats.get('HC', 0) or 0
                    h.discipline_pts += (
                        (stats.get('HF', 0) or 0) +
                        (stats.get('HY', 0) or 0) * 3 +
                        (stats.get('HR', 0) or 0) * 10
                    )
                    a.discipline_pts += (
                        (stats.get('AF', 0) or 0) +
                        (stats.get('AY', 0) or 0) * 3 +
                        (stats.get('AR', 0) or 0) * 10
                    )
            draw = (hg == ag)
            if hg > ag:
                h.wins += 1
                h.h_w += 1
                a.losses += 1
                h.pts += 3
                h.results.append(('W', hg, ag, ds))
                a.results.append(('L', ag, hg, ds))
                h.win_streak += 1
                h.loss_streak = 0
                h.unbeaten += 1
                a.win_streak = 0
                a.loss_streak += 1
                a.unbeaten = 0
            elif hg < ag:
                a.wins += 1
                a.a_w += 1
                h.losses += 1
                a.pts += 3
                h.results.append(('L', hg, ag, ds))
                a.results.append(('W', ag, hg, ds))
                a.win_streak += 1
                a.loss_streak = 0
                a.unbeaten += 1
                h.win_streak = 0
                h.loss_streak += 1
                h.unbeaten = 0
            else:
                h.draws += 1
                a.draws += 1
                h.h_d += 1
                a.a_d += 1
                h.pts += 1
                a.pts += 1
                h.results.append(('D', hg, ag, ds))
                a.results.append(('D', ag, hg, ds))
                h.win_streak = 0
                a.win_streak = 0
                h.loss_streak = 0
                a.loss_streak = 0
                h.unbeaten += 1
                a.unbeaten += 1
                h.consec_draws = (h.consec_draws + 1 if (draw and h._last_draw) else (1 if draw else 0))
                a.consec_draws = (a.consec_draws + 1 if (draw and a._last_draw) else (1 if draw else 0))
                h._last_draw = draw
                a._last_draw = draw
            key = f"{min(hid, aid)}_{max(hid, aid)}"
            self.h2h[key].append({
                'home_id': hid,
                'away_id': aid,
                'home_goals': hg,
                'away_goals': ag,
                'date': ds
            })
            self.fixes.append({
                'home_id': hid,
                'away_id': aid,
                'home_goals': hg,
                'away_goals': ag,
                'date': ds,
                'home_name': hn,
                'away_name': an,
                'stats': m.get('stats', {})
            })
            cnt += 1
        self.total += cnt
        self._avgs()
        self._rank()

    def _ext(self, m: dict):
        if m.get('status') != 'FINISHED':
            return None
        ht = m.get('homeTeam', {})
        at = m.get('awayTeam', {})
        hid = ht.get('id')
        aid = at.get('id')
        if not hid or not aid:
            return None
        hn = ht.get('shortName') or ht.get('name', '?')
        an = at.get('shortName') or at.get('name', '?')
        ft = m.get('score', {}).get('fullTime', {})
        hg = ft.get('home')
        ag = ft.get('away')
        if hg is None or ag is None:
            return None
        return (hid, hn, aid, an, int(hg), int(ag), m.get('utcDate', ''))

    def get_h2h(self, t1: int, t2: int) -> List[dict]:
        return self.h2h.get(f"{min(t1, t2)}_{max(t1, t2)}", [])

    def _avgs(self):
        th = sum(t.h_gf for t in self.teams.values())
        ta = sum(t.a_gf for t in self.teams.values())
        tm = sum(t.h_p for t in self.teams.values())
        if tm:
            self.avg_h = th / tm
            self.avg_a = ta / tm

    def _rank(self):
        for i, t in enumerate(
            sorted(self.teams.values(), key=lambda t: (t.pts, t.gd, t.gf), reverse=True), 1
        ):
            t.pos = i

    def team_by_name(self, name: str) -> Optional[Team]:
        lo = name.lower().strip()
        for t in self.teams.values():
            if t.name.lower() == lo:
                return t
        for t in self.teams.values():
            if lo in t.name.lower() or t.name.lower() in lo:
                return t
        return None

# ══════════════════════════════════════════════════════════════
# ML PREDICTOR
# ══════════════════════════════════════════════════════════════

class MLPred:
    N_FEATURES = N_FEATURES

    def __init__(self, model_file: str = None):
        self.pipeline: Optional[Pipeline] = None
        self.trained = False
        self.acc = 0.0
        self._external = False
        self.model_file = model_file or "models/default_model_v5.pkl"

    def feats(self, h: Team, a: Team, data: DataProc, md: datetime = None, derby: bool = False) -> List[float]:
        ah = max(data.avg_h, 0.5)
        aa = max(data.avg_a, 0.5)
        return [
            # ── Elo (3) ──────────────────────────────────────────
            h.elo,                          # 0: elo الهوم
            a.elo,                          # 1: elo الأواي
            h.elo - a.elo,                  # 2: الفارق
            # ── Form Score (4) ───────────────────────────────────
            h.form_score,                   # 3: فورم الهوم
            a.form_score,                   # 4: فورم الأواي
            h.form_score - a.form_score,    # 5: الفارق
            abs(h.form_score - a.form_score), # 6: القيمة المطلقة

            # ── هجوم وفورم التهديف (6) ──────────────────────────
            h.h_avg_gf,                     # 7: معدل تهديف الهوم في الملعب
            a.a_avg_gf,                     # 8: معدل تهديف الأواي خارجاً
            h.goal_form,                    # 9: فورم تهديف الهوم
            a.goal_form,                    # 10: فورم تهديف الأواي
            h.goal_form - a.goal_form,      # 11: الفارق
            h.h_avg_gf - a.a_avg_gf,        # 12: فارق الهجوم في الملعب

            # ── دفاع وفورم التلقي (6) ───────────────────────────
            h.h_avg_ga,                     # 13: معدل تلقي الهوم في الملعب
            a.a_avg_ga,                     # 14: معدل تلقي الأواي خارجاً
            h.defense_form,                 # 15: فورم دفاع الهوم
            a.defense_form,                 # 16: فورم دفاع الأواي
            h.defense_form - a.defense_form, # 17: الفارق
            h.h_avg_ga - a.a_avg_ga,        # 18: فارق الدفاع

            # ── نسب الهجوم/الدفاع (4) ─────────────────────────
            # ✅ الإصلاح: الهوم يُقسم على avg_h، الأواي على avg_a
            safe_div(h.h_avg_gf, ah, 1.0),  # 19: نسبة هجوم الهوم
            safe_div(a.a_avg_gf, aa, 1.0),  # 20: نسبة هجوم الأواي ✅
            safe_div(h.h_avg_ga, ah, 1.0),  # 21: نسبة دفاع الهوم
            safe_div(a.a_avg_ga, aa, 1.0),  # 22: نسبة دفاع الأواي ✅

            # ── نسب الفوز (4) ───────────────────────────────────
            h.h_wr,                         # 23: نسبة فوز الهوم في الملعب
            a.a_wr,                         # 24: نسبة فوز الأواي خارجاً
            h.wr,                           # 25: نسبة فوز الهوم الكلية
            a.wr,                           # 26: نسبة فوز الأواي الكلية

            # ── الترتيب والنقاط (7) ─────────────────────────────
            h.pos,                          # 27: مركز الهوم
            a.pos,                          # 28: مركز الأواي
            a.pos - h.pos,                  # 29: فارق المركز (양 = الأواي أضعف)
            h.pts,                          # 30: نقاط الهوم
            a.pts,                          # 31: نقاط الأواي
            h.ppg - a.ppg,                  # 32: فارق معدل النقاط
            h.gd,                           # 33: فارق أهداف الهوم

            # ── فارق أهداف الأواي (1) ───────────────────────────
            a.gd,                           # 34: فارق أهداف الأواي

            # ── الشباك النظيفة (4) ──────────────────────────────
            h.cs_r,                         # 35: نسبة شباك نظيفة الهوم
            a.cs_r,                         # 36: نسبة شباك نظيفة الأواي
            h.fts_r,                        # 37: نسبة المباريات دون تسجيل الهوم
            a.fts_r,                        # 38: نسبة المباريات دون تسجيل الأواي

            # ── الإرهاق (2) ─────────────────────────────────────
            Fatigue.score(h, md),           # 39: إرهاق الهوم
            Fatigue.score(a, md),           # 40: إرهاق الأواي

            # ── نسب التعادل (7) ─────────────────────────────────
            h.dr,                           # 41: نسبة تعادل الهوم الكلية
            a.dr,                           # 42: نسبة تعادل الأواي الكلية
            (h.dr + a.dr) / 2,             # 43: المتوسط
            h.draw_form,                    # 44: فورم تعادل الهوم
            a.draw_form,                    # 45: فورم تعادل الأواي
            h.h_dr,                         # 46: نسبة تعادل الهوم في الملعب
            a.a_dr,                         # 47: نسبة تعادل الأواي خارجاً

            # ── التقلب (2) ───────────────────────────────────────
            h.volatility,                   # 48: تقلب الهوم
            a.volatility,                   # 49: تقلب الأواي

            # ── الديربي (1) ──────────────────────────────────────
            1.0 if derby else 0.0,          # 50: هل هو ديربي

            # ── Elo مقياس (1) ────────────────────────────────────
            abs(h.elo - a.elo) / 100,       # 51: فارق Elo مُعاد قياسه

            # ── الزخم (7) ────────────────────────────────────────
            h.momentum / 100,               # 52: زخم الهوم
            a.momentum / 100,               # 53: زخم الأواي
            (h.momentum - a.momentum) / 100, # 54: فارق الزخم
            h.win_streak,                   # 55: سلسلة انتصارات الهوم
            a.win_streak,                   # 56: سلسلة انتصارات الأواي
            h.loss_streak,                  # 57: سلسلة خسائر الهوم
            a.loss_streak,                  # 58: سلسلة خسائر الأواي

            # ── الإحصائيات العميقة (9) ──────────────────────────
            h.avg_sot,                      # 59: متوسط تسديدات الهوم على المرمى
            a.avg_sot,                      # 60: متوسط تسديدات الأواي على المرمى
            h.avg_sot - a.avg_sot,          # 61: الفارق
            h.avg_corners,                  # 62: متوسط ركنيات الهوم
            a.avg_corners,                  # 63: متوسط ركنيات الأواي
            h.avg_corners - a.avg_corners,  # 64: الفارق
            h.avg_discipline,               # 65: معدل أخطاء وبطاقات الهوم
            a.avg_discipline,               # 66: معدل أخطاء وبطاقات الأواي
            h.avg_discipline - a.avg_discipline, # 67: الفارق
        ]
           # ✅ إجمالي: 68 ميزة (0-67)
       

    def _try_load_external(self) -> bool:
        if not Path(self.model_file).exists():
            return False
        try:
            with open(self.model_file, 'rb') as f:
                loaded = pickle.load(f)
            if not isinstance(loaded, Pipeline):
                return False
            final_step = loaded.steps[-1][1]
            n_feat = getattr(final_step, 'n_features_in_', None)
            if n_feat is not None and n_feat != self.N_FEATURES:
                print(f"⚠️ Model features mismatch: {n_feat} vs {self.N_FEATURES}")
                return False
            self.pipeline = loaded
            self.trained = True
            self._external = True
            return True
        except Exception:
            return False

    def train(self, data: DataProc, fixes: List[dict] = None, force_retrain: bool = False) -> bool:
        if not ML_AVAILABLE:
            return False
        if not force_retrain and self._try_load_external():
            return True
        fixes = fixes or data.fixes
        if len(fixes) < 40:
            return False
        X: List[List[float]] = []
        y: List[int] = []
        sim = DataProc(data.resources)
        sf = sorted(fixes, key=lambda f: f.get('date', ''))
        warm = int(len(sf) * 0.30)
        for idx, f in enumerate(sf):
            if idx >= warm:
                ht = sim.teams.get(f['home_id'])
                at = sim.teams.get(f['away_id'])
                if ht and at and ht.played >= 3 and at.played >= 3:
                    try:
                        md = parse_date(f.get('date', ''))
                        derby = False
                        if data.resources:
                            derby = bool(data.resources.is_derby(f['home_name'], f['away_name']))
                        ft = self.feats(ht, at, sim, md, derby)
                        if f['home_goals'] > f['away_goals']:
                            lb = 0
                        elif f['home_goals'] == f['away_goals']:
                            lb = 1
                        else:
                            lb = 2
                        X.append(ft)
                        y.append(lb)
                    except Exception:
                        pass
            fake = {
                'status': 'FINISHED',
                'homeTeam': {'id': f['home_id'], 'shortName': f['home_name']},
                'awayTeam': {'id': f['away_id'], 'shortName': f['away_name']},
                'score': {'fullTime': {'home': f['home_goals'], 'away': f['away_goals']}},
                'utcDate': f.get('date', ''),
                'stats': f.get('stats', {})
            }
            sim.process([fake], do_elo=True)
        if len(X) < 30:
            return False
        X_arr = np.array(X, dtype=np.float64)
        y_arr = np.array(y, dtype=np.int64)
        classes, counts = np.unique(y_arr, return_counts=True)
        min_count = int(counts.min())
        if min_count < MIN_SAMPLES_PER_CLASS:
            return False
        estimators = []
        base_rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        estimators.append(('rf', base_rf))
        if XGBOOST_AVAILABLE:
            freq = counts / counts.sum()
            spw = float(freq[0] / freq[1]) if len(freq) > 1 else 1.0
            base_xgb = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=spw,
                random_state=42,
                n_jobs=-1
            )
            estimators.append(('xgb', base_xgb))
        try:
            voter = VotingClassifier(estimators=estimators, voting='soft', n_jobs=1)
            full_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('model', voter)
            ])
            skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
            cv_scores = cross_val_score(
                full_pipeline, X_arr, y_arr,
                cv=skf, scoring='balanced_accuracy', n_jobs=1
            )
            self.acc = float(cv_scores.mean())
            full_pipeline.fit(X_arr, y_arr)
            self.pipeline = full_pipeline
        except Exception:
            fallback = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('model', RandomForestClassifier(
                    n_estimators=100,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                ))
            ])
            try:
                fallback.fit(X_arr, y_arr)
                self.pipeline = fallback
                self.acc = 0.0
            except Exception:
                return False
        self.trained = True
        return True

    def predict(self, h: Team, a: Team, data: DataProc, md: datetime = None, derby: bool = False) -> Optional[Tuple[float, float, float]]:
        if not self.trained or self.pipeline is None:
            return None
        try:
            ft = self.feats(h, a, data, md, derby)
            X = np.array([ft], dtype=np.float64)
            probs = self.pipeline.predict_proba(X)[0]
            classes = self.pipeline.classes_
            prob_dict = {cls: pr for cls, pr in zip(classes, probs)}
            return (
                float(prob_dict.get(0, 0.0)),
                float(prob_dict.get(1, 0.0)),
                float(prob_dict.get(2, 0.0))
            )
        except Exception:
            return None

    def save_pipeline(self):
        if self.pipeline is not None:
            Path(self.model_file).parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(self.model_file, 'wb') as f:
                    pickle.dump(self.pipeline, f)
                print(f"✅ Model saved: {self.model_file}")
            except Exception as e:
                print(f"⚠️ Save model error: {e}")

# ══════════════════════════════════════════════════════════════
# API CLIENT
# ══════════════════════════════════════════════════════════════

class FootballAPI:
    def __init__(self, token: str, base_url: str = "https://api.football-data.org/v4"):
        self.s = requests.Session()
        self.s.headers.update({'X-Auth-Token': token, 'Accept': 'application/json'})
        self.base_url = base_url
        self._c: Dict[str, dict] = {}
        self._t: float = 0.0

    def _rl(self):
        e = time.time() - self._t
        if e < 6.5:
            time.sleep(6.5 - e)
        self._t = time.time()

    def _get(self, ep: str, p: dict = None, cache: bool = True):
        p = p or {}
        k = hashlib.md5(f"{ep}|{json.dumps(p, sort_keys=True)}".encode()).hexdigest()
        if cache and k in self._c:
            return self._c[k]
        try:
            self._rl()
            r = self.s.get(f"{self.base_url}/{ep}", params=p, timeout=30)
            if r.status_code == 429:
                wait = int(r.headers.get('X-RequestCounter-Reset', 60)) + 1
                time.sleep(wait)
                return self._get(ep, p, cache)
            if r.status_code in (401, 403, 404):
                return None
            r.raise_for_status()
            d = r.json()
            if cache:
                self._c[k] = d
            return d
        except Exception:
            return None

    def season_year(self, league_code: str) -> Optional[int]:
        d = self._get(f"competitions/{league_code}")
        if d and d.get('currentSeason'):
            try:
                return int(d['currentSeason']['startDate'][:4])
            except Exception:
                pass
        return None

    def finished(self, league_code: str, season: int = None) -> List[dict]:
        p = {'status': 'FINISHED'}
        if season:
            p['season'] = season
        d = self._get(f"competitions/{league_code}/matches", p)
        if d and 'matches' in d:
            m = d['matches']
            m.sort(key=lambda x: x.get('utcDate', ''))
            return m
        return []

    def upcoming(self, league_code: str, days: int = 14) -> List[dict]:
        t = datetime.now().strftime('%Y-%m-%d')
        e = (datetime.now() + timedelta(days=days)).strftime('%Y-%m-%d')
        d = self._get(
            f"competitions/{league_code}/matches",
            {'status': 'SCHEDULED,TIMED', 'dateFrom': t, 'dateTo': e}
        )
        if d and 'matches' in d:
            m = d['matches']
            m.sort(key=lambda x: x.get('utcDate', ''))
            return m
        return []

class OddsAPI:
    def __init__(self, key: str, sport: str = "soccer_epl"):
        self.key = key
        self.sport = sport
        self.cache: Dict[str, dict] = {}

    def ok(self) -> bool:
        return bool(self.key) and len(self.key) > 10

    def fetch(self) -> Dict[str, dict]:
        if not self.ok():
            return {}
        try:
            r = requests.get(
                f"https://api.the-odds-api.com/v4/sports/{self.sport}/odds",
                params={
                    'apiKey': self.key,
                    'regions': 'uk,eu',
                    'markets': 'h2h,totals',
                    'oddsFormat': 'decimal'
                },
                timeout=15
            )
            if r.status_code != 200:
                return {}
            result: Dict[str, dict] = {}
            for ev in r.json():
                h = ev.get('home_team', '')
                a = ev.get('away_team', '')
                bms = ev.get('bookmakers', [])
                if not bms:
                    continue
                ah_list = []
                ad_list = []
                aa_list = []
                for bm in bms:
                    for mk in bm.get('markets', []):
                        if mk['key'] == 'h2h':
                            for o in mk.get('outcomes', []):
                                if o['name'] == h:
                                    ah_list.append(o['price'])
                                elif o['name'] == a:
                                    aa_list.append(o['price'])
                                elif o['name'] == 'Draw':
                                    ad_list.append(o['price'])
                if ah_list and ad_list and aa_list:
                    avh = sum(ah_list) / len(ah_list)
                    avd = sum(ad_list) / len(ad_list)
                    ava = sum(aa_list) / len(aa_list)
                    ih = 1 / avh
                    id_ = 1 / avd
                    ia = 1 / ava
                    result[f"{h}_vs_{a}".lower()] = {
                        'home_team': h,
                        'away_team': a,
                        'odds_home': round(avh, 2),
                        'odds_draw': round(avd, 2),
                        'odds_away': round(ava, 2),
                        'implied_home': round(ih, 4),
                        'implied_draw': round(id_, 4),
                        'implied_away': round(ia, 4),
                        'implied_1x': round(ih + id_, 4),
                        'implied_x2': round(ia + id_, 4),
                        'implied_12': round(ih + ia, 4),
                    }
            self.cache = result
            return result
        except Exception:
            return {}

    def find(self, hn: str, an: str) -> Optional[dict]:
        if not self.cache:
            self.fetch()
        hl = hn.lower()
        al = an.lower()
        for k, d in self.cache.items():
            oh = d['home_team'].lower()
            oa = d['away_team'].lower()
            hm = (hl in oh or oh in hl or any(w in oh for w in hl.split() if len(w) > 3))
            am = (al in oa or oa in al or any(w in oa for w in al.split() if len(w) > 3))
            if hm and am:
                return d
        return None

# ══════════════════════════════════════════════════════════════
# PREDICTION RESULT
# ══════════════════════════════════════════════════════════════

class Pred:
    def __init__(self):
        self.home = ""
        self.away = ""
        self.hid = 0
        self.aid = 0
        self.date = ""
        self.league = ""
        self.hp = 0.0
        self.dp = 0.0
        self.ap = 0.0
        self.raw_hp = 0.0
        self.raw_dp = 0.0
        self.raw_ap = 0.0
        self.hxg = 0.0
        self.axg = 0.0
        self.top_sc: List[Tuple] = []
        self.result = ""
        self.pred_sc = (0, 0)
        self.conf = 0.0
        self.btts = 0.0
        self.o15 = 0.0
        self.o25 = 0.0
        self.o35 = 0.0
        self.dc_1x = 0.0
        self.dc_x2 = 0.0
        self.dc_12 = 0.0
        self.dc_recommend = ""
        self.dc_value_bets: List[dict] = []
        self.value_bets: List[dict] = []
        self.h_form = ""
        self.a_form = ""
        self.h_pos = 0
        self.a_pos = 0
        self.h_elo = 0.0
        self.a_elo = 0.0
        self.h_fat = 0.0
        self.a_fat = 0.0
        self.h_rest = 0
        self.a_rest = 0
        self.h_momentum = 0
        self.a_momentum = 0
        self.models: Dict[str, Tuple] = {}
        self.odds = None
        self.ml_used = False
        self.ml_acc = 0.0
        self.calibrated = False
        self.is_derby = False
        self.derby_name = ""

# ══════════════════════════════════════════════════════════════
# ENGINE
# ══════════════════════════════════════════════════════════════

class Engine:
    def __init__(self, data: DataProc, resources: LeagueResources,
                 ml: MLPred = None, odds: OddsAPI = None, cal: Calibrator = None):
        self.data = data
        self.resources = resources
        self.ml = ml
        self.odds = odds
        self.cal = cal
        self.w = dict(WEIGHTS)
        if not ml or not ml.trained:
            mw = self.w.pop('ml', 0.15)
            rem = sum(self.w.values())
            if rem > 0:
                for k in self.w:
                    self.w[k] += mw * (self.w[k] / rem)

    def predict(self, hid: int, aid: int, date: str = "", md: datetime = None) -> Optional[Pred]:
        h = self.data.teams.get(hid)
        a = self.data.teams.get(aid)
        if not h or not a or h.played < 2 or a.played < 2:
            return None
        p = Pred()
        p.home = h.name
        p.away = a.name
        p.hid = hid
        p.aid = aid
        p.date = date
        p.league = self.resources.code if self.resources else ""
        p.h_form = h.form_string
        p.a_form = a.form_string
        p.h_pos = h.pos
        p.a_pos = a.pos
        p.h_elo = h.elo
        p.a_elo = a.elo
        if md is None and date:
            md = parse_date(date)
        md = md or datetime.now()
        derby = self.resources.is_derby(h.name, a.name) if self.resources else None
        p.is_derby = bool(derby)
        p.derby_name = derby or ""
        p.h_fat = Fatigue.score(h, md)
        p.a_fat = Fatigue.score(a, md)
        p.h_rest = h.days_rest(md)
        p.a_rest = a.days_rest(md)
        p.h_momentum = h.momentum
        p.a_momentum = a.momentum
        p.hxg = self._xg(h, a, True) * Fatigue.impact(h, md)
        p.axg = self._xg(a, h, False) * Fatigue.impact(a, md)
        if h.momentum > 40:
            p.hxg *= 1.05
        elif h.momentum < -40:
            p.hxg *= 0.95
        if a.momentum > 40:
            p.axg *= 1.05
        elif a.momentum < -40:
            p.axg *= 0.95
        models: Dict[str, Tuple[float, float, float]] = {}
        models['dixon_coles'] = DixonColes.predict(p.hxg, p.axg)
        models['elo'] = self.data.elo.predict(h, a)
        models['form'] = self._form(h, a)
        models['h2h'] = self._h2h(hid, aid)
        models['home_advantage'] = self._hadv(h, a)
        models['fatigue'] = Fatigue.predict(h, a, md)
        ha_val = self.resources.home_advantage if self.resources else 65
        ed = h.elo + ha_val - a.elo
        models['draw_model'] = DrawPredictor.predict(h, a, p.is_derby, ed)
        if self.ml and self.ml.trained:
            mp = self.ml.predict(h, a, self.data, md, p.is_derby)
            if mp:
                models['ml'] = mp
                p.ml_used = True
                p.ml_acc = self.ml.acc
        p.models = models
        hp = 0.0
        dp = 0.0
        ap = 0.0
        tw = 0.0
        for nm, probs in models.items():
            w = self.w.get(nm, 0)
            if w > 0:
                hp += probs[0] * w
                dp += probs[1] * w
                ap += probs[2] * w
                tw += w
        if tw > 0:
            hp /= tw
            dp /= tw
            ap /= tw
        t = hp + dp + ap
        if t > 0:
            hp /= t
            dp /= t
            ap /= t
        p.raw_hp = hp
        p.raw_dp = dp
        p.raw_ap = ap
        if self.cal and self.cal.ok:
            hp, dp, ap = self.cal.adjust((hp, dp, ap))
            p.calibrated = True
        p.hp = hp
        p.dp = dp
        p.ap = ap
        p.dc_1x = hp + dp
        p.dc_x2 = ap + dp
        p.dc_12 = hp + ap
        p.dc_recommend = self._dc_recommend(p)
        mx = DixonColes.matrix(p.hxg, p.axg)
        ss = sorted(mx.items(), key=lambda x: x[1], reverse=True)
        p.top_sc = [(s[0][0], s[0][1], s[1]) for s in ss[:6]]
        p.btts = sum(pr for (hh, aa), pr in mx.items() if hh > 0 and aa > 0)
        p.o15 = sum(pr for (hh, aa), pr in mx.items() if hh + aa > 1)
        p.o25 = sum(pr for (hh, aa), pr in mx.items() if hh + aa > 2)
        p.o35 = sum(pr for (hh, aa), pr in mx.items() if hh + aa > 3)
        pd_map = {'HOME': hp, 'DRAW': dp, 'AWAY': ap}
        p.result = max(pd_map, key=pd_map.get)
        p.conf = max(pd_map.values()) * 100
        if p.top_sc:
            p.pred_sc = (p.top_sc[0][0], p.top_sc[0][1])
        if self.odds and self.odds.ok():
            od = self.odds.find(h.name, a.name)
            if od:
                p.odds = od
                p.value_bets = self._value(p, od)
                p.dc_value_bets = self._dc_value(p, od)
        return p

    def _dc_recommend(self, p: Pred) -> str:
        recs = []
        if 0.40 <= p.hp <= 0.60 and p.dp > 0.20:
            recs.append(('1X', p.dc_1x, 'Home favored but draw possible'))
        if 0.30 <= p.ap <= 0.50 and p.dp > 0.20:
            recs.append(('X2', p.dc_x2, 'Away has real chance + draw likely'))
        if p.dp < 0.20:
            recs.append(('12', p.dc_12, 'Draw unlikely'))
        if p.is_derby and p.hp > p.ap:
            recs.append(('1X', p.dc_1x, f'{p.derby_name} - Home advantage'))
        if not recs:
            dc_vals = {'1X': p.dc_1x, 'X2': p.dc_x2, '12': p.dc_12}
            best = max(dc_vals, key=dc_vals.get)
            recs.append((best, dc_vals[best], 'Highest probability'))
        recs.sort(key=lambda x: -x[1])
        pct_str = f"{recs[0][1] * 100:.1f}%"
        return f"{recs[0][0]} ({pct_str}) - {recs[0][2]}"

    def _xg(self, t: Team, opp: Team, home: bool) -> float:
        ah = max(self.data.avg_h, 0.5)
        aa = max(self.data.avg_a, 0.5)

        if home:
            # هجوم الفريق المضيف في ملعبه
            att = safe_div(t.h_avg_gf, ah, 1.0)
            # دفاع الفريق الضيف خارج ملعبه - يُقسم على avg_a ✅
            df = safe_div(opp.a_avg_ga, ah, 1.0)
            base = ah
        else:
            # هجوم الفريق الضيف خارج ملعبه
            att = safe_div(t.a_avg_gf, aa, 1.0)
            # دفاع الفريق المضيف في ملعبه - يُقسم على avg_h ✅
            df = safe_div(opp.h_avg_ga, aa, 1.0)
            base = aa

        # وزن فورم التهديف
        fa = safe_div(t.goal_form, max(t.avg_gf, 0.5), 1.0)
        fa = 0.7 + 0.3 * min(fa, 2.0)

        raw_xg = att * df * base * fa
        return max(0.25, min(raw_xg, 3.5))  # الحد الأقصى 4.0 بدلاً من 4.5

    def _form(self, h: Team, a: Team) -> Tuple[float, float, float]:
        hf = h.form_score
        af = a.form_score
        t = hf + af
        if t == 0:
            return (0.4, 0.25, 0.35)
        hs = (hf / t) * 1.08
        a_s = af / t
        diff = abs(hs - a_s)
        d = max(0.15, 0.33 - diff * 0.4)
        rem = 1.0 - d
        sm = hs + a_s
        return (rem * hs / sm, d, rem * a_s / sm)

    def _h2h(self, hid: int, aid: int) -> Tuple[float, float, float]:
        default = (0.40, 0.25, 0.35)
        ms = self.data.get_h2h(hid, aid)
        if not ms:
            return default
        hw = 0
        dw = 0
        aw = 0
        for m in ms[-10:]:
            if m['home_goals'] > m['away_goals']:
                if m['home_id'] == hid:
                    hw += 1
                else:
                    aw += 1
            elif m['home_goals'] < m['away_goals']:
                if m['home_id'] == hid:
                    aw += 1
                else:
                    hw += 1
            else:
                dw += 1
        n = hw + dw + aw
        if n == 0:
            return default
        alpha = 1
        return (
            (hw + alpha) / (n + 3 * alpha),
            (dw + alpha) / (n + 3 * alpha),
            (aw + alpha) / (n + 3 * alpha)
        )

    def _hadv(self, h: Team, a: Team) -> Tuple[float, float, float]:
        hp = h.h_wr * 1.25
        ap = a.a_wr
        sm = hp + ap
        if sm > 0:
            hp /= sm
            ap /= sm
        d = 0.25
        hp *= 0.75
        ap *= 0.75
        t = hp + d + ap
        return (hp / t, d / t, ap / t)

    def _value(self, p: Pred, od: dict) -> List[dict]:
        vals = []
        markets = [
            ('Home', p.hp, od['implied_home'], od['odds_home']),
            ('Draw', p.dp, od['implied_draw'], od['odds_draw']),
            ('Away', p.ap, od['implied_away'], od['odds_away'])
        ]
        for nm, mp, ip, odd in markets:
            edge = (mp - ip) * 100
            kelly = (mp * odd - 1) / (odd - 1) if mp > 0 and odd > 1 else 0
            vals.append({
                'market': nm,
                'model': float(mp * 100),
                'implied': float(ip * 100),
                'odds': float(odd),
                'edge': float(edge),
                'kelly': float(max(0, kelly) * 100),
                'is_value': edge > 3
            })
        return vals

    def _dc_value(self, p: Pred, od: dict) -> List[dict]:
        vals = []
        markets = [
            ('1X', p.dc_1x, od.get('implied_1x'), od.get('odds_1x')),
            ('X2', p.dc_x2, od.get('implied_x2'), od.get('odds_x2')),
            ('12', p.dc_12, od.get('implied_12'), od.get('odds_12'))
        ]
        for nm, model_p, implied_p, odds_val in markets:
            if implied_p is None or odds_val is None:
                continue
            edge = (model_p - implied_p) * 100
            kelly = (model_p * odds_val - 1) / (odds_val - 1) if model_p > 0 and odds_val > 1 else 0
            vals.append({
                'market': f'DC {nm}',
                'model': float(model_p * 100),
                'implied': float(implied_p * 100),
                'odds': float(odds_val),
                'edge': float(edge),
                'kelly': float(max(0, kelly) * 100),
                'is_value': edge > 3
            })
        return vals

# ══════════════════════════════════════════════════════════════
# BACKTESTER - FIXED (no more 136% bug)
# ══════════════════════════════════════════════════════════════

class Backtester:
    def __init__(self):
        self.results: dict = {}
        self.cal = Calibrator()

    def run(self, matches: List[dict], resources: LeagueResources,
            split: float = BACKTEST_SPLIT) -> dict:

        fin = [m for m in matches if m.get('status') == 'FINISHED']
        fin.sort(key=lambda m: m.get('utcDate', ''))
        si = int(len(fin) * split)
        train = fin[:si]
        test = fin[si:]

        if len(train) < 30 or len(test) < 10:
            return {'error': 'Not enough data'}

        # ── تدريب النموذج على بيانات التدريب ──────────────────
        td = DataProc(resources)
        td.process(train)
        ml = None
        if ML_AVAILABLE:
            ml = MLPred(model_file=resources.model_file)
            ml.train(td, force_retrain=True)

        eng = Engine(td, resources, ml)

        # ── تقسيم مجموعة الاختبار إلى نصفين ──────────────────
        cs = len(test) // 2
        cal_set = test[:cs]
        eval_set = test[cs:]

        # ══════════════════════════════════════════════════════
        # المرحلة الأولى: Calibration
        # متغيرات خاصة بهذه المرحلة فقط
        # ══════════════════════════════════════════════════════
        cal_correct = 0
        cal_total = 0

        for m in cal_set:
            ht = m.get('homeTeam', {})
            at = m.get('awayTeam', {})
            ft = m.get('score', {}).get('fullTime', {})
            hid = ht.get('id')
            aid = at.get('id')
            ahg = ft.get('home')
            aag = ft.get('away')

            if not hid or not aid or ahg is None or aag is None:
                continue

            pr = eng.predict(hid, aid, m.get('utcDate', ''))
            if not pr:
                continue

            ahg = int(ahg)
            aag = int(aag)

            if ahg > aag:
                actual = 'HOME'
            elif ahg < aag:
                actual = 'AWAY'
            else:
                actual = 'DRAW'

            self.cal.add((pr.hp, pr.dp, pr.ap), actual)

            cal_total += 1
            if pr.result == actual:
                cal_correct += 1

            # تحديث النموذج بالمباراة الجديدة
            td.process([m])

        # بناء التحقق من صحة النموذج
        cal_ok = self.cal.calibrate()
        if cal_ok:
            eng2 = Engine(td, resources, ml, cal=self.cal)
        else:
            eng2 = eng

        # ══════════════════════════════════════════════════════
        # المرحلة الثانية: Evaluation
        # متغيرات منفصلة تماماً عن المرحلة الأولى
        # ══════════════════════════════════════════════════════
        eval_correct = 0
        eval_correct_sc = 0
        eval_total = 0
        preds = []

        # إحصائيات النتائج
        home_correct = 0
        home_total = 0
        draw_correct = 0
        draw_total = 0
        away_correct = 0
        away_total = 0

        # إحصائيات فئات الثقة
        hi_correct = 0
        hi_total = 0
        me_correct = 0
        me_total = 0
        lo_correct = 0
        lo_total = 0

        # إحصائيات Double Chance
        dc_1x_correct = 0
        dc_1x_total = 0
        dc_x2_correct = 0
        dc_x2_total = 0
        dc_12_correct = 0
        dc_12_total = 0

        for m in eval_set:
            ht = m.get('homeTeam', {})
            at = m.get('awayTeam', {})
            ft = m.get('score', {}).get('fullTime', {})
            hid = ht.get('id')
            aid = at.get('id')
            ahg = ft.get('home')
            aag = ft.get('away')

            if not hid or not aid or ahg is None or aag is None:
                continue

            hn = ht.get('shortName') or ht.get('name', '')
            an = at.get('shortName') or at.get('name', '')

            pr = eng2.predict(hid, aid, m.get('utcDate', ''))
            if not pr:
                continue

            ahg = int(ahg)
            aag = int(aag)

            if ahg > aag:
                actual = 'HOME'
            elif ahg < aag:
                actual = 'AWAY'
            else:
                actual = 'DRAW'

            # ── العدّادات الرئيسية ────────────────────────────
            eval_total += 1

            is_correct = (pr.result == actual)
            if is_correct:
                eval_correct += 1

            if pr.pred_sc[0] == ahg and pr.pred_sc[1] == aag:
                eval_correct_sc += 1

            # ── دقة كل نتيجة على حدة ─────────────────────────
            if actual == 'HOME':
                home_total += 1
                if is_correct:
                    home_correct += 1
            elif actual == 'DRAW':
                draw_total += 1
                if is_correct:
                    draw_correct += 1
            else:
                away_total += 1
                if is_correct:
                    away_correct += 1

            # ── دقة حسب مستوى الثقة ──────────────────────────
            conf = pr.conf
            if conf > 60:
                hi_total += 1
                if is_correct:
                    hi_correct += 1
            elif conf >= 45:
                me_total += 1
                if is_correct:
                    me_correct += 1
            else:
                lo_total += 1
                if is_correct:
                    lo_correct += 1

            # ── Double Chance ─────────────────────────────────
            dc_1x_total += 1
            if actual in ['HOME', 'DRAW']:
                dc_1x_correct += 1

            dc_x2_total += 1
            if actual in ['AWAY', 'DRAW']:
                dc_x2_correct += 1

            dc_12_total += 1
            if actual in ['HOME', 'AWAY']:
                dc_12_correct += 1

            # ── حفظ التنبؤ ───────────────────────────────────
            preds.append({
                'home': hn,
                'away': an,
                'predicted': pr.result,
                'actual': actual,
                'pred_score': pr.pred_sc,
                'actual_score': (ahg, aag),
                'confidence': float(pr.conf),
                'correct': is_correct,
                'probs': (float(pr.hp), float(pr.dp), float(pr.ap)),
                'dc_1x': float(pr.dc_1x),
                'dc_x2': float(pr.dc_x2),
                'dc_12': float(pr.dc_12),
                'calibrated': pr.calibrated
            })

            # تحديث النموذج بعد كل مباراة
            td.process([m])

        # ══════════════════════════════════════════════════════
        # حساب النتائج النهائية - كلها من eval فقط
        # ══════════════════════════════════════════════════════
        if eval_total == 0:
            return {'error': 'No evaluation matches found'}

        # ✅ الدقة الصحيحة - من eval_set فقط
        result_acc = (eval_correct / eval_total) * 100
        score_acc = (eval_correct_sc / eval_total) * 100

        # ── Brier Score ───────────────────────────────────────
        brier = 0.0
        for pred_item in preds:
            actual_vec = [0, 0, 0]
            outcome_index = ['HOME', 'DRAW', 'AWAY'].index(pred_item['actual'])
            actual_vec[outcome_index] = 1
            for i in range(3):
                brier += (pred_item['probs'][i] - actual_vec[i]) ** 2
        if eval_total > 0:
            brier /= (eval_total * 3)

        # ── دقة كل نتيجة ─────────────────────────────────────
        home_acc = (home_correct / home_total * 100) if home_total > 0 else 0.0
        draw_acc = (draw_correct / draw_total * 100) if draw_total > 0 else 0.0
        away_acc = (away_correct / away_total * 100) if away_total > 0 else 0.0

        # ── دقة فئات الثقة ───────────────────────────────────
        hi_acc = (hi_correct / hi_total * 100) if hi_total > 0 else 0.0
        me_acc = (me_correct / me_total * 100) if me_total > 0 else 0.0
        lo_acc = (lo_correct / lo_total * 100) if lo_total > 0 else 0.0

        # ── دقة Double Chance ─────────────────────────────────
        dc_1x_acc = (dc_1x_correct / dc_1x_total * 100) if dc_1x_total > 0 else 0.0
        dc_x2_acc = (dc_x2_correct / dc_x2_total * 100) if dc_x2_total > 0 else 0.0
        dc_12_acc = (dc_12_correct / dc_12_total * 100) if dc_12_total > 0 else 0.0

        # ── ML Accuracy ───────────────────────────────────────
        ml_acc_val = float(ml.acc * 100) if ml and ml.trained else 0.0

        self.results = {
            # ── أرقام أساسية ──
            'total': eval_total,
            'train': len(train),
            'test': len(test),
            'cal_size': cal_total,
            'eval_size': eval_total,

            # ── الدقة الرئيسية ──
            'result_acc': result_acc,
            'score_acc': score_acc,
            'brier': float(brier),
            'correct': eval_correct,
            'correct_sc': eval_correct_sc,

            # ── دقة كل نتيجة ──
            'home_acc': home_acc,
            'draw_acc': draw_acc,
            'away_acc': away_acc,
            'home_total': home_total,
            'draw_total': draw_total,
            'away_total': away_total,

            # ── دقة فئات الثقة ──
            'hi_acc': hi_acc,
            'me_acc': me_acc,
            'lo_acc': lo_acc,
            'hi_n': hi_total,
            'me_n': me_total,
            'lo_n': lo_total,

            # ── Double Chance ──
            'dc_1x_acc': dc_1x_acc,
            'dc_x2_acc': dc_x2_acc,
            'dc_12_acc': dc_12_acc,
            'dc_1x_n': dc_1x_total,
            'dc_x2_n': dc_x2_total,
            'dc_12_n': dc_12_total,

            # ── ML ──
            'ml_acc': ml_acc_val,
            'cal_used': cal_ok,

            'predictions': preds
        }
        return self.results

# ══════════════════════════════════════════════════════════════
# LEAGUE APP
# ══════════════════════════════════════════════════════════════

class LeagueApp:
    def __init__(self, league_code: str, api_token: str, odds_key: str = ""):
        self.code = league_code
        if league_code not in LEAGUES_CONFIG:
            raise ValueError(f"League '{league_code}' not found in {LEAGUES_CONFIG_FILE}")
        self.resources = LeagueResources(league_code, LEAGUES_CONFIG[league_code])
        self.api = FootballAPI(api_token, self.resources.api_url)
        self.data = DataProc(self.resources)
        self.ml = MLPred(model_file=self.resources.model_file)
        self.odds = OddsAPI(odds_key)
        self.cal = Calibrator()
        self.bt = Backtester()
        self.eng: Optional[Engine] = None
        self.raw: List[dict] = []
        self.last_preds: List[Pred] = []
        self._log: List[Tuple[str, str]] = []
        self.sy: Optional[int] = None

    def _log_msg(self, level: str, msg: str):
        full_msg = f"[{self.code}] {msg}"
        self._log.append((level, full_msg))
        print(f">>> {full_msg}", flush=True)

    def init(self) -> bool:
        self.raw = []

        # تحميل بيانات CSV
        self._log_msg('progress', f"Loading CSV data ({self.resources.name})...")
        csv_matches = self.resources.load_csv_data()
        if csv_matches:
            self.raw.extend(csv_matches)
            self._log_msg('success', f"CSV: {len(csv_matches)} matches with deep stats")
        else:
            self._log_msg('info', "No CSV data found, using API only")

        # تحميل بيانات API
        self.sy = self.api.season_year(self.resources.api_code)
        if self.sy:
            self._log_msg('progress', f"Loading API season {self.sy}...")
            api_matches = self.api.finished(self.resources.api_code, self.sy)
            if api_matches:
                self.raw.extend(api_matches)
                self._log_msg('success', f"API: {len(api_matches)} matches loaded")

        # دمج وإزالة المكرر - الأولوية لـ CSV
        unique_matches = {}
        for m in self.raw:
            date_key = m.get('utcDate', '')[:10]
            home_id = str(m['homeTeam'].get('id', ''))
            key = f"{date_key}_{home_id}"
            if key not in unique_matches or 'stats' in m:
                unique_matches[key] = m

        self.raw = sorted(unique_matches.values(), key=lambda x: x.get('utcDate', ''))

        if not self.raw:
            self._log_msg('error', "No data found!")
            return False

        self._log_msg('success', f"Total: {len(self.raw)} unique matches")

        # معالجة البيانات
        self._log_msg('progress', "Processing data + Elo + Deep Stats...")
        self.data.process(self.raw)
        self._log_msg('success', f"Teams processed: {len(self.data.teams)}")

        # تدريب النموذج
        if ML_AVAILABLE:
            model_type = "XGBoost+RF" if XGBOOST_AVAILABLE else "RF"
            self._log_msg('progress', f"Training {model_type} model ({N_FEATURES} features)...")
            if self.ml.train(self.data):
                src = "loaded" if self.ml._external else "trained"
                acc_val = self.ml.acc * 100
                acc_str = f"{acc_val:.1f}%"
                self._log_msg('success', f"ML model {src} | CV Acc: {acc_str}")
                if not self.ml._external:
                    self.ml.save_pipeline()
            else:
                self._log_msg('info', "ML training skipped (not enough data)")

        # تحميل التحقق
        if self.cal.load(self.resources.calibration_file):
            self._log_msg('success', "Calibration loaded")

        # تحميل الأوزان
        if self.odds.ok():
            self.odds.fetch()

        # تهيئة المحرك
        self.eng = Engine(self.data, self.resources, self.ml, self.odds, self.cal)
        self._log_msg('success', f"Engine Ready for {self.resources.name}!")
        return True

    def predict_upcoming(self, days: int = 14) -> List[Pred]:
        upcoming = self.api.upcoming(self.resources.api_code, days)
        if not upcoming:
            self._log_msg('info', "No upcoming matches from API")
            return []
        preds = []
        for m in upcoming:
            hid = m.get('homeTeam', {}).get('id')
            aid = m.get('awayTeam', {}).get('id')
            if hid and aid:
                pr = self.eng.predict(hid, aid, m.get('utcDate', ''))
                if pr:
                    preds.append(pr)
        self.last_preds = preds
        return preds

    def predict_custom(self, home_name: str, away_name: str) -> Optional[Pred]:
        ht = self.data.team_by_name(home_name)
        at = self.data.team_by_name(away_name)
        if not ht:
            self._log_msg('error', f"Team not found: {home_name}")
            return None
        if not at:
            self._log_msg('error', f"Team not found: {away_name}")
            return None
        pr = self.eng.predict(ht.id, at.id, "Custom")
        if pr:
            self.last_preds = [pr]
        return pr

    def run_backtest(self) -> dict:
        r = self.bt.run(self.raw, self.resources)
        if r.get('cal_used'):
            self.cal = self.bt.cal
            self.cal.save(self.resources.calibration_file)
            self.eng = Engine(self.data, self.resources, self.ml, self.odds, self.cal)
        return r

    def standings(self) -> List[Team]:
        return sorted(self.data.teams.values(), key=lambda t: t.pos)

    def export_predictions(self, preds: List[Pred] = None, filename: str = None) -> str:
        preds = preds or self.last_preds
        filename = filename or f"predictions_{self.code}_{datetime.now().strftime('%Y%m%d')}.json"
        out = []
        for p in preds:
            out.append({
                'league': p.league,
                'home': p.home,
                'away': p.away,
                'date': p.date,
                'prediction': p.result,
                'score': f"{p.pred_sc[0]}-{p.pred_sc[1]}",
                'confidence': round(float(p.conf), 1),
                'calibrated': p.calibrated,
                'derby': p.derby_name if p.is_derby else None,
                'probabilities': {
                    'home': round(float(p.hp * 100), 1),
                    'draw': round(float(p.dp * 100), 1),
                    'away': round(float(p.ap * 100), 1)
                },
                'double_chance': {
                    '1X': round(float(p.dc_1x * 100), 1),
                    'X2': round(float(p.dc_x2 * 100), 1),
                    '12': round(float(p.dc_12 * 100), 1),
                    'recommendation': p.dc_recommend
                },
                'xg': {
                    'home': round(p.hxg, 2),
                    'away': round(p.axg, 2)
                },
                'market': {
                    'btts': round(p.btts * 100, 1),
                    'over_1_5': round(p.o15 * 100, 1),
                    'over_2_5': round(p.o25 * 100, 1),
                    'over_3_5': round(p.o35 * 100, 1)
                }
            })
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        return filename

# ══════════════════════════════════════════════════════════════
# CLI DISPLAY
# ══════════════════════════════════════════════════════════════

class Disp:
    @staticmethod
    def header():
        print()
        print(C.cyan(" ╔══════════════════════════════════════════════════════════════╗"))
        print(C.cyan(" ║") + C.bold(" ⚽ FOOTBALL PREDICTOR PRO v5.0 (MULTI-LEAGUE FINAL) ⚽  ") + C.cyan("║"))
        print(C.cyan(" ║") + C.dim("  Multi-League • CSV Deep Stats • XGBoost • Backtest Fixed ") + C.cyan("║"))
        print(C.cyan(" ╚══════════════════════════════════════════════════════════════╝"))
        print()

    @staticmethod
    def section(t):
        print(f"\n {C.yellow(C.bold('══ ' + t + ' ══'))}\n")

    @staticmethod
    def leagues_menu(available: List[str]):
        Disp.section("Available Leagues")
        for i, code in enumerate(available, 1):
            cfg = LEAGUES_CONFIG.get(code, {})
            name = cfg.get('name', code)
            country = cfg.get('country', '')
            data_files = cfg.get('data_files', [])
            file_status = "✅" if any(Path(f).exists() for f in data_files) else "⚠️"
            print(f"  {C.cyan(str(i))}. {C.bold(code)} - {name} ({country}) {file_status}")
        print()

    @staticmethod
    def pred_card(p: Pred):
        w = 67
        print(f"\n {C.blue('┌' + '─' * w + '┐')}")
        if p.league:
            cfg = LEAGUES_CONFIG.get(p.league, {})
            league_name = cfg.get('name', p.league)
            print(box(f" 🏆 {C.magenta(league_name)}"))
        if p.is_derby:
            print(box(f" 🔥 {C.magenta(C.bold(p.derby_name))}"))
        print(box(f" {C.bold(C.green('🏠 ' + p.home))} {C.dim('vs')} {C.bold(C.red('✈️  ' + p.away))}"))
        if p.date and p.date != "Custom":
            dt = parse_date(p.date)
            ds = dt.strftime('%a %d %b %Y • %H:%M') if dt else p.date[:16]
            print(box(f" 📅 {ds}"))
        print(f" {C.blue('├' + '─' * w + '┤')}")
        print(box(f" {C.bold('📊 PROBABILITIES')}"))
        print(box(f" 🏠 Home: {C.green(f'{p.hp * 100:5.1f}%')} {C.pct_bar(p.hp, 25, C.G)}"))
        print(box(f" 🤝 Draw: {C.yellow(f'{p.dp * 100:5.1f}%')} {C.pct_bar(p.dp, 25, C.Y)}"))
        print(box(f" ✈️  Away: {C.red(f'{p.ap * 100:5.1f}%')} {C.pct_bar(p.ap, 25, C.R)}"))
        print(f" {C.blue('├' + '─' * w + '┤')}")
        hxg_str = f"{p.hxg:.2f}"
        axg_str = f"{p.axg:.2f}"
        total_xg_str = f"{p.hxg + p.axg:.2f}"
        print(box(f" ⚡ xG: {p.home}: {C.bold(hxg_str)} | {p.away}: {C.bold(axg_str)} | Total: {C.bold(total_xg_str)}"))
        print(box(f" 🎯 Predicted: {C.bold(p.result)} | Score: {p.pred_sc[0]}-{p.pred_sc[1]} | Conf: {p.conf:.1f}%"))
        print(f" {C.blue('├' + '─' * w + '┤')}")
        dc_1x_str = f"{p.dc_1x * 100:.1f}%"
        dc_12_str = f"{p.dc_12 * 100:.1f}%"
        dc_x2_str = f"{p.dc_x2 * 100:.1f}%"
        print(box(f" 🛡️  DC: 1X={dc_1x_str} | 12={dc_12_str} | X2={dc_x2_str}"))
        print(box(f" 💡 {C.bold(p.dc_recommend)}"))
        print(f" {C.blue('├' + '─' * w + '┤')}")
        print(box(f" {C.bold('🎯 TOP SCORES')}"))
        for i, (hg, ag, pr2) in enumerate(p.top_sc[:5]):
            mk = "👉" if i == 0 else "  "
            pr2_str = f"{pr2 * 100:.1f}%"
            print(box(f" {mk} {hg}-{ag} ({pr2_str})"))
        print(f" {C.blue('└' + '─' * w + '┘')}")

    @staticmethod
    def backtest_summary(r: dict, league_code: str = ""):
        Disp.section(f"Backtest Results [{league_code}]")
        if 'error' in r:
            print(f" {C.red('✖')} {r['error']}")
            return

        ra = r['result_acc']

        # تحقق من صحة النتيجة
        if ra > 100:
            print(f" {C.red('✖')} خطأ في الحساب: دقة تتجاوز 100% ({ra:.1f}%)")
            return

        rac = C.green if ra > 50 else (C.yellow if ra > 40 else C.red)

        ra_str = f"{ra:.1f}%"
        correct_str = f"{r['correct']}/{r['total']}"
        print(f"  📊 1X2 Accuracy   : {rac(ra_str)} ({correct_str})")

        sa_str = f"{r['score_acc']:.1f}%"
        print(f"  ⚽ Exact Score    : {sa_str}")

        bs = r['brier']
        bsc = C.green if bs < 0.15 else (C.yellow if bs < 0.22 else C.red)
        bs_str = f"{bs:.4f}"
        print(f"  📐 Brier Score    : {bsc(bs_str)}")

        ml_val = r.get('ml_acc', 0)
        if ml_val > 0:
            ml_str = f"{ml_val:.1f}%"
            print(f"  🤖 ML Bal. Acc    : {C.green(ml_str)}")

        print()
        ha_str = f"{r.get('home_acc', 0):.1f}%"
        da_str = f"{r.get('draw_acc', 0):.1f}%"
        aa_str = f"{r.get('away_acc', 0):.1f}%"
        print(f"  🏠 Home Win Acc   : {ha_str} ({r.get('home_total', 0)} matches)")
        print(f"  🤝 Draw Acc       : {da_str} ({r.get('draw_total', 0)} matches)")
        print(f"  ✈️  Away Win Acc   : {aa_str} ({r.get('away_total', 0)} matches)")

        print()
        hi_str = f"{r.get('hi_acc', 0):.1f}%"
        me_str = f"{r.get('me_acc', 0):.1f}%"
        lo_str = f"{r.get('lo_acc', 0):.1f}%"
        print(f"  🔥 High Conf >60% : {C.green(hi_str)} ({r.get('hi_n', 0)} matches)")
        print(f"  ⚡ Med Conf 45-60%: {C.yellow(me_str)} ({r.get('me_n', 0)} matches)")
        print(f"  ⚠️  Low Conf <45%  : {C.red(lo_str)} ({r.get('lo_n', 0)} matches)")

        print()
        dc1x_str = f"{r.get('dc_1x_acc', 0):.1f}%"
        dcx2_str = f"{r.get('dc_x2_acc', 0):.1f}%"
        dc12_str = f"{r.get('dc_12_acc', 0):.1f}%"
        print(f"  🛡️  DC 1X Acc      : {dc1x_str} ({r.get('dc_1x_n', 0)} matches)")
        print(f"  🛡️  DC X2 Acc      : {dcx2_str} ({r.get('dc_x2_n', 0)} matches)")
        print(f"  🛡️  DC 12 Acc      : {dc12_str} ({r.get('dc_12_n', 0)} matches)")

        print()
        cal_str = "Yes" if r.get('cal_used') else "No"
        print(f"  ✅ Calibrated     : {cal_str}")
        print(f"  📋 Train Matches  : {r.get('train', 0)}")
        print(f"  🧪 Cal Set        : {r.get('cal_size', 0)}")
        print(f"  🔬 Eval Set       : {r.get('eval_size', 0)}")

# ══════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════

def run_streamlit():
    st.set_page_config(
        page_title="Football Predictor Pro v5.0",
        page_icon="⚽",
        layout="wide"
    )

    st.markdown(
        "<h1 style='text-align:center;color:#00ff9d;'>⚽ Football Predictor Pro v5.0</h1>",
        unsafe_allow_html=True
    )
    st.caption("Multi-League • Deep Stats CSV • Per-League ML Models • 68 Features • Backtest Fixed")

    st.sidebar.title("⚙️ Settings")
    fb_key = st.sidebar.text_input(
        "🔑 Football-Data API Key",
        type="password",
        value=os.environ.get("FOOTBALL_DATA_KEY", "")
    )
    odds_key = st.sidebar.text_input(
        "🎰 Odds API Key (optional)",
        type="password",
        value=os.environ.get("ODDS_API_KEY", "")
    )

    available = list(LEAGUES_CONFIG.keys())
    if not available:
        st.warning("⚠️ No leagues configured! Check leagues_config.json")
        st.stop()

    selected_league = st.sidebar.selectbox(
        "🏆 Select League",
        available,
        format_func=lambda c: f"{c} - {LEAGUES_CONFIG.get(c, {}).get('name', c)}"
    )

    league_cfg = LEAGUES_CONFIG.get(selected_league, {})
    data_files = league_cfg.get('data_files', [])
    st.sidebar.markdown("**📁 Data Files:**")
    for f in data_files:
        exists = Path(f).exists()
        icon = "✅" if exists else "❌"
        st.sidebar.caption(f"{icon} {f}")

    init_key = f"app_{selected_league}_{fb_key[:8] if fb_key else 'nokey'}"

    if st.sidebar.button(f"🚀 Load {selected_league}") and fb_key:
        with st.spinner(f"Loading {selected_league}..."):
            try:
                app = LeagueApp(selected_league, fb_key, odds_key)
                if app.init():
                    st.session_state[init_key] = app
                    st.session_state['active_league'] = selected_league
                    st.sidebar.success(f"✅ {selected_league} Ready!")
                else:
                    st.sidebar.error("❌ Init failed")
            except Exception as e:
                st.sidebar.error(f"❌ {e}")

    if init_key not in st.session_state:
        st.info(f"👈 Enter API key and load {selected_league}")
        with st.expander("📖 Setup Guide"):
            st.markdown(f"""
### File Structure Required
leagues_config.json
data/{selected_league}_Master.csv
config/{selected_league}_aliases.json
config/{selected_league}_rivalries.json
config/{selected_league}_teams_map.json
models/ ← auto-generated
### CSV Format (football-data.co.uk)
Required: `Date, HomeTeam, AwayTeam, FTHG, FTAG`

Optional deep stats: `HST, AST, HC, AC, HF, AF, HY, AY, HR, AR`
            """)
        st.stop()

    app: LeagueApp = st.session_state[init_key]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🏆 League", app.resources.name)
    col2.metric("📊 Matches", app.data.total)
    col3.metric("👥 Teams", len(app.data.teams))
    col4.metric("🤖 ML Ready", "✅" if app.ml.trained else "❌")

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Predictions",
        "⚽ Custom Match",
        "📊 Standings",
        "🔬 Backtest"
    ])

    def render_pred_card(pr: Pred):
        title = f"{'🔥' if pr.is_derby else '⚽'} {pr.home} vs {pr.away}"
        hp_str = f"{pr.hp * 100:.1f}%"
        dp_str = f"{pr.dp * 100:.1f}%"
        ap_str = f"{pr.ap * 100:.1f}%"
        subtitle = f"{hp_str} / {dp_str} / {ap_str}"

        with st.expander(f"{title}  |  {subtitle}", expanded=True):
            if pr.is_derby:
                st.markdown(f"🔥 **{pr.derby_name}**")
            if pr.calibrated:
                st.caption("✅ Calibrated probabilities")

            c1, c2, c3 = st.columns(3)
            c1.metric("🏠 Home Win", hp_str)
            c2.metric("🤝 Draw", dp_str)
            c3.metric("✈️ Away Win", ap_str)

            st.progress(min(1.0, float(pr.hp)), text=f"Home {pr.home}")
            st.progress(min(1.0, float(pr.dp)), text="Draw")
            st.progress(min(1.0, float(pr.ap)), text=f"Away {pr.away}")

            st.divider()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("⚡ xG Home", f"{pr.hxg:.2f}")
            c2.metric("⚡ xG Away", f"{pr.axg:.2f}")
            c3.metric("🏆 Elo Home", f"{pr.h_elo:.0f}")
            c4.metric("🏆 Elo Away", f"{pr.a_elo:.0f}")

            st.markdown("#### 🛡️ Double Chance")
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("1X (Home/Draw)", f"{pr.dc_1x * 100:.1f}%")
            cc2.metric("12 (No Draw)", f"{pr.dc_12 * 100:.1f}%")
            cc3.metric("X2 (Away/Draw)", f"{pr.dc_x2 * 100:.1f}%")
            st.success(f"💡 **Recommendation:** {pr.dc_recommend}")

            st.markdown("#### 📈 Markets")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("BTTS", f"{pr.btts * 100:.1f}%")
            m2.metric("Over 1.5", f"{pr.o15 * 100:.1f}%")
            m3.metric("Over 2.5", f"{pr.o25 * 100:.1f}%")
            m4.metric("Over 3.5", f"{pr.o35 * 100:.1f}%")

            st.markdown("#### 🎯 Likely Scores")
            scores_html = " &nbsp; ".join([
                f"<span style='padding:4px 12px;background:#1a1a2e;color:#00ff9d;"
                f"border-radius:5px;font-weight:bold;'>"
                f"{hg}-{ag} ({pr2 * 100:.1f}%)</span>"
                for hg, ag, pr2 in pr.top_sc[:5]
            ])
            st.markdown(scores_html, unsafe_allow_html=True)

            if pr.value_bets:
                vb = [v for v in pr.value_bets if v['is_value']]
                if vb:
                    st.markdown("#### 💰 Value Bets")
                    for v in vb:
                        edge_str = f"+{v['edge']:.1f}%"
                        st.success(
                            f"**{v['market']}** | Odds: {v['odds']} | "
                            f"Model: {v['model']:.1f}% vs Implied: {v['implied']:.1f}% | "
                            f"Edge: {edge_str}"
                        )

    # ── Tab 1: Predictions ────────────────────────────────────
    with tab1:
        days = st.slider("Days ahead", 1, 30, 14)
        if st.button("🔍 Get Upcoming Predictions"):
            with st.spinner("Predicting..."):
                preds = app.predict_upcoming(days)
                st.session_state[f'preds_{selected_league}'] = preds

        preds_key = f'preds_{selected_league}'
        if preds_key in st.session_state:
            preds = st.session_state[preds_key]
            if preds:
                st.success(f"✅ {len(preds)} predictions ready")
                if st.button("📥 Export JSON"):
                    fn = app.export_predictions(preds)
                    st.success(f"Exported: {fn}")
                for pr in preds:
                    render_pred_card(pr)
            else:
                st.info("No upcoming matches found")

    # ── Tab 2: Custom Match ───────────────────────────────────
    with tab2:
        teams_list = sorted(t.name for t in app.data.teams.values())
        if not teams_list:
            st.warning("No teams data available")
        else:
            c1, c2 = st.columns(2)
            home = c1.selectbox("🏠 Home Team", teams_list, key=f"home_{selected_league}")
            away_default_idx = min(1, len(teams_list) - 1)
            away = c2.selectbox(
                "✈️ Away Team", teams_list,
                index=away_default_idx,
                key=f"away_{selected_league}"
            )
            if st.button("🔮 Predict"):
                if home == away:
                    st.error("Please select different teams!")
                else:
                    with st.spinner("Predicting..."):
                        pr = app.predict_custom(home, away)
                        if pr:
                            st.session_state[f'custom_{selected_league}'] = pr
                        else:
                            st.error("Prediction failed - not enough data for these teams")

            custom_key = f'custom_{selected_league}'
            if custom_key in st.session_state:
                render_pred_card(st.session_state[custom_key])

    # ── Tab 3: Standings ──────────────────────────────────────
    with tab3:
        teams = app.standings()
        if teams:
            rows = []
            for t in teams:
                rows.append({
                    'Pos': t.pos,
                    'Team': t.name,
                    'P': t.played,
                    'W': t.wins,
                    'D': t.draws,
                    'L': t.losses,
                    'GF': t.gf,
                    'GA': t.ga,
                    'GD': t.gd,
                    'Pts': t.pts,
                    'Elo': round(t.elo),
                    'PPG': round(t.ppg, 2),
                    'Form': t.form_string[-5:],
                    'xG For': round(t.avg_gf, 2),
                    'xG Ag': round(t.avg_ga, 2),
                    'SoT/g': round(t.avg_sot, 1),
                    'Corners/g': round(t.avg_corners, 1)
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Tab 4: Backtest ───────────────────────────────────────
    with tab4:
        split_pct = int(BACKTEST_SPLIT * 100)
        test_pct = 100 - split_pct
        st.info(f"Backtest uses {split_pct}% train / {test_pct}% test split")

        if st.button("▶️ Run Backtest"):
            with st.spinner("Running backtest..."):
                r = app.run_backtest()
                st.session_state[f'bt_{selected_league}'] = r

        bt_key = f'bt_{selected_league}'
        if bt_key in st.session_state:
            r = st.session_state[bt_key]
            if 'error' in r:
                st.error(r['error'])
            else:
                acc = r['result_acc']
                if acc > 100:
                    st.error(f"⚠️ Calculation error: {acc:.1f}% exceeds 100%")
                else:
                    # الصف الأول
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("🎯 1X2 Accuracy", f"{r['result_acc']:.1f}%")
                    c2.metric("⚽ Exact Score", f"{r['score_acc']:.1f}%")
                    c3.metric("📐 Brier Score", f"{r['brier']:.4f}")
                    ml_acc_display = f"{r.get('ml_acc', 0):.1f}%"
                    c4.metric("🤖 ML Acc", ml_acc_display)

                    # الصف الثاني
                    c5, c6, c7, c8 = st.columns(4)
                    c5.metric("📋 Train", r['train'])
                    c6.metric("🎚️ Cal Set", r.get('cal_size', 0))
                    c7.metric("🔬 Eval Set", r.get('eval_size', 0))
                    cal_display = "Yes" if r.get('cal_used') else "No"
                    c8.metric("✅ Calibrated", cal_display)

                    st.divider()

                    # دقة كل نتيجة
                    st.markdown("#### 📊 Accuracy by Outcome")
                    ca, cb, cc = st.columns(3)
                    ca.metric(
                        "🏠 Home Win",
                        f"{r.get('home_acc', 0):.1f}%",
                        f"{r.get('home_total', 0)} matches"
                    )
                    cb.metric(
                        "🤝 Draw",
                        f"{r.get('draw_acc', 0):.1f}%",
                        f"{r.get('draw_total', 0)} matches"
                    )
                    cc.metric(
                        "✈️ Away Win",
                        f"{r.get('away_acc', 0):.1f}%",
                        f"{r.get('away_total', 0)} matches"
                    )

                    # دقة فئات الثقة
                    st.markdown("#### 🎯 Accuracy by Confidence Level")
                    d1, d2, d3 = st.columns(3)
                    d1.metric(
                        "🔥 High >60%",
                        f"{r.get('hi_acc', 0):.1f}%",
                        f"{r.get('hi_n', 0)} matches"
                    )
                    d2.metric(
                        "⚡ Medium 45-60%",
                        f"{r.get('me_acc', 0):.1f}%",
                        f"{r.get('me_n', 0)} matches"
                    )
                    d3.metric(
                        "⚠️ Low <45%",
                        f"{r.get('lo_acc', 0):.1f}%",
                        f"{r.get('lo_n', 0)} matches"
                    )

                    # Double Chance
                    st.markdown("#### 🛡️ Double Chance Coverage")
                    e1, e2, e3 = st.columns(3)
                    e1.metric(
                        "1X Coverage",
                        f"{r.get('dc_1x_acc', 0):.1f}%",
                        f"{r.get('dc_1x_n', 0)} matches"
                    )
                    e2.metric(
                        "12 Coverage",
                        f"{r.get('dc_12_acc', 0):.1f}%",
                        f"{r.get('dc_12_n', 0)} matches"
                    )
                    e3.metric(
                        "X2 Coverage",
                        f"{r.get('dc_x2_acc', 0):.1f}%",
                        f"{r.get('dc_x2_n', 0)} matches"
                    )

                    # جدول التنبؤات
                    if r.get('predictions'):
                        st.markdown("#### 📋 Last 50 Test Predictions")
                        preds_rows = []
                        for pred_item in r['predictions'][-50:]:
                            preds_rows.append({
                                'Home': pred_item['home'],
                                'Away': pred_item['away'],
                                'Predicted': pred_item['predicted'],
                                'Actual': pred_item['actual'],
                                'Result': '✅' if pred_item['correct'] else '❌',
                                'Confidence': f"{pred_item['confidence']:.1f}%",
                                'P(Home)': f"{pred_item['probs'][0] * 100:.1f}%",
                                'P(Draw)': f"{pred_item['probs'][1] * 100:.1f}%",
                                'P(Away)': f"{pred_item['probs'][2] * 100:.1f}%"
                            })
                        preds_df = pd.DataFrame(preds_rows)
                        st.dataframe(preds_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════
# CLI MAIN
# ══════════════════════════════════════════════════════════════

def cli_main():
    Disp.header()

    available = list(LEAGUES_CONFIG.keys())
    if not available:
        print(f"{C.red('❌')} No leagues in {LEAGUES_CONFIG_FILE}")
        print(f"{C.yellow('💡')} A default config has been created. Please edit it.")
        return

    tok = os.environ.get("FOOTBALL_DATA_KEY", "")
    if not tok:
        tok = input(C.cyan(" 🔑 Football-Data API key: ")).strip()
    if not tok:
        print(C.red("❌ No API key provided"))
        return

    odds_key = os.environ.get("ODDS_API_KEY", "")

    Disp.leagues_menu(available)
    if len(available) == 1:
        league_code = available[0]
        print(f" Auto-selected: {C.bold(league_code)}")
    else:
        choice = input(C.cyan(f" Select league (1-{len(available)}): ")).strip()
        try:
            league_code = available[int(choice) - 1]
        except (ValueError, IndexError):
            league_code = available[0]

    print(f"\n {C.green('►')} Loading {C.bold(league_code)}...")

    try:
        app = LeagueApp(league_code, tok, odds_key)
        if not app.init():
            print(C.red("❌ Initialization failed"))
            return
    except Exception as e:
        print(C.red(f"❌ Error: {e}"))
        return

    while True:
        try:
            print(f"\n {C.cyan('─' * 50)}")
            print(f" {C.bold('Options:')}")
            print(f"  {C.cyan('1')} - Predict upcoming matches")
            print(f"  {C.cyan('2')} - Custom match prediction")
            print(f"  {C.cyan('3')} - League standings")
            print(f"  {C.cyan('4')} - Run backtest")
            print(f"  {C.cyan('5')} - Export last predictions")
            print(f"  {C.cyan('6')} - Switch league")
            print(f"  {C.cyan('0')} - Exit")

            ch = input(C.cyan("\n Choice: ")).strip()

            if ch == '1':
                Disp.section("Upcoming Predictions")
                preds = app.predict_upcoming(14)
                if preds:
                    for pr in preds:
                        Disp.pred_card(pr)
                else:
                    print(C.yellow(" No upcoming matches found"))

            elif ch == '2':
                Disp.section("Custom Match")
                teams = sorted(t.name for t in app.data.teams.values())
                print(f" Teams: {', '.join(teams)}")
                home = input(C.cyan(" Home team: ")).strip()
                away = input(C.cyan(" Away team: ")).strip()
                pr = app.predict_custom(home, away)
                if pr:
                    Disp.pred_card(pr)

            elif ch == '3':
                Disp.section(f"Standings - {app.resources.name}")
                standings = app.standings()
                print(f" {'#':>3} {'Team':<22} {'P':>3} {'W':>2} {'D':>2} {'L':>2} {'GD':>4} {'Pts':>4} {'Elo':>6}")
                print(f" {'─' * 60}")
                for t in standings:
                    pc = C.G if t.pos <= 4 else (C.R if t.pos >= len(standings) - 2 else C.W)
                    gd_str = f"{t.gd:+d}"
                    elo_str = f"{t.elo:.0f}"
                    print(f" {pc}{t.pos:>3}{C.E} {t.name:<22} {t.played:>3} {t.wins:>2} {t.draws:>2} {t.losses:>2} {gd_str:>4} {t.pts:>4} {elo_str:>6}")

            elif ch == '4':
                Disp.section("Backtest")
                r = app.run_backtest()
                Disp.backtest_summary(r, league_code)

            elif ch == '5':
                if app.last_preds:
                    fn = app.export_predictions()
                    print(C.green(f" ✅ Exported: {fn}"))
                else:
                    print(C.yellow(" No predictions to export"))

            elif ch == '6':
                Disp.leagues_menu(available)
                choice2 = input(C.cyan(f" Select (1-{len(available)}): ")).strip()
                try:
                    new_code = available[int(choice2) - 1]
                    new_app = LeagueApp(new_code, tok, odds_key)
                    if new_app.init():
                        app = new_app
                        league_code = new_code
                    else:
                        print(C.red("❌ Failed to load league"))
                except (ValueError, IndexError):
                    print(C.red("❌ Invalid choice"))

            elif ch == '0':
                print(C.green(" Goodbye! ⚽"))
                break

        except KeyboardInterrupt:
            print(C.green("\n Goodbye! ⚽"))
            break
        except Exception as e:
            print(C.red(f" Error: {e}"))

# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

if STREAMLIT_AVAILABLE:
    run_streamlit()
elif __name__ == "__main__":
    cli_main()
