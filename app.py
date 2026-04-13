#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║          ⚽ FOOTBALL PREDICTOR PRO v6.1 (ULTIMATE EDITION) ⚽        ║
║                                                                      ║
║  ✅ V6.1: Fixed Entry Point (Streamlit/CLI separation)               ║
║  ✅ V6.1: Fixed _advanced_xg_features (save rate calculation)        ║
║  ✅ V6.1: Fixed Calibrator.adjust() (negative values protection)     ║
║  ✅ V6.1: Fixed EloSystem.predict() (realistic draw probability)     ║
║  ✅ V6.1: Fixed MLPred.feats() (safe feature count)                  ║
║  ✅ V6.1: Fixed xG bounds (min 0.10 instead of 0.25)                 ║
║  ✅ V6.1: Improved _form() (draw-aware form scoring)                 ║
║  ✅ V6.1: Improved draw_model weight (0.13 → 0.20)                   ║
║  ✅ V6.1: Draw Correction in Engine.predict()                        ║
║  ✅ V6.1: Fixed unique_matches key (date+home+away)                  ║
║  ✅ V6.0: 126 Features (68 Original + 58 New)                        ║
║  ✅ V6.0: Stacking Classifier (RF + XGB + LR → Meta LR)              ║
║  ✅ V6.0: ImprovedDrawPredictor (Statistical Draw Model)             ║
║  ✅ V6.0: Multi-League Support + Separate Files per League           ║
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

# ── تحميل .env تلقائياً (يعمل مع Streamlit وCLI) ────────────
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)
except ImportError:
    # dotenv غير مثبت — نُحمّل .env يدوياً
    _env_file = Path(__file__).parent / ".env"
    if _env_file.exists():
        with open(_env_file, encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _, _v = _line.partition("=")
                    os.environ.setdefault(_k.strip(), _v.strip())

try:
    import pandas as pd
except ImportError:
    print("❌ Pandas missing! pip install pandas")
    sys.exit(1)

# ── Streamlit: اختياري فقط ────────────────────────────────────
STREAMLIT_AVAILABLE = False
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    pass

# ── ML: اختياري ───────────────────────────────────────────────
ML_AVAILABLE = False
XGBOOST_AVAILABLE = False
try:
    import numpy as np
    from sklearn.ensemble import (
        RandomForestClassifier,
        VotingClassifier,
        StackingClassifier,
    )
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
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

DEFAULT_GLOBAL_SETTINGS: Dict = {
    "elo_init": 1500,
    "elo_k": 32,
    "form_n": 8,
    "backtest_split": 0.70,
    "min_samples_per_class": 10,
    "n_features": 126,
}


def load_leagues_config() -> dict:
    if not Path(LEAGUES_CONFIG_FILE).exists():
        print(f"⚠️  {LEAGUES_CONFIG_FILE} not found! Creating default…")
        create_default_config()
    try:
        with open(LEAGUES_CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
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
                "model_file": "models/PL_model_v6.pkl",
                "calibration_file": "models/PL_calibration_v6.pkl",
                "elo_file": "models/PL_elo_v6.pkl",
                "teams_map_file": "config/PL_teams_map.json",
                "aliases_file": "config/PL_aliases.json",
                "rivalries_file": "config/PL_rivalries.json",
                "home_advantage": 65,
                "avg_home_goals": 1.53,
                "avg_away_goals": 1.16,
                "total_teams": 20,
                "total_rounds": 38,
            }
        },
        "global_settings": DEFAULT_GLOBAL_SETTINGS,
    }
    for folder in ["data", "models", "config"]:
        Path(folder).mkdir(exist_ok=True)
    with open(LEAGUES_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(default, f, indent=2, ensure_ascii=False)
    print(f"✅ Created {LEAGUES_CONFIG_FILE}")


_GLOBAL_CONFIG = load_leagues_config()
GLOBAL_SETTINGS: Dict = _GLOBAL_CONFIG.get("global_settings", DEFAULT_GLOBAL_SETTINGS)
LEAGUES_CONFIG: Dict = _GLOBAL_CONFIG.get("leagues", {})

ELO_INIT: float = float(GLOBAL_SETTINGS.get("elo_init", 1500))
ELO_K: float = float(GLOBAL_SETTINGS.get("elo_k", 32))
FORM_N: int = int(GLOBAL_SETTINGS.get("form_n", 8))
BACKTEST_SPLIT: float = float(GLOBAL_SETTINGS.get("backtest_split", 0.70))
MIN_SAMPLES_PER_CLASS: int = int(GLOBAL_SETTINGS.get("min_samples_per_class", 10))
N_FEATURES: int = int(GLOBAL_SETTINGS.get("n_features", 126))

# ── V6.1: رُفع وزن draw_model من 0.13 → 0.20 ─────────────────
# ── V6.1: خُفِّضت أوزان النماذج الأخرى توازناً ─────────────────
WEIGHTS: Dict[str, float] = {
    "dixon_coles":    0.18,   # ↓ (كان 0.22) يُقلّل Draw بطبيعته
    "elo":            0.13,   # ↓ (كان 0.16)
    "form":           0.08,   # ↓ (كان 0.10)
    "h2h":            0.07,   # ↓ (كان 0.08)
    "home_advantage": 0.06,   # ↓ (كان 0.07)
    "fatigue":        0.03,   # ↓ (كان 0.04)
    "draw_model":     0.20,   # ↑ (كان 0.13) أهم سلاح للتعادل
    "ml":             0.25,   # ↑ (كان 0.20)
}
# المجموع = 1.00 ✅

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
        c = s.replace("Z", "")
        fmt = "%Y-%m-%dT%H:%M:%S" if "T" in c else "%Y-%m-%d %H:%M:%S"
        return datetime.strptime(c[:19], fmt)
    except Exception:
        return None


def normalize_probs(hp: float, dp: float, ap: float) -> Tuple[float, float, float]:
    """تطبيع احتمالات ثلاثية مع حماية من القيم الصفرية"""
    hp = max(0.0, hp)
    dp = max(0.0, dp)
    ap = max(0.0, ap)
    t = hp + dp + ap
    if t <= 0:
        return (0.40, 0.25, 0.35)
    return (hp / t, dp / t, ap / t)


# ══════════════════════════════════════════════════════════════
# COLOUR HELPERS
# ══════════════════════════════════════════════════════════════
class C:
    H  = "\033[95m"
    B  = "\033[94m"
    CN = "\033[96m"
    G  = "\033[92m"
    Y  = "\033[93m"
    R  = "\033[91m"
    BD = "\033[1m"
    DM = "\033[2m"
    E  = "\033[0m"
    W  = "\033[97m"

    @staticmethod
    def bold(t):    return f"{C.BD}{t}{C.E}"
    @staticmethod
    def green(t):   return f"{C.G}{t}{C.E}"
    @staticmethod
    def red(t):     return f"{C.R}{t}{C.E}"
    @staticmethod
    def yellow(t):  return f"{C.Y}{t}{C.E}"
    @staticmethod
    def cyan(t):    return f"{C.CN}{t}{C.E}"
    @staticmethod
    def blue(t):    return f"{C.B}{t}{C.E}"
    @staticmethod
    def dim(t):     return f"{C.DM}{t}{C.E}"
    @staticmethod
    def magenta(t): return f"{C.H}{t}{C.E}"

    @staticmethod
    def form_char(ch: str) -> str:
        if ch == "W": return f"{C.G}{C.BD}W{C.E}"
        if ch == "D": return f"{C.Y}{C.BD}D{C.E}"
        if ch == "L": return f"{C.R}{C.BD}L{C.E}"
        return ch

    @staticmethod
    def form_str(s: str) -> str:
        return " ".join(C.form_char(c) for c in s)

    @staticmethod
    def pct_bar(v: float, w: int = 20, color: str = None) -> str:
        color = color or C.G
        f = int(max(0.0, min(1.0, v)) * w)
        e = w - f
        return f"{color}{'█' * f}{C.E}{C.DM}{'░' * e}{C.E}"


def box(t: str) -> str:
    return f" {C.blue('│')} {t}"


# ══════════════════════════════════════════════════════════════
# LEAGUE RESOURCES
# ══════════════════════════════════════════════════════════════
class LeagueResources:
    def __init__(self, league_code: str, config: dict):
        self.code            = league_code
        self.config          = config
        self.name            = config.get("name", league_code)
        self.country         = config.get("country", "")
        self.api_code        = config.get("api_code", league_code)
        self.api_url         = config.get("api_url", "https://api.football-data.org/v4")
        self.home_advantage  = config.get("home_advantage", 65)
        self.avg_home_goals  = config.get("avg_home_goals", 1.53)
        self.avg_away_goals  = config.get("avg_away_goals", 1.16)
        self.total_teams     = config.get("total_teams", 20)
        self.total_rounds    = config.get("total_rounds", 38)

        self.data_files:        List[str]              = config.get("data_files", [])
        self.model_file:        str = config.get("model_file",       f"models/{league_code}_model_v6.pkl")
        self.calibration_file:  str = config.get("calibration_file", f"models/{league_code}_calibration_v6.pkl")
        self.elo_file:          str = config.get("elo_file",         f"models/{league_code}_elo_v6.pkl")
        self.teams_map_file:    str = config.get("teams_map_file",   f"config/{league_code}_teams_map.json")
        self.aliases_file:      str = config.get("aliases_file",     f"config/{league_code}_aliases.json")
        self.rivalries_file:    str = config.get("rivalries_file",   f"config/{league_code}_rivalries.json")

        self.teams_map:   Dict[str, str]         = {}
        self.aliases:     Dict[str, str]         = {}
        self.rivalries:   Dict[frozenset, str]   = {}
        self.elo_ratings: Dict[str, float]       = {}

        # V6.1: بناء lookup مُسبق للأسماء المختصرة بدل O(n) عند كل بحث
        self._alias_prefix: Dict[str, str] = {}

        self._load_all()

    # ── تحميل ────────────────────────────────────────────────
    def _load_all(self):
        self._load_teams_map()
        self._load_aliases()
        self._load_rivalries()
        self._load_elo()

    def _load_teams_map(self):
        if Path(self.teams_map_file).exists():
            try:
                with open(self.teams_map_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self.teams_map = self._build_safe_teams_map(raw)
            except Exception as e:
                print(f"⚠️  [{self.code}] teams_map error: {e}")

    def _load_aliases(self):
        if Path(self.aliases_file).exists():
            try:
                with open(self.aliases_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self.aliases = {k.lower().strip(): v for k, v in raw.items()}
            except Exception as e:
                print(f"⚠️  [{self.code}] aliases error: {e}")
        else:
            self.aliases = self._get_default_aliases()
        # V6.1: بناء prefix-map لتسريع norm_name()
        self._build_prefix_map()

    def _build_prefix_map(self):
        """V6.1: lookup جزئي O(1) بدل O(n)"""
        self._alias_prefix = {}
        for k, v in self.aliases.items():
            # نُسجّل كل بادئة بطول ≥ 4 حروف
            for length in range(4, len(k) + 1):
                prefix = k[:length]
                if prefix not in self._alias_prefix:
                    self._alias_prefix[prefix] = v

    def _load_rivalries(self):
        if Path(self.rivalries_file).exists():
            try:
                with open(self.rivalries_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                for item in raw:
                    if isinstance(item, dict):
                        teams = item.get("teams", [])
                        derby_name = item.get("name", "Derby")
                        if len(teams) == 2:
                            self.rivalries[frozenset(teams)] = derby_name
            except Exception as e:
                print(f"⚠️  [{self.code}] rivalries error: {e}")

    def _load_elo(self):
        if Path(self.elo_file).exists():
            try:
                with open(self.elo_file, "rb") as f:
                    raw_elo = pickle.load(f)
                if isinstance(raw_elo, dict):
                    for key, val in raw_elo.items():
                        if isinstance(key, str) and isinstance(val, (int, float)):
                            self.elo_ratings[key] = float(val)
            except Exception as e:
                print(f"⚠️  [{self.code}] elo error: {e}")

    def save_elo(self, elo_data: Dict[str, float]):
        Path(self.elo_file).parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.elo_file, "wb") as f:
                pickle.dump(elo_data, f)
        except Exception as e:
            print(f"⚠️  [{self.code}] save elo error: {e}")

    # ── helpers ───────────────────────────────────────────────
    @staticmethod
    def _build_safe_teams_map(raw: dict) -> Dict[str, str]:
        safe = {}
        for k, v in raw.items():
            k_str = str(k).strip()
            if (
                " v " in k_str.lower()
                or any(ch.isdigit() for ch in k_str)
                or len(k_str) > 40
            ):
                continue
            safe[k_str.lower()] = str(v)
        return safe

    @staticmethod
    def _get_default_aliases() -> Dict[str, str]:
        return {
            "manchester united":        "Man United",
            "manchester city":          "Man City",
            "tottenham hotspur":        "Tottenham",
            "tottenham":                "Tottenham",
            "newcastle united":         "Newcastle",
            "west ham united":          "West Ham",
            "wolverhampton wanderers":  "Wolves",
            "wolverhampton":            "Wolves",
            "nottingham forest":        "Nottm Forest",
            "leicester city":           "Leicester",
            "brighton & hove albion":   "Brighton",
            "brighton":                 "Brighton",
            "crystal palace":           "Crystal Palace",
            "aston villa":              "Aston Villa",
            "arsenal":                  "Arsenal",
            "chelsea":                  "Chelsea",
            "liverpool":                "Liverpool",
            "everton":                  "Everton",
            "brentford":                "Brentford",
            "fulham":                   "Fulham",
            "bournemouth":              "Bournemouth",
            "luton town":               "Luton",
            "sheffield united":         "Sheffield Utd",
            "burnley":                  "Burnley",
            "ipswich town":             "Ipswich",
            "southampton":              "Southampton",
            "real madrid":              "Real Madrid",
            "fc barcelona":             "Barcelona",
            "atletico madrid":          "Atletico Madrid",
            "athletic bilbao":          "Athletic Club",
            "fc bayern münchen":        "Bayern Munich",
            "borussia dortmund":        "Dortmund",
            "bayer 04 leverkusen":      "Leverkusen",
            "inter milan":              "Inter",
            "ac milan":                 "Milan",
            "juventus fc":              "Juventus",
            "paris saint-germain":      "PSG",
            "olympique de marseille":   "Marseille",
            "olympique lyonnais":       "Lyon",
        }

    # V6.1: norm_name() أسرع بفضل prefix-map
    def norm_name(self, name: str) -> str:
        lo = name.lower().strip()
        # 1. تطابق تام
        if lo in self.aliases:
            return self.aliases[lo]
        # 2. بحث prefix مُسبق O(1)
        if lo in self._alias_prefix:
            return self._alias_prefix[lo]
        # 3. teams_map
        if self.teams_map:
            if lo in self.teams_map:
                return self.teams_map[lo]
            lo_words = set(lo.split())
            candidates = [
                v for k, v in self.teams_map.items()
                if set(k.split()) == lo_words
            ]
            if len(candidates) == 1:
                return candidates[0]
        return name

    def is_derby(self, home: str, away: str) -> Optional[str]:
        return self.rivalries.get(
            frozenset({self.norm_name(home), self.norm_name(away)})
        )

    # ── CSV loading ───────────────────────────────────────────
    def load_csv_data(self) -> List[dict]:
        all_matches: List[dict] = []
        for csv_path in self.data_files:
            if not Path(csv_path).exists():
                print(f"⚠️  [{self.code}] File not found: {csv_path}")
                continue
            matches = self._parse_csv(csv_path)
            all_matches.extend(matches)
            print(f"✅ [{self.code}] Loaded {len(matches)} matches from {csv_path}")
        all_matches.sort(key=lambda x: x.get("utcDate", ""))
        return all_matches

    def _parse_csv(self, csv_path: str) -> List[dict]:
        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
            matches: List[dict] = []
            for _, row in df.iterrows():
                try:
                    if pd.isna(row.get("FTHG")) or pd.isna(row.get("FTAG")):
                        continue
                    h_name = self.norm_name(str(row["HomeTeam"]))
                    a_name = self.norm_name(str(row["AwayTeam"]))
                    hid = (
                        int(hashlib.md5(f"{self.code}_{h_name}".encode()).hexdigest()[:8], 16)
                        % 100_000
                    )
                    aid = (
                        int(hashlib.md5(f"{self.code}_{a_name}".encode()).hexdigest()[:8], 16)
                        % 100_000
                    )
                    date_str = str(row["Date"])
                    try:
                        fmt = (
                            "%d/%m/%y"
                            if len(date_str.split("/")[-1]) == 2
                            else "%d/%m/%Y"
                        )
                        dt = datetime.strptime(date_str, fmt)
                    except ValueError:
                        try:
                            dt = pd.to_datetime(date_str).to_pydatetime()
                        except Exception:
                            continue

                    stats: Dict[str, float] = {}
                    for col in ["HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR"]:
                        val = row.get(col, 0)
                        stats[col] = 0.0 if pd.isna(val) else float(val)

                    matches.append({
                        "status":   "FINISHED",
                        "utcDate":  dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "homeTeam": {"id": hid, "shortName": h_name, "name": h_name},
                        "awayTeam": {"id": aid, "shortName": a_name, "name": a_name},
                        "score":    {"fullTime": {"home": int(row["FTHG"]), "away": int(row["FTAG"])}},
                        "stats":    stats,
                        "league":   self.code,
                    })
                except Exception:
                    continue
            return matches
        except Exception as e:
            print(f"❌ [{self.code}] CSV parse error {csv_path}: {e}")
            return []


# ══════════════════════════════════════════════════════════════
# TEAM
# ══════════════════════════════════════════════════════════════
class Team:
    __slots__ = (
        "id", "name", "played", "wins", "draws", "losses",
        "gf", "ga", "pts", "pos",
        "h_p", "h_w", "h_d", "h_gf", "h_ga",
        "a_p", "a_w", "a_d", "a_gf", "a_ga",
        "results", "elo", "elo_hist", "match_dates",
        "cs", "fts", "_last_draw", "consec_draws",
        "win_streak", "loss_streak", "unbeaten",
        "stats_played", "sot_for", "sot_against",
        "corners_for", "corners_against", "discipline_pts",
    )

    def __init__(self, tid: int, name: str, elo_init: float = None):
        self.id      = tid
        self.name    = name
        self.played  = 0
        self.wins    = 0
        self.draws   = 0
        self.losses  = 0
        self.gf      = 0
        self.ga      = 0
        self.pts     = 0
        self.pos     = 0

        self.h_p = self.h_w = self.h_d = self.h_gf = self.h_ga = 0
        self.a_p = self.a_w = self.a_d = self.a_gf = self.a_ga = 0

        self.results:     List[Tuple] = []
        self.elo:         float       = elo_init if elo_init is not None else ELO_INIT
        self.elo_hist:    List[float] = [self.elo]
        self.match_dates: List[datetime] = []

        self.cs = self.fts = 0
        self._last_draw  = False
        self.consec_draws = 0
        self.win_streak  = 0
        self.loss_streak = 0
        self.unbeaten    = 0

        self.stats_played    = 0
        self.sot_for         = 0.0
        self.sot_against     = 0.0
        self.corners_for     = 0.0
        self.corners_against = 0.0
        self.discipline_pts  = 0.0

    # ── properties ────────────────────────────────────────────
    @property
    def gd(self):          return self.gf - self.ga
    @property
    def avg_gf(self):      return safe_div(self.gf, self.played)
    @property
    def avg_ga(self):      return safe_div(self.ga, self.played)
    @property
    def h_avg_gf(self):    return safe_div(self.h_gf, self.h_p)
    @property
    def h_avg_ga(self):    return safe_div(self.h_ga, self.h_p)
    @property
    def a_avg_gf(self):    return safe_div(self.a_gf, self.a_p)
    @property
    def a_avg_ga(self):    return safe_div(self.a_ga, self.a_p)
    @property
    def h_wr(self):        return safe_div(self.h_w, self.h_p, 0.45)
    @property
    def a_wr(self):        return safe_div(self.a_w, self.a_p, 0.30)
    @property
    def wr(self):          return safe_div(self.wins, self.played)
    @property
    def dr(self):          return safe_div(self.draws, self.played)
    @property
    def h_dr(self):        return safe_div(self.h_d, self.h_p)
    @property
    def a_dr(self):        return safe_div(self.a_d, self.a_p)
    @property
    def cs_r(self):        return safe_div(self.cs, self.played)
    @property
    def fts_r(self):       return safe_div(self.fts, self.played)
    @property
    def ppg(self):         return safe_div(self.pts, self.played)
    @property
    def avg_sot(self):     return safe_div(self.sot_for, self.stats_played)
    @property
    def avg_corners(self): return safe_div(self.corners_for, self.stats_played)
    @property
    def avg_discipline(self): return safe_div(self.discipline_pts, self.stats_played)

    @property
    def form_score(self) -> float:
        rec = self.results[-FORM_N:]
        if not rec:
            return 50.0
        total = max_t = 0.0
        for i, r in enumerate(rec):
            w = math.exp(0.3 * (i - len(rec) + 1))
            total += {"W": 3, "D": 1, "L": 0}[r[0]] * w
            max_t += 3 * w
        return (total / max_t) * 100 if max_t else 50.0

    @property
    def goal_form(self) -> float:
        rec = self.results[-FORM_N:]
        if not rec:
            return self.avg_gf
        total = wt = 0.0
        for i, r in enumerate(rec):
            w = math.exp(0.2 * (i - len(rec) + 1))
            total += r[1] * w
            wt    += w
        return total / wt if wt else self.avg_gf

    @property
    def defense_form(self) -> float:
        rec = self.results[-FORM_N:]
        if not rec:
            return self.avg_ga
        total = wt = 0.0
        for i, r in enumerate(rec):
            w = math.exp(0.2 * (i - len(rec) + 1))
            total += r[2] * w
            wt    += w
        return total / wt if wt else self.avg_ga

    @property
    def draw_form(self) -> float:
        rec = self.results[-FORM_N:]
        if not rec:
            return self.dr
        return sum(1 for r in rec if r[0] == "D") / len(rec)

    @property
    def form_string(self) -> str:
        return "".join(r[0] for r in self.results[-6:])

    @property
    def momentum(self) -> int:
        if self.win_streak >= 5:  return 90
        if self.win_streak >= 3:  return 60 + self.win_streak * 5
        if self.win_streak >= 2:  return 40
        if self.unbeaten  >= 5:   return 30
        if self.loss_streak >= 4: return -80
        if self.loss_streak >= 3: return -50
        if self.loss_streak >= 2: return -25
        return 0

    @property
    def volatility(self) -> float:
        rec = self.results[-10:]
        if len(rec) < 4:
            return 0.5
        goals = [r[1] + r[2] for r in rec]
        mean  = sum(goals) / len(goals)
        var   = sum((g - mean) ** 2 for g in goals) / len(goals)
        return min(1.0, math.sqrt(var) / 2.0)

    def days_rest(self, ref: datetime = None) -> int:
        ref = ref or datetime.now()
        past = [d for d in self.match_dates if d < ref]
        if not past:
            return 7
        return max(0, (ref - max(past)).days)

    def matches_in(self, n: int = 14, ref: datetime = None) -> int:
        ref = ref or datetime.now()
        cut = ref - timedelta(days=n)
        return sum(1 for d in self.match_dates if cut <= d < ref)


# ══════════════════════════════════════════════════════════════
# ELO SYSTEM  — V6.1: معادلة Draw أكثر واقعية
# ══════════════════════════════════════════════════════════════
class EloSystem:
    def __init__(self, home_advantage: int = 65):
        self.k  = ELO_K
        self.ha = home_advantage

    def expected(self, ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400))

    def gd_mult(self, gd: int) -> float:
        gd = abs(gd)
        if gd <= 1: return 1.0
        if gd == 2: return 1.5
        return (11 + gd) / 8

    def update(self, h: "Team", a: "Team", hg: int, ag: int):
        ha = h.elo + self.ha
        eh = self.expected(ha, a.elo)
        ea = 1 - eh

        if hg > ag:   ah, aa = 1.0, 0.0
        elif hg < ag: ah, aa = 0.0, 1.0
        else:         ah, aa = 0.5, 0.5

        m  = self.gd_mult(hg - ag)
        kh = self.k * (1.5 if h.played < 5 else (0.85 if h.elo > 1600 else 1.0))
        ka = self.k * (1.5 if a.played < 5 else (0.85 if a.elo > 1600 else 1.0))

        h.elo += kh * m * (ah - eh)
        a.elo += ka * m * (aa - ea)
        h.elo_hist.append(h.elo)
        a.elo_hist.append(a.elo)

    # V6.1 FIX: معادلة Draw أكثر واقعية
    def predict(self, h: "Team", a: "Team") -> Tuple[float, float, float]:
        ha  = h.elo + self.ha
        eh  = self.expected(ha, a.elo)
        ea  = 1 - eh
        dd  = abs(ha - a.elo)

        # V6.1: db يتناسب عكسياً مع فارق Elo بشكل أكثر واقعية
        # عند dd=0 → db=0.30 | عند dd=400 → db=0.20 | حد أدنى 0.15
        db = max(0.15, 0.30 - dd / 800.0)

        hw = eh * (1 - db)
        aw = ea * (1 - db)
        return normalize_probs(hw, db, aw)


# ══════════════════════════════════════════════════════════════
# DIXON-COLES
# ══════════════════════════════════════════════════════════════
class DixonColes:
    @staticmethod
    def tau(hg: int, ag: int, lh: float, la: float, rho: float) -> float:
        if hg == 0 and ag == 0: return 1 - lh * la * rho
        if hg == 0 and ag == 1: return 1 + lh * rho
        if hg == 1 and ag == 0: return 1 + la * rho
        if hg == 1 and ag == 1: return 1 - rho
        return 1.0

    @staticmethod
    def prob(hg: int, ag: int, lh: float, la: float, rho: float = -0.13) -> float:
        b = poisson_pmf(hg, lh) * poisson_pmf(ag, la)
        return max(0.0, b * DixonColes.tau(hg, ag, lh, la, rho))

    @staticmethod
    def predict(
        hxg: float, axg: float, rho: float = -0.13, mg: int = 10
    ) -> Tuple[float, float, float]:
        hw = dr = aw = 0.0
        for i in range(mg):
            for j in range(mg):
                p = DixonColes.prob(i, j, hxg, axg, rho)
                if   i > j: hw += p
                elif i == j: dr += p
                else:        aw += p
        t = hw + dr + aw
        return (hw / t, dr / t, aw / t) if t > 0 else (0.40, 0.25, 0.35)

    @staticmethod
    def matrix(
        hxg: float, axg: float, rho: float = -0.13, mg: int = 10
    ) -> Dict[Tuple[int, int], float]:
        return {
            (i, j): DixonColes.prob(i, j, hxg, axg, rho)
            for i in range(mg)
            for j in range(mg)
        }


# ══════════════════════════════════════════════════════════════
# IMPROVED DRAW PREDICTOR
# ══════════════════════════════════════════════════════════════
class ImprovedDrawPredictor:
    """
    نموذج متخصص للتعادل بناءً على البحث الإحصائي.
    يأخذ في الاعتبار:
      1. فارق قوة الفريقين (Elo)
      2. الأسلوب الدفاعي لكلا الفريقين
      3. التاريخ المباشر (H2H)
      4. الديربي
      5. مرحلة الموسم
      6. الفورم المحايد
      7. انخفاض التقلب
      8. الزخم (جديد V6.1)
    """

    @staticmethod
    def predict_draw_prob(
        h: "Team",
        a: "Team",
        h2h_draw_rate: float = 0.25,
        elo_diff: float = 0.0,
        is_derby: bool = False,
        is_late_season: bool = False,
    ) -> float:
        base        = 0.26
        adjustments = 0.0

        # ── فارق القوة ────────────────────────────────────────
        elo_factor   = max(0.0, 1.0 - abs(elo_diff) / 300.0)
        adjustments += elo_factor * 0.08

        # ── الأسلوب الدفاعي ───────────────────────────────────
        avg_total_goals = (h.avg_gf + h.avg_ga + a.avg_gf + a.avg_ga) / 2.0
        defensive_style  = max(0.0, 2.5 - avg_total_goals) / 2.5
        adjustments     += defensive_style * 0.06

        # ── H2H ───────────────────────────────────────────────
        adjustments += (h2h_draw_rate - 0.25) * 0.40

        # ── ديربي ─────────────────────────────────────────────
        if is_derby:
            adjustments += 0.04

        # ── نهاية الموسم ──────────────────────────────────────
        if is_late_season and abs(h.pos - a.pos) <= 2:
            adjustments += 0.03

        # ── فورم محايد ────────────────────────────────────────
        h_neutral = abs(h.form_score - 50.0) < 15.0
        a_neutral = abs(a.form_score - 50.0) < 15.0
        if h_neutral and a_neutral:
            adjustments += 0.04

        # ── انخفاض التقلب ─────────────────────────────────────
        if (h.volatility + a.volatility) / 2.0 < 0.35:
            adjustments += 0.03

        # ── الزخم (V6.1): زخم قوي → تعادل أقل ───────────────
        if abs(h.momentum) > 50 or abs(a.momentum) > 50:
            adjustments -= 0.04

        # ── معدل تعادل الفريقين ───────────────────────────────
        team_draw_tendency = (h.draw_form + a.draw_form) / 2.0
        adjustments += (team_draw_tendency - 0.25) * 0.20

        return min(0.45, max(0.15, base + adjustments))

    @staticmethod
    def predict(
        h: "Team",
        a: "Team",
        h2h_draw_rate: float = 0.25,
        elo_diff: float = 0.0,
        is_derby: bool = False,
        is_late_season: bool = False,
    ) -> Tuple[float, float, float]:
        dp  = ImprovedDrawPredictor.predict_draw_prob(
            h, a, h2h_draw_rate, elo_diff, is_derby, is_late_season
        )
        rem = 1.0 - dp
        if   elo_diff > 0: hp, ap = rem * 0.58, rem * 0.42
        elif elo_diff < 0: hp, ap = rem * 0.42, rem * 0.58
        else:              hp, ap = rem * 0.50, rem * 0.50
        return normalize_probs(hp, dp, ap)


# ══════════════════════════════════════════════════════════════
# FATIGUE
# ══════════════════════════════════════════════════════════════
class Fatigue:
    @staticmethod
    def score(t: "Team", ref: datetime = None) -> float:
        ref = ref or datetime.now()
        rd  = t.days_rest(ref)
        m14 = t.matches_in(14, ref)
        m30 = t.matches_in(30, ref)

        rest_map = {0: 40, 1: 40, 2: 40, 3: 30, 4: 20, 5: 10}
        rs  = rest_map.get(rd, 0 if rd <= 7 else -5)
        d14 = 35 if m14 >= 5 else (25 if m14 >= 4 else (15 if m14 >= 3 else 0))
        d30 = 25 if m30 >= 9 else (15 if m30 >= 7 else 0)
        return max(0.0, min(100.0, rs + d14 + d30))

    @staticmethod
    def impact(t: "Team", ref: datetime = None) -> float:
        return 1.05 - (Fatigue.score(t, ref) / 100.0) * 0.17

    @staticmethod
    def predict(
        h: "Team", a: "Team", ref: datetime = None
    ) -> Tuple[float, float, float]:
        hi = Fatigue.impact(h, ref)
        ai = Fatigue.impact(a, ref)
        t  = hi + ai
        if t == 0:
            return (0.40, 0.25, 0.35)
        hp = hi / t
        ap = ai / t
        d  = max(0.18, 0.30 - abs(hp - ap) * 0.3)
        hp *= (1 - d)
        ap *= (1 - d)
        return normalize_probs(hp, d, ap)


# ══════════════════════════════════════════════════════════════
# CALIBRATOR  — V6.1: حماية من القيم السالبة
# ══════════════════════════════════════════════════════════════
class Calibrator:
    def __init__(self):
        self.ok     = False
        self.models: Dict[str, "LogisticRegression"] = {}
        self.hist:   List[dict] = []

    def add(self, probs: Tuple[float, float, float], actual: str):
        self.hist.append({"probs": probs, "actual": actual})

    def calibrate(self) -> bool:
        if not ML_AVAILABLE or len(self.hist) < 30:
            return False
        try:
            for idx, out in enumerate(["HOME", "DRAW", "AWAY"]):
                ps = np.array([h["probs"][idx] for h in self.hist]).reshape(-1, 1)
                ac = np.array([1 if h["actual"] == out else 0 for h in self.hist])
                lr = LogisticRegression(
                    C=1.0, solver="lbfgs", class_weight="balanced"
                )
                lr.fit(ps, ac)
                self.models[out] = lr
            self.ok = True
            return True
        except Exception:
            return False

    # V6.1 FIX: حماية من القيم السالبة قبل التطبيع
    def adjust(
        self, probs: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        if not self.ok:
            return probs
        try:
            adj = []
            for i, out in enumerate(["HOME", "DRAW", "AWAY"]):
                if out in self.models:
                    raw = float(
                        self.models[out].predict_proba([[probs[i]]])[0][1]
                    )
                    adj.append(max(0.0, raw))   # ← V6.1: لا قيم سالبة
                else:
                    adj.append(max(0.0, probs[i]))
            return normalize_probs(*adj)        # ← V6.1: تطبيع آمن
        except Exception:
            return probs

    def save(self, fn: str):
        Path(fn).parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(fn, "wb") as f:
                pickle.dump(
                    {"hist": self.hist, "ok": self.ok, "models": self.models}, f
                )
        except Exception:
            pass

    def load(self, fn: str) -> bool:
        try:
            if Path(fn).exists():
                with open(fn, "rb") as f:
                    d = pickle.load(f)
                self.hist   = d.get("hist", [])
                self.ok     = d.get("ok", False)
                self.models = d.get("models", {})
                return True
        except Exception:
            pass
        return False


# ══════════════════════════════════════════════════════════════
# DATA PROCESSOR
# ══════════════════════════════════════════════════════════════
class DataProc:
    def __init__(self, resources: "LeagueResources" = None):
        self.resources = resources
        self.teams:    Dict[int, Team]       = {}
        self.elo       = EloSystem(
            home_advantage=resources.home_advantage if resources else 65
        )
        self.avg_h = resources.avg_home_goals if resources else 1.53
        self.avg_a = resources.avg_away_goals if resources else 1.16
        self.total = 0
        self.fixes: List[dict] = []
        self.h2h:   Dict[str, List[dict]] = defaultdict(list)

    def _get_elo_init(self, name: str) -> float:
        if self.resources and name in self.resources.elo_ratings:
            return self.resources.elo_ratings[name]
        return ELO_INIT

    # V6.1: _avgs() و _rank() يُستدعيان مرة واحدة في النهاية
    def process(self, matches: List[dict], do_elo: bool = True):
        matches = sorted(matches, key=lambda m: m.get("utcDate", ""))
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

            h  = self.teams[hid]
            a  = self.teams[aid]
            md = parse_date(ds)
            if md:
                h.match_dates.append(md)
                a.match_dates.append(md)

            if do_elo:
                self.elo.update(h, a, hg, ag)

            # ── إحصائيات عامة ─────────────────────────────────
            h.played += 1;  a.played += 1
            h.gf += hg;     h.ga += ag
            a.gf += ag;     a.ga += hg
            h.h_p += 1;     h.h_gf += hg;   h.h_ga += ag
            a.a_p += 1;     a.a_gf += ag;   a.a_ga += hg

            # ── شباك نظيفة / فيلد تسجيل ───────────────────────
            if ag == 0: h.cs  += 1
            if hg == 0: a.cs  += 1
            if hg == 0: h.fts += 1
            if ag == 0: a.fts += 1

            # ── إحصائيات عميقة ────────────────────────────────
            stats = m.get("stats")
            if stats:
                hst = stats.get("HST", 0)
                if hst is not None and not (
                    isinstance(hst, float) and math.isnan(hst)
                ):
                    h.stats_played += 1
                    a.stats_played += 1
                    h.sot_for      += stats.get("HST", 0) or 0
                    h.sot_against  += stats.get("AST", 0) or 0
                    a.sot_for      += stats.get("AST", 0) or 0
                    a.sot_against  += stats.get("HST", 0) or 0
                    h.corners_for      += stats.get("HC", 0) or 0
                    h.corners_against  += stats.get("AC", 0) or 0
                    a.corners_for      += stats.get("AC", 0) or 0
                    a.corners_against  += stats.get("HC", 0) or 0
                    h.discipline_pts += (
                        (stats.get("HF", 0) or 0)
                        + (stats.get("HY", 0) or 0) * 3
                        + (stats.get("HR", 0) or 0) * 10
                    )
                    a.discipline_pts += (
                        (stats.get("AF", 0) or 0)
                        + (stats.get("AY", 0) or 0) * 3
                        + (stats.get("AR", 0) or 0) * 10
                    )

            # ── نتيجة ─────────────────────────────────────────
            draw = (hg == ag)
            if hg > ag:
                h.wins += 1; h.h_w += 1; h.pts += 3; a.losses += 1
                h.results.append(("W", hg, ag, ds))
                a.results.append(("L", ag, hg, ds))
                h.win_streak  += 1; h.loss_streak  = 0; h.unbeaten += 1
                a.win_streak   = 0; a.loss_streak += 1; a.unbeaten  = 0
            elif hg < ag:
                a.wins += 1; a.a_w += 1; a.pts += 3; h.losses += 1
                h.results.append(("L", hg, ag, ds))
                a.results.append(("W", ag, hg, ds))
                a.win_streak  += 1; a.loss_streak  = 0; a.unbeaten += 1
                h.win_streak   = 0; h.loss_streak += 1; h.unbeaten  = 0
            else:
                h.draws += 1; a.draws += 1
                h.h_d   += 1; a.a_d   += 1
                h.pts   += 1; a.pts   += 1
                h.results.append(("D", hg, ag, ds))
                a.results.append(("D", ag, hg, ds))
                h.win_streak = a.win_streak = 0
                h.loss_streak = a.loss_streak = 0
                h.unbeaten += 1; a.unbeaten += 1

            h.consec_draws = (
                h.consec_draws + 1 if (draw and h._last_draw) else (1 if draw else 0)
            )
            a.consec_draws = (
                a.consec_draws + 1 if (draw and a._last_draw) else (1 if draw else 0)
            )
            h._last_draw = draw
            a._last_draw = draw

            # ── H2H ───────────────────────────────────────────
            key = f"{min(hid, aid)}_{max(hid, aid)}"
            self.h2h[key].append({
                "home_id": hid, "away_id": aid,
                "home_goals": hg, "away_goals": ag,
                "date": ds,
            })

            self.fixes.append({
                "home_id": hid, "away_id": aid,
                "home_goals": hg, "away_goals": ag,
                "date": ds, "home_name": hn, "away_name": an,
                "stats": m.get("stats", {}),
            })
            cnt += 1

        self.total += cnt
        # V6.1: استدعاء مرة واحدة بعد كل batch
        self._avgs()
        self._rank()

    def _ext(self, m: dict):
        if m.get("status") != "FINISHED":
            return None
        ht  = m.get("homeTeam", {})
        at  = m.get("awayTeam", {})
        hid = ht.get("id")
        aid = at.get("id")
        if not hid or not aid:
            return None
        hn = ht.get("shortName") or ht.get("name", "?")
        an = at.get("shortName") or at.get("name", "?")
        ft = m.get("score", {}).get("fullTime", {})
        hg = ft.get("home")
        ag = ft.get("away")
        if hg is None or ag is None:
            return None
        return (hid, hn, aid, an, int(hg), int(ag), m.get("utcDate", ""))

    def get_h2h(self, t1: int, t2: int) -> List[dict]:
        return self.h2h.get(f"{min(t1, t2)}_{max(t1, t2)}", [])

    def _avgs(self):
        th = sum(t.h_gf for t in self.teams.values())
        ta = sum(t.a_gf for t in self.teams.values())
        tm = sum(t.h_p  for t in self.teams.values())
        if tm:
            self.avg_h = th / tm
            self.avg_a = ta / tm

    def _rank(self):
        for i, t in enumerate(
            sorted(
                self.teams.values(),
                key=lambda t: (t.pts, t.gd, t.gf),
                reverse=True,
            ),
            1,
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
# ML PREDICTOR — 126 FEATURES  V6.1
# ══════════════════════════════════════════════════════════════
class MLPred:
    N_FEATURES = 126

    def __init__(self, model_file: str = None):
        self.pipeline: Optional["Pipeline"] = None
        self.trained   = False
        self.acc       = 0.0
        self._external = False
        self.model_file = model_file or "models/default_model_v6.pkl"

    # ══════════════════════════════════════════════════════════
    # 68 ميزة أصلية
    # ══════════════════════════════════════════════════════════
    def _original_features(
        self,
        h: Team, a: Team, data: "DataProc",
        md: datetime = None, derby: bool = False,
    ) -> List[float]:
        ah = max(data.avg_h, 0.5)
        aa = max(data.avg_a, 0.5)
        return [
            # Elo (3)
            h.elo, a.elo, h.elo - a.elo,
            # Form Score (4)
            h.form_score, a.form_score,
            h.form_score - a.form_score,
            abs(h.form_score - a.form_score),
            # هجوم (6)
            h.h_avg_gf, a.a_avg_gf, h.goal_form, a.goal_form,
            h.goal_form - a.goal_form, h.h_avg_gf - a.a_avg_gf,
            # دفاع (6)
            h.h_avg_ga, a.a_avg_ga, h.defense_form, a.defense_form,
            h.defense_form - a.defense_form, h.h_avg_ga - a.a_avg_ga,
            # نسب هجوم/دفاع (4)
            safe_div(h.h_avg_gf, ah, 1.0), safe_div(a.a_avg_gf, aa, 1.0),
            safe_div(h.h_avg_ga, ah, 1.0), safe_div(a.a_avg_ga, aa, 1.0),
            # نسب الفوز (4)
            h.h_wr, a.a_wr, h.wr, a.wr,
            # الترتيب والنقاط (7)
            h.pos, a.pos, a.pos - h.pos,
            h.pts, a.pts, h.ppg - a.ppg, h.gd,
            # فارق أهداف أواي (1)
            a.gd,
            # الشباك النظيفة (4)
            h.cs_r, a.cs_r, h.fts_r, a.fts_r,
            # الإرهاق (2)
            Fatigue.score(h, md), Fatigue.score(a, md),
            # نسب التعادل (7)
            h.dr, a.dr, (h.dr + a.dr) / 2,
            h.draw_form, a.draw_form, h.h_dr, a.a_dr,
            # التقلب (2)
            h.volatility, a.volatility,
            # ديربي (1)
            1.0 if derby else 0.0,
            # Elo مقياس (1)
            abs(h.elo - a.elo) / 100.0,
            # الزخم (7)
            h.momentum / 100.0, a.momentum / 100.0,
            (h.momentum - a.momentum) / 100.0,
            h.win_streak, a.win_streak,
            h.loss_streak, a.loss_streak,
            # إحصائيات عميقة (9)
            h.avg_sot, a.avg_sot, h.avg_sot - a.avg_sot,
            h.avg_corners, a.avg_corners, h.avg_corners - a.avg_corners,
            h.avg_discipline, a.avg_discipline, h.avg_discipline - a.avg_discipline,
        ]  # 68 ميزة

    # ══════════════════════════════════════════════════════════
    # 12 ميزة زمنية
    # ══════════════════════════════════════════════════════════
    def _temporal_features(
        self, match_date: datetime, h: Team, a: Team
    ) -> List[float]:
        if match_date is None:
            return [0.0] * 12

        month       = match_date.month
        day_of_week = match_date.weekday()
        season_month = ((month - 8) % 12) + 1

        is_early_season = 1.0 if season_month <= 3 else 0.0
        is_late_season  = 1.0 if season_month >= 9 else 0.0
        is_mid_season   = 1.0 if 4 <= season_month <= 8 else 0.0
        is_weekend      = 1.0 if day_of_week >= 5 else 0.0
        is_midweek      = 1.0 if day_of_week in [1, 2, 3] else 0.0
        month_sin       = math.sin(2 * math.pi * month / 12.0)
        month_cos       = math.cos(2 * math.pi * month / 12.0)

        relegation_pressure_h = 0.0
        relegation_pressure_a = 0.0
        title_pressure_h      = 0.0
        title_pressure_a      = 0.0

        if is_late_season:
            if h.pos >= 17:
                relegation_pressure_h = min(1.0, (h.pos - 16) / 4.0)
            if h.pos <= 3:
                title_pressure_h = (4 - h.pos) / 3.0
            if a.pos >= 17:
                relegation_pressure_a = min(1.0, (a.pos - 16) / 4.0)
            if a.pos <= 3:
                title_pressure_a = (4 - a.pos) / 3.0

        both_relegation = 1.0 if (h.pos >= 15 and a.pos >= 15) else 0.0

        return [
            is_early_season,        # 68
            is_mid_season,          # 69
            is_late_season,         # 70
            is_weekend,             # 71
            is_midweek,             # 72
            month_sin,              # 73
            month_cos,              # 74
            relegation_pressure_h,  # 75
            relegation_pressure_a,  # 76
            title_pressure_h,       # 77
            title_pressure_a,       # 78
            both_relegation,        # 79
        ]  # 12 ميزة

    # ══════════════════════════════════════════════════════════
    # 14 ميزة زخم متقدم
    # ══════════════════════════════════════════════════════════
    def _advanced_momentum_features(self, h: Team, a: Team) -> List[float]:
        def weighted_form_stats(team: Team, last_n: int = 6) -> dict:
            results = team.results[-last_n:]
            if not results:
                return {"raw_pts": 0.0, "trend": 0.0, "consistency": 0.5}
            pts  = [3 if r[0] == "W" else (1 if r[0] == "D" else 0) for r in results]
            half = len(pts) // 2
            if half > 0:
                trend = (
                    sum(pts[half:]) / max(1, len(pts) - half)
                    - sum(pts[:half]) / half
                ) / 3.0
            else:
                trend = 0.0
            if len(pts) > 1:
                mean_pts = sum(pts) / len(pts)
                variance = sum((p - mean_pts) ** 2 for p in pts) / len(pts)
                consistency = 1.0 / (1.0 + variance)
            else:
                consistency = 0.5
            return {
                "raw_pts":    sum(pts) / (len(pts) * 3),
                "trend":      trend,
                "consistency": consistency,
            }

        def raw_pts_n(team: Team, n: int) -> float:
            results = team.results[-n:]
            if not results:
                return 0.0
            pts = [3 if r[0] == "W" else (1 if r[0] == "D" else 0) for r in results]
            return sum(pts) / (len(pts) * 3)

        def goal_trend(team: Team, last_n: int = 5) -> Tuple[float, float]:
            results = team.results[-last_n:]
            if not results:
                return (0.0, 0.0)
            gf_list = [r[1] for r in results]
            ga_list = [r[2] for r in results]
            n = len(gf_list)
            if n < 2:
                return (gf_list[0] if gf_list else 0.0, ga_list[0] if ga_list else 0.0)
            x_mean  = (n - 1) / 2.0
            gf_mean = sum(gf_list) / n
            ga_mean = sum(ga_list) / n
            denom   = sum((i - x_mean) ** 2 for i in range(n))
            if denom == 0:
                return (0.0, 0.0)
            gf_slope = sum((i - x_mean) * (gf_list[i] - gf_mean) for i in range(n)) / denom
            ga_slope = sum((i - x_mean) * (ga_list[i] - ga_mean) for i in range(n)) / denom
            return (gf_slope, ga_slope)

        h_stats = weighted_form_stats(h, 6)
        a_stats = weighted_form_stats(a, 6)
        h_hot   = raw_pts_n(h, 3) - raw_pts_n(h, 8)
        a_hot   = raw_pts_n(a, 3) - raw_pts_n(a, 8)
        h_gf_t, h_ga_t = goal_trend(h, 5)
        a_gf_t, a_ga_t = goal_trend(a, 5)

        return [
            h_stats["raw_pts"],                           # 80
            h_stats["trend"],                             # 81
            h_stats["consistency"],                       # 82
            a_stats["raw_pts"],                           # 83
            a_stats["trend"],                             # 84
            a_stats["consistency"],                       # 85
            h_stats["raw_pts"] - a_stats["raw_pts"],      # 86
            h_stats["trend"]   - a_stats["trend"],        # 87
            h_hot,                                        # 88
            a_hot,                                        # 89
            h_gf_t,                                       # 90
            h_ga_t,                                       # 91
            a_gf_t,                                       # 92
            a_ga_t,                                       # 93
        ]  # 14 ميزة

    # ══════════════════════════════════════════════════════════
    # 10 ميزة H2H متقدمة
    # ══════════════════════════════════════════════════════════
    def _advanced_h2h_features(
        self, h: Team, a: Team, data: "DataProc"
    ) -> List[float]:
        h2h_matches = data.get_h2h(h.id, a.id)
        if not h2h_matches:
            return [0.0, 0.0, 0.25, 1.3, 1.1, 0.5, 0.5, 0.0, 0.0, 0.0]

        recent = h2h_matches[-5:]
        h_wins = a_wins = draws = h_goals = a_goals = btts = over25 = 0

        for m in recent:
            hg = m["home_goals"] if m["home_id"] == h.id else m["away_goals"]
            ag = m["away_goals"] if m["home_id"] == h.id else m["home_goals"]
            if   hg > ag: h_wins += 1
            elif ag > hg: a_wins += 1
            else:          draws += 1
            h_goals += hg
            a_goals += ag
            if hg > 0 and ag > 0:   btts   += 1
            if hg + ag > 2:          over25 += 1

        n = len(recent)
        return [
            h_wins  / n,                         # 94 h_dominance
            a_wins  / n,                         # 95 a_dominance
            draws   / n,                         # 96 draw_tendency
            h_goals / n,                         # 97 h2h avg h goals
            a_goals / n,                         # 98 h2h avg a goals
            btts    / n,                         # 99 btts rate
            over25  / n,                         # 100 over2.5 rate
            1.0 if h_wins / n > 0.5 else 0.0,   # 101 favors home
            1.0 if a_wins / n > 0.5 else 0.0,   # 102 favors away
            min(1.0, len(h2h_matches) / 10.0),  # 103 sample weight
        ]  # 10 ميزة

    # ══════════════════════════════════════════════════════════
    # 11 ميزة xG متقدمة  — V6.1 FIX: save_rate محسوبة صحيح
    # ══════════════════════════════════════════════════════════
    def _advanced_xg_features(self, h: Team, a: Team) -> List[float]:
        NORMAL_CONVERSION = 0.30

        h_conversion = safe_div(h.gf, h.sot_for, NORMAL_CONVERSION)
        a_conversion = safe_div(a.gf, a.sot_for, NORMAL_CONVERSION)
        h_luck = h_conversion - NORMAL_CONVERSION
        a_luck = a_conversion - NORMAL_CONVERSION

        # V6.1 FIX: save_rate تعتمد على sot_against فقط (لا fallback خاطئ)
        h_saves_rate = 1.0 - safe_div(h.ga, h.sot_against, 0.70)
        a_saves_rate = 1.0 - safe_div(a.ga, a.sot_against, 0.70)

        # تقييد المدى المنطقي [0, 1]
        h_saves_rate = max(0.0, min(1.0, h_saves_rate))
        a_saves_rate = max(0.0, min(1.0, a_saves_rate))

        total_sot   = h.sot_for + a.sot_against
        h_shot_dom  = safe_div(h.sot_for, total_sot, 0.5)

        total_corn  = h.corners_for + a.corners_for
        h_corn_dom  = safe_div(h.corners_for, total_corn, 0.5)

        h_xg_est = h.sot_for * NORMAL_CONVERSION
        a_xg_est = a.sot_for * NORMAL_CONVERSION
        h_xg_diff = (
            (h.gf - h_xg_est) / max(h.played, 1) if h.stats_played > 0 else 0.0
        )
        a_xg_diff = (
            (a.gf - a_xg_est) / max(a.played, 1) if a.stats_played > 0 else 0.0
        )

        return [
            h_conversion,          # 104
            a_conversion,          # 105
            h_luck,                # 106
            a_luck,                # 107
            h_luck - a_luck,       # 108
            h_saves_rate,          # 109
            a_saves_rate,          # 110
            h_shot_dom,            # 111
            h_corn_dom,            # 112
            h_xg_diff,             # 113
            a_xg_diff,             # 114
        ]  # 11 ميزة

    # ══════════════════════════════════════════════════════════
    # 11 ميزة السياق والضغط
    # ══════════════════════════════════════════════════════════
    def _context_features(
        self, h: Team, a: Team,
        match_date: datetime, data: "DataProc",
    ) -> List[float]:
        elo_diff            = h.elo - a.elo
        elo_diff_normalized = elo_diff / 400.0
        elo_prob_h          = 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))

        def schedule_difficulty(team: Team, n: int = 5) -> float:
            recent = team.results[-n:]
            if not recent:
                return 0.0
            recent_pts = sum(
                3 if r[0] == "W" else (1 if r[0] == "D" else 0) for r in recent
            ) / max(len(recent), 1)
            return team.ppg - recent_pts

        h_schedule = schedule_difficulty(h, 5)
        a_schedule = schedule_difficulty(a, 5)

        h_stability = a_stability = 0.5
        if len(h.elo_hist) >= 5:
            re = h.elo_hist[-5:]
            h_stability = 1.0 / (1.0 + (max(re) - min(re)) / 100.0)
        if len(a.elo_hist) >= 5:
            re = a.elo_hist[-5:]
            a_stability = 1.0 / (1.0 + (max(re) - min(re)) / 100.0)

        total_rounds    = data.resources.total_rounds if data.resources else 38
        h_progress      = min(1.0, h.played / total_rounds)
        a_progress      = min(1.0, a.played / total_rounds)
        data_reliability = min(h_progress, a_progress)

        total_teams = max(len(data.teams), 20)
        h_urgency   = max(0.0, (h.pos - total_teams * 0.7) / (total_teams * 0.3))
        a_urgency   = max(0.0, (a.pos - total_teams * 0.7) / (total_teams * 0.3))

        return [
            elo_diff_normalized,       # 115
            elo_prob_h,                # 116
            h_schedule,                # 117
            a_schedule,                # 118
            h_stability,               # 119
            a_stability,               # 120
            h_progress,                # 121
            data_reliability,          # 122
            h_urgency,                 # 123
            a_urgency,                 # 124
            h_urgency - a_urgency,     # 125
        ]  # 11 ميزة

    # ══════════════════════════════════════════════════════════
    # feats() الرئيسية — V6.1: آمنة بدل assert
    # ══════════════════════════════════════════════════════════
    def feats(
        self, h: Team, a: Team,
        data: "DataProc", md: datetime = None, derby: bool = False,
    ) -> List[float]:
        all_feats = (
            self._original_features(h, a, data, md, derby)   # 68
            + self._temporal_features(md, h, a)              # 12
            + self._advanced_momentum_features(h, a)         # 14
            + self._advanced_h2h_features(h, a, data)        # 10
            + self._advanced_xg_features(h, a)               # 11
            + self._context_features(h, a, md, data)         # 11
        )  # = 126

        # V6.1 FIX: آمن في Production بدل assert
        n = len(all_feats)
        if n != self.N_FEATURES:
            # Pad أو Truncate
            all_feats = (all_feats + [0.0] * self.N_FEATURES)[: self.N_FEATURES]

        return all_feats

    # ══════════════════════════════════════════════════════════
    # بناء Pipelines
    # ══════════════════════════════════════════════════════════
    def _build_stacking_pipeline(self) -> "Pipeline":
        base = [
            ("rf", RandomForestClassifier(
                n_estimators=200, max_depth=8, min_samples_leaf=5,
                class_weight="balanced", random_state=42, n_jobs=-1,
            )),
            ("lr_base", LogisticRegression(
                C=0.1, class_weight="balanced", max_iter=1000,
                solver="lbfgs", random_state=42,
            )),
        ]
        if XGBOOST_AVAILABLE:
            base.append(("xgb", XGBClassifier(
                n_estimators=150, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1, verbosity=0,
            )))
        meta = LogisticRegression(
            C=0.5, class_weight="balanced", max_iter=1000,
            solver="lbfgs", random_state=42,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler",  StandardScaler()),
            ("model",   StackingClassifier(
                estimators=base, final_estimator=meta,
                cv=3, stack_method="predict_proba", n_jobs=1,
            )),
        ])

    def _build_voting_pipeline(self) -> "Pipeline":
        estimators = [
            ("rf", RandomForestClassifier(
                n_estimators=200, max_depth=8, min_samples_leaf=5,
                class_weight="balanced", random_state=42, n_jobs=-1,
            )),
        ]
        if XGBOOST_AVAILABLE:
            estimators.append(("xgb", XGBClassifier(
                n_estimators=150, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1, verbosity=0,
            )))
        return Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler",  StandardScaler()),
            ("model",   VotingClassifier(estimators=estimators, voting="soft", n_jobs=1)),
        ])

    def _try_load_external(self) -> bool:
        if not Path(self.model_file).exists():
            return False
        try:
            with open(self.model_file, "rb") as f:
                loaded = pickle.load(f)
            if not isinstance(loaded, Pipeline):
                return False
            # تحقق من عدد الميزات
            final = loaded.steps[-1][1]
            n_feat = getattr(final, "n_features_in_", None)
            if n_feat is None:
                try:
                    n_feat = getattr(final.final_estimator_, "n_features_in_", None)
                except Exception:
                    pass
            if n_feat is not None and n_feat != self.N_FEATURES:
                print(
                    f"⚠️  Model features mismatch: {n_feat} vs {self.N_FEATURES} "
                    "→ retraining"
                )
                return False
            self.pipeline  = loaded
            self.trained   = True
            self._external = True
            return True
        except Exception:
            return False

    def train(
        self,
        data: "DataProc",
        fixes: List[dict] = None,
        force_retrain: bool = False,
    ) -> bool:
        if not ML_AVAILABLE:
            return False
        if not force_retrain and self._try_load_external():
            return True

        fixes = fixes or data.fixes
        if len(fixes) < 40:
            return False

        X: List[List[float]] = []
        y: List[int]         = []

        sim  = DataProc(data.resources)
        sf   = sorted(fixes, key=lambda f: f.get("date", ""))
        warm = int(len(sf) * 0.30)

        for idx, f in enumerate(sf):
            if idx >= warm:
                ht = sim.teams.get(f["home_id"])
                at = sim.teams.get(f["away_id"])
                if ht and at and ht.played >= 3 and at.played >= 3:
                    try:
                        md_    = parse_date(f.get("date", ""))
                        derby_ = bool(
                            data.resources.is_derby(f["home_name"], f["away_name"])
                        ) if data.resources else False
                        ft     = self.feats(ht, at, sim, md_, derby_)
                        lb     = (
                            0 if f["home_goals"] > f["away_goals"] else
                            (1 if f["home_goals"] == f["away_goals"] else 2)
                        )
                        X.append(ft)
                        y.append(lb)
                    except Exception:
                        pass

            # تحديث sim تدريجياً
            sim.process([{
                "status":   "FINISHED",
                "homeTeam": {"id": f["home_id"], "shortName": f["home_name"]},
                "awayTeam": {"id": f["away_id"], "shortName": f["away_name"]},
                "score":    {"fullTime": {"home": f["home_goals"], "away": f["away_goals"]}},
                "utcDate":  f.get("date", ""),
                "stats":    f.get("stats", {}),
            }], do_elo=True)

        if len(X) < 30:
            return False

        X_arr = np.array(X, dtype=np.float64)
        y_arr = np.array(y, dtype=np.int64)

        _, counts = np.unique(y_arr, return_counts=True)
        if int(counts.min()) < MIN_SAMPLES_PER_CLASS:
            return False

        # Stacking → Voting → RF fallback
        for attempt, builder in enumerate([
            self._build_stacking_pipeline,
            self._build_voting_pipeline,
        ]):
            try:
                pipe = builder()
                n_splits = 3 if attempt == 0 else 2
                skf   = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                cv_sc = cross_val_score(
                    pipe, X_arr, y_arr,
                    cv=skf, scoring="balanced_accuracy", n_jobs=1,
                )
                self.acc = float(cv_sc.mean())
                pipe.fit(X_arr, y_arr)
                self.pipeline = pipe
                name = "Stacking" if attempt == 0 else "Voting"
                print(f"✅ {name} model trained | CV Balanced Acc: {self.acc * 100:.1f}%")
                self.trained = True
                return True
            except Exception as e:
                print(f"⚠️  Attempt {attempt + 1} failed: {e}")

        # RF fallback
        try:
            fallback = Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler",  StandardScaler()),
                ("model",   RandomForestClassifier(
                    n_estimators=200, class_weight="balanced",
                    random_state=42, n_jobs=-1,
                )),
            ])
            fallback.fit(X_arr, y_arr)
            self.pipeline = fallback
            self.acc      = 0.0
            self.trained  = True
            print("✅ RF fallback model trained")
            return True
        except Exception:
            return False

    def predict(
        self,
        h: Team, a: Team, data: "DataProc",
        md: datetime = None, derby: bool = False,
    ) -> Optional[Tuple[float, float, float]]:
        if not self.trained or self.pipeline is None:
            return None
        try:
            ft      = self.feats(h, a, data, md, derby)
            X       = np.array([ft], dtype=np.float64)
            probs   = self.pipeline.predict_proba(X)[0]
            classes = self.pipeline.classes_
            pd_     = {cls: pr for cls, pr in zip(classes, probs)}
            return (
                float(pd_.get(0, 0.0)),
                float(pd_.get(1, 0.0)),
                float(pd_.get(2, 0.0)),
            )
        except Exception:
            return None

    def save_pipeline(self):
        if self.pipeline is not None:
            Path(self.model_file).parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(self.model_file, "wb") as f:
                    pickle.dump(self.pipeline, f)
                print(f"✅ Model saved: {self.model_file}")
            except Exception as e:
                print(f"⚠️  Save model error: {e}")


# ══════════════════════════════════════════════════════════════
# API CLIENT
# ══════════════════════════════════════════════════════════════
class FootballAPI:
    def __init__(self, token: str, base_url: str = "https://api.football-data.org/v4"):
        self.s = requests.Session()
        self.s.headers.update({"X-Auth-Token": token, "Accept": "application/json"})
        self.base_url = base_url
        self._c:  Dict[str, dict] = {}
        self._t:  float           = 0.0

    def _rl(self):
        elapsed = time.time() - self._t
        if elapsed < 6.5:
            time.sleep(6.5 - elapsed)
        self._t = time.time()

    def _get(self, ep: str, p: dict = None, cache: bool = True):
        p = p or {}
        k = hashlib.md5(
            f"{ep}|{json.dumps(p, sort_keys=True)}".encode()
        ).hexdigest()
        if cache and k in self._c:
            return self._c[k]
        try:
            self._rl()
            r = self.s.get(f"{self.base_url}/{ep}", params=p, timeout=30)
            if r.status_code == 429:
                wait = int(r.headers.get("X-RequestCounter-Reset", 60)) + 1
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
        if d and d.get("currentSeason"):
            try:
                return int(d["currentSeason"]["startDate"][:4])
            except Exception:
                pass
        return None

    def finished(self, league_code: str, season: int = None) -> List[dict]:
        p = {"status": "FINISHED"}
        if season:
            p["season"] = season
        d = self._get(f"competitions/{league_code}/matches", p)
        if d and "matches" in d:
            m = d["matches"]
            m.sort(key=lambda x: x.get("utcDate", ""))
            return m
        return []

    def upcoming(self, league_code: str, days: int = 14) -> List[dict]:
        t = datetime.now().strftime("%Y-%m-%d")
        e = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
        d = self._get(
            f"competitions/{league_code}/matches",
            {"status": "SCHEDULED,TIMED", "dateFrom": t, "dateTo": e},
        )
        if d and "matches" in d:
            m = d["matches"]
            m.sort(key=lambda x: x.get("utcDate", ""))
            return m
        return []


class OddsAPI:
    def __init__(self, key: str, sport: str = "soccer_epl"):
        self.key   = key
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
                    "apiKey":      self.key,
                    "regions":     "uk,eu",
                    "markets":     "h2h,totals",
                    "oddsFormat":  "decimal",
                },
                timeout=15,
            )
            if r.status_code != 200:
                return {}

            result: Dict[str, dict] = {}
            for ev in r.json():
                h   = ev.get("home_team", "")
                a   = ev.get("away_team", "")
                bms = ev.get("bookmakers", [])
                if not bms:
                    continue
                ah_l, ad_l, aa_l = [], [], []
                for bm in bms:
                    for mk in bm.get("markets", []):
                        if mk["key"] == "h2h":
                            for o in mk.get("outcomes", []):
                                if o["name"] == h:       ah_l.append(o["price"])
                                elif o["name"] == a:     aa_l.append(o["price"])
                                elif o["name"] == "Draw": ad_l.append(o["price"])
                if ah_l and ad_l and aa_l:
                    avh = sum(ah_l) / len(ah_l)
                    avd = sum(ad_l) / len(ad_l)
                    ava = sum(aa_l) / len(aa_l)
                    ih, id_, ia = 1 / avh, 1 / avd, 1 / ava
                    # ── DC odds احتمالات مجمعة محسوبة ────────────
                    # 1X: لا تخسر إذا فاز أي منهما أو تعادلا
                    # X2: لا تخسر إذا فاز الضيف أو تعادلا
                    # 12: لا تخسر إذا لم يتعادلا
                    odds_1x  = round(1.0 / (ih + id_), 2) if (ih + id_) > 0 else None
                    odds_x2  = round(1.0 / (ia + id_), 2) if (ia + id_) > 0 else None
                    odds_12  = round(1.0 / (ih + ia),  2) if (ih + ia)  > 0 else None
                    result[f"{h}_vs_{a}".lower()] = {
                        "home_team":    h,
                        "away_team":    a,
                        "odds_home":    round(avh, 2),
                        "odds_draw":    round(avd, 2),
                        "odds_away":    round(ava, 2),
                        "implied_home": round(ih,  4),
                        "implied_draw": round(id_, 4),
                        "implied_away": round(ia,  4),
                        "implied_1x":   round(ih + id_, 4),
                        "implied_x2":   round(ia + id_, 4),
                        "implied_12":   round(ih + ia,  4),
                        "odds_1x":      odds_1x,
                        "odds_x2":      odds_x2,
                        "odds_12":      odds_12,
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
        for d in self.cache.values():
            oh = d["home_team"].lower()
            oa = d["away_team"].lower()
            hm = hl in oh or oh in hl or any(w in oh for w in hl.split() if len(w) > 3)
            am = al in oa or oa in al or any(w in oa for w in al.split() if len(w) > 3)
            if hm and am:
                return d
        return None


# ══════════════════════════════════════════════════════════════
# PREDICTION RESULT
# ══════════════════════════════════════════════════════════════
class Pred:
    __slots__ = (
        "home", "away", "hid", "aid", "date", "league",
        "hp", "dp", "ap", "raw_hp", "raw_dp", "raw_ap",
        "hxg", "axg", "top_sc", "result", "pred_sc", "conf",
        "btts", "o15", "o25", "o35",
        "dc_1x", "dc_x2", "dc_12", "dc_recommend",
        "dc_value_bets", "value_bets",
        "h_form", "a_form", "h_pos", "a_pos",
        "h_elo", "a_elo", "h_fat", "a_fat",
        "h_rest", "a_rest", "h_momentum", "a_momentum",
        "models", "odds", "ml_used", "ml_acc",
        "calibrated", "is_derby", "derby_name",
    )

    def __init__(self):
        self.home = self.away = self.date = self.league = ""
        self.hid  = self.aid  = 0
        self.hp = self.dp = self.ap = 0.0
        self.raw_hp = self.raw_dp = self.raw_ap = 0.0
        self.hxg = self.axg = 0.0
        self.top_sc: List[Tuple] = []
        self.result   = ""
        self.pred_sc  = (0, 0)
        self.conf     = 0.0
        self.btts = self.o15 = self.o25 = self.o35 = 0.0
        self.dc_1x = self.dc_x2 = self.dc_12 = 0.0
        self.dc_recommend = ""
        self.dc_value_bets: List[dict] = []
        self.value_bets:    List[dict] = []
        self.h_form = self.a_form = ""
        self.h_pos  = self.a_pos  = 0
        self.h_elo  = self.a_elo  = 0.0
        self.h_fat  = self.a_fat  = 0.0
        self.h_rest = self.a_rest = 0
        self.h_momentum = self.a_momentum = 0
        self.models: Dict[str, Tuple] = {}
        self.odds       = None
        self.ml_used    = False
        self.ml_acc     = 0.0
        self.calibrated = False
        self.is_derby   = False
        self.derby_name = ""


# ══════════════════════════════════════════════════════════════
# ENGINE  — V6.1: Draw Correction + _form() + _dc_recommend()
# ══════════════════════════════════════════════════════════════
class Engine:
    def __init__(
        self,
        data:      "DataProc",
        resources: "LeagueResources",
        ml:        "MLPred"    = None,
        odds:      "OddsAPI"   = None,
        cal:       "Calibrator"= None,
    ):
        self.data      = data
        self.resources = resources
        self.ml        = ml
        self.odds      = odds
        self.cal       = cal
        self.w         = dict(WEIGHTS)

        if not ml or not ml.trained:
            mw  = self.w.pop("ml", 0.25)
            rem = sum(self.w.values())
            if rem > 0:
                for k in self.w:
                    self.w[k] += mw * (self.w[k] / rem)

    def predict(
        self, hid: int, aid: int,
        date: str = "", md: datetime = None,
    ) -> Optional[Pred]:
        h = self.data.teams.get(hid)
        a = self.data.teams.get(aid)
        if not h or not a or h.played < 2 or a.played < 2:
            return None

        p         = Pred()
        p.home    = h.name
        p.away    = a.name
        p.hid     = hid
        p.aid     = aid
        p.date    = date
        p.league  = self.resources.code if self.resources else ""
        p.h_form  = h.form_string
        p.a_form  = a.form_string
        p.h_pos   = h.pos
        p.a_pos   = a.pos
        p.h_elo   = h.elo
        p.a_elo   = a.elo

        if md is None and date:
            md = parse_date(date)
        md = md or datetime.now()

        derby       = self.resources.is_derby(h.name, a.name) if self.resources else None
        p.is_derby  = bool(derby)
        p.derby_name = derby or ""

        p.h_fat     = Fatigue.score(h, md)
        p.a_fat     = Fatigue.score(a, md)
        p.h_rest    = h.days_rest(md)
        p.a_rest    = a.days_rest(md)
        p.h_momentum = h.momentum
        p.a_momentum = a.momentum

        p.hxg = self._xg(h, a, True)  * Fatigue.impact(h, md)
        p.axg = self._xg(a, h, False) * Fatigue.impact(a, md)

        if   h.momentum >  40: p.hxg *= 1.05
        elif h.momentum < -40: p.hxg *= 0.95
        if   a.momentum >  40: p.axg *= 1.05
        elif a.momentum < -40: p.axg *= 0.95

        season_month   = ((md.month - 8) % 12) + 1
        is_late_season = season_month >= 9

        h2h_matches    = self.data.get_h2h(hid, aid)
        h2h_draw_rate  = 0.25
        if h2h_matches:
            recent = h2h_matches[-10:]
            h2h_draw_rate = sum(
                1 for m in recent if m["home_goals"] == m["away_goals"]
            ) / len(recent)

        ha_val   = self.resources.home_advantage if self.resources else 65
        elo_diff = h.elo + ha_val - a.elo

        # ── تجميع النماذج ────────────────────────────────────
        models: Dict[str, Tuple[float, float, float]] = {}
        models["dixon_coles"]    = DixonColes.predict(p.hxg, p.axg)
        models["elo"]            = self.data.elo.predict(h, a)
        models["form"]           = self._form(h, a)
        models["h2h"]            = self._h2h(hid, aid)
        models["home_advantage"] = self._hadv(h, a)
        models["fatigue"]        = Fatigue.predict(h, a, md)
        models["draw_model"]     = ImprovedDrawPredictor.predict(
            h, a, h2h_draw_rate, elo_diff, p.is_derby, is_late_season
        )

        if self.ml and self.ml.trained:
            mp = self.ml.predict(h, a, self.data, md, p.is_derby)
            if mp:
                models["ml"] = mp
                p.ml_used    = True
                p.ml_acc     = self.ml.acc

        p.models = models

        # ── دمج الأوزان ──────────────────────────────────────
        hp = dp = ap = tw = 0.0
        for nm, probs in models.items():
            w = self.w.get(nm, 0)
            if w > 0:
                hp += probs[0] * w
                dp += probs[1] * w
                ap += probs[2] * w
                tw += w

        if tw > 0:
            hp /= tw; dp /= tw; ap /= tw

        hp, dp, ap = normalize_probs(hp, dp, ap)

        p.raw_hp = hp
        p.raw_dp = dp
        p.raw_ap = ap

        # V6.1: تصحيح انحياز التعادل
        hp, dp, ap = self._apply_draw_correction(hp, dp, ap, h, a)

        # ── Calibration ───────────────────────────────────────
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

        # ── مصفوفة الأهداف ───────────────────────────────────
        mx    = DixonColes.matrix(p.hxg, p.axg)
        ss    = sorted(mx.items(), key=lambda x: x[1], reverse=True)
        p.top_sc = [(s[0][0], s[0][1], s[1]) for s in ss[:6]]

        p.btts = sum(pr for (hh, aa), pr in mx.items() if hh > 0 and aa > 0)
        p.o15  = sum(pr for (hh, aa), pr in mx.items() if hh + aa > 1)
        p.o25  = sum(pr for (hh, aa), pr in mx.items() if hh + aa > 2)
        p.o35  = sum(pr for (hh, aa), pr in mx.items() if hh + aa > 3)

        pd_map   = {"HOME": hp, "DRAW": dp, "AWAY": ap}
        p.result = max(pd_map, key=pd_map.get)
        p.conf   = max(pd_map.values()) * 100

        if p.top_sc:
            p.pred_sc = (p.top_sc[0][0], p.top_sc[0][1])

        if self.odds and self.odds.ok():
            od = self.odds.find(h.name, a.name)
            if od:
                p.odds          = od
                p.value_bets    = self._value(p, od)
                p.dc_value_bets = self._dc_value(p, od)

        return p

    # ── V6.1: تصحيح انحياز التعادل ───────────────────────────
    def _apply_draw_correction(
        self,
        hp: float, dp: float, ap: float,
        h: Team, a: Team,
    ) -> Tuple[float, float, float]:
        """
        إذا كان فارق hp/ap مقارنةً بـ dp صغيراً
        → يُرجَّح التعادل أكثر.
        """
        max_ha = max(hp, ap)
        gap    = max_ha - dp

        if gap < 0.12:                             # فارق صغير → boost
            boost = min(0.05, gap * 0.3)
            dp   += boost
            hp   -= boost * 0.6
            ap   -= boost * 0.4

        # كلا الفريقين draw-prone تاريخياً
        team_draw_avg = (h.dr + a.dr) / 2.0
        if team_draw_avg > 0.30:
            extra = min(0.03, (team_draw_avg - 0.30) * 0.20)
            dp   += extra
            hp   -= extra * 0.55
            ap   -= extra * 0.45

        return normalize_probs(hp, dp, ap)

    # ── V6.1: _dc_recommend() بدون تعارض ─────────────────────
    def _dc_recommend(self, p: Pred) -> str:
        """
        نظام تسجيل نقاط بدل الشروط المتعارضة.
        """
        scores: Dict[str, float] = {
            "1X": p.dc_1x,
            "X2": p.dc_x2,
            "12": p.dc_12,
        }
        reasons: Dict[str, str] = {
            "1X": "Home favored but draw possible",
            "X2": "Away has real chance + draw likely",
            "12": "Draw unlikely",
        }

        # تعديلات حسب السياق
        if 0.40 <= p.hp <= 0.60 and p.dp > 0.20:
            scores["1X"] += 0.05
        if 0.30 <= p.ap <= 0.50 and p.dp > 0.20:
            scores["X2"] += 0.05
        if p.dp < 0.20:
            scores["12"] += 0.05
        if p.is_derby and p.hp > p.ap:
            scores["1X"] += 0.03
            reasons["1X"] = f"{p.derby_name} - Home advantage"

        best     = max(scores, key=scores.get)
        pct_str  = f"{scores[best] * 100:.1f}%"
        return f"{best} ({pct_str}) - {reasons[best]}"

    # ── xG ────────────────────────────────────────────────────
    def _xg(self, t: Team, opp: Team, home: bool) -> float:
        ah = max(self.data.avg_h, 0.5)
        aa = max(self.data.avg_a, 0.5)
        if home:
            att  = safe_div(t.h_avg_gf, ah, 1.0)
            df   = safe_div(opp.a_avg_ga, aa, 1.0)
            base = ah
        else:
            att  = safe_div(t.a_avg_gf, aa, 1.0)
            df   = safe_div(opp.h_avg_ga, ah, 1.0)
            base = aa

        fa      = safe_div(t.goal_form, max(t.avg_gf, 0.5), 1.0)
        fa      = 0.7 + 0.3 * min(fa, 2.0)
        raw_xg  = att * df * base * fa
        # V6.1 FIX: حد أدنى 0.10 بدل 0.25
        return max(0.10, min(raw_xg, 4.0))

    # ── V6.1: _form() يُدرك التعادل عند تقارب الفورم ──────────
    def _form(self, h: Team, a: Team) -> Tuple[float, float, float]:
        hf = h.form_score
        af = a.form_score
        t  = hf + af
        if t == 0:
            return (0.40, 0.25, 0.35)

        hs  = (hf / t) * 1.08
        a_s =  af / t
        diff = abs(hs - a_s)

        # V6.1 FIX: فورم متقارب → احتمال تعادل أعلى
        if   diff < 0.08: d = 0.35
        elif diff < 0.15: d = 0.30
        elif diff < 0.25: d = 0.25
        else:             d = max(0.15, 0.30 - diff * 0.40)

        rem = 1.0 - d
        sm  = hs + a_s
        if sm <= 0:
            return (rem * 0.5, d, rem * 0.5)
        return normalize_probs(rem * hs / sm, d, rem * a_s / sm)

    # ── H2H ───────────────────────────────────────────────────
    def _h2h(self, hid: int, aid: int) -> Tuple[float, float, float]:
        default = (0.40, 0.25, 0.35)
        ms = self.data.get_h2h(hid, aid)
        if not ms:
            return default
        hw = dw = aw = 0
        for m in ms[-10:]:
            if m["home_goals"] > m["away_goals"]:
                hw += 1 if m["home_id"] == hid else 0
                aw += 1 if m["home_id"] != hid else 0
            elif m["home_goals"] < m["away_goals"]:
                aw += 1 if m["home_id"] == hid else 0
                hw += 1 if m["home_id"] != hid else 0
            else:
                dw += 1
        n = hw + dw + aw
        if n == 0:
            return default
        alpha = 1
        return (
            (hw + alpha) / (n + 3 * alpha),
            (dw + alpha) / (n + 3 * alpha),
            (aw + alpha) / (n + 3 * alpha),
        )

    # ── Home Advantage ────────────────────────────────────────
    def _hadv(self, h: Team, a: Team) -> Tuple[float, float, float]:
        hp = h.h_wr * 1.25
        ap = a.a_wr
        sm = hp + ap
        if sm > 0:
            hp /= sm
            ap /= sm
        d  = 0.25
        hp *= 0.75
        ap *= 0.75
        return normalize_probs(hp, d, ap)

    # ── Value Bets ────────────────────────────────────────────
    def _value(self, p: Pred, od: dict) -> List[dict]:
        vals = []
        for nm, mp, ip, odd in [
            ("Home", p.hp, od["implied_home"], od["odds_home"]),
            ("Draw", p.dp, od["implied_draw"], od["odds_draw"]),
            ("Away", p.ap, od["implied_away"], od["odds_away"]),
        ]:
            edge  = (mp - ip) * 100
            kelly = (mp * odd - 1) / (odd - 1) if mp > 0 and odd > 1 else 0.0
            vals.append({
                "market":   nm,
                "model":    float(mp  * 100),
                "implied":  float(ip  * 100),
                "odds":     float(odd),
                "edge":     float(edge),
                "kelly":    float(max(0.0, kelly) * 100),
                "is_value": edge > 3,
            })
        return vals

    def _dc_value(self, p: Pred, od: dict) -> List[dict]:
        vals = []
        for nm, model_p, implied_p, odds_val in [
            ("1X", p.dc_1x, od.get("implied_1x"), od.get("odds_1x")),
            ("X2", p.dc_x2, od.get("implied_x2"), od.get("odds_x2")),
            ("12", p.dc_12, od.get("implied_12"), od.get("odds_12")),
        ]:
            if implied_p is None or odds_val is None:
                continue
            edge  = (model_p - implied_p) * 100
            kelly = (model_p * odds_val - 1) / (odds_val - 1) \
                    if model_p > 0 and odds_val > 1 else 0.0
            vals.append({
                "market":   f"DC {nm}",
                "model":    float(model_p  * 100),
                "implied":  float(implied_p * 100),
                "odds":     float(odds_val),
                "edge":     float(edge),
                "kelly":    float(max(0.0, kelly) * 100),
                "is_value": edge > 3,
            })
        return vals
# ══════════════════════════════════════════════════════════════
# BACKTESTER — V6.1: Data Leakage Fixed + DC Accuracy Fixed
#                    Brier Score Fixed + Confusion Matrix
# ══════════════════════════════════════════════════════════════
class Backtester:
    def __init__(self):
        self.results: dict = {}
        self.cal = Calibrator()

    def analyze_errors(self, preds: List[dict]) -> dict:
        """تحليل أنماط الأخطاء وطباعة Confusion Matrix"""
        errors = {
            "predicted_home_was_draw":  0,
            "predicted_home_was_away":  0,
            "predicted_draw_was_home":  0,
            "predicted_draw_was_away":  0,
            "predicted_away_was_home":  0,
            "predicted_away_was_draw":  0,
        }
        confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for p in preds:
            pred   = p["predicted"]
            actual = p["actual"]
            confusion[pred][actual] += 1
            if pred != actual:
                key = f"predicted_{pred.lower()}_was_{actual.lower()}"
                if key in errors:
                    errors[key] += 1

        print("\n ═══════ Confusion Matrix ═══════")
        print(f" {'Pred/Actual':>12} {'HOME':>8} {'DRAW':>8} {'AWAY':>8}")
        print(f" {'─' * 40}")
        for pred_label in ["HOME", "DRAW", "AWAY"]:
            row = f" {pred_label:>12}"
            for actual_label in ["HOME", "DRAW", "AWAY"]:
                val  = confusion[pred_label][actual_label]
                mark = "◼" if pred_label == actual_label else " "
                row += f" {mark}{val:>7}"
            print(row)
        print(f" {'─' * 40}")

        total_errors = sum(errors.values())
        print("\n ═══════ Error Analysis ═══════")
        for k, v in sorted(errors.items(), key=lambda x: -x[1]):
            pct = v / total_errors * 100 if total_errors > 0 else 0
            print(f" {k:<45}: {v:>4} ({pct:.1f}%)")

        home_pred  = sum(confusion["HOME"].values())
        draw_pred  = sum(confusion["DRAW"].values())
        away_pred  = sum(confusion["AWAY"].values())
        total_pred = home_pred + draw_pred + away_pred
        if total_pred > 0:
            print(f"\n ═══════ Prediction Bias ═══════")
            print(f" HOME predicted: {home_pred:>4} ({home_pred / total_pred * 100:.1f}%)")
            print(f" DRAW predicted: {draw_pred:>4} ({draw_pred / total_pred * 100:.1f}%)")
            print(f" AWAY predicted: {away_pred:>4} ({away_pred / total_pred * 100:.1f}%)")

        return {
            "confusion": {k: dict(v) for k, v in confusion.items()},
            "errors":    errors,
        }

    def run(
        self,
        matches:   List[dict],
        resources: "LeagueResources",
        split:     float = BACKTEST_SPLIT,
    ) -> dict:
        fin = [m for m in matches if m.get("status") == "FINISHED"]
        fin.sort(key=lambda m: m.get("utcDate", ""))

        si    = int(len(fin) * split)
        train = fin[:si]
        test  = fin[si:]

        if len(train) < 30 or len(test) < 10:
            return {"error": "Not enough data"}

        # ══════════════════════════════════════════════════════
        # V6.1 FIX: ثلاث نسخ منفصلة من DataProc لمنع Data Leakage
        # ══════════════════════════════════════════════════════

        # ── النموذج يُدرَّب على train فقط ────────────────────
        td_train = DataProc(resources)
        td_train.process(train)

        ml = None
        if ML_AVAILABLE:
            ml = MLPred(model_file=resources.model_file)
            ml.train(td_train, force_retrain=True)

        # ── تقسيم مجموعة الاختبار إلى نصفين ─────────────────
        cs      = len(test) // 2
        cal_set  = test[:cs]
        eval_set = test[cs:]

        # ══════════════════════════════════════════════════════
        # المرحلة الأولى: Calibration
        # td_cal يبدأ من train ويُضاف إليه cal_set تدريجياً
        # ══════════════════════════════════════════════════════
        td_cal = DataProc(resources)
        td_cal.process(train)
        eng_cal = Engine(td_cal, resources, ml)

        cal_total = 0
        for m in cal_set:
            ht  = m.get("homeTeam", {})
            at  = m.get("awayTeam", {})
            ft  = m.get("score", {}).get("fullTime", {})
            hid = ht.get("id")
            aid = at.get("id")
            ahg = ft.get("home")
            aag = ft.get("away")
            if not hid or not aid or ahg is None or aag is None:
                continue

            pr = eng_cal.predict(hid, aid, m.get("utcDate", ""))
            if not pr:
                td_cal.process([m])
                continue

            ahg = int(ahg)
            aag = int(aag)
            actual = "HOME" if ahg > aag else ("AWAY" if ahg < aag else "DRAW")

            self.cal.add((pr.hp, pr.dp, pr.ap), actual)
            cal_total += 1

            # تحديث td_cal بعد التنبؤ (لا قبله) ✅
            td_cal.process([m])

        cal_ok = self.cal.calibrate()

        # ══════════════════════════════════════════════════════
        # المرحلة الثانية: Evaluation
        # td_eval يبدأ من train + cal_set كاملاً
        # V6.1 FIX: نسخة جديدة نظيفة تمنع data leakage
        # ══════════════════════════════════════════════════════
        td_eval = DataProc(resources)
        td_eval.process(train + cal_set)   # كل البيانات السابقة للـ eval

        eng_eval = Engine(
            td_eval, resources, ml,
            cal=self.cal if cal_ok else None,
        )

        # ── عدّادات الأداء ────────────────────────────────────
        eval_correct    = 0
        eval_correct_sc = 0
        eval_total      = 0

        home_correct = home_total = 0
        draw_correct = draw_total = 0
        away_correct = away_total = 0

        hi_correct = hi_total = 0
        me_correct = me_total = 0
        lo_correct = lo_total = 0

        # V6.1 FIX: DC Accuracy — يعتمد على توقع النموذج لا على حدوث الحدث
        DC_THRESHOLD = 0.60
        dc_1x_correct = dc_1x_total = 0
        dc_x2_correct = dc_x2_total = 0
        dc_12_correct = dc_12_total = 0

        preds: List[dict] = []

        for m in eval_set:
            ht  = m.get("homeTeam", {})
            at  = m.get("awayTeam", {})
            ft  = m.get("score", {}).get("fullTime", {})
            hid = ht.get("id")
            aid = at.get("id")
            ahg = ft.get("home")
            aag = ft.get("away")
            if not hid or not aid or ahg is None or aag is None:
                continue

            hn = ht.get("shortName") or ht.get("name", "")
            an = at.get("shortName") or at.get("name", "")

            pr = eng_eval.predict(hid, aid, m.get("utcDate", ""))
            if not pr:
                td_eval.process([m])
                continue

            ahg    = int(ahg)
            aag    = int(aag)
            actual = "HOME" if ahg > aag else ("AWAY" if ahg < aag else "DRAW")

            eval_total += 1
            is_correct  = pr.result == actual
            if is_correct:
                eval_correct += 1

            if pr.pred_sc[0] == ahg and pr.pred_sc[1] == aag:
                eval_correct_sc += 1

            # ── دقة حسب النتيجة ───────────────────────────────
            if actual == "HOME":
                home_total += 1
                if is_correct: home_correct += 1
            elif actual == "DRAW":
                draw_total += 1
                if is_correct: draw_correct += 1
            else:
                away_total += 1
                if is_correct: away_correct += 1

            # ── دقة حسب الثقة ─────────────────────────────────
            conf = pr.conf
            if conf > 60:
                hi_total += 1
                if is_correct: hi_correct += 1
            elif conf >= 45:
                me_total += 1
                if is_correct: me_correct += 1
            else:
                lo_total += 1
                if is_correct: lo_correct += 1

            # ── V6.1 FIX: DC Accuracy بناءً على توقع النموذج ─
            # 1X: نتوقع HOME أو DRAW
            if pr.dc_1x > DC_THRESHOLD:
                dc_1x_total += 1
                if actual in ("HOME", "DRAW"):
                    dc_1x_correct += 1

            # X2: نتوقع AWAY أو DRAW
            if pr.dc_x2 > DC_THRESHOLD:
                dc_x2_total += 1
                if actual in ("AWAY", "DRAW"):
                    dc_x2_correct += 1

            # 12: نتوقع HOME أو AWAY (لا تعادل)
            if pr.dc_12 > DC_THRESHOLD:
                dc_12_total += 1
                if actual in ("HOME", "AWAY"):
                    dc_12_correct += 1

            preds.append({
                "home":        hn,
                "away":        an,
                "predicted":   pr.result,
                "actual":      actual,
                "pred_score":  pr.pred_sc,
                "actual_score":(ahg, aag),
                "confidence":  float(pr.conf),
                "correct":     is_correct,
                "probs":       (float(pr.hp), float(pr.dp), float(pr.ap)),
                "dc_1x":       float(pr.dc_1x),
                "dc_x2":       float(pr.dc_x2),
                "dc_12":       float(pr.dc_12),
                "calibrated":  pr.calibrated,
            })

            # تحديث td_eval بعد التنبؤ (لا قبله) ✅
            td_eval.process([m])

        if eval_total == 0:
            return {"error": "No evaluation matches found"}

        # ══════════════════════════════════════════════════════
        # V6.1 FIX: Brier Score — التقسيم على N فقط [0,2]
        # ثم على 2 للتطبيع إلى [0,1] (Multi-class Brier)
        # ══════════════════════════════════════════════════════
        brier_raw = 0.0
        for pred_item in preds:
            actual_vec    = [0, 0, 0]
            outcome_index = ["HOME", "DRAW", "AWAY"].index(pred_item["actual"])
            actual_vec[outcome_index] = 1
            for i in range(3):
                brier_raw += (pred_item["probs"][i] - actual_vec[i]) ** 2

        # Multi-class Brier مُطبَّع [0,1]:  BS = Σ / (N * 2)
        brier = brier_raw / (eval_total * 2) if eval_total > 0 else 0.0

        # ── احتساب النسب ──────────────────────────────────────
        result_acc = eval_correct    / eval_total * 100
        score_acc  = eval_correct_sc / eval_total * 100

        home_acc = home_correct / home_total * 100 if home_total > 0 else 0.0
        draw_acc = draw_correct / draw_total * 100 if draw_total > 0 else 0.0
        away_acc = away_correct / away_total * 100 if away_total > 0 else 0.0

        hi_acc   = hi_correct / hi_total * 100 if hi_total > 0 else 0.0
        me_acc   = me_correct / me_total * 100 if me_total > 0 else 0.0
        lo_acc   = lo_correct / lo_total * 100 if lo_total > 0 else 0.0

        dc_1x_acc = dc_1x_correct / dc_1x_total * 100 if dc_1x_total > 0 else 0.0
        dc_x2_acc = dc_x2_correct / dc_x2_total * 100 if dc_x2_total > 0 else 0.0
        dc_12_acc = dc_12_correct / dc_12_total * 100 if dc_12_total > 0 else 0.0

        ml_acc_val = float(ml.acc * 100) if ml and ml.trained else 0.0

        # ── Confusion Matrix ──────────────────────────────────
        print()
        error_analysis = self.analyze_errors(preds)

        # ── حفظ Elo من td_eval (الأحدث) ──────────────────────
        elo_data = {t.name: t.elo for t in td_eval.teams.values()}
        resources.save_elo(elo_data)

        self.results = {
            "total":        eval_total,
            "train":        len(train),
            "test":         len(test),
            "cal_size":     cal_total,
            "eval_size":    eval_total,
            "result_acc":   result_acc,
            "score_acc":    score_acc,
            "brier":        float(brier),
            "correct":      eval_correct,
            "correct_sc":   eval_correct_sc,
            "home_acc":     home_acc,
            "draw_acc":     draw_acc,
            "away_acc":     away_acc,
            "home_total":   home_total,
            "draw_total":   draw_total,
            "away_total":   away_total,
            "hi_acc":       hi_acc,
            "me_acc":       me_acc,
            "lo_acc":       lo_acc,
            "hi_n":         hi_total,
            "me_n":         me_total,
            "lo_n":         lo_total,
            "dc_1x_acc":    dc_1x_acc,
            "dc_x2_acc":    dc_x2_acc,
            "dc_12_acc":    dc_12_acc,
            "dc_1x_n":      dc_1x_total,
            "dc_x2_n":      dc_x2_total,
            "dc_12_n":      dc_12_total,
            "dc_threshold": DC_THRESHOLD,
            "ml_acc":       ml_acc_val,
            "cal_used":     cal_ok,
            "confusion":    error_analysis.get("confusion", {}),
            "error_analysis": error_analysis.get("errors", {}),
            "predictions":  preds,
        }
        return self.results


# ══════════════════════════════════════════════════════════════
# LEAGUE APP — V6.1: unique_matches key مُصلح
# ══════════════════════════════════════════════════════════════
class LeagueApp:
    def __init__(
        self,
        league_code: str,
        api_token:   str,
        odds_key:    str = "",
    ):
        self.code = league_code
        if league_code not in LEAGUES_CONFIG:
            raise ValueError(
                f"League '{league_code}' not found in {LEAGUES_CONFIG_FILE}"
            )
        self.resources = LeagueResources(league_code, LEAGUES_CONFIG[league_code])
        self.api       = FootballAPI(api_token, self.resources.api_url)
        self.data      = DataProc(self.resources)
        self.ml        = MLPred(model_file=self.resources.model_file)
        _sport_map = {
            "PL": "soccer_epl",
            "LL": "soccer_spain_la_liga",
            "SA": "soccer_italy_serie_a",
            "BL": "soccer_germany_bundesliga",
            "L1": "soccer_france_ligue_one",
        }
        self.odds      = OddsAPI(odds_key, _sport_map.get(league_code, "soccer_epl"))
        self.cal       = Calibrator()
        self.bt        = Backtester()
        self.eng:      Optional[Engine] = None
        self.raw:      List[dict]       = []
        self.last_preds: List[Pred]     = []
        self._log:     List[Tuple[str, str]] = []
        self.sy:       Optional[int]    = None

    def _log_msg(self, level: str, msg: str):
        full_msg = f"[{self.code}] {msg}"
        self._log.append((level, full_msg))
        print(f">>> {full_msg}", flush=True)

    def init(self) -> bool:
        self.raw = []

        # ── CSV ──────────────────────────────────────────────
        self._log_msg("progress", f"Loading CSV data ({self.resources.name})…")
        csv_matches = self.resources.load_csv_data()
        if csv_matches:
            self.raw.extend(csv_matches)
            self._log_msg("success", f"CSV: {len(csv_matches)} matches with deep stats")
        else:
            self._log_msg("info", "No CSV data found, using API only")

        # ── API ──────────────────────────────────────────────
        self.sy = self.api.season_year(self.resources.api_code)
        if self.sy:
            self._log_msg("progress", f"Loading API season {self.sy}…")
            api_matches = self.api.finished(self.resources.api_code, self.sy)
            if api_matches:
                self.raw.extend(api_matches)
                self._log_msg("success", f"API: {len(api_matches)} matches loaded")

        # ── V6.1 FIX: مفتاح التكرار يشمل home + away + date ─
        unique_matches: Dict[str, dict] = {}
        for m in self.raw:
            date_key = m.get("utcDate", "")[:10]
            home_id  = str(m["homeTeam"].get("id", ""))
            away_id  = str(m["awayTeam"].get("id", ""))
            key      = f"{date_key}_{home_id}_{away_id}"   # ← V6.1 FIX
            # CSV (مع stats) يأخذ الأولوية على API (بدون stats)
            if key not in unique_matches or "stats" in m:
                unique_matches[key] = m

        self.raw = sorted(
            unique_matches.values(),
            key=lambda x: x.get("utcDate", ""),
        )

        if not self.raw:
            self._log_msg("error", "No data found!")
            return False

        self._log_msg("success", f"Total: {len(self.raw)} unique matches")

        # ── معالجة ───────────────────────────────────────────
        self._log_msg("progress", "Processing data + Elo + Deep Stats…")
        self.data.process(self.raw)
        self._log_msg("success", f"Teams processed: {len(self.data.teams)}")

        # ── تدريب ML ─────────────────────────────────────────
        if ML_AVAILABLE:
            model_type = (
                "Stacking(RF+XGB+LR)" if XGBOOST_AVAILABLE else "Stacking(RF+LR)"
            )
            self._log_msg(
                "progress",
                f"Training {model_type} | {N_FEATURES} features…",
            )
            if self.ml.train(self.data):
                src     = "loaded" if self.ml._external else "trained"
                acc_str = f"{self.ml.acc * 100:.1f}%"
                self._log_msg("success", f"ML model {src} | CV Acc: {acc_str}")
                if not self.ml._external:
                    self.ml.save_pipeline()
            else:
                self._log_msg("info", "ML training skipped (not enough data)")

        # ── Calibration ──────────────────────────────────────
        if self.cal.load(self.resources.calibration_file):
            self._log_msg("success", "Calibration loaded")

        # ── Odds ─────────────────────────────────────────────
        if self.odds.ok():
            self._log_msg("progress", f"Fetching live odds ({self.odds.sport})...")
            odds_res = self.odds.fetch()
            if odds_res:
                self._log_msg("success", f"Live odds loaded for {len(odds_res)} matches")
            else:
                self._log_msg("info", "No live odds available right now (empty/failed response)")

        # ── Engine ───────────────────────────────────────────
        self.eng = Engine(
            self.data, self.resources, self.ml, self.odds, self.cal
        )
        self._log_msg(
            "success",
            f"Engine Ready for {self.resources.name}! (v6.1 | 126 features)",
        )
        return True

    def predict_upcoming(self, days: int = 14) -> List[Pred]:
        upcoming = self.api.upcoming(self.resources.api_code, days)
        if not upcoming:
            self._log_msg("info", "No upcoming matches from API")
            return []

        preds: List[Pred] = []
        for m in upcoming:
            hid = m.get("homeTeam", {}).get("id")
            aid = m.get("awayTeam", {}).get("id")
            if hid and aid:
                pr = self.eng.predict(hid, aid, m.get("utcDate", ""))
                if pr:
                    preds.append(pr)

        self.last_preds = preds
        return preds

    def predict_custom(self, home_name: str, away_name: str) -> Optional[Pred]:
        ht = self.data.team_by_name(home_name)
        at = self.data.team_by_name(away_name)
        if not ht:
            self._log_msg("error", f"Team not found: {home_name}")
            return None
        if not at:
            self._log_msg("error", f"Team not found: {away_name}")
            return None
        pr = self.eng.predict(ht.id, at.id, "Custom")
        if pr:
            self.last_preds = [pr]
        return pr

    def run_backtest(self) -> dict:
        r = self.bt.run(self.raw, self.resources)
        if r.get("cal_used"):
            self.cal = self.bt.cal
            self.cal.save(self.resources.calibration_file)
            # إعادة بناء Engine مع Calibration المحدّثة
            self.eng = Engine(
                self.data, self.resources, self.ml, self.odds, self.cal
            )
        return r

    def standings(self) -> List[Team]:
        return sorted(self.data.teams.values(), key=lambda t: t.pos)

    def export_predictions(
        self,
        preds:    List[Pred] = None,
        filename: str        = None,
    ) -> str:
        preds    = preds    or self.last_preds
        filename = filename or (
            f"predictions_{self.code}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        )
        out = []
        for p in preds:
            out.append({
                "league":     p.league,
                "home":       p.home,
                "away":       p.away,
                "date":       p.date,
                "prediction": p.result,
                "score":      f"{p.pred_sc[0]}-{p.pred_sc[1]}",
                "confidence": round(float(p.conf),  1),
                "calibrated": p.calibrated,
                "derby":      p.derby_name if p.is_derby else None,
                "probabilities": {
                    "home": round(float(p.hp * 100), 1),
                    "draw": round(float(p.dp * 100), 1),
                    "away": round(float(p.ap * 100), 1),
                },
                "double_chance": {
                    "1X":            round(float(p.dc_1x * 100), 1),
                    "X2":            round(float(p.dc_x2 * 100), 1),
                    "12":            round(float(p.dc_12 * 100), 1),
                    "recommendation": p.dc_recommend,
                },
                "xg": {
                    "home":  round(p.hxg, 2),
                    "away":  round(p.axg, 2),
                    "total": round(p.hxg + p.axg, 2),
                },
                "market": {
                    "btts":      round(p.btts  * 100, 1),
                    "over_1_5":  round(p.o15   * 100, 1),
                    "over_2_5":  round(p.o25   * 100, 1),
                    "over_3_5":  round(p.o35   * 100, 1),
                },
            })
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        return filename


# ══════════════════════════════════════════════════════════════
# CLI DISPLAY
# ══════════════════════════════════════════════════════════════
class Disp:
    @staticmethod
    def header():
        print()
        print(C.cyan(" ╔══════════════════════════════════════════════════════════════════╗"))
        print(C.cyan(" ║") + C.bold(" ⚽  FOOTBALL PREDICTOR PRO v6.1 (ULTIMATE EDITION) ⚽  ") + C.cyan("║"))
        print(C.cyan(" ║") + C.dim("  126 Features • Stacking ML • Draw-Aware • Leakage-Free BT  ") + C.cyan("║"))
        print(C.cyan(" ║") + C.dim("  Multi-League • CSV Deep Stats • Per-League Models • v6.1  ") + C.cyan("║"))
        print(C.cyan(" ╚══════════════════════════════════════════════════════════════════╝"))
        print()

    @staticmethod
    def section(t: str):
        print(f"\n {C.yellow(C.bold('══ ' + t + ' ══'))}\n")

    @staticmethod
    def leagues_menu(available: List[str]):
        Disp.section("Available Leagues")
        for i, code in enumerate(available, 1):
            cfg         = LEAGUES_CONFIG.get(code, {})
            name        = cfg.get("name", code)
            country     = cfg.get("country", "")
            data_files  = cfg.get("data_files", [])
            file_status = "✅" if any(Path(f).exists() for f in data_files) else "⚠️"
            print(f" {C.cyan(str(i))}. {C.bold(code)} - {name} ({country}) {file_status}")
        print()

    @staticmethod
    def pred_card(p: Pred):
        w = 68
        print(f"\n {C.blue('┌' + '─' * w + '┐')}")

        if p.league:
            cfg         = LEAGUES_CONFIG.get(p.league, {})
            league_name = cfg.get("name", p.league)
            print(box(f" 🏆 {C.magenta(league_name)}"))

        if p.is_derby:
            print(box(f" 🔥 {C.magenta(C.bold(p.derby_name))}"))

        print(box(
            f" {C.bold(C.green('🏠 ' + p.home))} "
            f"{C.dim('vs')} "
            f"{C.bold(C.red('✈️  ' + p.away))}"
        ))

        if p.date and p.date != "Custom":
            dt = parse_date(p.date)
            ds = dt.strftime("%a %d %b %Y • %H:%M") if dt else p.date[:16]
            print(box(f" 📅 {ds}"))

        print(f" {C.blue('├' + '─' * w + '┤')}")
        print(box(f" {C.bold('📊 PROBABILITIES')}"))
        print(box(f" 🏠 Home: {C.green(f'{p.hp * 100:5.1f}%')} {C.pct_bar(p.hp, 25, C.G)}"))
        print(box(f" 🤝 Draw: {C.yellow(f'{p.dp * 100:5.1f}%')} {C.pct_bar(p.dp, 25, C.Y)}"))
        print(box(f" ✈️  Away: {C.red(f'{p.ap * 100:5.1f}%')} {C.pct_bar(p.ap, 25, C.R)}"))

        print(f" {C.blue('├' + '─' * w + '┤')}")
        print(box(
            f" ⚡ xG: {p.home}: {C.bold(f'{p.hxg:.2f}')} | "
            f"{p.away}: {C.bold(f'{p.axg:.2f}')} | "
            f"Total: {C.bold(f'{p.hxg + p.axg:.2f}')}"
        ))
        print(box(
            f" 🎯 Predicted: {C.bold(p.result)} | "
            f"Score: {p.pred_sc[0]}-{p.pred_sc[1]} | "
            f"Conf: {p.conf:.1f}%"
        ))

        # V6.1: label صحيح CV Acc
        cal_str = " ✅ Calibrated" if p.calibrated else ""
        acc_lbl = f"CV Acc: {p.ml_acc * 100:.1f}%" if p.ml_used else "N/A"
        print(box(f" 🤖 ML: {'✅' if p.ml_used else '❌'} | {acc_lbl}{cal_str}"))

        print(f" {C.blue('├' + '─' * w + '┤')}")
        print(box(
            f" 🛡️  DC: 1X={p.dc_1x * 100:.1f}% | "
            f"12={p.dc_12 * 100:.1f}% | "
            f"X2={p.dc_x2 * 100:.1f}%"
        ))
        print(box(f" 💡 {C.bold(p.dc_recommend)}"))

        print(f" {C.blue('├' + '─' * w + '┤')}")
        print(box(f" {C.bold('🎯 TOP SCORES')}"))
        for i, (hg, ag, pr2) in enumerate(p.top_sc[:5]):
            mk = "👉" if i == 0 else "  "
            print(box(f" {mk} {hg}-{ag} ({pr2 * 100:.1f}%)"))

        print(f" {C.blue('├' + '─' * w + '┤')}")
        print(box(
            f" 📈 Markets: BTTS={p.btts * 100:.1f}% | "
            f"O1.5={p.o15 * 100:.1f}% | "
            f"O2.5={p.o25 * 100:.1f}% | "
            f"O3.5={p.o35 * 100:.1f}%"
        ))
        print(f" {C.blue('└' + '─' * w + '┘')}")

    @staticmethod
    def backtest_summary(r: dict, league_code: str = ""):
        Disp.section(f"Backtest Results [{league_code}] — v6.1")
        if "error" in r:
            print(f" {C.red('✖')} {r['error']}")
            return

        ra  = r["result_acc"]
        rac = C.green if ra > 52 else (C.yellow if ra > 47 else C.red)
        print(f" 📊 1X2 Accuracy  : {rac(f'{ra:.1f}%')} ({r['correct']}/{r['total']})")
        print(f" ⚽ Exact Score   : {r['score_acc']:.1f}%")

        bs  = r["brier"]
        # V6.1: Brier مُطبَّع [0,1] → عتبات أصغر
        bsc = C.green if bs < 0.08 else (C.yellow if bs < 0.12 else C.red)
        print(f" 📐 Brier Score   : {bsc(f'{bs:.4f}')}  [0=perfect, 1=worst]")

        if r.get("ml_acc", 0) > 0:
            print(f" 🤖 ML CV Bal.Acc : {C.green(f'{r["ml_acc"]:.1f}%')}")

        print()
        print(f" 🏠 Home Win Acc  : {r.get('home_acc', 0):.1f}%  ({r.get('home_total', 0)} matches)")
        print(f" 🤝 Draw Acc      : {r.get('draw_acc', 0):.1f}%  ({r.get('draw_total', 0)} matches)")
        print(f" ✈️  Away Win Acc  : {r.get('away_acc', 0):.1f}%  ({r.get('away_total', 0)} matches)")

        print()
        print(f" 🔥 High Conf >60%   : {C.green(f'{r.get("hi_acc", 0):.1f}%')}  ({r.get('hi_n', 0)} matches)")
        print(f" ⚡ Med  Conf 45-60% : {C.yellow(f'{r.get("me_acc", 0):.1f}%')}  ({r.get('me_n', 0)} matches)")
        print(f" ⚠️  Low  Conf <45%  : {C.red(f'{r.get("lo_acc", 0):.1f}%')}  ({r.get('lo_n', 0)} matches)")

        thr = r.get("dc_threshold", 0.60)
        print()
        print(f" 🛡️  DC 1X Acc (>{thr:.0%}): {r.get('dc_1x_acc', 0):.1f}%  ({r.get('dc_1x_n', 0)} bets)")
        print(f" 🛡️  DC X2 Acc (>{thr:.0%}): {r.get('dc_x2_acc', 0):.1f}%  ({r.get('dc_x2_n', 0)} bets)")
        print(f" 🛡️  DC 12 Acc (>{thr:.0%}): {r.get('dc_12_acc', 0):.1f}%  ({r.get('dc_12_n', 0)} bets)")

        print()
        print(f" ✅ Calibrated    : {'Yes' if r.get('cal_used') else 'No'}")
        print(f" 📋 Train Matches : {r.get('train', 0)}")
        print(f" 🧪 Cal Set       : {r.get('cal_size', 0)}")
        print(f" 🔬 Eval Set      : {r.get('eval_size', 0)}")


# ══════════════════════════════════════════════════════════════
# STREAMLIT UI — V6.1: Memory leak fix + Brier label fix
# ══════════════════════════════════════════════════════════════
def run_streamlit():
    st.set_page_config(
        page_title="Football Predictor Pro v6.1",
        page_icon="⚽",
        layout="wide",
    )
    st.markdown(
        "<h1 style='text-align:center;color:#00ff9d;'>"
        "⚽ Football Predictor Pro v6.1</h1>",
        unsafe_allow_html=True,
    )
    st.caption(
        "126 Features • Stacking ML (RF+XGB+LR) • Draw-Aware Engine • "
        "Data-Leakage-Free Backtest • Confusion Matrix"
    )

    st.sidebar.title("⚙️ Settings")
    fb_key = st.sidebar.text_input(
        "🔑 Football-Data API Key",
        type="password",
        value=os.environ.get("FOOTBALL_DATA_KEY", ""),
    )
    odds_key = st.sidebar.text_input(
        "🎰 Odds API Key (optional)",
        type="password",
        value=os.environ.get("ODDS_API_KEY", ""),
    )

    available = list(LEAGUES_CONFIG.keys())
    if not available:
        st.warning("⚠️ No leagues configured!")
        st.stop()

    selected_league = st.sidebar.selectbox(
        "🏆 Select League",
        available,
        format_func=lambda c: f"{c} - {LEAGUES_CONFIG.get(c, {}).get('name', c)}",
    )
    league_cfg = LEAGUES_CONFIG.get(selected_league, {})
    data_files = league_cfg.get("data_files", [])
    st.sidebar.markdown("**📁 Data Files:**")
    for f in data_files:
        exists = Path(f).exists()
        st.sidebar.caption(f"{'✅' if exists else '❌'} {f}")

    init_key = (
        f"app_{selected_league}_{fb_key[:8] if fb_key else 'nokey'}"
    )

    if st.sidebar.button(f"🚀 Load {selected_league}") and fb_key:
        # V6.1 FIX: تنظيف نسخ قديمة من session_state
        old_keys = [
            k for k in st.session_state
            if k.startswith("app_") and k != init_key
        ]
        for k in old_keys:
            del st.session_state[k]

        with st.spinner(f"Loading {selected_league}…"):
            try:
                app = LeagueApp(selected_league, fb_key, odds_key)
                if app.init():
                    st.session_state[init_key] = app
                    st.session_state["active_league"] = selected_league
                    st.sidebar.success(f"✅ {selected_league} Ready!")
                else:
                    st.sidebar.error("❌ Init failed")
            except Exception as e:
                st.sidebar.error(f"❌ {e}")

    if init_key not in st.session_state:
        st.info(f"👈 Enter API key and load {selected_league}")
        with st.expander("📖 Setup Guide — v6.1"):
            st.markdown(f"""
            ### v6.1 Fixes
            - **Data Leakage Fixed**: Separate DataProc for train / cal / eval
            - **DC Accuracy Fixed**: Based on model prediction (threshold {60}%), not event occurrence
            - **Brier Score Fixed**: Normalized Multi-class [0, 1]
            - **Draw Correction**: Automatic bias adjustment
            - **_form() Draw-Aware**: Higher draw prob when form is close
            - **xG bounds**: min 0.10 (was 0.25)
            - **Entry Point**: Streamlit/CLI properly separated

            ### File Structure
            ```
            leagues_config.json
            data/{selected_league}_Master.csv
            config/{selected_league}_aliases.json
            models/   ← auto-generated
            ```
            ### CSV Required Columns
            `Date, HomeTeam, AwayTeam, FTHG, FTAG`
            Optional: `HST, AST, HC, AC, HF, AF, HY, AY, HR, AR`
            """)
        st.stop()

    app: LeagueApp = st.session_state[init_key]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🏆 League",   app.resources.name)
    col2.metric("📊 Matches",  app.data.total)
    col3.metric("👥 Teams",    len(app.data.teams))
    col4.metric("🤖 ML Ready", "✅" if app.ml.trained else "❌")
    col5.metric("🎯 Features", "126")

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Predictions", "⚽ Custom Match", "📊 Standings", "🔬 Backtest",
    ])

    # ── helper: render_pred_card ──────────────────────────────
    def render_pred_card(pr: Pred):
        title    = f"{'🔥' if pr.is_derby else '⚽'} {pr.home} vs {pr.away}"
        subtitle = f"{pr.hp*100:.1f}% / {pr.dp*100:.1f}% / {pr.ap*100:.1f}%"
        with st.expander(f"{title} | {subtitle}", expanded=True):

            if pr.is_derby:
                st.markdown(f"🔥 **{pr.derby_name}**")
            if pr.calibrated:
                st.caption("✅ Calibrated probabilities")
            if pr.ml_used:
                st.caption(f"🤖 ML Stacking | CV Acc: {pr.ml_acc*100:.1f}%")

            c1, c2, c3 = st.columns(3)
            c1.metric("🏠 Home Win", f"{pr.hp*100:.1f}%")
            c2.metric("🤝 Draw",     f"{pr.dp*100:.1f}%")
            c3.metric("✈️ Away Win",  f"{pr.ap*100:.1f}%")

            st.progress(min(1.0, float(pr.hp)), text=f"Home {pr.home}")
            st.progress(min(1.0, float(pr.dp)), text="Draw")
            st.progress(min(1.0, float(pr.ap)), text=f"Away {pr.away}")

            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("⚡ xG Home",  f"{pr.hxg:.2f}")
            c2.metric("⚡ xG Away",  f"{pr.axg:.2f}")
            c3.metric("🏆 Elo Home", f"{pr.h_elo:.0f}")
            c4.metric("🏆 Elo Away", f"{pr.a_elo:.0f}")

            st.markdown("#### 🛡️ Double Chance")
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("1X (Home/Draw)", f"{pr.dc_1x*100:.1f}%")
            cc2.metric("12 (No Draw)",   f"{pr.dc_12*100:.1f}%")
            cc3.metric("X2 (Away/Draw)", f"{pr.dc_x2*100:.1f}%")
            st.success(f"💡 **Recommendation:** {pr.dc_recommend}")

            st.markdown("#### 📈 Markets")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("BTTS",    f"{pr.btts*100:.1f}%")
            m2.metric("Over 1.5",f"{pr.o15*100:.1f}%")
            m3.metric("Over 2.5",f"{pr.o25*100:.1f}%")
            m4.metric("Over 3.5",f"{pr.o35*100:.1f}%")

            st.markdown("#### 🎯 Likely Scores")
            scores_html = " &nbsp; ".join([
                f"<span style='padding:4px 12px;background:#1a1a2e;"
                f"color:#00ff9d;border-radius:5px;font-weight:bold;'>"
                f"{hg}-{ag} ({pr2*100:.1f}%)</span>"
                for hg, ag, pr2 in pr.top_sc[:5]
            ])
            st.markdown(scores_html, unsafe_allow_html=True)

            if pr.value_bets:
                vb = [v for v in pr.value_bets if v["is_value"]]
                if vb:
                    st.markdown("#### 💰 Value Bets")
                    for v in vb:
                        st.success(
                            f"**{v['market']}** | Odds: {v['odds']} | "
                            f"Model: {v['model']:.1f}% vs "
                            f"Implied: {v['implied']:.1f}% | "
                            f"Edge: +{v['edge']:.1f}%"
                        )

    # ── Tab 1: Upcoming ───────────────────────────────────────
    with tab1:
        days = st.slider("Days ahead", 1, 30, 14)
        if st.button("🔍 Get Upcoming Predictions"):
            with st.spinner("Predicting…"):
                preds = app.predict_upcoming(days)
                st.session_state[f"preds_{selected_league}"] = preds

        preds_key = f"preds_{selected_league}"
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
            home = c1.selectbox(
                "🏠 Home Team", teams_list,
                key=f"home_{selected_league}",
            )
            away = c2.selectbox(
                "✈️ Away Team", teams_list,
                index=min(1, len(teams_list) - 1),
                key=f"away_{selected_league}",
            )
            if st.button("🔮 Predict"):
                if home == away:
                    st.error("Please select different teams!")
                else:
                    with st.spinner("Predicting…"):
                        pr = app.predict_custom(home, away)
                        if pr:
                            st.session_state[f"custom_{selected_league}"] = pr
                        else:
                            st.error(
                                "Prediction failed — not enough data for these teams"
                            )

        custom_key = f"custom_{selected_league}"
        if custom_key in st.session_state:
            render_pred_card(st.session_state[custom_key])

    # ── Tab 3: Standings ──────────────────────────────────────
    with tab3:
        teams = app.standings()
        if teams:
            rows = []
            for t in teams:
                rows.append({
                    "Pos":        t.pos,
                    "Team":       t.name,
                    "P":          t.played,
                    "W":          t.wins,
                    "D":          t.draws,
                    "L":          t.losses,
                    "GF":         t.gf,
                    "GA":         t.ga,
                    "GD":         t.gd,
                    "Pts":        t.pts,
                    "Elo":        round(t.elo),
                    "PPG":        round(t.ppg,         2),
                    "Form":       t.form_string[-5:],
                    "xG For":     round(t.avg_gf,      2),
                    "xG Ag":      round(t.avg_ga,      2),
                    "SoT/g":      round(t.avg_sot,     1),
                    "Corners/g":  round(t.avg_corners, 1),
                    "Momentum":   t.momentum,
                })
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )

    # ── Tab 4: Backtest ───────────────────────────────────────
    with tab4:
        split_pct = int(BACKTEST_SPLIT * 100)
        st.info(
            f"Backtest v6.1: {split_pct}% train / {100 - split_pct}% test | "
            "Data-Leakage-Free | 126 features"
        )
        if st.button("▶️ Run Backtest"):
            with st.spinner("Running backtest…"):
                r = app.run_backtest()
                st.session_state[f"bt_{selected_league}"] = r

        bt_key = f"bt_{selected_league}"
        if bt_key in st.session_state:
            r = st.session_state[bt_key]
            if "error" in r:
                st.error(r["error"])
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("🎯 1X2 Accuracy", f"{r['result_acc']:.1f}%")
                c2.metric("⚽ Exact Score",  f"{r['score_acc']:.1f}%")
                # V6.1: Brier label مُصلح
                c3.metric(
                    "📐 Brier [0→1]",
                    f"{r['brier']:.4f}",
                    help="Multi-class Brier Score, normalized [0,1]. Lower is better.",
                )
                c4.metric("🤖 ML CV Acc", f"{r.get('ml_acc', 0):.1f}%")

                c5, c6, c7, c8 = st.columns(4)
                c5.metric("📋 Train",    r["train"])
                c6.metric("🎚️ Cal Set",  r.get("cal_size", 0))
                c7.metric("🔬 Eval Set", r.get("eval_size", 0))
                c8.metric("✅ Calibrated", "Yes" if r.get("cal_used") else "No")

                st.divider()
                st.markdown("#### 📊 Accuracy by Outcome")
                ca, cb, cc = st.columns(3)
                ca.metric("🏠 Home Win", f"{r.get('home_acc', 0):.1f}%",
                          f"{r.get('home_total', 0)} matches")
                cb.metric("🤝 Draw",     f"{r.get('draw_acc', 0):.1f}%",
                          f"{r.get('draw_total', 0)} matches")
                cc.metric("✈️ Away Win", f"{r.get('away_acc', 0):.1f}%",
                          f"{r.get('away_total', 0)} matches")

                st.markdown("#### 🎯 Accuracy by Confidence")
                d1, d2, d3 = st.columns(3)
                d1.metric("🔥 High >60%",     f"{r.get('hi_acc', 0):.1f}%",
                          f"{r.get('hi_n', 0)} matches")
                d2.metric("⚡ Medium 45-60%", f"{r.get('me_acc', 0):.1f}%",
                          f"{r.get('me_n', 0)} matches")
                d3.metric("⚠️ Low <45%",      f"{r.get('lo_acc', 0):.1f}%",
                          f"{r.get('lo_n', 0)} matches")

                thr = r.get("dc_threshold", 0.60)
                st.markdown(
                    f"#### 🛡️ Double Chance (threshold >{thr:.0%})"
                )
                e1, e2, e3 = st.columns(3)
                e1.metric("1X", f"{r.get('dc_1x_acc', 0):.1f}%",
                          f"{r.get('dc_1x_n', 0)} bets")
                e2.metric("12", f"{r.get('dc_12_acc', 0):.1f}%",
                          f"{r.get('dc_12_n', 0)} bets")
                e3.metric("X2", f"{r.get('dc_x2_acc', 0):.1f}%",
                          f"{r.get('dc_x2_n', 0)} bets")

                # Confusion Matrix
                if r.get("confusion"):
                    st.markdown("#### 🔍 Confusion Matrix")
                    conf    = r["confusion"]
                    cm_data = []
                    for pred_label in ["HOME", "DRAW", "AWAY"]:
                        row = {"Predicted ↓ / Actual →": pred_label}
                        for act_label in ["HOME", "DRAW", "AWAY"]:
                            row[f"Actual {act_label}"] = (
                                conf.get(pred_label, {}).get(act_label, 0)
                            )
                        cm_data.append(row)
                    st.dataframe(
                        pd.DataFrame(cm_data),
                        use_container_width=True,
                        hide_index=True,
                    )

                if r.get("predictions"):
                    st.markdown("#### 📋 Last 50 Test Predictions")
                    preds_rows = []
                    for pred_item in r["predictions"][-50:]:
                        preds_rows.append({
                            "Home":       pred_item["home"],
                            "Away":       pred_item["away"],
                            "Predicted":  pred_item["predicted"],
                            "Actual":     pred_item["actual"],
                            "Result":     "✅" if pred_item["correct"] else "❌",
                            "Confidence": f"{pred_item['confidence']:.1f}%",
                            "P(Home)":    f"{pred_item['probs'][0]*100:.1f}%",
                            "P(Draw)":    f"{pred_item['probs'][1]*100:.1f}%",
                            "P(Away)":    f"{pred_item['probs'][2]*100:.1f}%",
                            "Calibrated": "✅" if pred_item.get("calibrated") else "❌",
                        })
                    st.dataframe(
                        pd.DataFrame(preds_rows),
                        use_container_width=True,
                        hide_index=True,
                    )


# ══════════════════════════════════════════════════════════════
# CLI MAIN
# ══════════════════════════════════════════════════════════════
def cli_main():
    Disp.header()
    available = list(LEAGUES_CONFIG.keys())
    if not available:
        print(f"{C.red('❌')} No leagues in {LEAGUES_CONFIG_FILE}")
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

    print(f"\n {C.green('►')} Loading {C.bold(league_code)} v6.1…")
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
            print(f"\n {C.cyan('─' * 55)}")
            print(f" {C.bold('Options:')}")
            print(f" {C.cyan('1')} - Predict upcoming matches")
            print(f" {C.cyan('2')} - Custom match prediction")
            print(f" {C.cyan('3')} - League standings")
            print(f" {C.cyan('4')} - Run backtest (v6.1 — Leakage-Free)")
            print(f" {C.cyan('5')} - Export last predictions")
            print(f" {C.cyan('6')} - Switch league")
            print(f" {C.cyan('0')} - Exit")

            ch = input(C.cyan("\n Choice: ")).strip()

            if ch == "1":
                Disp.section("Upcoming Predictions")
                preds = app.predict_upcoming(14)
                if preds:
                    for pr in preds:
                        Disp.pred_card(pr)
                else:
                    print(C.yellow(" No upcoming matches found"))

            elif ch == "2":
                Disp.section("Custom Match")
                teams = sorted(t.name for t in app.data.teams.values())
                print(f" Teams: {', '.join(teams)}")
                home = input(C.cyan(" Home team: ")).strip()
                away = input(C.cyan(" Away team: ")).strip()
                pr   = app.predict_custom(home, away)
                if pr:
                    Disp.pred_card(pr)

            elif ch == "3":
                Disp.section(f"Standings — {app.resources.name}")
                standings = app.standings()
                print(
                    f" {'#':>3} {'Team':<22} {'P':>3} {'W':>2} "
                    f"{'D':>2} {'L':>2} {'GD':>4} {'Pts':>4} "
                    f"{'Elo':>6} {'Mom':>5}"
                )
                print(f" {'─' * 65}")
                total_t = len(standings)
                for t in standings:
                    pc = (
                        C.G if t.pos <= 4
                        else (C.R if t.pos >= total_t - 2 else C.W)
                    )
                    gd_str = f"{t.gd:+d}"
                    print(
                        f" {pc}{t.pos:>3}{C.E} {t.name:<22} "
                        f"{t.played:>3} {t.wins:>2} {t.draws:>2} "
                        f"{t.losses:>2} {gd_str:>4} {t.pts:>4} "
                        f"{t.elo:>6.0f} {t.momentum:>5}"
                    )

            elif ch == "4":
                Disp.section("Backtest v6.1 — Data-Leakage-Free")
                r = app.run_backtest()
                Disp.backtest_summary(r, league_code)

            elif ch == "5":
                if app.last_preds:
                    fn = app.export_predictions()
                    print(C.green(f" ✅ Exported: {fn}"))
                else:
                    print(C.yellow(" No predictions to export"))

            elif ch == "6":
                Disp.leagues_menu(available)
                choice2 = input(C.cyan(f" Select (1-{len(available)}): ")).strip()
                try:
                    new_code = available[int(choice2) - 1]
                    new_app  = LeagueApp(new_code, tok, odds_key)
                    if new_app.init():
                        app         = new_app
                        league_code = new_code
                    else:
                        print(C.red("❌ Failed to load league"))
                except (ValueError, IndexError):
                    print(C.red("❌ Invalid choice"))

            elif ch == "0":
                print(C.green(" Goodbye! ⚽"))
                break

        except KeyboardInterrupt:
            print(C.green("\n Goodbye! ⚽"))
            break
        except Exception as e:
            print(C.red(f" Error: {e}"))


# ══════════════════════════════════════════════════════════════
# ENTRY POINT — V6.1 FIX: Streamlit/CLI separation
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # ══════════════════════════════════════════════════════
    # ENTRY POINT — V6.1 FIX: منطق واضح بدون تعارض
    # ──────────────────────────────────────────────────────
    # python appp.py               → CLI
    # python appp.py --streamlit   → Streamlit (يُطلق run_streamlit)
    # streamlit run appp.py        → Streamlit (sys.argv[0] = 'appp.py')
    # ══════════════════════════════════════════════════════
    _want_streamlit = False
    if STREAMLIT_AVAILABLE:
        if "--streamlit" in sys.argv:
            _want_streamlit = True
        else:
            try:
                from streamlit.runtime.scriptrunner import get_script_run_ctx
                if get_script_run_ctx() is not None:
                    _want_streamlit = True
            except Exception:
                pass

    if _want_streamlit:
        run_streamlit()
    else:
        cli_main()