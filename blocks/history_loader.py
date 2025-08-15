# blocks/history_loader.py
from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import re, glob
import pandas as pd

SEASON_RX = re.compile(r"^\d{4}-\d{2}$")

def find_repo_root(relative_to: Path) -> Path:
    c1 = relative_to.parents[1] / "Fantasy-Premier-League-master"
    c2 = relative_to.parents[1] / "Fantasy-Premier-League"
    for cand in (c1, c2):
        if (cand / "data").exists():
            return cand
    p = relative_to
    for _ in range(5):
        if (p / "Fantasy-Premier-League-master" / "data").exists():
            return p / "Fantasy-Premier-League-master"
        if (p / "Fantasy-Premier-League" / "data").exists():
            return p / "Fantasy-Premier-League"
        p = p.parent
    raise FileNotFoundError(
        "Vaastav repo root not found; place it under your project as 'Fantasy-Premier-League-master'"
    )

# ---------- robust readers ----------
def _read_csv_robust(path: Path, as_str_fallback: bool = True) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip", encoding=enc)
        except Exception as e:
            last_err = e
            if as_str_fallback:
                try:
                    return pd.read_csv(path, engine="python", on_bad_lines="skip",
                                       encoding=enc, dtype=str)
                except Exception as e2:
                    last_err = e2
            continue
    raise last_err if last_err else RuntimeError(f"Failed to read {path}")

def _read_gw_files(season_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(season_dir / "gws" / "gw*.csv")))
    parts = []
    for f in files:
        parts.append(_read_csv_robust(Path(f)))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

def _read_merged_gw_csv(season_dir: Path) -> pd.DataFrame:
    p = season_dir / "gws" / "merged_gw.csv"
    return _read_csv_robust(p) if p.exists() else pd.DataFrame()

def _read_fixtures(season_dir: Path) -> pd.DataFrame:
    p = season_dir / "fixtures.csv"
    if not p.exists():
        return pd.DataFrame()
    fx = _read_csv_robust(p)
    fx.columns = [c.strip().lower() for c in fx.columns]
    # common Vaastav columns: id, event, team_h, team_a, team_h_difficulty, team_a_difficulty
    return fx.rename(columns={
        "id": "fixture_id",
        "event": "gw",
        "team_h": "team_h_id",
        "team_a": "team_a_id",
    })

# ---------- normalize + fill team via fixtures ----------
def _normalize(df_raw: pd.DataFrame, season: str, fixtures: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    if df_raw.empty:
        return pd.DataFrame()
    df = df_raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    def num(s): return pd.to_numeric(s, errors="coerce")

    out = pd.DataFrame({
        "season": season,
        "gw":           num(df.get("round", df.get("gw"))).astype("Int64"),
        "player_id":    num(df.get("element", df.get("player_id"))).astype("Int64"),
        "team_id":      num(df.get("team", df.get("team_id"))).astype("Int64"),
        "opponent_id":  num(df.get("opponent_team", df.get("opponent_id"))).astype("Int64"),
        "fixture":      num(df.get("fixture", df.get("fixture_id"))).astype("Int64"),
        "was_home":     (num(df.get("was_home", 0)).fillna(0) > 0).astype("int8"),
        "minutes":      num(df.get("minutes", 0)).fillna(0).astype("int16"),
        "total_points": num(df.get("total_points", df.get("points", 0))).fillna(0).astype("int16"),
        "goals_scored": num(df.get("goals_scored", 0)).fillna(0).astype("int16"),
        "assists":      num(df.get("assists", 0)).fillna(0).astype("int16"),
        "clean_sheets": num(df.get("clean_sheets", 0)).fillna(0).astype("int16"),
        "goals_conceded": num(df.get("goals_conceded", 0)).fillna(0).astype("int16"),
        "saves":        num(df.get("saves", 0)).fillna(0).astype("int16"),
        "yellow_cards": num(df.get("yellow_cards", 0)).fillna(0).astype("int16"),
        "red_cards":    num(df.get("red_cards", 0)).fillna(0).astype("int16"),
        "bonus":        num(df.get("bonus", 0)).fillna(0).astype("int16"),
    })

    # Fill team_id from fixtures if missing
    if fixtures is not None and not fixtures.empty and out["team_id"].isna().any():
        fx = fixtures[["fixture_id","team_h_id","team_a_id"]].dropna()
        tmp = out.merge(fx, left_on="fixture", right_on="fixture_id", how="left")
        # primary fill: was_home flag
        home_mask = tmp["team_id"].isna() & (tmp["was_home"] == 1) & tmp["team_h_id"].notna()
        away_mask = tmp["team_id"].isna() & (tmp["was_home"] == 0) & tmp["team_a_id"].notna()
        tmp.loc[home_mask, "team_id"] = tmp.loc[home_mask, "team_h_id"]
        tmp.loc[away_mask, "team_id"] = tmp.loc[away_mask, "team_a_id"]

        # secondary fill: if opponent_id equals team_h ⇒ team is team_a, and vice-versa
        opp = tmp["opponent_id"]
        hids = tmp["team_h_id"]
        aids = tmp["team_a_id"]
        mask2 = tmp["team_id"].isna() & opp.notna() & hids.notna() & aids.notna()
        tmp.loc[mask2 & (opp == hids), "team_id"] = aids[mask2 & (opp == hids)]
        tmp.loc[mask2 & (opp == aids), "team_id"] = hids[mask2 & (opp == aids)]

        out["team_id"] = tmp["team_id"].astype("Int64")

    miss = out[["gw","player_id","team_id","opponent_id"]].isna().sum().to_dict()
    if verbose and any(v > 0 for v in miss.values()):
        print(f"  [warn] missing counts {miss}")

    out = out.dropna(subset=["gw","player_id","team_id","opponent_id"])
    out["gw"] = out["gw"].astype(int)
    out["player_id"] = out["player_id"].astype(int)
    out["team_id"] = out["team_id"].astype(int)
    out["opponent_id"] = out["opponent_id"].astype(int)
    return out.sort_values(["season","gw","player_id"]).reset_index(drop=True)

# ---------- public API ----------
def detect_seasons(repo_root: Path) -> List[str]:
    data_dir = repo_root / "data"
    seasons = [p.name for p in sorted(data_dir.iterdir())
               if p.is_dir() and SEASON_RX.fullmatch(p.name)]
    if not seasons:
        raise RuntimeError(f"No seasons found under {data_dir}")
    return seasons

def seasons_with_gws(repo_root: Path) -> List[str]:
    """Return seasons that actually contain at least one gw CSV (or merged)."""
    out = []
    for s in detect_seasons(repo_root):
        sd = repo_root / "data" / s
        if list((sd / "gws").glob("gw*.csv")) or (sd / "gws" / "merged_gw.csv").exists():
            out.append(s)
    return out

def load_history_multi_season(
    repo_root: str | Path,
    seasons: Optional[List[str]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    repo_root = Path(repo_root)
    if seasons is None:
        seasons = seasons_with_gws(repo_root)
    parts = []
    for s in seasons:
        season_dir = repo_root / "data" / s
        if verbose: print(f"[{s}] reading …")
        df = _read_gw_files(season_dir)
        if df.empty:
            df = _read_merged_gw_csv(season_dir)
            if verbose: print(f"  used merged_gw.csv: {not df.empty}, gw*.csv count: 0")
        else:
            if verbose: print(f"  gw*.csv rows: {len(df)}")
        fixtures = _read_fixtures(season_dir)
        norm = _normalize(df, s, fixtures, verbose)
        if verbose: print(f"  normalized rows: {len(norm)}")
        if not norm.empty:
            parts.append(norm)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
