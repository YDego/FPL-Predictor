from __future__ import annotations
from pathlib import Path
from typing import List
import glob
import pandas as pd

def _read_season_gws(season_dir: Path) -> pd.DataFrame:
    gws_dir = season_dir / "gws"
    files = sorted(glob.glob(str(gws_dir / "gw*.csv")))
    parts = []
    for f in files:
        try:
            df = pd.read_csv(f, engine="python", on_bad_lines="skip")
        except Exception:
            df = pd.read_csv(f, engine="python", on_bad_lines="skip", dtype=str)
        parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

def _normalize(df_raw: pd.DataFrame, season: str) -> pd.DataFrame:
    if df_raw.empty:
        return pd.DataFrame()
    df = df_raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    def num(s):
        return pd.to_numeric(s, errors="coerce")

    out = pd.DataFrame({
        "season": season,
        "gw":           num(df.get("round", df.get("gw"))).astype("Int64"),
        "player_id":    num(df.get("element", df.get("player_id"))).astype("Int64"),
        "team_id":      num(df.get("team", df.get("team_id"))).astype("Int64"),
        "opponent_id":  num(df.get("opponent_team", df.get("opponent_id"))).astype("Int64"),
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
    out = out.dropna(subset=["gw","player_id","team_id","opponent_id"])
    out["gw"] = out["gw"].astype(int)
    out["player_id"] = out["player_id"].astype(int)
    out["team_id"] = out["team_id"].astype(int)
    out["opponent_id"] = out["opponent_id"].astype(int)
    return out.sort_values(["season","gw","player_id"]).reset_index(drop=True)

def load_history_multi_season(repo_root: str, seasons: List[str]) -> pd.DataFrame:
    root = Path(repo_root)
    all_parts = []
    for s in seasons:
        season_dir = root / "data" / s
        raw = _read_season_gws(season_dir)
        norm = _normalize(raw, s)
        all_parts.append(norm)
    return pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame()
