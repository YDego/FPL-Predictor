# blocks/meta.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

_POS_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

def _coerce_int(s):
    return pd.to_numeric(s, errors="coerce").astype("Int64")

def _build_web_name(df: pd.DataFrame) -> pd.Series:
    cols = {c.lower(): c for c in df.columns}
    # try direct
    if "web_name" in cols:
        wn = df[cols["web_name"]].astype(str)
        if wn.notna().any():
            return wn
    # fallback 1: second_name
    if "second_name" in cols:
        sn = df[cols["second_name"]].astype(str)
        # if also have first_name, use "F. Second"
        if "first_name" in cols:
            fn = df[cols["first_name"]].astype(str)
            initial = fn.str.slice(0, 1).str.upper().fillna("")
            out = (initial.where(initial.eq(""), initial + ". ") + sn).str.strip()
            return out.where(out != "", sn)
        return sn
    # fallback 2: first + last name variants
    if "first_name" in cols and "last_name" in cols:
        fn = df[cols["first_name"]].astype(str)
        ln = df[cols["last_name"]].astype(str)
        return (fn.str.slice(0, 1).str.upper() + ". " + ln).str.strip()
    # last resort: stringified id
    id_col = cols.get("id", "id")
    return "P" + df[id_col].astype(str)

def load_player_prices_positions(season_dir: str | Path) -> pd.DataFrame:
    """
    Returns: player_id, position (GKP/DEF/MID/FWD), now_cost (float £m), team_id, web_name
    """
    season_dir = Path(season_dir)
    p = season_dir / "players_raw.csv"
    if not p.exists():
        raise FileNotFoundError(f"players_raw.csv not found at {p}")

    df = pd.read_csv(p)
    cols = {c.lower(): c for c in df.columns}
    id_col   = cols.get("id", "id")
    et_col   = cols.get("element_type", "element_type")
    team_col = cols.get("team", "team")
    cost_col = cols.get("now_cost", "now_cost")

    out = pd.DataFrame({
        "player_id": _coerce_int(df[id_col]),
        "team_id":   _coerce_int(df[team_col]),
        "position":  pd.to_numeric(df[et_col], errors="coerce").map(_POS_MAP),
        "now_cost":  pd.to_numeric(df[cost_col], errors="coerce").fillna(0) / 10.0,
    })
    out["web_name"] = _build_web_name(df)
    out = out.dropna(subset=["player_id", "team_id", "position"]).reset_index(drop=True)
    return out[["player_id", "position", "now_cost", "team_id", "web_name"]]

def load_player_meta(season_dir: str | Path) -> pd.DataFrame:
    """Lightweight: player_id, team_id, web_name (same fallbacks)."""
    season_dir = Path(season_dir)
    p = season_dir / "players_raw.csv"
    if not p.exists():
        raise FileNotFoundError(f"players_raw.csv not found at {p}")
    df = pd.read_csv(p)
    cols = {c.lower(): c for c in df.columns}
    id_col   = cols.get("id", "id")
    team_col = cols.get("team", "team")
    out = pd.DataFrame({
        "player_id": _coerce_int(df[id_col]),
        "team_id":   _coerce_int(df[team_col]),
    })
    out["web_name"] = _build_web_name(df)
    return out

def load_teams_labels(season_dir: str | Path) -> pd.DataFrame:
    """
    Returns: team_id, team_short
    Guarantees team_short by falling back to 'short_name', 'name', or a 3-letter code.
    """
    season_dir = Path(season_dir)
    p = season_dir / "teams.csv"
    if not p.exists():
        raise FileNotFoundError(f"teams.csv not found at {p}")
    df = pd.read_csv(p)
    cols = {c.lower(): c for c in df.columns}
    id_col   = cols.get("id", "id")
    shortcol = cols.get("short_name", None)
    namecol  = cols.get("name", None)

    out = pd.DataFrame({"team_id": _coerce_int(df[id_col])})
    if shortcol:
        team_short = df[shortcol].astype(str)
    elif namecol:
        team_short = df[namecol].astype(str).str.slice(0, 3).str.upper()
    else:
        team_short = ("T" + df[id_col].astype(str))
    out["team_short"] = team_short
    out = out.dropna(subset=["team_id"]).drop_duplicates("team_id").reset_index(drop=True)
    return out[["team_id", "team_short"]]
