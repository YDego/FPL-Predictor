# blocks/team_strength.py
from __future__ import annotations
import pandas as pd
import numpy as np

def _safe_ewm_mean(s: pd.Series, alpha: float) -> pd.Series:
    return s.ewm(alpha=alpha, adjust=False).mean()

def compute_team_strength(
    team_matches: pd.DataFrame,
    alpha_att: float = 0.35,
    alpha_def: float = 0.35,
) -> pd.DataFrame:
    """
    Input: team_matches(season, gw, team_id, opponent_id, was_home, gf, ga)
    Output: wide per (season, gw, team_id) with both home/away columns,
            plus indicators has_home / has_away.
    NOTE: we DO NOT fill missing side columns with 1.0 here; we keep NaN.
          The 'as of' selection will pick the latest row for each side separately.
    """
    req = {"season","gw","team_id","opponent_id","was_home","gf","ga"}
    missing = req - set(team_matches.columns)
    if missing:
        raise ValueError(f"team_matches missing columns: {missing}")

    tm = team_matches.copy().sort_values(["season","team_id","was_home","gw"]).reset_index(drop=True)

    def _by_team_side(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g["att_prev"] = _safe_ewm_mean(g["gf"], alpha_att).shift(1).fillna(0.0)
        g["def_prev"] = _safe_ewm_mean(g["ga"], alpha_def).shift(1).fillna(0.0)
        return g

    side = tm.groupby(["season","team_id","was_home"], group_keys=False).apply(_by_team_side)

    # Normalize per (season, gw, side) to league mean ~1
    mean_att = side.groupby(["season","gw","was_home"])["att_prev"].transform(
        lambda s: s.replace(0, np.nan).mean()
    )
    mean_def = side.groupby(["season","gw","was_home"])["def_prev"].transform(
        lambda s: s.replace(0, np.nan).mean()
    )
    side["att_norm"] = (side["att_prev"] / mean_att).where(mean_att.notna(), np.nan)
    side["def_norm"] = (side["def_prev"] / mean_def).where(mean_def.notna(), np.nan)

    # Split to home/away and pivot to wide
    home = side[side["was_home"]==1][["season","gw","team_id","att_norm","def_norm"]].copy()
    home.rename(columns={"att_norm":"att_home_norm","def_norm":"def_home_norm"}, inplace=True)
    home["has_home"] = 1

    away = side[side["was_home"]==0][["season","gw","team_id","att_norm","def_norm"]].copy()
    away.rename(columns={"att_norm":"att_away_norm","def_norm":"def_away_norm"}, inplace=True)
    away["has_away"] = 1

    wide = pd.merge(home, away, on=["season","gw","team_id"], how="outer").sort_values(["season","gw","team_id"])
    # Keep NaNs here (don’t fill with 1.0). We'll handle defaults at “as-of” selection.
    wide["has_home"] = wide["has_home"].fillna(0).astype(int)
    wide["has_away"] = wide["has_away"].fillna(0).astype(int)

    return wide.reset_index(drop=True)
