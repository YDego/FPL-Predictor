# blocks/recency.py
from __future__ import annotations
import numpy as np
import pandas as pd

# Tunables (recency)
ALPHA_PTS = 0.30  # EWM for points
ALPHA_MIN = 0.30  # EWM for minutes / start-rate

def build_recency_baseline(
    history: pd.DataFrame,
    horizon: int = 4,
    player_key_col: str = "player_key",   # <= use 'code' field wired in via player_key
) -> pd.DataFrame:
    """
    Leak-safe recency baseline. For each player_key, compute EWM of points/minutes
    up to each GW, then produce predictions for the next `horizon` GWs after the
    last observed GW in the latest season.

    Returns columns: ['player_key','gw','exp_points','reliability']
    """
    if history is None or history.empty:
        return pd.DataFrame(columns=["player_key","gw","exp_points","reliability"])

    needed = {player_key_col,"season","gw","minutes","total_points"}
    missing = needed - set(history.columns)
    if missing:
        raise KeyError(f"history missing columns: {missing}")

    df = history[[player_key_col,"season","gw","minutes","total_points"]].copy()
    df = df.dropna(subset=[player_key_col,"gw"])
    df["gw"] = pd.to_numeric(df["gw"], errors="coerce").astype(int)

    def _group_apply(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(["season","gw"]).copy()
        pts_ewm  = g["total_points"].ewm(alpha=ALPHA_PTS, adjust=False).mean().shift(1).fillna(0.0)
        mins_ewm = g["minutes"].ewm(alpha=ALPHA_MIN, adjust=False).mean().shift(1).fillna(0.0)
        start_rate_ewm = (g["minutes"] >= 60).astype(float).ewm(alpha=ALPHA_MIN, adjust=False).mean().shift(1).fillna(0.0)
        rel = 0.6*start_rate_ewm + 0.4*np.clip(mins_ewm/90.0, 0.0, 1.0)
        out = pd.DataFrame({
            "player_key": g[player_key_col].values,
            "season": g["season"].values,
            "gw": g["gw"].values,
            "exp_points": pts_ewm.values,
            "reliability": rel.values,
        })
        return out

    out = df.groupby(player_key_col, group_keys=False).apply(_group_apply).reset_index(drop=True)

    # Next horizon after the last observed GW of the latest season
    latest_season = str(sorted(history["season"].unique())[-1])
    max_gw = int(history.loc[history["season"]==latest_season,"gw"].max())
    future_gws = list(range(max_gw+1, max_gw+1+horizon))

    latest_by_key = (out[out["season"]==latest_season]
                     .sort_values(["player_key","gw"])
                     .groupby("player_key", as_index=False).tail(1)[["player_key","exp_points","reliability"]])

    preds = []
    for gw in future_gws:
        tmp = latest_by_key.copy()
        tmp["gw"] = gw
        preds.append(tmp)
    preds = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(columns=["player_key","gw","exp_points","reliability"])
    return preds[["player_key","gw","exp_points","reliability"]]
