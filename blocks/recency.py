# blocks/recency.py
from __future__ import annotations
import pandas as pd
import numpy as np

# ---------- helpers ----------
def _season_start_year(season_str: str) -> int:
    # "2024-25" -> 2024
    try:
        return int(season_str.split("-")[0])
    except Exception:
        return -1

def _safe_ewm_mean(s: pd.Series, alpha: float) -> pd.Series:
    # pandas ewm.mean supports adjust=False; we also shift(1) outside
    return s.ewm(alpha=alpha, adjust=False).mean()

def _safe_ewm_sum(s: pd.Series, alpha: float) -> pd.Series:
    # emulate ewm(sum) via ewm(mean) * ewm(count of ones)
    # but we only need mean & rates here, so we won't use sum for now
    return s.ewm(alpha=alpha, adjust=False).mean() * 1.0

# ---------- core EWMA features ----------
def compute_recency_features(
    player_history: pd.DataFrame,
    alpha_pts: float = 0.20,      # how fast points EWMA decays
    alpha_mins: float = 0.20,     # minutes EWMA
    alpha_avail: float = 0.20,    # appearance-rate EWMA
    min_games_window: int = 8     # rolling window for last-N games minutes sum (contextual)
) -> pd.DataFrame:
    """
    Returns a per-row table with leak-safe EWMAs:
      pts_ewm_prev, mins_ewm_prev, appear_rate_prev, mins_lastN_prev
    All *_prev are shifted(1) so they only use history strictly before the row GW.
    Expect player_history with columns:
      season, gw, player_id, minutes, total_points
    """
    req = {"season","gw","player_id","minutes","total_points"}
    missing = req - set(player_history.columns)
    if missing:
        raise ValueError(f"player_history missing required columns: {missing}")

    df = player_history.copy()
    # order by season start year then gw to carry EWM across seasons
    df["season_year"] = df["season"].astype(str).map(_season_start_year)
    df = df.sort_values(["player_id","season_year","gw"]).reset_index(drop=True)

    # build helpers
    df["appeared"] = (df["minutes"] > 0).astype(int)

    def _group_apply(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        # EWMA means (no leakage): compute then shift(1)
        g["pts_ewm_prev"]    = _safe_ewm_mean(g["total_points"], alpha_pts).shift(1)
        g["mins_ewm_prev"]   = _safe_ewm_mean(g["minutes"],      alpha_mins).shift(1)
        g["appear_rate_prev"]= _safe_ewm_mean(g["appeared"],     alpha_avail).shift(1)
        # last-N games minutes (rolling sum) then shift(1)
        g["mins_lastN_prev"] = g["minutes"].rolling(min_games_window, min_periods=1).sum().shift(1)
        return g

    out = df.groupby("player_id", group_keys=False).apply(_group_apply)
    # fill NaNs for first appearances
    for c in ["pts_ewm_prev","mins_ewm_prev","appear_rate_prev","mins_lastN_prev"]:
        if c in out.columns:
            out[c] = out[c].fillna(0.0)

    # reliability: combine expected minutes and appearance rate
    # clip minutes to [0,90], scale to [0,1], multiply by appearance rate
    exp_mins01 = out["mins_ewm_prev"].clip(0, 90) / 90.0
    out["reliability"] = (exp_mins01 * out["appear_rate_prev"]).clip(0.0, 1.0)

    return out

# ---------- horizon predictions (no fixtures yet) ----------
def build_recency_predictions(
    player_history: pd.DataFrame,
    horizon_gws: int = 4,
    alpha_pts: float = 0.20,
    alpha_mins: float = 0.20,
    alpha_avail: float = 0.20,
    min_games_window: int = 8,
    season_filter: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      preds  : columns [player_id, gw, exp_points, reliability, basis='recency']
      latest : last-known per-player row with [player_id, last_gw, pts_ewm_prev, reliability, season]
    Strategy:
      - Compute leak-safe EWMAs on the history.
      - For each player's last observed GW (optionally within a season), take pts_ewm_prev & reliability.
      - Predict a flat value for next H GWs: exp_points = pts_ewm_prev * reliability.
    """
    hist = player_history.copy()
    if season_filter is not None:
        hist = hist[hist["season"].astype(str) == str(season_filter)].copy()
    if hist.empty:
        return pd.DataFrame(columns=["player_id","gw","exp_points","reliability","basis"]), \
               pd.DataFrame(columns=["player_id","last_gw","season","pts_ewm_prev","reliability"])

    feats = compute_recency_features(
        hist, alpha_pts=alpha_pts, alpha_mins=alpha_mins,
        alpha_avail=alpha_avail, min_games_window=min_games_window
    )

    # last observed per player
    idx = feats.groupby("player_id")["gw"].idxmax()
    latest = feats.loc[idx, ["player_id","season","gw","pts_ewm_prev","reliability"]].rename(columns={"gw":"last_gw"})
    latest["pts_ewm_prev"] = latest["pts_ewm_prev"].clip(lower=0.0)
    latest["reliability"]  = latest["reliability"].clip(0.0, 1.0)

    if latest.empty:
        return pd.DataFrame(columns=["player_id","gw","exp_points","reliability","basis"]), latest

    # horizon gws
    max_gw = int(feats["gw"].max())
    future_gws = list(range(max_gw + 1, max_gw + 1 + horizon_gws))

    # cartesian: each player x each future gw
    preds = latest.assign(key=1).merge(
        pd.DataFrame({"gw": future_gws, "key":[1]*len(future_gws)}),
        on="key", how="outer"
    ).drop(columns=["key"])

    preds["exp_points"] = (preds["pts_ewm_prev"] * preds["reliability"]).astype(float)
    preds["basis"] = "recency"

    # tidy
    preds = preds[["player_id","gw","exp_points","reliability","basis"]].sort_values(["gw","exp_points"], ascending=[True, False])
    return preds.reset_index(drop=True), latest.reset_index(drop=True)

def build_recency_for_gws(
    player_history: pd.DataFrame,
    future_gws: list[int],
    season_filter: str | None = None,
    alpha_pts: float = 0.20,
    alpha_mins: float = 0.20,
    alpha_avail: float = 0.20,
    min_games_window: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Like build_recency_predictions, but you pass the exact future GW numbers
    (e.g., from fixtures). Returns:
      preds  : [player_id, gw, exp_points, reliability, basis='recency']
      latest : [player_id, last_gw, season, pts_ewm_prev, reliability]
    """
    hist = player_history.copy()
    if season_filter is not None:
        hist = hist[hist["season"].astype(str) == str(season_filter)].copy()
    if hist.empty or not future_gws:
        return pd.DataFrame(columns=["player_id","gw","exp_points","reliability","basis"]), \
               pd.DataFrame(columns=["player_id","last_gw","season","pts_ewm_prev","reliability"])

    feats = compute_recency_features(
        hist, alpha_pts=alpha_pts, alpha_mins=alpha_mins,
        alpha_avail=alpha_avail, min_games_window=min_games_window
    )

    idx = feats.groupby("player_id")["gw"].idxmax()
    latest = feats.loc[idx, ["player_id","season","gw","pts_ewm_prev","reliability"]].rename(columns={"gw":"last_gw"})
    latest["pts_ewm_prev"] = latest["pts_ewm_prev"].clip(lower=0.0)
    latest["reliability"]  = latest["reliability"].clip(0.0, 1.0)

    if latest.empty:
        return pd.DataFrame(columns=["player_id","gw","exp_points","reliability","basis"]), latest

    # Cartesian with provided future_gws
    gws_df = pd.DataFrame({"gw": list(sorted(set(int(g) for g in future_gws)))})
    latest["key"] = 1; gws_df["key"] = 1
    preds = latest.merge(gws_df, on="key", how="left").drop(columns=["key"])

    preds["exp_points"] = (preds["pts_ewm_prev"] * preds["reliability"]).astype(float)
    preds["basis"] = "recency"
    preds = preds[["player_id","gw","exp_points","reliability","basis"]].sort_values(["gw","exp_points"], ascending=[True, False])
    return preds.reset_index(drop=True), latest.reset_index(drop=True)

