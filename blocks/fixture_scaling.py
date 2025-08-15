# blocks/fixture_scaling.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

from blocks.data_io import _load_fixtures  # reuse your fixture loader


def latest_team_strength_asof(ts_wide: pd.DataFrame, season: str, asof_gw: int) -> pd.DataFrame:
    """
    Select, for each team, the latest HOME norms from the latest row with has_home==1 (<= asof_gw),
    and the latest AWAY norms from the latest row with has_away==1 (<= asof_gw).
    Returns a DataFrame indexed by team_id with columns:
      att_home_norm, def_home_norm, att_away_norm, def_away_norm
    Missing side -> filled with neutral 1.0.
    """
    cur = ts_wide[(ts_wide["season"] == season) & (ts_wide["gw"] <= asof_gw)].copy()
    if cur.empty:
        return pd.DataFrame(columns=["att_home_norm", "def_home_norm", "att_away_norm", "def_away_norm"]).set_index(
            pd.Index([], name="team_id")
        )

    # latest HOME row per team
    home_latest = (
        cur[cur.get("has_home", 0) == 1]
        .sort_values(["team_id", "gw"])
        .groupby("team_id")
        .tail(1)[["team_id", "att_home_norm", "def_home_norm"]]
    )

    # latest AWAY row per team
    away_latest = (
        cur[cur.get("has_away", 0) == 1]
        .sort_values(["team_id", "gw"])
        .groupby("team_id")
        .tail(1)[["team_id", "att_away_norm", "def_away_norm"]]
    )

    merged = pd.merge(home_latest, away_latest, on="team_id", how="outer")

    for c in ["att_home_norm", "def_home_norm", "att_away_norm", "def_away_norm"]:
        merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(1.0).astype(float)

    merged = merged.drop_duplicates(subset=["team_id"]).set_index("team_id")
    merged.index.name = "team_id"
    return merged


def compute_team_fixture_factors(
    repo_root: str | Path,
    season: str,
    team_strength_wide: pd.DataFrame,
    current_gw: int,
    horizon: int,
    asof_gw_override: int | None = None,
) -> pd.DataFrame:
    """
    Build per-(team, future_gw) fixture factors using team strengths as of a given GW.

    Factor per fixture:
      - for HOME team: att_home_norm(home, asof) * def_away_norm(away, asof)
      - for AWAY team: att_away_norm(away, asof) * def_home_norm(home, asof)

    For blanks/doubles, aggregate per (team, gw):
      factor_mean = mean over fixtures, n_fixtures = count

    Returns columns: season, gw, team_id, factor_mean, n_fixtures
    """
    repo_root = Path(repo_root)
    season_dir = repo_root / "data" / season
    fx = _load_fixtures(season_dir)
    if fx.empty or "event" not in fx.columns:
        return pd.DataFrame(columns=["season", "gw", "team_id", "factor_mean", "n_fixtures"])

    # Determine "as of" GW
    asof_gw = int(asof_gw_override) if asof_gw_override is not None else int(current_gw)

    # Coerce fixture dtypes
    fx = fx.copy()
    fx["event"] = pd.to_numeric(fx["event"], errors="coerce").astype("Int64")
    fx["team_h"] = pd.to_numeric(fx["team_h"], errors="coerce").astype("Int64")
    fx["team_a"] = pd.to_numeric(fx["team_a"], errors="coerce").astype("Int64")
    fx = fx.dropna(subset=["event", "team_h", "team_a"]).astype({"event": int, "team_h": int, "team_a": int})

    # Choose future GWs
    future_events = sorted(e for e in fx["event"].unique().tolist() if e > asof_gw)[:horizon]
    if not future_events:
        return pd.DataFrame(columns=["season", "gw", "team_id", "factor_mean", "n_fixtures"])

    # Get side-aware strengths as of 'asof_gw' (indexed by team_id)
    ts_asof = latest_team_strength_asof(team_strength_wide, season, asof_gw)
    if ts_asof.empty:
        return pd.DataFrame(columns=["season", "gw", "team_id", "factor_mean", "n_fixtures"])

    needed_cols = ["att_home_norm", "att_away_norm", "def_home_norm", "def_away_norm"]
    for c in needed_cols:
        if c not in ts_asof.columns:
            ts_asof[c] = 1.0
    if ts_asof.index.name != "team_id":
        ts_asof.index.name = "team_id"

    # Compute per-fixture factor
    f = fx[fx["event"].isin(future_events)][["event", "team_h", "team_a"]].rename(columns={"event": "gw"}).copy()

    def get_norm(team_id: int, col: str) -> float:
        try:
            return float(ts_asof.loc[int(team_id), col])
        except Exception:
            return 1.0

    rows = []
    for _, r in f.iterrows():
        gw = int(r["gw"]); th = int(r["team_h"]); ta = int(r["team_a"])
        # home perspective
        home_factor = get_norm(th, "att_home_norm") * get_norm(ta, "def_away_norm")
        rows.append({"season": season, "gw": gw, "team_id": th, "factor": home_factor})
        # away perspective
        away_factor = get_norm(ta, "att_away_norm") * get_norm(th, "def_home_norm")
        rows.append({"season": season, "gw": gw, "team_id": ta, "factor": away_factor})

    per_team_gw = (
        pd.DataFrame(rows)
        .groupby(["season", "gw", "team_id"], as_index=False)
        .agg(factor_mean=("factor", "mean"), n_fixtures=("factor", "size"))
    )
    return per_team_gw


def apply_fixture_factors_to_preds(
    preds: pd.DataFrame,
    player_meta: pd.DataFrame,   # columns: player_id, team_id
    team_factors: pd.DataFrame,  # columns: season, gw, team_id, factor_mean, n_fixtures
    season: str
) -> pd.DataFrame:
    """
    Merge preds with player's team and team fixture factors per (season, team_id, gw).
    New columns:
      factor_mean, n_fixtures, exp_points_scaled = exp_points * factor_mean * n_fixtures
    """
    if preds.empty:
        return preds.copy()

    out = preds.copy()
    out["season"] = str(season)

    # attach team_id to each player
    meta = player_meta.copy()
    meta["player_id"] = pd.to_numeric(meta["player_id"], errors="coerce").astype("Int64")
    meta["team_id"]   = pd.to_numeric(meta["team_id"],   errors="coerce").astype("Int64")
    out = out.merge(meta[["player_id", "team_id"]], on="player_id", how="left")

    # enforce dtypes for merge keys
    out["gw"]      = pd.to_numeric(out["gw"], errors="coerce").astype(int)
    out["team_id"] = pd.to_numeric(out["team_id"], errors="coerce").astype("Int64")

    tf = team_factors.copy()
    tf["season"]  = tf["season"].astype(str)
    tf["gw"]      = pd.to_numeric(tf["gw"], errors="coerce").astype(int)
    tf["team_id"] = pd.to_numeric(tf["team_id"], errors="coerce").astype("Int64")

    out = out.merge(tf, on=["season", "team_id", "gw"], how="left")

    out["factor_mean"] = out["factor_mean"].fillna(1.0)
    out["n_fixtures"]  = out["n_fixtures"].fillna(1).astype(int)
    out["exp_points_scaled"] = out["exp_points"] * out["factor_mean"] * out["n_fixtures"]
    return out
