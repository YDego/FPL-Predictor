# blocks/fixture_fdr.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
from typing import Iterable, Tuple

def _fdr_to_factor(d: float) -> float:
    """
    Map FDR (1 easiest … 5 hardest) to a multiplicative factor around 1.0.
    Gentle, symmetric scaling to avoid blowing up recency estimates.
      1 -> 1.30, 2 -> 1.15, 3 -> 1.00, 4 -> 0.85, 5 -> 0.70
    """
    if pd.isna(d):  # missing
        return 1.0
    return {1: 1.30, 2: 1.15, 3: 1.00, 4: 0.85, 5: 0.70}.get(int(d), 1.0)

def compute_team_fixture_factors_from_fdr(
    fixtures_csv_path: Path,
    season: str,
    future_gws: Iterable[int],
) -> pd.DataFrame:
    """
    Build per-(gw, team_id) fixture scaling factors from FDR in Vaastav fixtures.csv.
    Handles blanks/doubles (n_fixtures >= 0).
    Returns columns: ['season','gw','team_id','factor_mean','n_fixtures']
    """
    fx = pd.read_csv(fixtures_csv_path)
    fx.columns = [c.strip().lower() for c in fx.columns]

    # expected columns in Vaastav fixtures.csv
    need = {"event","team_h","team_a","team_h_difficulty","team_a_difficulty"}
    missing = need - set(fx.columns)
    if missing:
        raise KeyError(f"fixtures.csv missing columns: {missing}")

    fx = fx.rename(columns={
        "event": "gw",
        "team_h": "team_h_id",
        "team_a": "team_a_id",
        "team_h_difficulty": "h_fdr",
        "team_a_difficulty": "a_fdr",
    })
    fx["season"] = season
    fx = fx[fx["gw"].isin(list(future_gws))].copy()

    # explode to team rows (home + away)
    home = fx[["season","gw","team_h_id","h_fdr"]].copy()
    home["team_id"] = pd.to_numeric(home["team_h_id"], errors="coerce").astype("Int64")
    home["factor"]  = home["h_fdr"].map(_fdr_to_factor)

    away = fx[["season","gw","team_a_id","a_fdr"]].copy()
    away["team_id"] = pd.to_numeric(away["team_a_id"], errors="coerce").astype("Int64")
    away["factor"]  = away["a_fdr"].map(_fdr_to_factor)

    teams = pd.concat([
        home[["season","gw","team_id","factor"]],
        away[["season","gw","team_id","factor"]],
    ], ignore_index=True)

    # aggregate (mean for doubles, count fixtures)
    out = (teams.groupby(["season","gw","team_id"], as_index=False)
                .agg(factor_mean=("factor","mean"),
                     n_fixtures=("factor","size")))
    return out
