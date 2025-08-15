# scripts/fixture_aware_recency.py
from __future__ import annotations

from pathlib import Path
import pandas as pd

from blocks.history_loader import (
    find_repo_root,
    seasons_with_gws,
    load_history_multi_season,
)
from blocks.ids import attach_codes_to_history              # adds stable player_key (FPL `code`)
from blocks.recency import build_recency_baseline           # recency baseline, horizon-aware
from blocks.fixture_fdr import compute_team_fixture_factors_from_fdr
from blocks.meta import load_live_prices_positions          # live FPL API: names, teams, prices

# ==== Tunables ====
H = 4  # horizon in GWs (you can change; file name no longer encodes H)

OUT_PRED_PER_GW = Path("data_cache/preds_fixture_aware.parquet")
OUT_HORIZON_AGG = Path("data_cache/horizon_xpts_fixture_aware.parquet")


def main():
    # 0) Resolve Vaastav repo root and select seasons that actually have GW files
    repo_root = find_repo_root(Path(__file__).resolve())
    print(f"[info] Vaastav repo: {repo_root}")

    detected = seasons_with_gws(repo_root)
    seasons = detected[-3:]  # last 3 seasons with real GW files
    print(f"[info] seasons selected: {seasons}")

    # 1) Load multi-season history and attach stable player_key (FPL `code`)
    hist = load_history_multi_season(repo_root, seasons, verbose=True)
    if hist.empty:
        raise RuntimeError("History is empty. Check the repo layout: data/<season>/gws/gw*.csv")

    # Map to stable key (FPL 'code') so IDs don't change across seasons
    hist = attach_codes_to_history(hist, str(repo_root))
    if "player_key" not in hist.columns:
        raise RuntimeError("attach_codes_to_history failed to add 'player_key'")

    # Sanity: identify the latest season and last played GW in that season
    latest_season = str(sorted(hist["season"].unique())[-1])
    last_gw = int(hist.loc[hist["season"] == latest_season, "gw"].max())
    print(f"[info] latest season: {latest_season} | last_gw: {last_gw}")

    # 2) Build recency baseline for the next H GWs (keyed by player_key)
    #    Output columns: ['player_key','gw','exp_points','reliability',...]
    base = build_recency_baseline(history=hist, horizon=H, player_key_col="player_key")
    future_gws = sorted(base["gw"].unique())
    print(f"[info] future GWs from recency baseline: {future_gws}")

    # 3) For fixture scaling, map each player to their latest team_id in the latest season
    latest_team = (
        hist[hist["season"] == latest_season]
        .sort_values(["player_key", "gw"])
        .groupby("player_key", as_index=False)
        .tail(1)[["player_key", "team_id"]]
    )

    preds = base.merge(latest_team, on="player_key", how="left")

    # 4) FDR-based fixture scaling from Vaastav fixtures.csv (latest season)
    fixtures_csv = repo_root / "data" / latest_season / "fixtures.csv"
    if not fixtures_csv.exists():
        raise FileNotFoundError(f"fixtures.csv not found: {fixtures_csv}")

    team_factors = compute_team_fixture_factors_from_fdr(
        fixtures_csv_path=fixtures_csv,
        season=latest_season,
        future_gws=future_gws,
    )
    # team_factors: ['season','gw','team_id','factor_mean','n_fixtures']

    preds = preds.merge(
        team_factors[["gw", "team_id", "factor_mean", "n_fixtures"]],
        on=["gw", "team_id"],
        how="left",
    )
    preds["factor_mean"] = preds["factor_mean"].fillna(1.0)
    preds["n_fixtures"] = preds["n_fixtures"].fillna(1).astype(int)
    preds["exp_points_scaled"] = preds["exp_points"] * preds["factor_mean"]

    # 5) Save per-GW predictions (keep player_key for later joins)
    OUT_PRED_PER_GW.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(OUT_PRED_PER_GW, index=False)

    # 6) Aggregate horizon: sum over GWs
    agg = (
        preds.groupby("player_key", as_index=False)
        .agg(
            horizon_xpts=("exp_points_scaled", "sum"),
            mean_reliability=("reliability", "mean"),
        )
    )

    # 7) Live metadata preview (current season only): names, price, team short name
    try:
        live = load_live_prices_positions().rename(columns={"code": "player_key"})
        show = (
            agg.merge(live[["player_key", "web_name", "team_short", "now_cost"]], on="player_key", how="left")
            .sort_values("horizon_xpts", ascending=False)
            .head(15)
        )
        print("\nTop 15 (fixture-aware via FDR):")
        print(
            show[["web_name", "team_short", "now_cost", "horizon_xpts", "mean_reliability"]]
            .to_string(index=False)
        )
    except Exception as e:
        print(f"[warn] live preview skipped: {e}")

    # 8) Save horizon aggregation (with player_key)
    OUT_HORIZON_AGG.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(OUT_HORIZON_AGG, index=False)

    print(f"\nSaved:\n  {OUT_PRED_PER_GW}\n  {OUT_HORIZON_AGG}")


if __name__ == "__main__":
    main()
