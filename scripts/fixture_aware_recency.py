# scripts/fixture_aware_recency.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

from blocks.history_loader import load_history_multi_season
from blocks.ids import attach_codes_to_history
from blocks.recency import build_recency_baseline
from blocks.fixture_fdr import compute_team_fixture_factors_from_fdr
from blocks.meta import load_live_prices_positions

REPO_ROOT = r"C:\Users\hoday\PycharmProjects\FPL_Project\Fantasy-Premier-League-master"
SEASONS   = ["2022-23","2023-24","2024-25"]
H         = 4

OUT_PRED_PER_GW = Path("data_cache/preds_fixture_aware.parquet")
OUT_HORIZON_AGG = Path("data_cache/horizon_xpts_fixture_aware.parquet")

def main():
    # 1) Load history & attach stable player_key (FPL `code`)
    hist = load_history_multi_season(REPO_ROOT, SEASONS)
    hist = attach_codes_to_history(hist, REPO_ROOT)
    if hist.empty:
        raise RuntimeError("History is empty. Check repo path / seasons.")

    # latest season + last played GW in that season
    latest_season = str(sorted(hist["season"].unique())[-1])
    last_gw = int(hist.loc[hist["season"]==latest_season, "gw"].max())
    future_gws = list(range(last_gw + 1, last_gw + 1 + H))

    # 2) Recency baseline keyed by player_key (code)
    base = build_recency_baseline(
        history=hist,
        horizon=H,
        player_key_col="player_key",
    )   # -> ['player_key','gw','exp_points','reliability']

    # 3) For FUTURE GWs, assign each player's CURRENT team_id from the latest season
    latest_team = (hist[hist["season"]==latest_season]
                   .sort_values(["player_key","gw"])
                   .groupby("player_key", as_index=False)
                   .tail(1)[["player_key","team_id"]])

    preds = base.merge(latest_team, on="player_key", how="left")

    # 4) Fixture FDR scaling (no team strength needed)
    fixtures_csv = Path(REPO_ROOT) / "data" / latest_season / "fixtures.csv"
    team_factors = compute_team_fixture_factors_from_fdr(
        fixtures_csv_path=fixtures_csv,
        season=latest_season,
        future_gws=future_gws,
    )  # -> ['season','gw','team_id','factor_mean','n_fixtures']

    preds = preds.merge(
        team_factors[["gw","team_id","factor_mean","n_fixtures"]],
        on=["gw","team_id"], how="left"
    )
    preds["factor_mean"] = preds["factor_mean"].fillna(1.0)
    preds["n_fixtures"]  = preds["n_fixtures"].fillna(1).astype(int)
    preds["exp_points_scaled"] = preds["exp_points"] * preds["factor_mean"]

    # 5) Save per-GW (WITH player_key)
    OUT_PRED_PER_GW.parent.mkdir(exist_ok=True)
    preds.to_parquet(OUT_PRED_PER_GW, index=False)

    # 6) Aggregate horizon (sum over GWs)
    agg = (preds.groupby("player_key", as_index=False)
                 .agg(horizon_xpts=("exp_points_scaled","sum"),
                      mean_reliability=("reliability","mean")))

    # 7) Pretty preview using LIVE meta via code → player_key
    try:
        live = load_live_prices_positions().rename(columns={"code":"player_key"})
        show = (agg.merge(live[["player_key","web_name","team_short","now_cost"]], on="player_key", how="left")
                    .sort_values("horizon_xpts", ascending=False).head(20))
        print(f"Top 20 (fixture-aware via FDR, H={H}):")
        print(show[["web_name","team_short","now_cost","horizon_xpts","mean_reliability"]].to_string(index=False))
    except Exception as e:
        print(f"[warn] live meta preview unavailable: {e}")

    # 8) Write horizon parquet WITH `player_key`
    agg.to_parquet(OUT_HORIZON_AGG, index=False)
    print(f"\nSaved:\n  {OUT_PRED_PER_GW}\n  {OUT_HORIZON_AGG}")

if __name__ == "__main__":
    main()
