# scripts/fixture_aware_recency.py
from pathlib import Path
import pandas as pd  # <-- needed

from blocks.data_io import build_multi_season_history, build_team_matches_from_players
from blocks.meta import load_player_meta, load_teams_labels
from blocks.recency import build_recency_for_gws
from blocks.team_strength import compute_team_strength
from blocks.fixture_scaling import compute_team_fixture_factors, apply_fixture_factors_to_preds

REPO_ROOT = r"C:\Users\hoday\PycharmProjects\FPL_Project\Fantasy-Premier-League-master"
SEASONS    = ["2022-23","2023-24","2024-25"]
SEASON     = "2024-25"
HORIZON    = 4
OVERRIDE_AS_OF_GW = 30   # set to None to auto, or e.g. 30 to simulate mid-season

def main():
    # 1) history & team matches
    hist = build_multi_season_history(REPO_ROOT, SEASONS, verbose=False)
    tm   = build_team_matches_from_players(hist, verbose=False)

    # 2) team strength (leak-safe)
    ts = compute_team_strength(tm, alpha_att=0.35, alpha_def=0.35)

    # 3) determine current_gw for this season (latest observed)
    ts_this = ts[ts["season"]==SEASON]
    if ts_this.empty:
        print(f"No team strength rows for {SEASON}.")
        return
    current_gw = int(ts_this["gw"].max())

    # 4) compute team fixture factors with optional override
    team_factors = compute_team_fixture_factors(
        REPO_ROOT, SEASON, ts,
        current_gw=current_gw,
        horizon=HORIZON,
        asof_gw_override=OVERRIDE_AS_OF_GW
    )
    print(team_factors.head(12).to_string(index=False))
    print("factor_mean uniques (sample):", team_factors["factor_mean"].unique()[:10])

    future_gws = sorted(team_factors["gw"].unique().tolist())
    if not future_gws:
        print(f"No future fixtures beyond GW{OVERRIDE_AS_OF_GW or current_gw} in {SEASON}.")
        return
    print(f"Future fixture GWs detected: {future_gws}")

    # 5) recency baseline exactly for those fixture GWs
    preds_base, latest = build_recency_for_gws(
        hist, future_gws=future_gws, season_filter=SEASON,
        alpha_pts=0.20, alpha_mins=0.20, alpha_avail=0.20, min_games_window=8
    )

    # 6) names/teams & apply fixture factors
    season_dir = Path(REPO_ROOT) / "data" / SEASON
    meta  = load_player_meta(season_dir)      # columns: player_id, web_name, team_id
    teams = load_teams_labels(season_dir)     # columns: team_id, team_name, team_short

    # enforce types before joining
    meta["player_id"] = pd.to_numeric(meta["player_id"], errors="coerce").astype("Int64")
    meta["team_id"]   = pd.to_numeric(meta["team_id"],   errors="coerce").astype("Int64")
    teams["team_id"]  = pd.to_numeric(teams["team_id"],  errors="coerce").astype("Int64")

    preds_scaled = apply_fixture_factors_to_preds(
        preds_base,
        meta[["player_id","team_id"]],
        team_factors,
        season=SEASON,  # IMPORTANT: merge on (season, team_id, gw)
    )

    # Save per-GW predictions (fixture-aware)
    Path("data_cache").mkdir(exist_ok=True)
    preds_scaled[["player_id", "gw", "exp_points_scaled", "reliability"]].to_parquet(
        "data_cache/preds_fixture_aware.parquet",
        index=False
    )
    print("Saved per-GW fixture-aware predictions → data_cache/preds_fixture_aware.parquet")

    # 7) pretty print next GW
    next_gw = min(future_gws)
    show = preds_scaled[preds_scaled["gw"] == next_gw].copy()

    # attach names + team short without duplicating team_id
    show = show.merge(meta[["player_id","web_name"]], on="player_id", how="left") \
               .merge(teams[["team_id","team_short"]], on="team_id", how="left")

    # guard if merges somehow miss
    if "web_name" not in show.columns:
        show["web_name"] = show["player_id"].astype(str)
    if "team_short" not in show.columns:
        show["team_short"] = show["team_id"].astype(str)

    cols = ["gw","web_name","team_short","exp_points","factor_mean","n_fixtures","exp_points_scaled","reliability"]
    show = show[cols].sort_values("exp_points_scaled", ascending=False).head(20)

    print(f"Top 20 for GW{next_gw} (fixture-aware):")
    print(show.to_string(index=False))

    # ---- Aggregate across horizon (GW31..GW34 here) ----
    horizon_set = set(future_gws)
    agg = (preds_scaled[preds_scaled["gw"].isin(horizon_set)]
           .groupby("player_id", as_index=False)
           .agg(
               horizon_xpts=("exp_points_scaled", "sum"),
               mean_reliability=("reliability", "mean"),
               gws=("gw", "nunique")
           )
           .merge(meta[["player_id","web_name","team_id"]], on="player_id", how="left")
           .merge(teams[["team_id","team_short"]], on="team_id", how="left")
           .sort_values("horizon_xpts", ascending=False))

    print("\nTop 20 across horizon (fixture-aware total):")
    print(agg[["web_name","team_short","horizon_xpts","mean_reliability","gws"]]
          .head(20)
          .to_string(index=False))

    # cache for optimizer
    Path("data_cache").mkdir(exist_ok=True)
    agg.to_csv("data_cache/horizon_xpts_fixture_aware.csv", index=False)
    agg.to_parquet("data_cache/horizon_xpts_fixture_aware.parquet", index=False)
    print("\nSaved: data_cache/horizon_xpts_fixture_aware.[csv|parquet]")


if __name__ == "__main__":
    main()
