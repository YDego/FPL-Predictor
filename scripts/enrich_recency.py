# scripts/enrich_recency.py
from pathlib import Path
from blocks.data_io import build_multi_season_history
from blocks.recency import build_recency_predictions
from blocks.meta import load_player_meta, load_teams_labels

REPO_ROOT = r"C:\Users\hoday\PycharmProjects\FPL_Project\Fantasy-Premier-League-master"
SEASON = "2024-25"
SEASONS = ["2022-23", "2023-24", "2024-25"]
HORIZON = 4

def main():
    # 1) history
    hist = build_multi_season_history(REPO_ROOT, SEASONS, verbose=False)

    # 2) baseline recency predictions
    preds, latest = build_recency_predictions(hist, horizon_gws=HORIZON, season_filter=SEASON)

    # 3) names & team labels
    season_dir = Path(REPO_ROOT) / "data" / SEASON
    meta  = load_player_meta(season_dir)
    teams = load_teams_labels(season_dir)

    preds_named = preds.merge(meta[["player_id","web_name","team_id"]], on="player_id", how="left") \
                       .merge(teams, on="team_id", how="left")

    # 4) show the next GW nicely
    if preds_named.empty:
        print("No predictions.")
        return
    next_gw = int(preds_named["gw"].min())
    cols = ["gw","web_name","team_short","exp_points","reliability"]
    print(f"Top 20 for GW{next_gw} (H={HORIZON}):")
    print(preds_named[preds_named["gw"]==next_gw][cols].nlargest(20, "exp_points").to_string(index=False))

if __name__ == "__main__":
    main()
