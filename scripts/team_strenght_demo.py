# scripts/team_strength_demo.py
from blocks.data_io import build_multi_season_history, build_team_matches_from_players
from blocks.team_strength import compute_team_strength

REPO = r"C:\Users\hoday\PycharmProjects\FPL_Project\Fantasy-Premier-League-master"
SEASONS = ["2022-23","2023-24","2024-25"]

def main():
    hist = build_multi_season_history(REPO, SEASONS, verbose=False)
    tm   = build_team_matches_from_players(hist, verbose=False)

    ts = compute_team_strength(tm, alpha_att=0.35, alpha_def=0.35)

    # peek latest gw in 2024-25
    ts_2425 = ts[ts["season"]=="2024-25"]
    if ts_2425.empty:
        print("No 2024-25 rows.")
        return
    last_gw = int(ts_2425["gw"].max())
    view = ts_2425[ts_2425["gw"]==last_gw] \
        .sort_values("att_home_norm", ascending=False) \
        .head(10)[["team_id","att_home_norm","att_away_norm","def_home_norm","def_away_norm"]]
    print(f"Top 10 attack-home at GW{last_gw} (norm ~1):")
    print(view.to_string(index=False))
    chk = ts[ts["season"] == "2024-25"].groupby(["gw"]).agg(
        mean_att_home=("att_home_norm", "mean"),
        mean_att_away=("att_away_norm", "mean"),
        mean_def_home=("def_home_norm", "mean"),
        mean_def_away=("def_away_norm", "mean"),
    )
    print(chk.tail())


if __name__ == "__main__":
    main()
