# scripts/plan_transfers.py
from pathlib import Path
import pandas as pd

from blocks.transfer_planner import plan_transfers_greedy
from blocks.meta import load_player_prices_positions, load_teams_labels

# Inputs
REPO_ROOT = r"C:\Users\hoday\PycharmProjects\FPL_Project\Fantasy-Premier-League-master"
SEASON     = "2024-25"
SQUAD_PATH = Path("data_cache/initial_squad.csv")
PREDS_PATH = Path("data_cache/preds_fixture_aware.parquet")
OUT_PLAN   = Path("data_cache/transfer_plan.csv")
START_BANK = 0.0   # adjust if you have ITB bank

# Horizon to plan (first N upcoming GWs from preds)
H = 4


def build_market(repo_root: str, season: str) -> pd.DataFrame:
    season_dir = Path(repo_root) / "data" / season
    pricepos = load_player_prices_positions(season_dir)[
        ["player_id", "position", "now_cost", "team_id", "web_name"]
    ]
    teams = load_teams_labels(season_dir)[["team_id", "team_short"]]
    market = pricepos.merge(teams, on="team_id", how="left")
    return market


def main():
    if not SQUAD_PATH.exists():
        raise FileNotFoundError(f"{SQUAD_PATH} not found. Run scripts/optimize_squad.py first.")
    if not PREDS_PATH.exists():
        raise FileNotFoundError(f"{PREDS_PATH} not found. Run scripts/fixture_aware_recency.py first.")

    squad = pd.read_csv(SQUAD_PATH)
    preds = pd.read_parquet(PREDS_PATH)

    # derive horizon GW list from preds (upcoming)
    gws_upcoming = sorted(int(g) for g in preds["gw"].unique())
    gw_list = gws_upcoming[:H]

    market = build_market(REPO_ROOT, SEASON)

    plan_df, final_squad = plan_transfers_greedy(
        squad=squad,
        preds=preds,
        market=market,
        gw_list=gw_list,
        start_bank=START_BANK,
        topK_per_pos=60,
    )

    # Pretty print
    print("\n=== Transfer Plan (1 FT per GW, greedy) ===")
    if plan_df.empty:
        print("(no plan rows)")
    else:
        show = plan_df.copy()
        show["gain"] = show["gain"].map(lambda x: f"{x:.2f}")
        show["base_xi_xpts"] = show["base_xi_xpts"].map(lambda x: f"{x:.2f}")
        show["new_xi_xpts"]  = show["new_xi_xpts"].map(lambda x: f"{x:.2f}")
        show["bank_after"]   = show["bank_after"].map(lambda x: f"{x:.1f}m")
        print(show[["gw", "action", "base_xi_xpts", "new_xi_xpts", "gain", "bank_after"]]
              .to_string(index=False))

    Path("data_cache").mkdir(exist_ok=True)
    plan_df.to_csv(OUT_PLAN, index=False)
    print(f"\nSaved transfer plan → {OUT_PLAN}")

    # (optional) save final squad after all transfers
    final_squad.to_csv("data_cache/final_squad_after_plan.csv", index=False)
    print("Saved final squad → data_cache/final_squad_after_plan.csv")


if __name__ == "__main__":
    main()
