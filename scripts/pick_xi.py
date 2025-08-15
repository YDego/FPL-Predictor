# scripts/pick_xi.py
from pathlib import Path
import pandas as pd
from blocks.xi_picker import pick_starting_xi

SQUAD_PATH = Path("data_cache/initial_squad.csv")               # from optimize_squad.py
PREDS_PATH = Path("data_cache/preds_fixture_aware.parquet")     # from fixture_aware_recency.py
TARGET_GW  = 31  # change as needed

def main():
    if not SQUAD_PATH.exists():
        raise FileNotFoundError(f"Squad not found: {SQUAD_PATH}. Run scripts/optimize_squad.py first.")
    if not PREDS_PATH.exists():
        raise FileNotFoundError(f"Per-GW predictions not found: {PREDS_PATH}. "
                                "Run scripts/fixture_aware_recency.py with the save snippet.")

    squad = pd.read_csv(SQUAD_PATH)
    preds = pd.read_parquet(PREDS_PATH)

    # coerce types
    for df in (squad, preds):
        if "player_id" in df.columns:
            df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")

    # sanity
    if "position" not in squad.columns:
        raise RuntimeError("Squad CSV missing 'position'. Re-run optimize_squad.py so it saves full metadata.")

    result = pick_starting_xi(squad, preds, target_gw=TARGET_GW)

    print(f"\n=== Starting XI for GW{result['target_gw']} (obj xPts: {result['objective']:.2f}) ===")
    print(result["xi"][["web_name","team_short","position","xpts_gw"]]
          .rename(columns={"web_name":"Player","team_short":"Tm","position":"Pos","xpts_gw":"xPts"})
          .to_string(index=False, formatters={"xPts":"{:,.2f}".format}))

    c = result["captain"]; v = result["vice"]
    print(f"\nCaptain: {c['web_name']} ({c['team_short']}, {c['position']}) – {c['xpts_gw']:.2f} xPts")
    print(f"   Vice: {v['web_name']} ({v['team_short']}, {v['position']}) – {v['xpts_gw']:.2f} xPts")

    print("\nBench GK:")
    print(result["bench_gk"][["web_name","team_short","xpts_gw"]]
          .rename(columns={"web_name":"Player","team_short":"Tm","xpts_gw":"xPts"})
          .to_string(index=False, formatters={"xPts":"{:,.2f}".format}))

    print("\nBench (outfield, order 1→3):")
    print(result["bench_outfield"][["web_name","team_short","position","xpts_gw"]]
          .rename(columns={"web_name":"Player","team_short":"Tm","position":"Pos","xpts_gw":"xPts"})
          .to_string(index=False, formatters={"xPts":"{:,.2f}".format}))

if __name__ == "__main__":
    main()
