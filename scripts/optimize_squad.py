from pathlib import Path
import pandas as pd
from blocks.optimizer import pick_initial_squad
from blocks.meta import load_player_prices_positions, load_teams_labels, _build_web_name

REPO_ROOT = r"C:\Users\hoday\PycharmProjects\FPL_Project\Fantasy-Premier-League-master"
SEASON     = "2024-25"
AGG_PATH   = Path("data_cache/horizon_xpts_fixture_aware.parquet")
BUDGET     = 100.0
RELIAB_W   = 0.30


def _ensure_player_id(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure 'player_id' is a normal column (not in the index), or raise a clear error."""
    if "player_id" in df.columns:
        return df
    idx_name = getattr(df.index, "name", None)
    if idx_name == "player_id":
        return df.reset_index()
    if hasattr(df.index, "names") and df.index.names and "player_id" in df.index.names:
        return df.reset_index()
    if "id" in df.columns:  # last resort
        return df.rename(columns={"id": "player_id"})
    raise KeyError("Horizon file is missing 'player_id' as a column or index. "
                   "Check the aggregation step in fixture_aware_recency.py.")


def main():
    # ---- 0) Inputs & guards ----
    if not AGG_PATH.exists():
        raise FileNotFoundError(f"{AGG_PATH} not found. Run scripts/fixture_aware_recency.py first.")

    # ---- 1) Load horizon xPts and ensure 'player_id' column exists ----
    agg = pd.read_parquet(AGG_PATH)

    # Ensure player_id is a proper column
    if "player_id" not in agg.columns:
        if agg.index.name == "player_id":
            agg = agg.reset_index()
        elif hasattr(agg.index, "names") and "player_id" in (agg.index.names or []):
            agg = agg.reset_index()
        else:
            raise KeyError("Horizon file missing 'player_id'. Check fixture_aware_recency aggregation.")

    if "horizon_xpts" not in agg.columns:
        raise KeyError("Horizon file missing 'horizon_xpts'. Re-run fixture_aware_recency with aggregation.")
    if "mean_reliability" not in agg.columns:
        agg["mean_reliability"] = 1.0

    # Avoid any accidental 'team_id' carried in agg (we'll bring a clean one from players_raw)
    if "team_id" in agg.columns:
        agg = agg.drop(columns=["team_id"])

    # ---- 2) Season metadata from players_raw + teams ----
    season_dir = Path(REPO_ROOT) / "data" / SEASON

    # Single source of truth for id / team / position / price / web_name
    pricepos = load_player_prices_positions(season_dir)[
        ["player_id", "position", "now_cost", "team_id", "web_name"]
    ]
    teams = load_teams_labels(season_dir)[["team_id", "team_short"]]

    # ---- 3) Coerce merge-key dtypes ----
    for df in (agg, pricepos, teams):
        if "player_id" in df.columns:
            df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
        if "team_id" in df.columns:
            df["team_id"]   = pd.to_numeric(df["team_id"], errors="coerce").astype("Int64")

    # ---- 4) Build candidate pool (robust merges) ----
    cand = (
        agg.merge(pricepos, on="player_id", how="left", validate="m:1")
           .merge(teams, on="team_id", how="left", validate="m:1")
    )

    # ---- 5) Ensure display columns exist even if CSVs are sparse ----
    # web_name: create column if missing, then fill from players_raw.csv mapping
    if "web_name" not in cand.columns:
        cand["web_name"] = pd.NA
    if cand["web_name"].isna().any():
        # rebuild from players_raw quickly, defensively
        raw = pd.read_csv(season_dir / "players_raw.csv")
        # try web_name; else synthesize "F. Last" from first_name/second_name
        def _derive_web(row):
            wn = str(row.get("web_name", "")).strip()
            if wn:
                return wn
            first = str(row.get("first_name", "")).strip()
            second = str(row.get("second_name", "")).strip()
            if second:
                init = (first[:1].upper() + ". ") if first else ""
                return (init + second).strip()
            return None
        name_map = pd.Series(
            (raw.apply(_derive_web, axis=1)).values,
            index=pd.to_numeric(raw["id"], errors="coerce")
        ).to_dict()
        mask = cand["web_name"].isna()
        cand.loc[mask, "web_name"] = cand.loc[mask, "player_id"].map(name_map)
        cand["web_name"] = cand["web_name"].fillna("Unknown")

    # team_short: create column if missing, then fill from teams.csv mapping, fallback to "T{id}"
    if "team_short" not in cand.columns:
        cand["team_short"] = pd.NA
    if cand["team_short"].isna().any():
        teams2 = load_teams_labels(season_dir)
        tmap = teams2.set_index("team_id")["team_short"].to_dict()
        mask = cand["team_short"].isna()
        cand.loc[mask, "team_short"] = cand.loc[mask, "team_id"].map(tmap)
        cand["team_short"] = cand["team_short"].fillna(
            cand["team_id"].apply(lambda x: f"T{int(x)}" if pd.notna(x) else "T?")
        )

    # ---- 6) Sanity filters ----
    if "position" not in cand.columns or "now_cost" not in cand.columns:
        raise RuntimeError("Missing 'position' or 'now_cost' after merges. Check players_raw.csv content.")
    cand = cand.dropna(subset=["position", "now_cost"])
    cand = cand[cand["now_cost"] > 3.5]            # drop unusable placeholders
    cand["now_cost"] = cand["now_cost"].round(1)

    if cand.empty:
        raise RuntimeError("Candidate pool is empty after merges/filters. "
                           "Ensure season in SEASON matches your horizon data season.")

    # ---- 7) Optimize 15-man squad ----
    result = pick_initial_squad(
        candidates=cand,
        budget=BUDGET,
        reliability_weight=RELIAB_W
    )

    print(f"\nStatus: {result['status']}")
    print(f"Objective (score): {result['totals']['score']}   "
          f"Cost: {result['totals']['cost']}m   "
          f"Horizon xPts: {result['totals']['horizon_xpts']}")

    # ---- 8) Pretty print by position ----
    for pos in ["GKP", "DEF", "MID", "FWD"]:
        dfp = result["by_position"][pos]
        if dfp.empty:
            continue
        out = dfp.rename(columns={
            "web_name": "Player",
            "team_short": "Tm",
            "now_cost": "£m",
            "horizon_xpts": "HxPts",
            "mean_reliability": "Rel",
            "score": "Score",
        })
        print(f"\n{pos} ({len(out)}):")
        print(out.to_string(
            index=False,
            formatters={
                "£m": "{:,.1f}".format,
                "HxPts": "{:,.2f}".format,
                "Rel": "{:,.2f}".format,
                "Score": "{:,.2f}".format,
            },
        ))

    # ---- 9) Save chosen squad ----
    Path("data_cache").mkdir(exist_ok=True)
    out_path = Path("data_cache/initial_squad.csv")
    result["selected"].to_csv(out_path, index=False)
    print(f"\nSaved chosen squad to {out_path}")



if __name__ == "__main__":
    main()
