# scripts/optimize_squad.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Iterable, Dict
import pandas as pd

from blocks.meta import load_live_prices_positions

# -------- CONFIG --------
HORIZON_AGG_PATH = "data_cache/horizon_xpts_fixture_aware.parquet"

BUDGET        = 100.0
RELIAB_WEIGHT = 0.30

DROP_STATUSES: set[str] = {"u"}
MIN_CHANCE: Optional[float] = 10.0
MIN_RELIABILITY = 0.60

TOPK_PER_POS: Dict[str, int] = {"GKP": 20, "DEF": 50, "MID": 80, "FWD": 40}
# ------------------------


def _print_block(title: str, rows: pd.DataFrame, fields: Iterable[str]):
    print(f"\n{title}")
    if rows.empty:
        print("(none)")
        return
    print(" " + "  ".join(f"{h:>10}" for h in fields))
    for _, r in rows.iterrows():
        vals = []
        for h in fields:
            v = r[h]
            if h in {"£m","now_cost"}:
                vals.append(f"{float(v):.1f}")
            elif h in {"HxPts","Rel","Score","horizon_xpts","mean_reliability","score"}:
                vals.append(f"{float(v):.2f}")
            else:
                vals.append(str(v))
        print(" " + "  ".join(f"{v:>10}" for v in vals))


def main():
    # 0) Live meta first (has code + current team/price)
    meta = load_live_prices_positions().rename(columns={"code": "player_key"})
    print(f"[meta] LIVE players: {len(meta)}")

    # Availability filtering (live status)
    if DROP_STATUSES:
        meta = meta[~meta["status"].isin(DROP_STATUSES)].copy()
    if MIN_CHANCE is not None:
        meta = meta[(meta["chance_of_playing_next_round"].isna()) |
                    (meta["chance_of_playing_next_round"] >= float(MIN_CHANCE))].copy()

    # 1) Load horizon predictions
    agg_path = Path(HORIZON_AGG_PATH)
    if not agg_path.exists():
        print(f"[error] Missing horizon xPts file: {agg_path}")
        print("        Run your fixture-aware prediction script first.")
        sys.exit(1)

    agg = pd.read_parquet(agg_path)

    # 2) Ensure we have a stable key to join on (player_key = code)
    if "player_key" not in agg.columns:
        # Best-effort recovery: if horizon has player_id, map to code via LIVE meta (current season).
        if "player_id" in agg.columns:
            print("[warn] horizon file lacks 'player_key'; creating it from 'player_id' using LIVE API map (current season only).")
            pid_to_code = meta.set_index("player_id")["player_key"].to_dict()
            agg["player_key"] = agg["player_id"].map(pid_to_code)
        else:
            raise KeyError("Horizon file has neither 'player_key' nor 'player_id'. "
                           "Regenerate predictions to include 'player_key' (FPL code).")

    need = {"player_key","horizon_xpts","mean_reliability"}
    missing = need - set(agg.columns)
    if missing:
        raise KeyError(f"horizon file missing columns after recovery: {missing}")

    # 3) Build candidate pool — JOIN BY CODE
    cand = (
        agg.merge(meta, on="player_key", how="inner")
           .dropna(subset=["team_id","team_short","position","now_cost"])
           .copy()
    )

    # Guardrails
    cand = cand[(cand["now_cost"] >= 4.0) & (cand["now_cost"] <= 15.5)]
    cand = cand[cand["mean_reliability"] >= MIN_RELIABILITY]

    # Debug: premium mids should show up
    mids_dbg = (cand[cand["position"]=="MID"]
                .sort_values("horizon_xpts", ascending=False)
                .head(10)[["web_name","team_short","now_cost","horizon_xpts","mean_reliability"]])
    print("\n[debug] Top 10 MID by horizon_xpts:")
    print(mids_dbg.to_string(index=False))
    print("\n[debug] Candidate team codes:", sorted(cand["team_short"].dropna().unique().tolist()))

    # 4) Prune to top-K per position
    parts = []
    for pos, k in TOPK_PER_POS.items():
        parts.append(cand[cand["position"] == pos].sort_values("horizon_xpts", ascending=False).head(k))
    cand = pd.concat(parts, ignore_index=True) if parts else cand

    # 5) Optimize
    from blocks.optimizer import pick_initial_squad

    result = pick_initial_squad(
        candidates=cand,
        budget=BUDGET,
        pos_limits=None,
        club_limit=3,
        reliability_weight=RELIAB_WEIGHT,
        solver_time_limit=None,
    )
    status = result.get("status", "Unknown")
    sel: pd.DataFrame = result["selected"].copy()
    totals = result.get("totals", {})

    print(f"\nStatus: {status}")
    print(f"Objective (score): {totals.get('score', 0):.2f}   Cost: {totals.get('cost', 0):.1f}m   Horizon xPts: {totals.get('horizon_xpts', 0):.2f}")

    def _fmt(df: pd.DataFrame) -> pd.DataFrame:
        out = df[["web_name","team_short","now_cost","horizon_xpts","mean_reliability","score"]].copy()
        out.columns = ["Player","Tm","£m","HxPts","Rel","Score"]
        return out

    def _blk(name, df):
        _print_block(f"\n\n{name}", _fmt(df), ["Player","Tm","£m","HxPts","Rel","Score"])

    _blk("GKP (2):", sel[sel["position"]=="GKP"])
    _blk("DEF (5):", sel[sel["position"]=="DEF"])
    _blk("MID (5):", sel[sel["position"]=="MID"])
    _blk("FWD (3):", sel[sel["position"]=="FWD"])

    Path("data_cache").mkdir(exist_ok=True)
    sel.to_csv(Path("data_cache")/"initial_squad.csv", index=False, encoding="utf-8")
    print("\nSaved chosen squad to data_cache/initial_squad.csv")


if __name__ == "__main__":
    main()
