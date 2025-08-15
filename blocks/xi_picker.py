# blocks/xi_picker.py
from __future__ import annotations
import pandas as pd

try:
    from pulp import LpProblem, LpVariable, LpBinary, LpMaximize, lpSum, PULP_CBC_CMD, LpStatus
except ImportError as e:
    raise ImportError("PuLP is required. Install with: pip install pulp") from e


def pick_starting_xi(
    squad: pd.DataFrame,
    gw_preds: pd.DataFrame,
    target_gw: int,
    min_def: int = 3,
    min_mid: int = 2,
    min_fwd: int = 1,
    solver_time_limit: int | None = None,
):
    """
    Choose a legal XI for target_gw.

    squad: DataFrame of your 15 with columns:
        player_id, web_name, team_short, position in {GKP,DEF,MID,FWD}, now_cost
    gw_preds: per-GW predictions with columns:
        player_id, gw, exp_points_scaled
    target_gw: int

    Returns dict with: xi, bench_gk, bench_outfield, captain, vice
    """
    # ensure keys
    for df in (squad, gw_preds):
        if "player_id" in df.columns:
            df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")

    # join next-GW xPts
    p = gw_preds[gw_preds["gw"] == target_gw][["player_id", "exp_points_scaled"]].rename(
        columns={"exp_points_scaled": "xpts_gw"}
    )
    pool = squad.merge(p, on="player_id", how="left")
    pool["xpts_gw"] = pd.to_numeric(pool["xpts_gw"], errors="coerce").fillna(0.0)

    # sanity: require exactly 15, and positions look ok
    if len(pool) != 15:
        raise ValueError(f"Expected squad of 15, got {len(pool)} rows.")
    if not set(pool["position"]).issubset({"GKP", "DEF", "MID", "FWD"}):
        raise ValueError("Unknown positions present in squad.")

    # MILP
    model = LpProblem("FPL_XI_Selection", LpMaximize)
    idx = list(pool.index)
    y = {i: LpVariable(f"y_{i}", lowBound=0, upBound=1, cat=LpBinary) for i in idx}

    # objective
    model += lpSum(pool.loc[i, "xpts_gw"] * y[i] for i in idx)

    # constraints
    model += lpSum(y[i] for i in idx) == 11, "xi_size_11"

    # exactly 1 GK
    gk_idx = [i for i in idx if pool.loc[i, "position"] == "GKP"]
    model += lpSum(y[i] for i in gk_idx) == 1, "gk_exactly_one"

    # min position constraints (outfield)
    def_idx = [i for i in idx if pool.loc[i, "position"] == "DEF"]
    mid_idx = [i for i in idx if pool.loc[i, "position"] == "MID"]
    fwd_idx = [i for i in idx if pool.loc[i, "position"] == "FWD"]

    model += lpSum(y[i] for i in def_idx) >= min_def, "min_def"
    model += lpSum(y[i] for i in mid_idx) >= min_mid, "min_mid"
    model += lpSum(y[i] for i in fwd_idx) >= min_fwd, "min_fwd"

    cmd = PULP_CBC_CMD(msg=False, timeLimit=solver_time_limit) if solver_time_limit else PULP_CBC_CMD(msg=False)
    model.solve(cmd)

    status = LpStatus[model.status]
    chosen_idx = [i for i in idx if y[i].value() == 1]
    xi = pool.loc[chosen_idx].copy()
    bench = pool.loc[[i for i in idx if i not in chosen_idx]].copy()

    # captain & vice: top two by xPts among starters
    xi_sorted = xi.sort_values("xpts_gw", ascending=False).reset_index(drop=True)
    captain = xi_sorted.iloc[0][["player_id", "web_name", "team_short", "position", "xpts_gw"]].to_dict()
    vice    = xi_sorted.iloc[1][["player_id", "web_name", "team_short", "position", "xpts_gw"]].to_dict()

    # bench: GK is the non-starting GK; outfield bench ordered by xPts desc
    bench_gk = bench[bench["position"] == "GKP"].copy()
    if len(bench_gk) != 1:
        # in case of any data weirdness, take the lowest xPts GK as bench
        bench_gk = pool[pool["position"] == "GKP"].sort_values("xpts_gw").head(1).copy()

    bench_outfield = bench[bench["position"] != "GKP"].copy().sort_values("xpts_gw", ascending=False)

    # pretty ordering for XI
    xi = xi.sort_values(["position", "xpts_gw"], ascending=[True, False]).reset_index(drop=True)

    return {
        "status": status,
        "xi": xi,
        "bench_gk": bench_gk.reset_index(drop=True),
        "bench_outfield": bench_outfield.reset_index(drop=True),
        "captain": captain,
        "vice": vice,
        "target_gw": int(target_gw),
        "objective": float(sum(xi["xpts_gw"])),
    }
