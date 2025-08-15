# blocks/optimizer.py
from __future__ import annotations
from typing import Dict, Iterable
import pandas as pd

try:
    from pulp import LpProblem, LpVariable, LpBinary, LpMaximize, lpSum, PULP_CBC_CMD, LpStatus
except ImportError as e:
    raise ImportError("PuLP is required. Install with: pip install pulp") from e


PositionLimits = Dict[str, int]

DEFAULT_POS_LIMITS: PositionLimits = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}


def _validate_pool(pool: pd.DataFrame) -> pd.DataFrame:
    req = {"player_id", "web_name", "team_id", "team_short", "position", "now_cost", "horizon_xpts"}
    missing = req - set(pool.columns)
    if missing:
        raise ValueError(f"optimizer: candidate pool missing columns: {missing}")

    pool = pool.copy()

    # types & sanitization
    pool["player_id"] = pd.to_numeric(pool["player_id"], errors="coerce").astype(int)
    pool["team_id"]   = pd.to_numeric(pool["team_id"], errors="coerce").astype(int)
    pool["now_cost"]  = pd.to_numeric(pool["now_cost"], errors="coerce")
    pool["horizon_xpts"] = pd.to_numeric(pool["horizon_xpts"], errors="coerce")

    pool = pool.dropna(subset=["now_cost", "horizon_xpts"])
    pool = pool[pool["position"].isin(["GKP","DEF","MID","FWD"])]

    # drop clear duplicates (same FPL id)
    pool = pool.sort_values("horizon_xpts", ascending=False).drop_duplicates("player_id")
    return pool.reset_index(drop=True)


def pick_initial_squad(
    candidates: pd.DataFrame,
    budget: float = 100.0,
    pos_limits: PositionLimits = None,
    club_limit: int = 3,
    reliability_weight: float = 0.3,
    solver_time_limit: int | None = None,
) -> dict:
    """
    Choose a legal 15-man squad maximizing (horizon_xpts * (0.7 + 0.3*reliability)).
    - candidates: DataFrame with columns:
        player_id, web_name, team_id, team_short, position (GKP/DEF/MID/FWD),
        now_cost (float, in £m), horizon_xpts (float), mean_reliability (0..1 optional)
    - budget: total budget in £m
    - pos_limits: dict like {"GKP":2, "DEF":5, "MID":5, "FWD":3}
    - club_limit: max 3 per club
    - reliability_weight: blends reliability in objective:
        score_i = xpts * (1 - w) + xpts * w * rel = xpts * [ (1-w) + w*rel ]
    - solver_time_limit: seconds for CBC (optional)

    Returns dict with:
      status, obj, selected (DataFrame of 15), by_position (dict), totals (dict)
    """
    if pos_limits is None:
        pos_limits = DEFAULT_POS_LIMITS

    pool = _validate_pool(candidates)

    # ensure reliability column
    if "mean_reliability" not in pool.columns:
        pool["mean_reliability"] = 1.0
    pool["mean_reliability"] = pd.to_numeric(pool["mean_reliability"], errors="coerce").fillna(1.0).clip(0, 1)

    # effective score
    w = float(reliability_weight)
    pool["score"] = pool["horizon_xpts"] * ((1 - w) + w * pool["mean_reliability"])

    # price to integer pennies (avoid floating solver issues): multiply by 10 (FPL costs are in £0.1m steps)
    pool["price10"] = (pool["now_cost"] * 10 + 1e-6).astype(int)
    budget10 = int(round(budget * 10))

    # Variables
    model = LpProblem("FPL_Initial_Squad", LpMaximize)
    x = {i: LpVariable(f"x_{int(i)}", lowBound=0, upBound=1, cat=LpBinary) for i in pool.index}

    # Objective
    model += lpSum(pool.loc[i, "score"] * x[i] for i in pool.index)

    # Squad size
    model += lpSum(x.values()) == 15, "squad_size_15"

    # Budget
    model += lpSum(pool.loc[i, "price10"] * x[i] for i in pool.index) <= budget10, "budget"

    # Position constraints
    for pos, k in pos_limits.items():
        idx = [i for i in pool.index if pool.loc[i, "position"] == pos]
        model += lpSum(x[i] for i in idx) == k, f"count_{pos}_{k}"

    # Club constraints (max 3 per team)
    for t in pool["team_id"].unique():
        idx = [i for i in pool.index if pool.loc[i, "team_id"] == t]
        model += lpSum(x[i] for i in idx) <= club_limit, f"club_cap_{t}"

    # Solve
    cmd = PULP_CBC_CMD(msg=False, timeLimit=solver_time_limit) if solver_time_limit else PULP_CBC_CMD(msg=False)
    model.solve(cmd)

    status = LpStatus[model.status]
    selected_idx: Iterable[int] = [i for i in pool.index if x[i].value() == 1]
    chosen = pool.loc[selected_idx].copy().sort_values(["position", "score"], ascending=[True, False])

    # Totals
    total_cost = chosen["now_cost"].sum()
    total_xpts = chosen["horizon_xpts"].sum()
    total_score = chosen["score"].sum()

    by_pos = {
        p: chosen[chosen["position"] == p][
            ["web_name", "team_short", "now_cost", "horizon_xpts", "mean_reliability", "score"]
        ].reset_index(drop=True)
        for p in ["GKP", "DEF", "MID", "FWD"]
    }

    return {
        "status": status,
        "obj": total_score,
        "selected": chosen.reset_index(drop=True),
        "by_position": by_pos,
        "totals": {
            "cost": round(total_cost, 1),
            "horizon_xpts": round(total_xpts, 2),
            "score": round(total_score, 2),
        }
    }
