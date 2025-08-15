# blocks/optimizer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Iterable, Dict

import pandas as pd
from pulp import LpProblem, LpVariable, LpBinary, LpMaximize, lpSum, PULP_CBC_CMD


@dataclass
class PositionLimits:
    GKP: int = 2
    DEF: int = 5
    MID: int = 5
    FWD: int = 3
    SQUAD: int = 15


def _validate_pool(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the candidate pool has everything we need.
    Accepts either 'player_id' OR 'player_key' (stable code).
    Creates a 'uid' column used internally by the solver.
    """
    required = {
        "web_name", "team_short", "team_id", "position",
        "now_cost", "horizon_xpts", "mean_reliability"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"optimizer: candidate pool missing columns: {missing}")

    if ("player_id" not in df.columns) and ("player_key" not in df.columns):
        raise ValueError("optimizer: need one of 'player_id' or 'player_key' in candidates")

    pool = df.copy()

    # build a unique id we can always rely on
    if "player_id" in pool.columns and pool["player_id"].notna().any():
        pool["uid"] = pool["player_id"].astype("Int64").astype(str)
    else:
        # fallback to stable code (player_key)
        pool["uid"] = pool["player_key"].astype("Int64").astype(str)

    # clean types
    pool["now_cost"] = pd.to_numeric(pool["now_cost"], errors="coerce").fillna(0.0)
    pool["horizon_xpts"] = pd.to_numeric(pool["horizon_xpts"], errors="coerce").fillna(0.0)
    pool["mean_reliability"] = pd.to_numeric(pool["mean_reliability"], errors="coerce").fillna(0.0)

    # restrict to valid positions
    pool = pool[pool["position"].isin(["GKP", "DEF", "MID", "FWD"])].copy()

    # guardrails on costs
    pool = pool[(pool["now_cost"] >= 3.5) & (pool["now_cost"] <= 15.5)].copy()

    # de-dup by uid (keep best by xPts, then cheaper)
    pool = (pool.sort_values(["horizon_xpts", "now_cost"], ascending=[False, True])
                 .drop_duplicates(subset=["uid"], keep="first"))

    return pool.reset_index(drop=True)


def pick_initial_squad(
    candidates: pd.DataFrame,
    budget: float = 100.0,
    pos_limits: PositionLimits = None,
    club_limit: int = 3,
    reliability_weight: float = 0.3,
    solver_time_limit: Optional[int] = None,
) -> dict:
    """
    MILP: choose a 15-player squad maximizing reliability-weighted horizon xPts
    subject to budget, position counts, and club limit.

    candidates must include:
      web_name, team_short, team_id, position, now_cost, horizon_xpts, mean_reliability
      and either player_id or player_key (stable FPL 'code')
    """
    pool = _validate_pool(candidates)
    if pos_limits is None:
        pos_limits = PositionLimits()

    # weighted score
    # score = HxPts * ( (1 - w) + w * reliability )
    w = float(reliability_weight)
    pool = pool.copy()
    pool["score"] = pool["horizon_xpts"] * ((1.0 - w) + w * pool["mean_reliability"])

    # decision vars
    prob = LpProblem("FPL_Initial_Squad", LpMaximize)
    x = {uid: LpVariable(f"x_{uid}", lowBound=0, upBound=1, cat=LpBinary)
         for uid in pool["uid"]}

    # objective
    prob += lpSum(pool.set_index("uid").loc[uid, "score"] * x[uid] for uid in x)

    # budget
    prob += lpSum(pool.set_index("uid").loc[uid, "now_cost"] * x[uid] for uid in x) <= budget

    # position counts
    by_pos = pool.groupby("position")["uid"].apply(list).to_dict()
    prob += lpSum(x[uid] for uid in by_pos.get("GKP", [])) == pos_limits.GKP
    prob += lpSum(x[uid] for uid in by_pos.get("DEF", [])) == pos_limits.DEF
    prob += lpSum(x[uid] for uid in by_pos.get("MID", [])) == pos_limits.MID
    prob += lpSum(x[uid] for uid in by_pos.get("FWD", [])) == pos_limits.FWD

    # total squad size
    prob += lpSum(x[uid] for uid in x) == pos_limits.SQUAD

    # club limit: max N per team_id
    for team_id, uids in pool.groupby("team_id")["uid"]:
        prob += lpSum(x[uid] for uid in uids) <= club_limit

    # solve
    solver = PULP_CBC_CMD(timeLimit=solver_time_limit) if solver_time_limit else PULP_CBC_CMD()
    prob.solve(solver)

    status = prob.status  # 1 = Optimal
    pool_idx = pool.set_index("uid")

    chosen_uids = [uid for uid in x if x[uid].value() == 1]
    selected = pool_idx.loc[chosen_uids].reset_index()

    # objective components for reporting
    total_cost = float(selected["now_cost"].sum()) if not selected.empty else 0.0
    total_xpts = float(selected["horizon_xpts"].sum()) if not selected.empty else 0.0
    total_score = float(selected["score"].sum()) if not selected.empty else 0.0

    return {
        "status": {1: "Optimal", -1: "Infeasible"}.get(status, str(status)),
        "selected": selected,  # includes uid, player_id (if present), player_key (if present)
        "totals": {
            "cost": total_cost,
            "horizon_xpts": total_xpts,
            "score": total_score,
        },
    }
