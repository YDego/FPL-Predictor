# blocks/transfer_planner.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import pandas as pd

from blocks.xi_picker import pick_starting_xi


@dataclass
class TransferDecision:
    gw: int
    action: str                 # "HOLD" or "BUY X / SELL Y"
    buy_id: Optional[int]
    buy_name: Optional[str]
    buy_team: Optional[str]
    buy_pos: Optional[str]
    buy_price: Optional[float]
    sell_id: Optional[int]
    sell_name: Optional[str]
    sell_team: Optional[str]
    sell_pos: Optional[str]
    sell_price: Optional[float]
    bank_after: float
    base_xi_xpts: float
    new_xi_xpts: float
    gain: float


def _enforce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "player_id" in df.columns:
        df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    if "team_id" in df.columns:
        df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce").astype("Int64")
    if "now_cost" in df.columns:
        df["now_cost"] = pd.to_numeric(df["now_cost"], errors="coerce")
    return df


def _club_counts(squad: pd.DataFrame) -> Dict[int, int]:
    cc = squad["team_id"].value_counts(dropna=False).to_dict()
    return {int(k): int(v) for k, v in cc.items() if pd.notna(k)}


def _valid_after_transfer(
    squad: pd.DataFrame,
    sell_row: pd.Series,
    buy_row: pd.Series,
    max_per_club: int = 3
) -> bool:
    """Check ≤3 per club is preserved after swapping sell_row with buy_row."""
    cc = _club_counts(squad)
    sell_team = int(sell_row["team_id"])
    buy_team  = int(buy_row["team_id"])

    # removing seller
    cc[sell_team] = cc.get(sell_team, 0) - 1
    if cc[sell_team] <= 0:
        cc.pop(sell_team, None)

    # adding buyer
    cc[buy_team] = cc.get(buy_team, 0) + 1

    return all(v <= max_per_club for v in cc.values())


def _apply_transfer(squad: pd.DataFrame, sell_idx: int, buy_row: pd.Series) -> pd.DataFrame:
    """Return a copy with one row replaced by buy_row's fields."""
    new = squad.copy()
    cols = ["player_id", "web_name", "team_short", "team_id", "position", "now_cost"]
    for c in cols:
        new.at[sell_idx, c] = buy_row[c]
    return new


def _best_single_transfer_for_gw(
    squad: pd.DataFrame,
    market: pd.DataFrame,
    gw_preds: pd.DataFrame,
    gw: int,
    bank: float,
    topK_per_pos: int = 60,
) -> Tuple[TransferDecision, pd.DataFrame, float]:
    """
    Evaluate all 1-for-1 same-position swaps and return the best improvement (or HOLD).
    Returns: (decision, updated_squad, updated_bank)
    """
    # Base XI value
    base = pick_starting_xi(squad, gw_preds, target_gw=gw)
    base_obj = base["objective"]

    # Build position-sliced market, topK by that GW xPts (speed)
    p = gw_preds[gw_preds["gw"] == gw][["player_id", "exp_points_scaled"]].rename(
        columns={"exp_points_scaled": "xpts_gw"}
    )
    market_gw = market.merge(p, on="player_id", how="left")
    market_gw["xpts_gw"] = pd.to_numeric(market_gw["xpts_gw"], errors="coerce").fillna(0.0)

    cand_pos = {
        "GKP": market_gw[market_gw["position"] == "GKP"].nlargest(topK_per_pos, "xpts_gw"),
        "DEF": market_gw[market_gw["position"] == "DEF"].nlargest(topK_per_pos, "xpts_gw"),
        "MID": market_gw[market_gw["position"] == "MID"].nlargest(topK_per_pos, "xpts_gw"),
        "FWD": market_gw[market_gw["position"] == "FWD"].nlargest(topK_per_pos, "xpts_gw"),
    }

    best_gain = 0.0
    best = None
    best_squad = squad
    best_bank = bank

    # Iterate over current squad; only consider same-position buys
    for idx, row in squad.iterrows():
        pos = row["position"]
        out_cost = float(row["now_cost"])
        out_team = int(row["team_id"])
        out_pid  = int(row["player_id"])

        # candidate buys in same position, not already in squad
        already = set(int(x) for x in squad["player_id"].tolist() if pd.notna(x))
        pool = cand_pos[pos][~cand_pos[pos]["player_id"].isin(already)].copy()

        # budget filter: price_in <= bank + out_cost
        pool = pool[pool["now_cost"] <= bank + out_cost]
        if pool.empty:
            continue

        # club limit filter (≤3)
        for _, buy in pool.iterrows():
            if not _valid_after_transfer(squad, row, buy, max_per_club=3):
                continue

            trial_squad = _apply_transfer(squad, idx, buy)
            trial = pick_starting_xi(trial_squad, gw_preds, target_gw=gw)
            gain = trial["objective"] - base_obj
            if gain > best_gain + 1e-9:
                best_gain = float(gain)
                best = (row, buy, trial["objective"])
                best_squad = trial_squad
                best_bank  = bank + out_cost - float(buy["now_cost"])

    if best is None:
        # HOLD
        dec = TransferDecision(
            gw=gw, action="HOLD",
            buy_id=None, buy_name=None, buy_team=None, buy_pos=None, buy_price=None,
            sell_id=None, sell_name=None, sell_team=None, sell_pos=None, sell_price=None,
            bank_after=bank, base_xi_xpts=base_obj, new_xi_xpts=base_obj, gain=0.0
        )
        return dec, squad, bank

    sell, buy, new_obj = best
    dec = TransferDecision(
        gw=gw,
        action=f"BUY {buy['web_name']} / SELL {sell['web_name']}",
        buy_id=int(buy["player_id"]),
        buy_name=str(buy["web_name"]),
        buy_team=str(buy["team_short"]),
        buy_pos=str(buy["position"]),
        buy_price=float(buy["now_cost"]),
        sell_id=int(sell["player_id"]),
        sell_name=str(sell["web_name"]),
        sell_team=str(sell["team_short"]),
        sell_pos=str(sell["position"]),
        sell_price=float(sell["now_cost"]),
        bank_after=best_bank,
        base_xi_xpts=base_obj,
        new_xi_xpts=float(new_obj),
        gain=float(new_obj - base_obj),
    )
    return dec, best_squad, best_bank


def plan_transfers_greedy(
    squad: pd.DataFrame,
    preds: pd.DataFrame,
    market: pd.DataFrame,
    gw_list: List[int],
    start_bank: float = 0.0,
    topK_per_pos: int = 60,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Greedy 1-FT planner: for each GW in gw_list, take the best single transfer if it improves XI xPts.
    Returns:
      - plan_df: rows with action per GW
      - final_squad: squad after applying the sequence
    """
    squad = _enforce_types(squad)
    preds = _enforce_types(preds)
    market = _enforce_types(market)

    bank = float(start_bank)
    records: List[Dict] = []
    cur_squad = squad.copy()

    for gw in gw_list:
        dec, cur_squad, bank = _best_single_transfer_for_gw(
            cur_squad, market, preds, gw=gw, bank=bank, topK_per_pos=topK_per_pos
        )
        records.append(dec.__dict__)

    plan_df = pd.DataFrame(records)
    return plan_df, cur_squad
