# blocks/meta.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

API_BOOTSTRAP = "https://fantasy.premierleague.com/api/bootstrap-static/"

def load_live_prices_positions() -> pd.DataFrame:
    """
    Pull current players from the official FPL API with live prices & availability.
    Returns current PL players only, with a STABLE player key `code`.
    Columns:
      code, player_id, web_name, team_id, team_short, position, now_cost,
      status, chance_of_playing_next_round, chance_of_playing_this_round
    """
    import requests
    r = requests.get(API_BOOTSTRAP, timeout=20)
    r.raise_for_status()
    data = r.json()

    elems = pd.DataFrame(data["elements"])
    teams = pd.DataFrame(data["teams"])
    etyp  = pd.DataFrame(data["element_types"])

    pos_map_short = etyp.set_index("id")["singular_name_short"].to_dict()  # {1:GKP,2:DEF,3:MID,4:FWD}
    team_short_map = teams.set_index("id")["short_name"].to_dict()
    valid_teams = set(team_short_map.keys())

    m = pd.DataFrame({
        "player_id": elems["id"].astype("int64"),          # season-volatile
        "code":      elems["code"].astype("int64"),         # season-stable ✅
        "web_name":  elems["web_name"].astype(str),
        "team_id":   elems["team"].astype("int64"),
        "element_type": elems["element_type"].astype("int64"),
        "now_cost":  elems["now_cost"].astype("float") / 10.0,  # tenths → £m
        "status":    elems["status"].astype(str),
        "chance_of_playing_next_round": pd.to_numeric(elems["chance_of_playing_next_round"], errors="coerce"),
        "chance_of_playing_this_round": pd.to_numeric(elems["chance_of_playing_this_round"], errors="coerce"),
    })

    # keep only current PL squads
    m = m[m["team_id"].isin(valid_teams)].copy()

    # enrich
    m["position"]   = m["element_type"].map(pos_map_short)
    m["team_short"] = m["team_id"].map(team_short_map)

    keep = ["code","player_id","web_name","team_id","team_short","position","now_cost",
            "status","chance_of_playing_next_round","chance_of_playing_this_round"]
    return m[keep].reset_index(drop=True)


def load_fallback_prices_positions(repo_root: str | Path, season: str) -> pd.DataFrame:
    """
    Fallback to Vaastav CSVs if the API is unreachable.
    (Prices may be stale; `code` still present and is stable.)
    """
    sdir = Path(repo_root) / "data" / season
    players_csv = sdir / "players_raw.csv"
    teams_csv   = sdir / "teams.csv"

    df = pd.read_csv(players_csv)
    df.columns = [c.lower() for c in df.columns]
    teams = pd.read_csv(teams_csv)
    teams.columns = [c.lower() for c in teams.columns]
    team_map = teams.set_index("id")["short_name"].to_dict()

    pos_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    out = pd.DataFrame({
        "player_id": pd.to_numeric(df["id"], errors="coerce").astype("Int64"),
        "code":      pd.to_numeric(df["code"], errors="coerce").astype("Int64"),
        "web_name":  df["web_name"].astype(str),
        "team_id":   pd.to_numeric(df["team"], errors="coerce").astype("Int64"),
        "position":  pd.to_numeric(df["element_type"], errors="coerce").map(pos_map),
        "now_cost":  pd.to_numeric(df.get("now_cost", df.get("value", None)), errors="coerce")/10.0,
        "status":    df.get("status", pd.Series([None]*len(df))),
        "chance_of_playing_next_round": pd.to_numeric(df.get("chance_of_playing_next_round", None), errors="coerce"),
        "chance_of_playing_this_round": pd.to_numeric(df.get("chance_of_playing_this_round", None), errors="coerce"),
    }).dropna(subset=["player_id","code","team_id","position","now_cost"])
    out["team_short"] = out["team_id"].map(team_map)
    return out.reset_index(drop=True)
