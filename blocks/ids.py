from __future__ import annotations
from pathlib import Path
import pandas as pd

def attach_codes_to_history(history: pd.DataFrame, repo_root: str) -> pd.DataFrame:
    """
    Attach stable FPL `code` to each row using season's players_raw.csv.
    Adds: code (Int64), player_key (=code).
    """
    if history.empty:
        return history.assign(code=pd.NA, player_key=pd.NA)

    maps = []
    for s in sorted(history["season"].unique()):
        pr = pd.read_csv(Path(repo_root)/"data"/s/"players_raw.csv")
        pr.columns = [c.lower() for c in pr.columns]
        m = pr[["id","code"]].rename(columns={"id":"player_id"})
        m["season"] = s
        maps.append(m)
    m = pd.concat(maps, ignore_index=True)
    m["player_id"] = pd.to_numeric(m["player_id"], errors="coerce").astype("Int64")
    m["code"]      = pd.to_numeric(m["code"], errors="coerce").astype("Int64")

    out = history.merge(m, on=["season","player_id"], how="left")
    out["player_key"] = out["code"]
    return out
