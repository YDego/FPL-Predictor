# blocks/data_io.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import pandas as pd

def _read_csv_robust(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.read_csv(p, engine="python", on_bad_lines="skip")

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    if "round" in df.columns and "gw" not in df.columns:
        df["gw"] = df["round"]
    if "player_id" in df.columns and "element" not in df.columns:
        df["element"] = df["player_id"]
    if "team_id" in df.columns and "team" not in df.columns:
        df["team"] = df["team_id"]
    if "opponent_id" in df.columns and "opponent_team" not in df.columns:
        df["opponent_team"] = df["opponent_id"]
    return df

def _to_bool01(s: pd.Series) -> pd.Series:
    s = s.copy()
    if s.dtype == bool:
        return s.astype("int8")
    if s.dtype == object:
        m = {"true":1,"false":0,"t":1,"f":0,"1":1,"0":0,"yes":1,"no":0}
        s = s.astype(str).str.lower().map(m)
    return pd.to_numeric(s, errors="coerce").fillna(0).astype("int8")

def _load_fixtures(season_dir: Path) -> pd.DataFrame:
    fx = _read_csv_robust(season_dir / "fixtures.csv")
    fx = _norm_cols(fx)
    # keep flexible schema
    keep = [c for c in ["event","id","code","team_h","team_a"] if c in fx.columns]
    fx = fx[keep].copy()
    for c in keep:
        if c in ["event","id","code"]:
            fx[c] = pd.to_numeric(fx[c], errors="coerce")
    return fx

def _load_teams_map(season_dir: Path) -> pd.DataFrame:
    """Map names/short_name to numeric team id."""
    t = _read_csv_robust(season_dir / "teams.csv")
    t = _norm_cols(t)
    out = pd.DataFrame()
    out["id"] = pd.to_numeric(t["id"], errors="coerce").astype("Int64")
    out["name_lower"]  = t["name"].astype(str).str.lower().str.strip() if "name" in t.columns else pd.Series(dtype=str)
    out["short_lower"] = t["short_name"].astype(str).str.lower().str.strip() if "short_name" in t.columns else pd.Series(dtype=str)
    return out.dropna(subset=["id"]).reset_index(drop=True)

def _value_to_team_id(val: pd.Series, teams_map: pd.DataFrame) -> pd.Series:
    num = pd.to_numeric(val, errors="coerce")
    out = num.copy()
    mask = out.isna() & val.notna()
    if mask.any() and not teams_map.empty:
        names = val[mask].astype(str).str.lower().str.strip()
        name2id  = dict(zip(teams_map["name_lower"], teams_map["id"]))
        short2id = dict(zip(teams_map["short_lower"], teams_map["id"]))
        mapped = names.map(name2id)
        still = mapped.isna()
        if still.any():
            mapped.loc[still] = names[still].map(short2id)
        out.loc[mask] = mapped
    return pd.to_numeric(out, errors="coerce")

def normalize_single_gw(season_dir: str | Path, gw_csv: str | Path, season: str) -> pd.DataFrame:
    """
    Build canonical rows for a single GW file using fixtures to derive team_id.
    Returns columns:
      season, gw, player_id, team_id, opponent_id, was_home, minutes, total_points, goals_scored, ...
    """
    season_dir = Path(season_dir)
    df = _read_csv_robust(Path(gw_csv))
    df = _norm_cols(df)
    # ensure 'gw'
    if "gw" not in df.columns or df["gw"].isna().all():
        # infer from filename like gw1.csv
        try:
            stem = Path(gw_csv).stem
            gw_infer = int("".join(ch for ch in stem if ch.isdigit()))
        except Exception:
            gw_infer = None
        if gw_infer is not None:
            df["gw"] = gw_infer

    # coercions
    for c in ["gw","element","opponent_team","fixture","team"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["was_home"] = _to_bool01(df.get("was_home", 0))

    fixtures = _load_fixtures(season_dir)
    teams_map = _load_teams_map(season_dir)

    # 1) try fixture id/code
    if "fixture" in df.columns and not fixtures.empty:
        left = df.copy()
        left["fixture"] = pd.to_numeric(left["fixture"], errors="coerce")
        for key in ["id","code"]:
            if key in fixtures.columns:
                t = left.merge(
                    fixtures.rename(columns={key:"_fxkey"}),
                    left_on="fixture", right_on="_fxkey", how="left", suffixes=("","_fx")
                )
                team_from_fix = np.where(t["was_home"]==1, t.get("team_h"), t.get("team_a"))
                if "team" not in t.columns:
                    t["team"] = team_from_fix
                else:
                    t["team"] = t["team"].mask(t["team"].isna(), team_from_fix)
                left = t.drop(columns=["_fxkey"], errors="ignore")
        df = left

    # 2) fallback via (event==gw, opponent_team, was_home)
    need_team = ("team" not in df.columns) or (df["team"].isna().mean() > 0.5)
    if need_team and not fixtures.empty and "event" in fixtures.columns:
        fx = fixtures.copy()
        fx["event"] = pd.to_numeric(fx["event"], errors="coerce")
        base = df.copy()
        base["gw"] = pd.to_numeric(base["gw"], errors="coerce").astype("Int64")
        base["opponent_team"] = pd.to_numeric(base["opponent_team"], errors="coerce").astype("Int64")
        base["was_home"] = _to_bool01(base.get("was_home", 0))
        # home rows
        home = base[base["was_home"]==1].merge(
            fx[["event","team_h","team_a"]].rename(columns={"event":"gw"}),
            left_on=["gw","opponent_team"], right_on=["gw","team_a"], how="left"
        )
        home["team_from_gw"] = home["team_h"]
        # away rows
        away = base[base["was_home"]==0].merge(
            fx[["event","team_h","team_a"]].rename(columns={"event":"gw"}),
            left_on=["gw","opponent_team"], right_on=["gw","team_h"], how="left"
        )
        away["team_from_gw"] = away["team_a"]
        fb = pd.concat([home, away], ignore_index=True, sort=False)
        if "team" in df.columns:
            df = df.merge(
                fb[["element","gw","opponent_team","was_home","team_from_gw"]],
                on=["element","gw","opponent_team","was_home"], how="left"
            )
            df["team"] = df["team"].mask(df["team"].isna(), df["team_from_gw"])
            df = df.drop(columns=["team_from_gw"], errors="ignore")
        else:
            df = fb.copy()
            df["team"] = df["team_from_gw"]
            df = df.drop(columns=["team_from_gw"], errors="ignore")

    # map team → numeric team_id using teams.csv (handles name/short_name cases)
    df["team_id"] = _value_to_team_id(df.get("team"), teams_map)
    df["opponent_id"] = pd.to_numeric(df.get("opponent_team"), errors="coerce")

    out = pd.DataFrame({
        "season": season,
        "gw": df.get("gw"),
        "player_id": df.get("element"),
        "team_id": df.get("team_id"),
        "opponent_id": df.get("opponent_id"),
        "was_home": df.get("was_home"),
        "minutes": df.get("minutes", 0),
        "total_points": df.get("total_points", df.get("points", 0)),
        "goals_scored": df.get("goals_scored", 0),
        "assists": df.get("assists", 0),
        "clean_sheets": df.get("clean_sheets", 0),
        "goals_conceded": df.get("goals_conceded", 0),
        "saves": df.get("saves", 0),
        "yellow_cards": df.get("yellow_cards", 0),
        "red_cards": df.get("red_cards", 0),
        "bonus": df.get("bonus", 0),
    })

    # drop rows missing keys & finalize dtypes
    out = out.dropna(subset=["gw","player_id","team_id","opponent_id"])
    int_cols = ["gw","player_id","team_id","opponent_id","minutes","total_points",
                "goals_scored","assists","clean_sheets","goals_conceded","saves",
                "yellow_cards","red_cards","bonus","was_home"]
    for c in int_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)

    return out.reset_index(drop=True)

def build_season_history(season_dir: str | Path, season: str, verbose: bool = True) -> pd.DataFrame:
    """
    Loop over all gw*.csv in <season_dir>/gws and return a canonical
    player_history for a single season.
    """
    season_dir = Path(season_dir)
    gw_files = sorted(glob.glob(str(season_dir / "gws" / "gw*.csv")))
    parts = []
    for f in gw_files:
        try:
            hist_gw = normalize_single_gw(season_dir, f, season)
            if not hist_gw.empty:
                parts.append(hist_gw)
            elif verbose:
                print(f"[{season}] empty after normalize: {f}")
        except Exception as e:
            if verbose:
                print(f"[{season}] failed {f}: {e}")
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["season","gw","player_id","team_id","opponent_id","was_home",
                 "minutes","total_points","goals_scored","assists","clean_sheets",
                 "goals_conceded","saves","yellow_cards","red_cards","bonus"]
    )
    if verbose:
        print(f"[{season}] season rows: {len(out)} from {len(gw_files)} GW files")
    return out

def build_multi_season_history(repo_root: str | Path, seasons: list[str] | None = None, verbose: bool = True) -> pd.DataFrame:
    """
    Build player_history across multiple seasons.
    repo_root points to the Vaastav repo root (the folder that contains data/<season>).
    """
    repo_root = Path(repo_root)
    data_dir = repo_root / "data"
    if seasons is None:
        # discover seasons by directory name like 2024-25
        seasons = sorted([p.name for p in data_dir.iterdir() if p.is_dir() and "-" in p.name])
    parts = []
    for s in seasons:
        sd = data_dir / s
        if not sd.exists():
            if verbose: print(f"[warn] missing season dir: {sd}")
            continue
        one = build_season_history(sd, s, verbose=verbose)
        if not one.empty:
            parts.append(one)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["season","gw","player_id","team_id","opponent_id","was_home",
                 "minutes","total_points","goals_scored","assists","clean_sheets",
                 "goals_conceded","saves","yellow_cards","red_cards","bonus"]
    )
    if verbose:
        print(f"[ALL] multi-season rows: {len(out)} across {len(parts)} seasons")
    return out

# --- team matches from fixtures (preferred) ---
def build_team_matches_from_fixtures(repo_root: str | Path,
                                     seasons: list[str] | None = None,
                                     verbose: bool = True) -> pd.DataFrame:
    """
    Create a team-level match table using fixtures, not player rows.
    Output columns: season, gw, team_id, opponent_id, was_home, gf, ga
    """
    repo_root = Path(repo_root)
    data_dir = repo_root / "data"
    if seasons is None:
        seasons = sorted([p.name for p in data_dir.iterdir() if p.is_dir() and "-" in p.name])

    parts = []
    for s in seasons:
        sd = data_dir / s
        fx = _load_fixtures(sd)  # uses your helper above
        if fx.empty:
            if verbose: print(f"[{s}] fixtures.csv missing/empty; skipping")
            continue

        # robust score column detection
        score_cols = [("team_h_score","team_a_score"),
                      ("home_score","away_score"),
                      ("fh","fa")]  # fallback aliases if any
        gf_h = ga_h = gf_a = ga_a = None
        for h_col, a_col in score_cols:
            if h_col in fx.columns and a_col in fx.columns:
                gf_h, ga_h = h_col, a_col
                gf_a, ga_a = a_col, h_col
                break
        if gf_h is None:
            if verbose: print(f"[{s}] no score columns found in fixtures; skipping")
            continue

        # we need: event (gw), team_h, team_a, scores
        need = {"event","team_h","team_a", gf_h, ga_h}
        if not need.issubset(set(fx.columns)):
            if verbose: print(f"[{s}] fixtures missing columns {need - set(fx.columns)}; skipping")
            continue

        # build two rows per fixture
        base = fx[["event","team_h","team_a", gf_h, ga_h]].copy()
        base.rename(columns={"event":"gw"}, inplace=True)
        # home rows
        home = base.rename(columns={
            "team_h":"team_id", "team_a":"opponent_id", gf_h:"gf", ga_h:"ga"
        }).copy()
        home["was_home"] = 1
        # away rows
        away = base.rename(columns={
            "team_a":"team_id", "team_h":"opponent_id", gf_h:"gf", ga_h:"ga"
        }).copy()
        away["was_home"] = 0
        out = pd.concat([home, away], ignore_index=True)
        out["season"] = s

        int_cols = ["gw","team_id","opponent_id","gf","ga","was_home"]
        for c in int_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)

        parts.append(out[["season","gw","team_id","opponent_id","was_home","gf","ga"]])

    tm = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["season","gw","team_id","opponent_id","was_home","gf","ga"]
    )

    if verbose and not tm.empty:
        chk = tm.groupby(["season","gw"]).agg(GF=("gf","sum"), GA=("ga","sum")).reset_index()
        bad = chk[chk["GF"] != chk["GA"]]
        if bad.empty:
            print("[ok] GF equals GA for every (season, gw) from fixtures ✅")
        else:
            print("[warn] fixtures-derived GF != GA somewhere (unexpected):")
            print(bad.head())
    return tm

# --- fallback if you ever need to derive from player rows (do NOT use goals_conceded) ---
def build_team_matches_from_players(player_history: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Build team matches only from goals_scored:
      - team GF = sum(goals_scored) by (season, gw, team_id, opponent_id, was_home)
      - team GA = opponent's GF (join on swapped team/opponent)
    Guarantees GF==GA by construction.
    """
    if player_history.empty:
        return pd.DataFrame(columns=["season","gw","team_id","opponent_id","was_home","gf","ga"])

    gf_tbl = (player_history.groupby(["season","gw","team_id","opponent_id","was_home"], as_index=False)
              .agg(gf=("goals_scored","sum")))
    # swap team/opponent to get opponent GF as our GA
    opp_gf = gf_tbl.rename(columns={
        "team_id":"opponent_id","opponent_id":"team_id",
        "was_home":"_opp_home","gf":"ga"
    })
    # for GA, home/away is inverted
    opp_gf["was_home"] = (1 - opp_gf["_opp_home"].astype(int))
    opp_gf = opp_gf.drop(columns=["_opp_home"])

    tm = gf_tbl.merge(
        opp_gf[["season","gw","team_id","opponent_id","was_home","ga"]],
        on=["season","gw","team_id","opponent_id","was_home"],
        how="left"
    ).fillna({"ga":0})

    if verbose:
        chk = tm.groupby(["season","gw"]).agg(GF=("gf","sum"), GA=("ga","sum")).reset_index()
        bad = chk[chk["GF"] != chk["GA"]]
        if bad.empty:
            print("[ok] GF equals GA for every (season, gw) from players’ goals ✅")
        else:
            print("[warn] GF != GA even with players’ goals (investigate):")
            print(bad.head())
    return tm
