# scripts/01_normalize_one_gw.py
from pathlib import Path
from blocks.data_io import normalize_single_gw

if __name__ == "__main__":
    season_dir = r"C:\Users\hoday\PycharmProjects\FPL_Project\Fantasy-Premier-League-master\data\2024-25"
    gw_csv     = str(Path(season_dir) / "gws" / "gw1.csv")
    season     = "2024-25"

    hist = normalize_single_gw(season_dir, gw_csv, season)
    print(hist.head(10))
    print("rows:", len(hist))
    # quick acceptance checks:
    assert hist["team_id"].notna().all(), "team_id has nulls"
    assert hist["opponent_id"].notna().all(), "opponent_id has nulls"
    assert hist["minutes"].between(0, 120).all(), "minutes out of range"
