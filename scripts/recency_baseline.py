# scripts/recency_baseline.py
from blocks.data_io import build_multi_season_history
from blocks.recency import build_recency_predictions

# --- config you can tweak inline ---
REPO_ROOT = r"C:\Users\hoday\PycharmProjects\FPL_Project\Fantasy-Premier-League-master"
SEASONS = ["2022-23", "2023-24", "2024-25"]
SEASON_FILTER = "2024-25"
HORIZON = 4  # default; change here when needed

def main():
    print("Loading history…")
    hist = build_multi_season_history(REPO_ROOT, SEASONS, verbose=False)
    print(f"Rows: {len(hist)}")

    print(f"Computing recency baseline (H={HORIZON})…")
    preds, latest = build_recency_predictions(
        hist,
        horizon_gws=HORIZON,
        season_filter=SEASON_FILTER
    )

    if not preds.empty:
        next_gw = int(preds["gw"].min())
        top = preds[preds["gw"] == next_gw].nlargest(15, "exp_points")
        print(f"Top 15 for GW{next_gw} (H={HORIZON}):")
        print(top.to_string(index=False))
    else:
        print("No predictions produced (is history empty?).")

if __name__ == "__main__":
    main()
