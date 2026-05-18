import pandas as pd
from pathlib import Path

REALTIME_PATH = Path("data/processed_snapshots/processed_realtime.csv")
STATIC_DIR = Path("data/static_gtfs")
OUT_DIR = Path("data/final_dataset")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def time_to_seconds(t):
    h, m, s = map(int, t.split(":"))
    return h * 3600 + m * 60 + s

def snapshot_to_seconds(t):
    time_part = t.split("_")[1]
    h = int(time_part[0:2])
    m = int(time_part[2:4])
    s = int(time_part[4:6])
    return h * 3600 + m * 60 + s

realtime = pd.read_csv(REALTIME_PATH)

stop_times = pd.read_csv(STATIC_DIR / "stop_times.txt")
trips = pd.read_csv(STATIC_DIR / "trips.txt")
stops = pd.read_csv(STATIC_DIR / "stops.txt")

planned = stop_times.merge(trips, on="trip_id", how="left")


planned["arrival_seconds"] = planned["arrival_time"].apply(time_to_seconds)
planned["departure_seconds"] = planned["departure_time"].apply(time_to_seconds)

merged = realtime.merge(
    stops,
    on="stop_id",
    how="left"
)

merged["snapshot_seconds"] = merged["snapshot_time"].apply(snapshot_to_seconds)

merged.to_csv(OUT_DIR / "merged_static_realtime.csv", index=False)

print(merged.shape)
print(merged.head())
