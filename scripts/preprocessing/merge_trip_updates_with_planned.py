import os
import pandas as pd

GTFS_DIR = "data/static_gtfs"

stop_times = pd.read_csv(f"{GTFS_DIR}/stop_times.txt")
trips = pd.read_csv(f"{GTFS_DIR}/trips.txt")
stops = pd.read_csv(f"{GTFS_DIR}/stops.txt")

planned = (
    stop_times
    .merge(trips, on="trip_id", how="left")
    .merge(stops, on="stop_id", how="left")
)

planned["trip_id_short"] = planned["trip_id"].str.extract(
    r"(\d{6}_.+)$"
)

trip_updates = pd.read_csv(
    "data/processed_snapshots/processed_realtime.csv"
)

planned_small = planned[
    [
        "trip_id",
        "trip_id_short",
        "stop_id",
        "arrival_time",
        "departure_time",
        "stop_sequence",
        "route_id",
        "service_id",
        "direction_id",
        "shape_id",
        "stop_name",
        "stop_lat",
        "stop_lon"
    ]
]

merged = planned_small.merge(
    trip_updates,
    left_on=["trip_id_short", "stop_id"],
    right_on=["trip_id", "stop_id"],
    how="inner"
)

merged["delay_seconds"] = merged["arrival_delay"].fillna(
    merged["departure_delay"]
)

merged["delay_minutes"] = merged["delay_seconds"] / 60

os.makedirs("data/final_dataset", exist_ok=True)

merged.to_csv(
    "data/final_dataset/merged_trip_updates_planned_v1.csv",
    index=False
)

print("merged shape:")
print(merged.shape)

print("unique snapshots:")
print(merged["snapshot_time"].nunique())

print("unique trips:")
print(merged["trip_id_short"].nunique())

print(merged.isnull().sum())