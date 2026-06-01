import os
from pathlib import Path
import pandas as pd
from google.transit import gtfs_realtime_pb2

merged = pd.read_csv(
    "data/final_dataset/merged_trip_updates_planned_v1.csv",
    low_memory=False
)

rows = []

snapshot_dir = Path("data/raw_snapshots")

for snapshot_file in snapshot_dir.glob("*.pb"):
    feed = gtfs_realtime_pb2.FeedMessage()

    with open(snapshot_file, "rb") as f:
        feed.ParseFromString(f.read())

    snapshot_file_time = snapshot_file.name.replace("snapshot_", "").replace(".pb", "")

    for entity in feed.entity:
        if entity.HasField("vehicle"):
            vehicle = entity.vehicle

            rows.append({
                "snapshot_file_time": snapshot_file_time,
                "trip_id_realtime": vehicle.trip.trip_id,
                "route_id_vehicle": vehicle.trip.route_id,
                "stop_id_vehicle": vehicle.stop_id,
                "timestamp": vehicle.timestamp,
                "latitude": vehicle.position.latitude,
                "longitude": vehicle.position.longitude,
                "current_stop_sequence": vehicle.current_stop_sequence,
                "current_status": vehicle.current_status
            })

vehicle_df = pd.DataFrame(rows)

print("vehicle rows:", vehicle_df.shape)
print("unique vehicle timestamps:", vehicle_df["timestamp"].nunique())

merged["snapshot_time"] = merged["snapshot_time"].astype(str)
vehicle_df["snapshot_file_time"] = vehicle_df["snapshot_file_time"].astype(str)

final_merged = merged.merge(
    vehicle_df,
    left_on=["trip_id_y", "snapshot_time"],
    right_on=["trip_id_realtime", "snapshot_file_time"],
    how="inner"
)

print("final merged shape:", final_merged.shape)
print("unique snapshots:", final_merged["snapshot_time"].nunique())
print("unique trips:", final_merged["trip_id_short"].nunique())

os.makedirs("data/final_dataset", exist_ok=True)

final_merged.to_csv(
    "data/final_dataset/final_merged_vehicle_trip_updates.csv",
    index=False
)

print("saved successfully")