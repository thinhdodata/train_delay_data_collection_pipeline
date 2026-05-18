from pathlib import Path
import pandas as pd
from google.transit import gtfs_realtime_pb2

RAW_DIR = Path("data/raw_snapshots")
OUT_DIR = Path("data/processed_snapshots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []

for file_path in RAW_DIR.glob("*.pb"):
    feed = gtfs_realtime_pb2.FeedMessage()

    with open(file_path, "rb") as f:
        feed.ParseFromString(f.read())

    snapshot_time = file_path.stem.replace("snapshot_", "")

    for entity in feed.entity:
        if entity.HasField("trip_update"):
            trip = entity.trip_update.trip

            for stop_update in entity.trip_update.stop_time_update:
                rows.append({
                    "snapshot_time": snapshot_time,
                    "trip_id": trip.trip_id,
                    "route_id": trip.route_id,
                    "start_time": trip.start_time,
                    "start_date": trip.start_date,
                    "stop_id": stop_update.stop_id,
                    "stop_sequence": stop_update.stop_sequence,
                    "arrival_delay": stop_update.arrival.delay if stop_update.HasField("arrival") else None,
                    "departure_delay": stop_update.departure.delay if stop_update.HasField("departure") else None
                })

df = pd.DataFrame(rows)

df.to_csv(OUT_DIR / "processed_realtime.csv", index=False)

print(df.shape)
print(df.head())