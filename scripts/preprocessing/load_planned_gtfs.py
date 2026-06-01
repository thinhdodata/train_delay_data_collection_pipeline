import pandas as pd

GTFS_DIR = "../data/static_gtfs"

stop_times = pd.read_csv(f"{GTFS_DIR}/stop_times.txt")
trips = pd.read_csv(f"{GTFS_DIR}/trips.txt")
stops = pd.read_csv(f"{GTFS_DIR}/stops.txt")
routes = pd.read_csv(f"{GTFS_DIR}/routes.txt")

planned = (
    stop_times
    .merge(trips, on="trip_id", how="left")
    .merge(stops, on="stop_id", how="left")
)

print(planned.head())
print(planned.columns)
print(planned.shape)