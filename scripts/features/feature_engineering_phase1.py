import pandas as pd

# -----------------------------------
# LOAD DATASET
# -----------------------------------

df = pd.read_csv(
    "data/final_dataset/final_refined_operational_dataset.csv",
    low_memory=False
)

print("initial shape:")
print(df.shape)

# -----------------------------------
# DATETIME FEATURES
# -----------------------------------

df["datetime_utc"] = pd.to_datetime(
    df["timestamp"],
    unit="s",
    utc=True
)

df["datetime"] = (
    df["datetime_utc"]
    .dt.tz_convert("America/New_York")
)

df["hour"] = df["datetime"].dt.hour

df["weekday"] = df["datetime"].dt.weekday

df["is_weekend"] = (
    df["weekday"] >= 5
).astype(int)

# -----------------------------------
# REAL DELAY TARGET
# -----------------------------------

def gtfs_time_to_seconds(t):
    h, m, s = str(t).split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)

df["planned_arrival_seconds"] = (
    df["arrival_time"]
    .apply(gtfs_time_to_seconds)
)

df["actual_seconds"] = (
    df["datetime"].dt.hour * 3600
    + df["datetime"].dt.minute * 60
    + df["datetime"].dt.second
)

df["real_delay_minutes"] = (
    df["actual_seconds"]
    - df["planned_arrival_seconds"]
) / 60

df["planned_day_offset"] = (
    df["planned_arrival_seconds"] // 86400
)

df["actual_seconds_adjusted"] = (
    df["actual_seconds"]
    + df["planned_day_offset"] * 86400
)

df["real_delay_minutes"] = (
    df["actual_seconds_adjusted"]
    - df["planned_arrival_seconds"]
) / 60

df.loc[df["real_delay_minutes"] > 720, "real_delay_minutes"] -= 1440
df.loc[df["real_delay_minutes"] < -720, "real_delay_minutes"] += 1440

df = df[
    (
        df["real_delay_minutes"] >= -30
    )
    &
    (
        df["real_delay_minutes"] <= 60
    )
]

# -----------------------------------
# RUSH HOUR
# -----------------------------------

df["rush_hour_flag"] = (
    (
        (df["hour"] >= 7)
        & (df["hour"] <= 10)
    )
    |
    (
        (df["hour"] >= 16)
        & (df["hour"] <= 19)
    )
).astype(int)

# -----------------------------------
# ROUTE CONGESTION
# -----------------------------------

df["train_count_same_route"] = (
    df.groupby("route_id_x")["trip_id_short"]
    .transform("count")
)

# -----------------------------------
# ROLLING DELAY
# -----------------------------------

df["rolling_avg_delay"] = (
    df.groupby("route_id_x")["delay_minutes"]
    .transform("mean")
)

# -----------------------------------
# STOP PROGRESS
# -----------------------------------

df["stop_progress_diff"] = (
    df["stop_sequence_x"]
    - df["current_stop_sequence"]
)

# -----------------------------------
# SORT FOR HEADWAY
# -----------------------------------

df = df.sort_values(
    by=[
        "route_id_x",
        "direction_id",
        "stop_id",
        "timestamp"
    ]
)

# -----------------------------------
# REMOVE REPEATED SAME TRAIN SNAPSHOTS
# -----------------------------------

headway_df = df.drop_duplicates(
    subset=[
        "route_id_x",
        "direction_id",
        "stop_id",
        "trip_id_short"
    ]
).copy()

# -----------------------------------
# PREVIOUS TRAIN TIME
# -----------------------------------

headway_df["previous_train_time"] = (
    headway_df.groupby(
        [
            "route_id_x",
            "direction_id",
            "stop_id"
        ]
    )["timestamp"]
    .shift(1)
)

headway_df["headway_seconds"] = (
    headway_df["timestamp"]
    - headway_df["previous_train_time"]
)

headway_df["headway_minutes"] = (
    headway_df["headway_seconds"] / 60
)

# -----------------------------------
# MERGE BACK
# -----------------------------------

df = df.merge(
    headway_df[
        [
            "route_id_x",
            "direction_id",
            "stop_id",
            "trip_id_short",
            "headway_minutes"
        ]
    ],
    on=[
        "route_id_x",
        "direction_id",
        "stop_id",
        "trip_id_short"
    ],
    how="left"
)

# -----------------------------------
# CLEAN HEADWAY OUTLIERS
# -----------------------------------

df.loc[df["headway_minutes"] > 720, "headway_minutes"] -= 1440

df = df[
    (
        df["headway_minutes"].isna()
    )
    |
    (
        df["headway_minutes"] >= 0
    )
]

df = df[
    (
        df["headway_minutes"].isna()
    )
    |
    (
        df["headway_minutes"] <= 60
    )
]

# -----------------------------------
# CONGESTION FEATURE
# -----------------------------------

df["route_train_count"] = (
    df.groupby(
        [
            "route_id_x",
            "direction_id",
            "timestamp"
        ]
    )["trip_id_short"]
    .transform("nunique")
)

# -----------------------------------
# DWELL PROXY FEATURE
# -----------------------------------

dwell_df = (
    df.groupby(
        [
            "trip_id_short",
            "stop_id"
        ]
    )["timestamp"]
    .agg(["min", "max"])
    .reset_index()
)

dwell_df["dwell_proxy_seconds"] = (
    dwell_df["max"]
    - dwell_df["min"]
)

dwell_df["dwell_proxy_minutes"] = (
    dwell_df["dwell_proxy_seconds"] / 60
)

df = df.merge(
    dwell_df[
        [
            "trip_id_short",
            "stop_id",
            "dwell_proxy_minutes"
        ]
    ],
    on=[
        "trip_id_short",
        "stop_id"
    ],
    how="left"
)

df.loc[
    df["dwell_proxy_minutes"] > 720,
    "dwell_proxy_minutes"
] -= 1440

df = df[
    (
        df["dwell_proxy_minutes"] >= 0
    )
    &
    (
        df["dwell_proxy_minutes"] <= 30
    )
]

# -----------------------------------
# FUTURE TARGET
# -----------------------------------

df = df.sort_values(
    by=[
        "trip_id_short",
        "timestamp"
    ]
)

df["future_delay_minutes"] = (
    df.groupby(
        "trip_id_short"
    )["real_delay_minutes"]
    .shift(-1)
)

df = df[
    df["future_delay_minutes"].notna()
]

df["headway_minutes"] = (
    df["headway_minutes"]
    .fillna(
        df["headway_minutes"].median()
    )
)

# -----------------------------------
# SAVE
# -----------------------------------

df.to_csv(
    "data/final_dataset/final_phase1_dataset.csv",
    index=False
)
print("\nsaved successfully")
