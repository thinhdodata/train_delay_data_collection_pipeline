import pandas as pd
from pathlib import Path

DATA_PATH = "data/final_dataset/final_phase1_dataset.csv"

OUTPUT_DIR = Path("data/graph")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH, low_memory=False)

# -----------------------------
# BASIC CLEANING
# -----------------------------

df["headway_minutes"] = (
    df["headway_minutes"]
    .fillna(df["headway_minutes"].median())
)

# -----------------------------
# BUILD NODES
# -----------------------------

nodes = (
    df[
        [
            "stop_id",
            "stop_name",
            "stop_lat",
            "stop_lon"
        ]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)

nodes["node_idx"] = range(len(nodes))

# -----------------------------
# BUILD EDGES
# -----------------------------

edge_rows = []

for trip_id, group in df.groupby("trip_id_x"):

    group = group.sort_values("stop_sequence_x")

    stops = group["stop_id"].tolist()

    for i in range(len(stops) - 1):

        source = stops[i]
        target = stops[i + 1]

        if source != target:
            edge_rows.append(
                {
                    "source": source,
                    "target": target
                }
            )

edges = pd.DataFrame(edge_rows).drop_duplicates()

# -----------------------------
# NODE MAPPING
# -----------------------------

node_map = dict(
    zip(
        nodes["stop_id"],
        nodes["node_idx"]
    )
)

edges["source_idx"] = edges["source"].map(node_map)
edges["target_idx"] = edges["target"].map(node_map)

edges = edges.dropna(
    subset=[
        "source_idx",
        "target_idx"
    ]
)

edges["source_idx"] = edges["source_idx"].astype(int)
edges["target_idx"] = edges["target_idx"].astype(int)

# -----------------------------
# SAVE GRAPH STRUCTURE
# -----------------------------

nodes.to_csv(
    OUTPUT_DIR / "nodes.csv",
    index=False
)

edges.to_csv(
    OUTPUT_DIR / "edges.csv",
    index=False
)

# -----------------------------
# BUILD NODE FEATURES
# -----------------------------
df["delay_change_minutes"] = (
    df["future_delay_minutes"]
    - df["real_delay_minutes"]
)

feature_cols = [
    "snapshot_time",
    "stop_id",
    "real_delay_minutes",
    "headway_minutes",
    "route_train_count",
    "dwell_proxy_minutes",
    "rush_hour_flag",
    "hour",
    "weekday",
    "is_weekend",
    "stop_sequence_x",
    "direction_id",
    "future_delay_minutes",
    "delay_change_minutes"
]

node_features = df[feature_cols].copy()

node_features = (
    node_features
    .groupby(
        [
            "snapshot_time",
            "stop_id"
        ]
    )
    .mean(numeric_only=True)
    .reset_index()
)

node_features["node_idx"] = (
    node_features["stop_id"].map(node_map)
)

node_features = node_features.dropna(
    subset=["node_idx"]
)

node_features["node_idx"] = (
    node_features["node_idx"].astype(int)
)

# -----------------------------
# SAVE NODE FEATURES
# -----------------------------

node_features.to_csv(
    OUTPUT_DIR / "node_features.csv",
    index=False
)

print("nodes:", nodes.shape)
print("edges:", edges.shape)
print("node_features:", node_features.shape)
print("saved graph files")