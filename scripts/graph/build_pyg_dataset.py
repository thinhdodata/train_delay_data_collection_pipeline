import pandas as pd
import numpy as np
import torch

from torch_geometric.data import Data

# -----------------------------
# LOAD GRAPH FILES
# -----------------------------

nodes = pd.read_csv("data/graph/nodes.csv")
edges = pd.read_csv("data/graph/edges.csv")
node_features = pd.read_csv(
    "data/graph/node_features.csv",
    low_memory=False
)

# -----------------------------
# FEATURE + TARGET COLUMNS
# -----------------------------

feature_cols = [
    "real_delay_minutes",
    "headway_minutes",
    "route_train_count",
    "dwell_proxy_minutes",
    "rush_hour_flag",
    "hour",
    "weekday",
    "is_weekend",
    "stop_sequence_x",
    "direction_id"
]

target_col = "delay_change_minutes"

# -----------------------------
# CLEAN FEATURES
# -----------------------------

node_features[feature_cols] = (
    node_features[feature_cols]
    .astype(float)
    .fillna(0)
)

node_features[target_col] = (
    node_features[target_col]
    .astype(float)
    .fillna(0)
)

# -----------------------------
# BUILD EDGE INDEX
# -----------------------------

edge_index = torch.tensor(
    np.array([
        edges["source_idx"].values,
        edges["target_idx"].values
    ]),
    dtype=torch.long
)

# -----------------------------
# BASIC INFO
# -----------------------------

snapshots = sorted(
    node_features["snapshot_time"].unique()
)

num_nodes = len(nodes)

graph_list = []

# -----------------------------
# BUILD GRAPH SNAPSHOTS
# -----------------------------

for snapshot in snapshots:

    snapshot_df = node_features[
        node_features["snapshot_time"] == snapshot
    ]

    x = torch.zeros(
        (num_nodes, len(feature_cols)),
        dtype=torch.float
    )

    y = torch.zeros(
        num_nodes,
        dtype=torch.float
    )

    mask = torch.zeros(
        num_nodes,
        dtype=torch.bool
    )

    for _, row in snapshot_df.iterrows():

        node_idx = int(row["node_idx"])

        features = (
        row[feature_cols]
        .astype(float)
        .to_numpy()
    )

        x[node_idx] = torch.tensor(
        features,
        dtype=torch.float
    )

        y[node_idx] = float(row[target_col])

        mask[node_idx] = True

    graph = Data(
        x=x,
        edge_index=edge_index,
        y=y
    )

    graph.train_mask = mask
    graph.snapshot_time = snapshot

    graph_list.append(graph)

# -----------------------------
# SAVE
# -----------------------------

print("number of graphs:", len(graph_list))
print("first graph:", graph_list[0])
print("last graph:", graph_list[-1])

print("x shape:", graph_list[0].x.shape)
print("edge_index shape:", graph_list[0].edge_index.shape)
print("y shape:", graph_list[0].y.shape)

torch.save(
    graph_list,
    "data/graph/pyg_graph_list_future.pt"
)

print("saved updated PyG graph list")

