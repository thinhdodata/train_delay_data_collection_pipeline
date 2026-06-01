import torch
import pandas as pd
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch.nn import Linear

# -----------------------------
# LOAD DATA
# -----------------------------

graphs = torch.load(
    "data/graph/pyg_graph_list_future.pt",
    weights_only=False
)

train_size = int(len(graphs) * 0.8)

test_graphs = graphs[train_size:]

# -----------------------------
# MODEL
# -----------------------------

class GCN(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels):

        super().__init__()

        self.conv1 = GCNConv(
            in_channels,
            hidden_channels
        )

        self.conv2 = GCNConv(
            hidden_channels,
            hidden_channels
        )

        self.linear = Linear(
            hidden_channels,
            1
        )

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.linear(x)

        return x.squeeze()

# -----------------------------
# LOAD TRAINED MODEL
# -----------------------------

model = GCN(
    in_channels=10,
    hidden_channels=32
)

model.load_state_dict(
    torch.load(
        "data/graph/gcn_delay_change_model.pt"
    )
)

model.eval()

# -----------------------------
# INSPECT PREDICTIONS
# -----------------------------

results = []

with torch.no_grad():

    for graph in test_graphs[:5]:

        pred = model(
            graph.x,
            graph.edge_index
        )

        mask_indices = (
            graph.train_mask
            .nonzero(as_tuple=True)[0]
        )

        for idx in mask_indices[:10]:

            results.append({
                "snapshot_time": graph.snapshot_time,
                "node_idx": int(idx),
                "actual_delay_change":
                    round(
                        graph.y[idx].item(),
                        3
                    ),
                "predicted_delay_change":
                    round(
                        pred[idx].item(),
                        3
                    )
            })

results_df = pd.DataFrame(results)

results_df.to_csv(
    "data/graph/prediction_inspection.csv",
    index=False
)

print("saved prediction inspection csv")


print(results_df)

print("\nsummary:")
print(results_df.describe())