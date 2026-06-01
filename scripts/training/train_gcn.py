import torch
import torch.nn.functional as F

from torch.nn import Linear
from torch_geometric.nn import GCNConv


# -----------------------------
# CONFIG
# -----------------------------

GRAPH_PATH = "data/graph/pyg_graph_list_future.pt"
MODEL_PATH = "data/graph/gcn_delay_change_model.pt"

IN_CHANNELS = 10
HIDDEN_CHANNELS = 32
LEARNING_RATE = 0.001
EPOCHS = 50
TRAIN_RATIO = 0.8


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
# LOAD DATA
# -----------------------------

graphs = torch.load(
    GRAPH_PATH,
    weights_only=False
)

train_size = int(len(graphs) * TRAIN_RATIO)

train_graphs = graphs[:train_size]
test_graphs = graphs[train_size:]

print("graphs:", len(graphs))
print("train graphs:", len(train_graphs))
print("test graphs:", len(test_graphs))


# -----------------------------
# INITIALIZE MODEL
# -----------------------------

model = GCN(
    in_channels=IN_CHANNELS,
    hidden_channels=HIDDEN_CHANNELS
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

loss_fn = torch.nn.MSELoss()


# -----------------------------
# TRAIN
# -----------------------------

for epoch in range(1, EPOCHS + 1):

    model.train()
    total_loss = 0

    for graph in train_graphs:

        optimizer.zero_grad()

        pred = model(
            graph.x,
            graph.edge_index
        )

        loss = loss_fn(
            pred[graph.train_mask],
            graph.y[graph.train_mask]
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_graphs)

    if epoch % 5 == 0:
        print(
            f"Epoch {epoch}, Train Loss: {avg_loss:.4f}"
        )


# -----------------------------
# EVALUATE GCN
# -----------------------------

model.eval()

total_mae = 0
total_rmse = 0

with torch.no_grad():

    for graph in test_graphs:

        pred = model(
            graph.x,
            graph.edge_index
        )

        mask = graph.train_mask

        error = pred[mask] - graph.y[mask]

        mae = error.abs().mean()
        rmse = torch.sqrt((error ** 2).mean())

        total_mae += mae.item()
        total_rmse += rmse.item()

gcn_mae = total_mae / len(test_graphs)
gcn_rmse = total_rmse / len(test_graphs)

print("GCN MAE:", gcn_mae)
print("GCN RMSE:", gcn_rmse)


# -----------------------------
# NAIVE BASELINE
# -----------------------------

naive_total_mae = 0
naive_total_rmse = 0

with torch.no_grad():

    for graph in test_graphs:

        mask = graph.train_mask

        naive_pred = torch.zeros_like(graph.y)

        error = naive_pred[mask] - graph.y[mask]

        mae = error.abs().mean()
        rmse = torch.sqrt((error ** 2).mean())

        naive_total_mae += mae.item()
        naive_total_rmse += rmse.item()

naive_mae = naive_total_mae / len(test_graphs)
naive_rmse = naive_total_rmse / len(test_graphs)

print("Naive MAE:", naive_mae)
print("Naive RMSE:", naive_rmse)


# -----------------------------
# SAVE MODEL
# -----------------------------

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print("saved model:", MODEL_PATH)