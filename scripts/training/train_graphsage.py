import torch
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv
from torch.nn import Linear

graphs = torch.load(
    "data/graph/pyg_graph_list_future.pt",
    weights_only=False
)

train_size = int(len(graphs) * 0.8)

train_graphs = graphs[:train_size]
test_graphs = graphs[train_size:]

print("graphs:", len(graphs))
print("train graphs:", len(train_graphs))
print("test graphs:", len(test_graphs))


class GraphSAGE(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels):
        super().__init__()

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.linear = Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.linear(x)

        return x.squeeze()


model = GraphSAGE(
    in_channels=10,
    hidden_channels=32
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

loss_fn = torch.nn.MSELoss()

for epoch in range(1, 51):

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
        print(f"Epoch {epoch}, Train Loss: {avg_loss:.4f}")


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

print("GraphSAGE MAE:", total_mae / len(test_graphs))
print("GraphSAGE RMSE:", total_rmse / len(test_graphs))


torch.save(
    model.state_dict(),
    "data/graph/graphsage_delay_change_model.pt"
)

print("saved GraphSAGE model")