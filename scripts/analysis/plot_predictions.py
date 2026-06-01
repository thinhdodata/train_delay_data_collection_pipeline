import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/graph/prediction_inspection.csv")

plt.figure(figsize=(7, 6))

plt.scatter(
    df["actual_delay_change"],
    df["predicted_delay_change"],
    alpha=0.6
)

plt.xlabel("Actual delay change")
plt.ylabel("Predicted delay change")
plt.title("GCN Prediction vs Actual Delay Change")

plt.axhline(0)
plt.axvline(0)

plt.savefig(
    "data/graph/prediction_vs_actual.png",
    dpi=200,
    bbox_inches="tight"
)

print("saved prediction_vs_actual.png")

