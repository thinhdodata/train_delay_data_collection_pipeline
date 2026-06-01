import pandas as pd

merged = pd.read_csv(
    "data/final_dataset/final_merged_vehicle_trip_updates.csv",
    low_memory=False
)

print("before cleaning:")
print(merged.shape)

merged["route_id_x"] = merged["route_id_x"].astype(str)

filtered = merged[
    merged["route_id_x"] == merged["route_id_vehicle"]
]

print("after route filtering:")
print(filtered.shape)

filtered = filtered.drop_duplicates(
    subset=["trip_id_short", "timestamp"]
)

print("after dedup:")
print(filtered.shape)

filtered.to_csv(
    "data/final_dataset/final_clean_operational_dataset.csv",
    index=False
)

print("saved successfully")