import pandas as pd

# -----------------------------------
# LOAD CLEAN DATASET
# -----------------------------------

df = pd.read_csv(
    "data/final_dataset/final_clean_operational_dataset.csv"
)

print("before filtering:")
print(df.shape)

# -----------------------------------
# COMPUTE SEQUENCE DIFFERENCE
# -----------------------------------

df["sequence_diff"] = abs(
    df["stop_sequence_x"]
    - df["current_stop_sequence"]
)

# -----------------------------------
# KEEP ONLY NEARBY TRAIN STATES
# -----------------------------------

df = df[
    df["sequence_diff"] <= 5
]

print("\nafter sequence filtering:")
print(df.shape)

# -----------------------------------
# SORT
# -----------------------------------

df = df.sort_values(
    by=["trip_id_short", "sequence_diff"]
)

# -----------------------------------
# KEEP BEST MATCH PER TRAIN
# -----------------------------------

df = df.drop_duplicates(
    subset=["trip_id_short", "timestamp"],
    keep="first"
)

print("\nafter dedup:")
print(df.shape)

# -----------------------------------
# OUTPUT
# -----------------------------------

print("\npreview:")
print(df.head())

df.to_csv(
    "data/final_dataset/final_refined_operational_dataset.csv",
    index=False
)

print("\nsaved successfully")