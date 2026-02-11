import pandas as pd

input_path = "data/raw/ml-100k/u.data"
output_path = "data/ml-100k.csv"

df = pd.read_csv(
    input_path,
    sep="\t",
    names=["user_id", "item_id", "rating", "timestamp"]
)

df.to_csv(output_path, index=False)

print("Converted! Saved to", output_path)
