import json
import random

# Load dataset
with open("dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Shuffle dataset
random.shuffle(data)

# Split ratios
train_size = int(0.8 * len(data))
val_size = int(0.1 * len(data))

train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]

# Function to save JSONL
def save_jsonl(filename, dataset):
    with open(filename, "w", encoding="utf-8") as f:
        for item in dataset:
            json.dump(item, f)
            f.write("\n")

# Save files
save_jsonl("data/train.jsonl", train_data)
save_jsonl("data/val.jsonl", val_data)
save_jsonl("data/test.jsonl", test_data)

print("✅ Preprocessing Done!")
print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")