import pandas as pd
from collections import Counter

file_path = r"datasets\PaySim\PS_20174392719_1491204439457_log.csv"

chunk_size = 500_000

pair_counts = Counter()

for chunk in pd.read_csv(
    file_path,
    chunksize=chunk_size,
    usecols=["nameOrig", "nameDest"]
):
    pairs = zip(chunk["nameOrig"], chunk["nameDest"])
    pair_counts.update(pairs)

print("\n========== SENDER → RECEIVER ANALYSIS ==========")

total_pairs = len(pair_counts)

repeated_pairs = sum(
    count > 1 for count in pair_counts.values()
)

pairs_5_plus = sum(
    count >= 5 for count in pair_counts.values()
)

pairs_10_plus = sum(
    count >= 10 for count in pair_counts.values()
)

print("Unique sender-receiver pairs:", total_pairs)
print("Repeated pairs:", repeated_pairs)
print("Pairs with 5+ transactions:", pairs_5_plus)
print("Pairs with 10+ transactions:", pairs_10_plus)

print("\nTop 20 repeated sender-receiver pairs:")

for pair, count in pair_counts.most_common(20):
    print(pair, "→", count, "transactions")