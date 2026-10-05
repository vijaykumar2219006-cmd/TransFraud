import pandas as pd
from collections import Counter

file_path = r"datasets\PaySim\PS_20174392719_1491204439457_log.csv"

chunk_size = 500_000

sender_counts = Counter()
receiver_counts = Counter()

for chunk in pd.read_csv(
    file_path,
    chunksize=chunk_size,
    usecols=["nameOrig", "nameDest"]
):
    sender_counts.update(chunk["nameOrig"])
    receiver_counts.update(chunk["nameDest"])


print("\n========== SENDER ANALYSIS ==========")

print("Unique senders:", len(sender_counts))

sender_frequency = Counter(sender_counts.values())

print("\nSender transaction frequency:")
for transactions, senders in sorted(sender_frequency.items()):
    print(f"{transactions} transaction(s): {senders} sender(s)")


print("\n========== RECEIVER ANALYSIS ==========")

print("Unique receivers:", len(receiver_counts))

receiver_frequency = Counter(receiver_counts.values())

print("\nReceiver transaction frequency:")

# Print only the first 20 frequencies
for transactions, receivers in sorted(receiver_frequency.items())[:20]:
    print(f"{transactions} transaction(s): {receivers} receiver(s)")


print("\n========== SUMMARY ==========")

print(
    "Senders with more than 1 transaction:",
    sum(count > 1 for count in sender_counts.values())
)

print(
    "Receivers with more than 1 transaction:",
    sum(count > 1 for count in receiver_counts.values())
)

print(
    "Senders with 5+ transactions:",
    sum(count >= 5 for count in sender_counts.values())
)

print(
    "Receivers with 5+ transactions:",
    sum(count >= 5 for count in receiver_counts.values())
)

print(
    "Senders with 10+ transactions:",
    sum(count >= 10 for count in sender_counts.values())
)

print(
    "Receivers with 10+ transactions:",
    sum(count >= 10 for count in receiver_counts.values())
)