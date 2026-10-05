import pandas as pd

file_path = r"datasets\PaySim\PS_20174392719_1491204439457_log.csv"

chunk_size = 500_000

fraud_by_step = {}
total_by_step = {}

for chunk in pd.read_csv(
    file_path,
    chunksize=chunk_size,
    usecols=["step", "isFraud"]
):

    total_counts = chunk["step"].value_counts()
    fraud_counts = chunk[chunk["isFraud"] == 1]["step"].value_counts()

    for step, count in total_counts.items():
        total_by_step[step] = total_by_step.get(step, 0) + count

    for step, count in fraud_counts.items():
        fraud_by_step[step] = fraud_by_step.get(step, 0) + count


print("\n========== FRAUD BY TIME ==========")

print("\nTotal steps:", len(total_by_step))

print("\nTop 20 steps by fraud count:")

top_steps = sorted(
    fraud_by_step.items(),
    key=lambda x: x[1],
    reverse=True
)[:20]

for step, count in top_steps:
    total = total_by_step[step]
    rate = (count / total) * 100

    print(
        f"Step {step}: "
        f"Fraud={count}, "
        f"Total={total}, "
        f"Fraud Rate={rate:.4f}%"
    )

print("\nOverall fraud rate by selected steps:")

for step in sorted(fraud_by_step)[:20]:
    total = total_by_step[step]
    fraud = fraud_by_step[step]
    rate = (fraud / total) * 100

    print(
        f"Step {step}: "
        f"Fraud={fraud}, "
        f"Total={total}, "
        f"Rate={rate:.4f}%"
    )