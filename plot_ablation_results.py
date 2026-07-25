import pandas as pd
import matplotlib.pyplot as plt
import os


os.makedirs(
    "results/figures",
    exist_ok=True
)


df = pd.read_csv(
    "results/comparison/final_ablation_table.csv"
)


# ==========================
# Peak Power
# ==========================

plt.figure(figsize=(8,5))

plt.bar(
    df["Method"],
    df["Peak Power (kW)"]
)

plt.ylabel("Peak Power (kW)")
plt.title("Peak Power Comparison")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "results/figures/Fig_Ablation_Peak.png",
    dpi=300
)

plt.close()



# ==========================
# Cost
# ==========================

plt.figure(figsize=(8,5))

plt.bar(
    df["Method"],
    df["Cost"]
)

plt.ylabel("Cost")
plt.title("Charging Cost Comparison")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "results/figures/Fig_Ablation_Cost.png",
    dpi=300
)

plt.close()



# ==========================
# Overloads
# ==========================

plt.figure(figsize=(8,5))

plt.bar(
    df["Method"],
    df["Soft Overloads"]
)

plt.ylabel("Soft Overload Hours")
plt.title("Overload Comparison")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "results/figures/Fig_Ablation_Overloads.png",
    dpi=300
)

plt.close()


print("Ablation figures generated")