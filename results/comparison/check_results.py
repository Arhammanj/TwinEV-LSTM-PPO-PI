import pandas as pd

df = pd.read_csv(
    "results/comparison/final_comparison_test.csv"
)

print(
    df[
        [
            "Model",
            "Energy Delivered (kWh)",
            "Total Cost ($)",
            "Cost Saved ($)",
            "Cost Reduction %"
        ]
    ]
)