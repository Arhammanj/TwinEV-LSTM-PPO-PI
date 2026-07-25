"""
==========================================
TwinEV v2

Complete Controller Comparison

Compares:
1. Raw EV Charging Demand
2. FCFS Baseline
3. Round Robin Baseline
4. Greedy Peak Limiting
5. Proposed LSTM + PPO + PI Controller
==========================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from baseline import (
    FCFSBaseline,
    RoundRobinBaseline,
    GreedyBaseline
)

from twinev_env import (
    TwinEVEnv,
    MAX_GRID_KW,
    OVERLOAD_KW
)

from stable_baselines3 import PPO


os.makedirs("results/comparison", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)


print("Loading test data...")


df = pd.read_csv(
    "ev_synthetic_hourly_load.csv",
    index_col=0,
    parse_dates=True
)


power = df["power_kw"].values
prices = df["price_eur_kwh"].values


test_start = int(len(power)*0.85)

test_power = power[test_start:]
test_prices = prices[test_start:]


print("Test hours:", len(test_power))


def calculate_metrics(power_output, name, prices=None):

    power_output = np.array(power_output)


    cost = np.nan

    if prices is not None:
        cost = np.sum(
            power_output[:len(prices)]
            *
            prices[:len(power_output)]
        )


    return {

        "Model": name,

        "Energy Delivered (kWh)":
            round(power_output.sum(),2),

        "Peak Power (kW)":
            round(power_output.max(),2),

        "Mean Power (kW)":
            round(power_output.mean(),2),

        "Variance":
            round(power_output.var(),2),

        "Cost":
            round(cost,2),

        "Hard Overloads":
            int(np.sum(power_output > MAX_GRID_KW)),

        "Soft Overloads":
            int(np.sum(power_output > OVERLOAD_KW))
    }



results=[]


# -------------------------------
# RAW
# -------------------------------

print("Evaluating Raw Demand")

results.append(
    calculate_metrics(
        test_power,
        "Raw Demand",
        test_prices
    )
)


# -------------------------------
# FCFS
# -------------------------------

print("Evaluating FCFS")

fcfs = FCFSBaseline()

fcfs_power = fcfs.predict(test_power)

results.append(
    calculate_metrics(
        fcfs_power,
        "FCFS",
        test_prices
    )
)


# -------------------------------
# ROUND ROBIN
# -------------------------------

print("Evaluating Round Robin")

rr = RoundRobinBaseline(
    max_power=150
)

rr_power = rr.predict(test_power)


results.append(
    calculate_metrics(
        rr_power,
        "Round Robin",
        test_prices
    )
)



# -------------------------------
# GREEDY
# -------------------------------

print("Evaluating Greedy")

greedy = GreedyBaseline(
    limit=130
)

greedy_power = greedy.predict(test_power)


results.append(
    calculate_metrics(
        greedy_power,
        "Greedy",
        test_prices
    )
)



# -------------------------------
# PPO + PI
# -------------------------------

print("Evaluating LSTM + PPO + PI")


ppo_model = PPO.load(
    "twinev_ppo_agent"
)


env = TwinEVEnv(
    mode="test"
)


obs,_ = env.reset()


ppo_power=[]


for i in range(len(test_power)):

    action,_ = ppo_model.predict(
        obs,
        deterministic=True
    )


    obs,reward,done,_,info = env.step(action)


    ppo_power.append(
        info["actual_power"]
    )


    if done:
        obs,_=env.reset()



results.append(
    calculate_metrics(
        ppo_power,
        "LSTM + PPO + PI",
        test_prices
    )
)



# -------------------------------
# SAVE
# -------------------------------


comparison = pd.DataFrame(results)


print("\n================================")
print("FINAL CONTROLLER COMPARISON")
print("================================")

print(comparison)



comparison.to_csv(
    "results/comparison/final_controller_comparison.csv",
    index=False
)



print(
"\nSaved comparison CSV"
)



# -------------------------------
# PLOT
# -------------------------------


plt.figure(figsize=(10,5))


plt.bar(
    comparison["Model"],
    comparison["Peak Power (kW)"]
)


plt.xticks(
    rotation=45
)

plt.ylabel(
    "Peak Power (kW)"
)

plt.title(
    "Peak Power Comparison"
)

plt.grid(
    axis="y",
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    "results/figures/peak_power_comparison.png",
    dpi=300
)


plt.close()


print(
"Figure saved successfully"
)