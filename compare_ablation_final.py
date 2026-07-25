import os
import numpy as np
import pandas as pd

from stable_baselines3 import PPO

from twinev_env import TwinEVEnv, MAX_GRID_KW, OVERLOAD_KW


os.makedirs(
    "results/comparison",
    exist_ok=True
)


# ==========================
# METRICS
# ==========================

def metrics(power, name, forecast, rl, pi):

    power = np.array(power)

    baseline_peak = test_power.max()

    peak_reduction = (
        (baseline_peak - power.max())
        /
        baseline_peak
    ) * 100


    cost = np.sum(
        power[:len(prices[test_start:])] *
        prices[test_start:test_start+len(power)]
    )


    return {
        "Method": name,
        "Forecast": forecast,
        "RL": rl,
        "PI": pi,

        "Peak Power (kW)": round(power.max(),2),

        "Peak Reduction (%)":
            round(peak_reduction,2),

        "Energy Delivered (kWh)":
            round(power.sum(),2),

        "Cost":
            round(cost,2),

        "Hard Overloads":
            int(np.sum(power > MAX_GRID_KW)),

        "Soft Overloads":
            int(np.sum(power > OVERLOAD_KW))
    }
results = []


# ==========================
# LOAD RAW DATA
# ==========================

df=pd.read_csv(
    "ev_synthetic_hourly_load.csv",
    index_col=0,
    parse_dates=True
)

raw=df["power_kw"].values
prices = df["price_eur_kwh"].values



test_start=int(len(raw)*0.85)

test_power=raw[test_start:]


# ==========================
# 1. NO CONTROL
# ==========================

print("Running No Control")


results.append(
    metrics(
        test_power,
        "No Control",
        "No",
        "No",
        "No"
    )
)



# ==========================
# LOAD PPO
# ==========================

print("Loading PPO")

ppo=PPO.load(
    "twinev_ppo_agent"
)



# ==========================
# FUNCTION FOR PPO
# ==========================

def run_ppo(use_pi=False):

    env=TwinEVEnv(
        mode="test"
    )

    obs,_=env.reset()

    output=[]


    for i in range(len(test_power)):

        action,_=ppo.predict(
            obs,
            deterministic=True
        )


        obs,reward,done,_,info=env.step(action)


        output.append(
            info["actual_power"]
        )


        if done:
            obs,_=env.reset()


    return output



# ==========================
# PPO ONLY
# ==========================

print("Running PPO")


ppo_output=run_ppo(False)


results.append(
    metrics(
        ppo_output,
        "PPO",
        "No",
        "Yes",
        "No"
    )
)



# ==========================
# PPO + PI
# ==========================

print("Running PPO + PI")


ppo_pi_output=run_ppo(True)


results.append(
    metrics(
        ppo_pi_output,
        "PPO + PI",
        "No",
        "Yes",
        "Yes"
    )
)



# ==========================
# LSTM + PPO + PI
# ==========================

print("Running LSTM + PPO + PI")


lstm=pd.read_csv(
    "lstm_full_predictions.csv"
)


lstm_forecast = lstm["predicted"].values[-len(test_power):]


env=TwinEVEnv(
    mode="test"
)

obs,_=env.reset()

lstm_output=[]


for i in range(len(test_power)):

    # inject LSTM forecast into observation
    obs[4] = lstm_forecast[i] / MAX_GRID_KW

    action,_ = ppo.predict(
        obs,
        deterministic=True
    )

    obs,reward,done,_,info = env.step(action)

    power = info["actual_power"]

    lstm_output.append(power)

    if done:
        obs,_ = env.reset()


lstm_output = np.array(lstm_output)


results.append(
    metrics(
        lstm_output,
        "LSTM + PPO + PI",
        "Yes",
        "Yes",
        "Yes"
    )
)



# ==========================
# SAVE
# ==========================


final=pd.DataFrame(results)


print("\nFINAL ABLATION TABLE")
print(final)


final.to_csv(
    "results/comparison/final_ablation_table.csv",
    index=False
)


print(
    "\nSaved final_ablation_table.csv"
)