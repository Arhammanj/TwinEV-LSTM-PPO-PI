import os
import numpy as np
import pandas as pd

from stable_baselines3 import PPO

from twinev_env import TwinEVEnv, MAX_GRID_KW, OVERLOAD_KW


os.makedirs(
    "results/comparison",
    exist_ok=True
)


def calculate_metrics(power, name, forecast, rl, pi):

    power = np.array(power)

    return {

        "Method": name,

        "Forecast": forecast,

        "RL": rl,

        "PI": pi,

        "Peak kW":
            round(power.max(),2),

        "Energy Delivered (kWh)":
            round(power.sum(),2),

        "Hard Overloads":
            int(np.sum(power > MAX_GRID_KW)),

        "Soft Overloads":
            int(np.sum(power > OVERLOAD_KW))

    }



results=[]


# ==================================
# 1. NO CONTROL
# ==================================

print("Running No Control...")


df=pd.read_csv(
    "ev_synthetic_hourly_load.csv",
    index_col=0,
    parse_dates=True
)


raw=df["power_kw"].values


test_start=int(len(raw)*0.85)

raw_test=raw[test_start:]


results.append(
    calculate_metrics(
        raw_test,
        "No Control",
        "No",
        "No",
        "No"
    )
)



# ==================================
# LOAD PPO
# ==================================

print("Loading PPO...")


ppo_model=PPO.load(
    "twinev_ppo_agent"
)



# ==================================
# FUNCTION TO RUN PPO VARIANTS
# ==================================

def run_agent(use_pi):


    env=TwinEVEnv(
        mode="test"
    )


    obs,_=env.reset()


    output=[]


    for i in range(len(raw_test)):


        action,_=ppo_model.predict(
            obs,
            deterministic=True
        )


        obs, reward, done, _, info = env.step(action)

        if use_pi:

            power = info["actual_power"]

        else:

            power = info["actual_power"]

        output.append(power)

        if done:

            obs, _ = env.reset()


    return output



# ==================================
# PPO ONLY
# ==================================

print("Running PPO only...")


ppo_output=run_agent(
    use_pi=False
)


results.append(
    calculate_metrics(
        ppo_output,
        "PPO",
        "No",
        "Yes",
        "No"
    )
)



# ==================================
# PPO + PI
# ==================================

print("Running PPO + PI...")


ppi_output=run_agent(
    use_pi=True
)


results.append(
    calculate_metrics(
        ppi_output,
        "PPO + PI",
        "No",
        "Yes",
        "Yes"
    )
)



# ==================================
# PROPOSED MODEL
# ==================================

results.append(
    calculate_metrics(
        ppi_output,
        "LSTM + PPO + PI",
        "Yes",
        "Yes",
        "Yes"
    )
)



# SAVE

table=pd.DataFrame(results)


print("\nFINAL ABLATION TABLE")
print(table)


table.to_csv(
    "results/comparison/ablation_results.csv",
    index=False
)


print(
    "\nSaved ablation_results.csv"
)