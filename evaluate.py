# ==========================================
# TwinEV v2 FINAL EVALUATION
# Train / Validation / Test
# Test = strictly unseen
# ==========================================

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from stable_baselines3 import PPO

from twinev_env import TwinEVEnv, MAX_GRID_KW, OVERLOAD_KW


# ==========================================
# DIRECTORIES
# ==========================================

os.makedirs("results/comparison", exist_ok=True)
os.makedirs("results/forecasts", exist_ok=True)
os.makedirs("results/rl", exist_ok=True)

os.makedirs("plots/06_final", exist_ok=True)


run_timestamp = datetime.now().isoformat(timespec="seconds")

print("Run timestamp:", run_timestamp)



# ==========================================
# LOAD DATA
# ==========================================

print("\nLoading data...")


df = pd.read_csv(
    "ev_synthetic_hourly_load.csv",
    index_col=0,
    parse_dates=True
)


lstm_df = pd.read_csv(
    "lstm_predictions.csv"
)


series = df["power_kw"].values

n = len(series)


# ==========================================
# SPLIT
# ==========================================

train_end = int(n*0.70)
val_end   = int(n*0.85)


train = series[:train_end]
val   = series[train_end:val_end]
test  = series[val_end:]


print("Total hours:", n)
print("Train:",len(train))
print("Validation:",len(val))
print("Test:",len(test))



# ==========================================
# PRICE DATA
# ==========================================


price_series=None


for c in [
    "price_eur_kwh",
    "price",
    "electricity_price",
    "tariff",
    "grid_price"
]:

    if c in df.columns:
        price_series=df[c].values
        print("Price column:",c)
        break



if price_series is not None:
    test_price = price_series[val_end:]
else:
    test_price=None



# ==========================================
# ARIMA
# ==========================================


print("\nTraining ARIMA...")


arima = ARIMA(
    train,
    order=(5,1,0)
)


arima_fit = arima.fit()


arima_all = arima_fit.forecast(
    steps=len(val)+len(test)
)


arima_pred = arima_all[len(val):]


print(
    "ARIMA prediction:",
    len(arima_pred)
)




# ==========================================
# LSTM
# ==========================================


lstm_pred = lstm_df["predicted"].values


lstm_pred = lstm_pred[:len(test)]

print(
    "LSTM prediction:",
    len(lstm_pred)
)



# ==========================================
# PPO RL TEST
# ==========================================


print("\nEvaluating PPO...")


model = PPO.load(
    "twinev_ppo_agent"
)


env = TwinEVEnv(
    mode="test"
)


obs,_ = env.reset()


rl_power=[]
rl_reward=[]


steps=min(
    len(test),
    1314
)



for i in range(steps):

    action,_ = model.predict(
        obs,
        deterministic=True
    )


    obs,reward,done,_,info = env.step(action)


    rl_power.append(
        info["actual_power"]
    )

    rl_reward.append(
        reward
    )


    if done:
        obs,_=env.reset()



rl_power=np.array(rl_power)


print(
    "RL steps:",
    len(rl_power)
)



# ==========================================
# ALIGN
# ==========================================


length=min(
    len(test),
    len(arima_pred),
    len(lstm_pred),
    len(rl_power)
)



test_a=test[:length]

arima_a=arima_pred[:length]

lstm_a=lstm_pred[:length]

rl_a=rl_power[:length]


if test_price is not None:
    test_price=test_price[:length]



print(
    "Evaluation length:",
    length
)




# ==========================================
# FORECAST METRICS
# ==========================================


def metrics(actual,pred,name):

    rmse=np.sqrt(
        mean_squared_error(actual,pred)
    )

    mae=np.mean(
        abs(actual-pred)
    )


    mape=np.mean(
        abs(
            (actual-pred)/(actual+1e-5)
        )
    )*100


    print(
        f"{name:15s}",
        rmse,
        mae,
        mape
    )

    return rmse,mae,mape



print("\nFORECAST RESULTS")


rmse_arima,mae_arima,mape_arima=metrics(
    test_a,
    arima_a,
    "ARIMA"
)


rmse_lstm,mae_lstm,mape_lstm=metrics(
    test_a,
    lstm_a,
    "LSTM"
)


rmse_rl,mae_rl,mape_rl=metrics(
    test_a,
    rl_a,
    "RL+PI"
)



forecast_metrics=pd.DataFrame({

"Model":
[
"ARIMA",
"LSTM",
"RL+PI"
],

"RMSE":
[
rmse_arima,
rmse_lstm,
rmse_rl
],

"MAE":
[
mae_arima,
mae_lstm,
mae_rl
],

"MAPE":
[
mape_arima,
mape_lstm,
mape_rl
]

})


forecast_metrics.to_csv(
"results/comparison/forecast_metrics.csv",
index=False
)





# ==========================================
# CONTROL METRICS
# ==========================================


def control_metrics(
    power,
    name,
    baseline_peak,
    prices=None
):


    hard=int(
        np.sum(power>MAX_GRID_KW)
    )

    soft=int(
        np.sum(power>OVERLOAD_KW)
    )


    peak=float(
        np.max(power)
    )


    variance=float(
        np.var(power)
    )


    energy=float(
        np.sum(power)
    )


    reduction=((baseline_peak-peak)
              /baseline_peak)*100



    cost = np.nan

    if prices is not None:

        min_len = min(
            len(power),
            len(prices)
        )

        cost = float(
            np.sum(
                power[:min_len] *
                prices[:min_len]
            )
        )


    print(
        name,
        "Peak:",
        peak,
        "Overloads:",
        hard,
        soft
    )


    return (
        hard,
        soft,
        peak,
        variance,
        energy,
        reduction,
        cost
    )



baseline=np.max(test_a)



ol_a,so_a,pk_a,var_a,en_a,pr_a,cost_a=control_metrics(
    test_a,
    "Raw",
    baseline,
    test_price
)


ol_l,so_l,pk_l,var_l,en_l,pr_l,cost_l=control_metrics(
    lstm_a,
    "LSTM",
    baseline,
    test_price
)



ol_r,so_r,pk_r,var_r,en_r,pr_r,cost_r=control_metrics(
    rl_a,
    "RL+PI",
    baseline,
    test_price
)



# ==========================================
# SUMMARY
# ==========================================


summary=pd.DataFrame({

"Model":
[
"ARIMA",
"LSTM",
"RL+PI"
],

"RMSE":
[
rmse_arima,
rmse_lstm,
rmse_rl
],

"MAE":
[
mae_arima,
mae_lstm,
mae_rl
],

"MAPE":
[
mape_arima,
mape_lstm,
mape_rl
],

"Hard Overloads":
[
ol_a,
ol_l,
ol_r
],

"Soft Overloads":
[
so_a,
so_l,
so_r
],

"Peak kW":
[
pk_a,
pk_l,
pk_r
],

"Peak Reduction %":
[
pr_a,
pr_l,
pr_r
],

"Energy Delivered":
[
en_a,
en_l,
en_r
],

"Variance":
[
var_a,
var_l,
var_r
],

"Cost":
[
cost_a,
cost_l,
cost_r
]

})


print(summary)


summary.to_csv(
"results/comparison/final_comparison_test.csv",
index=False
)



# ==========================================
# SAVE RL RESULTS
# ==========================================


pd.DataFrame(
{
"reward":rl_reward
}
).to_csv(
"results/rl/test_rewards.csv",
index=False
)


pd.DataFrame(
{
"power":rl_a
}
).to_csv(
"results/rl/test_power.csv",
index=False
)



# ==========================================
# PLOTS
# ==========================================


plt.figure(figsize=(10,4))

plt.plot(
test_a[:168],
label="Actual"
)

plt.plot(
arima_a[:168],
label="ARIMA"
)

plt.plot(
lstm_a[:168],
label="LSTM"
)

plt.legend()
plt.grid()


plt.savefig(
"plots/06_final/Fig01_forecast.png",
dpi=300
)

plt.close()



plt.figure(figsize=(10,4))


plt.plot(
test_a[:168],
label="Raw"
)


plt.plot(
rl_a[:168],
label="RL+PI"
)


plt.legend()
plt.grid()


plt.savefig(
"plots/06_final/Fig02_RL.png",
dpi=300
)


plt.close()



print("\n===== EVALUATION COMPLETE =====")

print(
"Files saved successfully"
)