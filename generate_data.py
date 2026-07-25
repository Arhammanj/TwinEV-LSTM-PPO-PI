# ==========================================
# TwinEV v2 — Synthetic Data Generator
# Run this FIRST before anything else
# ==========================================
# pip install simpy numpy pandas matplotlib
# ==========================================

import simpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

np.random.seed(42)
from plot_style import set_style

set_style()

# ==========================================
# CONFIG
# ==========================================

NUM_CHARGERS        = 10
MAX_POWER_KW        = 150       # total station capacity kW
CHARGE_POWER_KW     = 22        # per EV (Level 2 AC) kW
INTER_ARRIVAL_MEAN  = 4         # minutes between EV arrivals
CHARGE_DURATION_MEAN= 45        # avg minutes per session
SIM_DURATION        = 60 * 24 * 365  # 1 year in minutes

log = []

# ==========================================
# SOLAR GENERATION PROFILE
# Peak during midday, zero at night
# ==========================================

def solar_kw_at_minute(t_minutes):
    hour = (t_minutes / 60) % 24
    # Bell curve centered at noon
    if 6 <= hour <= 20:
        return max(0, 150 * np.exp(-0.5 * ((hour - 13) / 3.5) ** 2)
                   + np.random.normal(0, 5))
    return 0.0

# ==========================================
# ENERGY PRICE PROFILE
# Peak pricing 7-9am and 5-8pm
# ==========================================

def price_at_minute(t_minutes):
    hour = (t_minutes / 60) % 24
    if 7 <= hour <= 9 or 17 <= hour <= 20:
        return round(np.random.uniform(0.35, 0.55), 3)   # peak tariff
    elif 22 <= hour or hour <= 5:
        return round(np.random.uniform(0.05, 0.12), 3)   # off-peak
    else:
        return round(np.random.uniform(0.15, 0.30), 3)   # mid

# ==========================================
# BATTERY SoC GENERATOR
# Arriving EVs have random SoC 10-80%
# ==========================================

def random_soc():
    return round(np.random.uniform(0.10, 0.80), 2)

# ==========================================
# SIMPY EV PROCESS
# ==========================================

def ev_process(env, ev_id, chargers):
    arrival_time = env.now
    soc_on_arrival = random_soc()

    with chargers.request() as req:
        yield req
        wait_time   = env.now - arrival_time
        charge_time = max(10, np.random.normal(CHARGE_DURATION_MEAN, 10))
        power       = np.random.uniform(7, CHARGE_POWER_KW)
        solar       = solar_kw_at_minute(env.now)
        price       = price_at_minute(env.now)

        yield env.timeout(charge_time)

        log.append({
            'timestamp'     : env.now,
            'ev_id'         : ev_id,
            'arrival_min'   : arrival_time,
            'wait_min'      : wait_time,
            'charge_min'    : charge_time,
            'power_kw'      : power,
            'soc_arrival'   : soc_on_arrival,
            'solar_kw'      : solar,
            'price_eur_kwh' : price,
        })

# ==========================================
# SIMPY EV GENERATOR
# ==========================================

def ev_generator(env, chargers):
    ev_id = 0
    while True:
        # Arrival rate varies by hour (busier during day)
        hour = (env.now / 60) % 24
        if 8 <= hour <= 20:
            lam = INTER_ARRIVAL_MEAN * 0.6    # busier
        else:
            lam = INTER_ARRIVAL_MEAN * 2.0    # quieter

        yield env.timeout(np.random.exponential(lam))
        env.process(ev_process(env, ev_id, chargers))
        ev_id += 1

# ==========================================
# RUN SIMULATION
# ==========================================

print("Running SimPy EV simulation (1 year)...")

env      = simpy.Environment()
chargers = simpy.Resource(env, capacity=NUM_CHARGERS)
env.process(ev_generator(env, chargers))
env.run(until=SIM_DURATION)

df_raw = pd.DataFrame(log)
df_raw['timestamp'] = (
    pd.to_datetime('2024-01-01')
    + pd.to_timedelta(df_raw['timestamp'], unit='m')
)
df_raw.set_index('timestamp', inplace=True)

print(f"Total EVs simulated: {len(df_raw)}")

# ==========================================
# RESAMPLE TO HOURLY
# ==========================================

hourly = df_raw.resample('h').agg({
    'power_kw'      : 'sum',
    'solar_kw'      : 'mean',
    'price_eur_kwh' : 'mean',
    'wait_min'      : 'mean',
    'soc_arrival'   : 'mean',
    'ev_id'         : 'count',
}).fillna(0)

hourly.rename(columns={'ev_id': 'ev_count'}, inplace=True)

# Add time features
hourly['hour']      = hourly.index.hour
hourly['dayofweek'] = hourly.index.dayofweek
hourly['month']     = hourly.index.month

# ==========================================
# SAVE FILES
# ==========================================

hourly.to_csv('ev_synthetic_hourly_load.csv')
df_raw.to_csv('ev_raw_simulation_log.csv')

print("Saved: ev_synthetic_hourly_load.csv")
print("Saved: ev_raw_simulation_log.csv")

# ==========================================
# QUICK PLOTS
# ==========================================

fig, axes = plt.subplots(2, 2, figsize=(16, 8))
fig.suptitle('TwinEV Synthetic Data Overview', fontsize=14)

# 1. Annual load
axes[0,0].plot(hourly['power_kw'], linewidth=0.5, color='#2c7be5')
axes[0,0].set_title('Annual EV Charging Load')
axes[0,0].set_ylabel('Power (kW)')

# 2. Avg daily pattern
daily = hourly.groupby('hour')['power_kw'].mean()
axes[0,1].bar(daily.index, daily.values, color='#f6c23e')
axes[0,1].set_title('Average Load by Hour of Day')
axes[0,1].set_xlabel('Hour')

# 3. Solar profile
solar_daily = hourly.groupby('hour')['solar_kw'].mean()
axes[1,0].plot(solar_daily.index, solar_daily.values,
               color='#e74a3b', linewidth=2)
axes[1,0].set_title('Average Solar Generation by Hour')
axes[1,0].set_xlabel('Hour')
axes[1,0].set_ylabel('Solar (kW)')

# 4. Price profile
price_daily = hourly.groupby('hour')['price_eur_kwh'].mean()
axes[1,1].plot(price_daily.index, price_daily.values,
               color='#1cc88a', linewidth=2)
axes[1,1].set_title('Average Energy Price by Hour')
axes[1,1].set_xlabel('Hour')
axes[1,1].set_ylabel('Price (EUR/kWh)')

plt.tight_layout()
plt.savefig('plot_data_overview.png', dpi=150)
plt.show()

print("\nData generation complete. Run lstm_forecast.py next.")
