\# TwinEV v2

\## AI-Based Digital Twin for EV Charging Station Management using LSTM Forecasting, PPO Reinforcement Learning and PI Control



\---



\## Overview



TwinEV is an AI-driven digital twin framework designed for intelligent management of Electric Vehicle Charging Stations (EVCS).



The framework combines:



\- Deep Learning based load forecasting using Bidirectional LSTM

\- Reinforcement Learning based charging optimization using Proximal Policy Optimization (PPO)

\- Classical PI control for stable power tracking

\- Multi-objective reward optimization considering:

&#x20; - Grid constraints

&#x20; - Solar utilization

&#x20; - EV queue management

&#x20; - Electricity price

&#x20; - Tracking error



The objective is to reduce grid overloads while maintaining efficient EV charging operation.



\---



\# System Architecture





EV Charging Data

|

|

v



Synthetic Digital Twin Environment

|

|

+----------------+

| |

v v



LSTM Forecasting Real-time State



&#x20;   |

&#x20;   |

&#x20;   v



PPO Reinforcement Learning Agent



&#x20;   |

&#x20;   |

&#x20;   v



Charging Power Setpoint



&#x20;   |

&#x20;   |

&#x20;   v



PI Controller



&#x20;   |

&#x20;   |

&#x20;   v



Actual EV Charging Power





\---



\# Dataset



The digital twin generates hourly EV charging station data.



Simulation duration:





1 Year

8760 hourly samples





Features:



| Feature | Description |

|-|-|

| power\_kw | EV charging demand |

| solar\_kw | Renewable generation |

| price\_eur\_kwh | Electricity price |

| ev\_count | Number of connected EVs |

| soc\_arrival | Battery state of charge |



\---



\# Module Description



\## 1. Data Generation



File:





generate\_data.py





Creates synthetic EV charging station operation data.



Output:





ev\_synthetic\_hourly\_load.csv





\---



\# 2. LSTM Forecasting



File:





lstm\_forecast.py





A Bidirectional LSTM predicts future charging demand.



Input:



Previous 24 hours sequence



Features:



\- Historical load

\- Solar generation

\- Electricity price

\- EV arrival count

\- SOC information

\- Lag features

\- Rolling statistics

\- Time encoding





Architecture:





BiLSTM(128)



Dropout



LSTM(64)



Dropout



LSTM(32)



Dense Layers



Prediction





Output:





lstm\_predictions.csv

lstm\_full\_predictions.csv





\---



\# 3. Reinforcement Learning Environment



File:





twinev\_env.py





Implemented using Gymnasium.



Observation Space:



7 states:





Current load

Electricity price

Solar generation

EV queue

LSTM forecast

Hour sin

Hour cos





Action:



Continuous charging power decision:





30 kW - 150 kW





\---



\# 4. PPO Agent



File:





train\_rl.py





Algorithm:



Proximal Policy Optimization (PPO)



Network:





256

|

256

|

128

|

Action





Training:





500,000 timesteps





\---



\# 5. PI Controller



File:





pi\_controller.py





The PI controller converts RL decisions into smooth physical charging commands.



Purpose:



\- Prevent sudden power changes

\- Maintain grid stability

\- Reduce oscillations



\---



\# 6. Baselines



File:





baseline.py





Compared against:



\## FCFS



First Come First Serve charging.





\## Round Robin



Equal power distribution.





\## Greedy



Peak limiting controller.



\---



\# Evaluation



File:





evaluate.py





The model is evaluated on unseen test data.



Dataset split:





Train : 70%



Validation : 15%



Test : 15%





The test set is never used during training.



\---



\# Performance Metrics



Forecast metrics:



\- RMSE

\- MAE

\- MAPE





Operational metrics:



\- Peak power reduction

\- Energy delivered

\- Grid overload events

\- Power variance

\- Charging cost





\---



\# Technologies



Python



Libraries:





TensorFlow

PyTorch

Gymnasium

Stable-Baselines3

Scikit-learn

Pandas

NumPy

Matplotlib





\---



\# Execution Pipeline



Run in order:





python generate\_data.py



python lstm\_forecast.py



python train\_rl.py



python evaluate.py





\---



\# Research Contribution



TwinEV integrates:



1\. Forecast-informed reinforcement learning



2\. Digital twin simulation



3\. Multi-objective EV charging optimization



4\. Hybrid RL + classical control architecture





The framework demonstrates how AI forecasting and reinforcement learning can improve EV charging station reliability while respecting grid constraints.

