# ==========================================
# TwinEV v2 — Gymnasium Environment
# Multi-Objective RL + PI Controller
# FIXED: proper train / val / test split
# ==========================================

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from pi_controller import PIController

# ==========================================
# REWARD WEIGHTS
# ==========================================

REWARD_WEIGHTS = {
    'grid'     : 1.5,
    'solar'    : 1.5,
    'queue'    : 5.0,
    'price'    : 1.0,
    'pi_error' : 0.3,
}
# ==========================================
# STATION LIMITS
# ==========================================

MAX_GRID_KW  = 150.0
OVERLOAD_KW  = 130.0
SOLAR_MAX_KW = 200.0
MAX_QUEUE    = 100
MAX_PRICE    = 0.55

# ==========================================
# DATA SPLIT RATIOS
# ==========================================
#
#   Train : 0.00 → 0.70   (70%)
#   Val   : 0.70 → 0.85   (15%)
#   Test  : 0.85 → 1.00   (15%) ← strictly unseen
#
# ==========================================


class TwinEVEnv(gym.Env):
    """
    TwinEV Digital Twin — EV Charging Station RL Environment

    Modes
    -----
    train : episodes sampled from first 70% of data
    val   : episodes sampled from 70-85% of data
            used for reward weight tuning
    test  : episodes sampled from last 15% of data
            NEVER seen during training — final evaluation only
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        data_path     = 'ev_synthetic_hourly_load.csv',
        forecast_path = 'lstm_full_predictions.csv',
        mode          = 'train',   # 'train' | 'val' | 'test'
        max_steps     = 168,       # 1 week per episode
    ):
        super().__init__()

        assert mode in ('train', 'val', 'test'), \
            f"mode must be 'train', 'val', or 'test'. Got: {mode}"

        self.max_steps = max_steps
        self.mode      = mode

        # ------------------------------------------
        # LOAD DATA
        # ------------------------------------------

        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        fc = pd.read_csv(forecast_path)

        self.grid_load     = df['power_kw'].values.astype(np.float32)
        self.solar         = df['solar_kw'].values.astype(np.float32)
        self.price         = df['price_eur_kwh'].values.astype(np.float32)
        self.ev_queue      = df['ev_count'].values.astype(np.float32)
        self.lstm_forecast = fc['predicted'].values.astype(np.float32)

        min_len = min(len(self.grid_load), len(self.lstm_forecast))
        self.grid_load     = self.grid_load[:min_len]
        self.solar         = self.solar[:min_len]
        self.price         = self.price[:min_len]
        self.ev_queue      = self.ev_queue[:min_len]
        self.lstm_forecast = self.lstm_forecast[:min_len]

        self.n_steps_total = min_len

        # ------------------------------------------
        # PROPER TRAIN / VAL / TEST BOUNDARIES
        # ------------------------------------------

        self.train_end = int(self.n_steps_total * 0.70)  # hour 6132
        self.val_end   = int(self.n_steps_total * 0.85)  # hour 7446

        # Episode start boundaries per mode
        if self.mode == 'train':
            self.start_min = 0
            self.start_max = self.train_end - self.max_steps - 1
        elif self.mode == 'val':
            self.start_min = self.train_end
            self.start_max = self.val_end - self.max_steps - 1
        else:  # test — strictly unseen
            self.start_min = self.val_end
            self.start_max = self.n_steps_total - self.max_steps - 1

        print(f"Environment [{mode}] loaded: {min_len} total hours")
        print(f"  Train: 0 → {self.train_end} hrs")
        print(f"  Val  : {self.train_end} → {self.val_end} hrs")
        print(f"  Test : {self.val_end} → {self.n_steps_total} hrs")
        print(f"  This episode range: {self.start_min} → {self.start_max}")

        # ------------------------------------------
        # SPACES
        # ------------------------------------------

        self.observation_space = spaces.Box(
            low   = np.zeros(7, dtype=np.float32),
            high  = np.ones(7,  dtype=np.float32),
            dtype = np.float32
        )

        # Minimum 0.2 → at least 30 kW (prevents zero-charging hack)
        self.action_space = spaces.Box(
            low   = np.array([0.2], dtype=np.float32),
            high  = np.array([1.0], dtype=np.float32),
            dtype = np.float32
        )

        # ------------------------------------------
        # PI CONTROLLER
        # ------------------------------------------

        self.pi = PIController(
            Kp           = 0.6,
            Ki           = 0.15,
            dt           = 1.0,
            output_min   = 0.0,
            output_max   = MAX_GRID_KW,
            windup_limit = 40.0
        )

        # ------------------------------------------
        # INTERNAL STATE
        # ------------------------------------------

        self.current_step  = 0
        self.episode_start = 0
        self.actual_power  = 0.0

        self.ep_rewards   = []
        self.ep_actions   = []
        self.ep_actual    = []
        self.ep_setpoints = []

    # ==========================================
    # RESET
    # ==========================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.pi.reset()
        self.current_step = 0
        self.ep_rewards   = []
        self.ep_actions   = []
        self.ep_actual    = []
        self.ep_setpoints = []

        # Sample episode start strictly within mode boundary
        low  = self.start_min
        high = max(self.start_min + 1, self.start_max)

        self.episode_start = int(
            self.np_random.integers(low, high)
        )

        idx = self.episode_start
        self.actual_power = float(self.grid_load[idx])

        return self._get_obs(idx), {}

    # ==========================================
    # STEP
    # ==========================================

    def step(self, action):

        idx = self.episode_start + self.current_step

        action_scalar = float(np.clip(action[0], 0.2, 1.0))
        setpoint_kw   = action_scalar * MAX_GRID_KW

        actual_power = self.pi.step(
            setpoint = setpoint_kw,
            actual   = self.actual_power
        )
        self.actual_power = actual_power

        grid_load    = float(self.grid_load[idx])
        solar_gen    = float(self.solar[idx])
        energy_price = float(self.price[idx])
        queue_len    = float(self.ev_queue[idx])
        lstm_next    = float(self.lstm_forecast[
                            min(idx+1, self.n_steps_total-1)])

        reward, reward_info = self._compute_reward(
            actual_power  = actual_power,
            setpoint_kw   = setpoint_kw,
            grid_load     = grid_load,
            solar_gen     = solar_gen,
            energy_price  = energy_price,
            queue_len     = queue_len,
            lstm_forecast = lstm_next,
        )

        self.ep_rewards.append(reward)
        self.ep_actions.append(action_scalar)
        self.ep_actual.append(actual_power)
        self.ep_setpoints.append(setpoint_kw)

        self.current_step += 1
        done = self.current_step >= self.max_steps

        next_idx = min(
            self.episode_start + self.current_step,
            self.n_steps_total - 1
        )

        obs  = self._get_obs(next_idx)
        info = {**reward_info, 'step': self.current_step}

        return obs, reward, done, False, info

    # ==========================================
    # OBSERVATION
    # ==========================================

    def _get_obs(self, idx: int) -> np.ndarray:
        hour = idx % 24
        return np.array([
            np.clip(self.grid_load[idx]    / MAX_GRID_KW,  0, 1),
            np.clip(self.price[idx]         / MAX_PRICE,    0, 1),
            np.clip(self.solar[idx]         / SOLAR_MAX_KW, 0, 1),
            np.clip(self.ev_queue[idx]      / MAX_QUEUE,    0, 1),
            np.clip(self.lstm_forecast[idx] / MAX_GRID_KW,  0, 1),
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
        ], dtype=np.float32)

    # ==========================================
    # REWARD FUNCTION
    # ==========================================

    def _compute_reward(
        self,
        actual_power  : float,
        setpoint_kw   : float,
        grid_load     : float,
        solar_gen     : float,
        energy_price  : float,
        queue_len     : float,
        lstm_forecast : float,
    ) -> tuple:

        W = REWARD_WEIGHTS

        # ------------------------------------------
        # R1 — GRID
        # Sweet spot at 70% (105 kW) — not too low, not too high
        # ------------------------------------------

        if actual_power > MAX_GRID_KW:
            r_grid = -10.0 * (actual_power - MAX_GRID_KW) / MAX_GRID_KW
        elif actual_power > OVERLOAD_KW:
            r_grid = -3.0 * (actual_power - OVERLOAD_KW) / (
                MAX_GRID_KW - OVERLOAD_KW)
        else:
            utilization = actual_power / MAX_GRID_KW
            r_grid = 1.0 - abs(utilization - 0.70)

        # Penalize ignoring EV queue
        if queue_len > 5 and actual_power < 30:
            r_grid -= 2.0 * (queue_len / MAX_QUEUE)

        # Reward actively serving queue
        if queue_len > 5 and actual_power >= 50:
            r_grid += 0.4 * (actual_power / MAX_GRID_KW)

        # ------------------------------------------
        # R2 — SOLAR
        # ------------------------------------------

        solar_utilized = min(actual_power, solar_gen)
        r_solar = solar_utilized / (SOLAR_MAX_KW + 1e-5)

        if solar_gen > 80 and actual_power > 60:
            r_solar += 0.5

        if solar_gen > 80 and actual_power < 30:
            r_solar -= 0.5

        # ------------------------------------------
        # R3 — QUEUE
        # ------------------------------------------

        r_queue = -(queue_len / MAX_QUEUE)

        if queue_len > 30 and actual_power < 50:
            r_queue -= 1.0

        if queue_len > 10 and actual_power > 80:
            r_queue += 0.3

        # ------------------------------------------
        # R4 — PRICE
        # ------------------------------------------

        price_norm = energy_price / MAX_PRICE
        if actual_power > 30:
            r_price = (1.0 - price_norm)
        else:
            r_price = -0.3

        # ------------------------------------------
        # R5 — PI TRACKING ERROR
        # ------------------------------------------

        r_pi_error = -(self.pi.tracking_error / MAX_GRID_KW)

        # ------------------------------------------
        # LOOKAHEAD BONUS (LSTM)
        # ------------------------------------------

        lookahead_bonus = 0.0
        if lstm_forecast > OVERLOAD_KW and actual_power < 80:
            lookahead_bonus = 0.3

        # ------------------------------------------
        # WEIGHTED SUM
        # ------------------------------------------

        total_reward = (
            W['grid']     * r_grid
          + W['solar']    * r_solar
          + W['queue']    * r_queue
          + W['price']    * r_price
          + W['pi_error'] * r_pi_error
          + lookahead_bonus
        )

        reward_info = {
            'r_grid'       : round(r_grid,         4),
            'r_solar'      : round(r_solar,        4),
            'r_queue'      : round(r_queue,        4),
            'r_price'      : round(r_price,        4),
            'r_pi_error'   : round(r_pi_error,     4),
            'lookahead'    : round(lookahead_bonus, 4),
            'total_reward' : round(total_reward,   4),
            'actual_power' : round(actual_power,   2),
            'setpoint_kw'  : round(setpoint_kw,    2),
        }

        return float(total_reward), reward_info

    # ==========================================
    # EPISODE SUMMARY
    # ==========================================

    def episode_summary(self) -> dict:
        return {
            'mean_reward'  : round(np.mean(self.ep_rewards), 4),
            'total_reward' : round(np.sum(self.ep_rewards),  4),
            'mean_power'   : round(np.mean(self.ep_actual),  2),
            'max_power'    : round(np.max(self.ep_actual),   2),
            'overloads'    : int(np.sum(
                [1 for p in self.ep_actual if p > MAX_GRID_KW]
            )),
        }


# ==========================================
# SANITY CHECK
# ==========================================

if __name__ == '__main__':

    print("=" * 50)
    print("Testing all 3 modes...")
    print("=" * 50)

    for mode in ['train', 'val', 'test']:
        env = TwinEVEnv(mode=mode)
        obs, _ = env.reset()
        total_r = 0
        for _ in range(168):
            action = env.action_space.sample()
            obs, reward, done, _, info = env.step(action)
            total_r += reward
            if done:
                break
        summary = env.episode_summary()
        print(f"\n[{mode}] Total reward : {total_r:.2f}")
        print(f"[{mode}] Mean power   : {summary['mean_power']} kW")
        print(f"[{mode}] Overloads    : {summary['overloads']}")

    print("\nAll 3 modes OK.")