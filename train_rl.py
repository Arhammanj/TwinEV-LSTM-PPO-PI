# ==========================================
# TwinEV v2 — PPO Training Script
# ==========================================
# pip install stable-baselines3[extra] gymnasium
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from twinev_env import TwinEVEnv
from plot_style import set_style

set_style()

# ==========================================
# CONFIG
# ==========================================

TOTAL_TIMESTEPS = 500_000   # increase to 500k for better results
EVAL_FREQ       = 5_000     # evaluate every N steps
N_ENVS          = 4         # parallel environments

LOG_DIR         = './ppo_logs/'
MODEL_SAVE_PATH = 'twinev_ppo_agent'

os.makedirs(LOG_DIR, exist_ok=True)

# ==========================================
# REWARD LOGGER CALLBACK
# ==========================================

class RewardLoggerCallback(BaseCallback):
    """
    Logs episode rewards and component breakdown
    to CSV during training.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Collect episode info from Monitor wrapper
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])

        return True

    def save_log(self, path='training_reward_log.csv'):
        pd.DataFrame({
            'episode_reward' : self.episode_rewards,
            'episode_length' : self.episode_lengths,
        }).to_csv(path, index=False)
        print(f"Reward log saved: {path}")


# ==========================================
# BUILD ENVIRONMENTS
# ==========================================

def make_env(mode='train'):
    def _init():
        env = TwinEVEnv(mode=mode)
        env = Monitor(env)
        return env
    return _init

print("Building training environments...")

train_env = DummyVecEnv([make_env('train')] * N_ENVS)
eval_env  = DummyVecEnv([make_env('val')])

# ==========================================
# PPO MODEL
# ==========================================
# Using MlpPolicy with a larger network
# since we have 7-dim state + continuous action
# ==========================================

model = PPO(
    policy          = 'MlpPolicy',
    env             = train_env,
    verbose         = 1,
    tensorboard_log = LOG_DIR,

    # Hyperparameters tuned for EV charging task
    learning_rate   = 3e-4,
    n_steps         = 2048,
    batch_size      = 256,
    n_epochs        = 10,
    gamma           = 0.99,       # discount factor
    gae_lambda      = 0.95,       # GAE smoothing
    clip_range      = 0.2,        # PPO clipping
    ent_coef        = 0.01,       # entropy bonus for exploration
    vf_coef         = 0.5,
    max_grad_norm   = 0.5,

    # Network architecture
    policy_kwargs   = dict(
        net_arch = [256, 256, 128]   # 3-layer MLP
    )
)

print(model.policy)

# ==========================================
# CALLBACKS
# ==========================================

reward_logger = RewardLoggerCallback()

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path = './best_model/',
    log_path             = LOG_DIR,
    eval_freq            = EVAL_FREQ,
    n_eval_episodes      = 5,
    deterministic        = True,
    verbose              = 1
)

checkpoint_callback = CheckpointCallback(
    save_freq   = 10_000,
    save_path   = './checkpoints/',
    name_prefix = 'twinev_ppo'
)

# ==========================================
# TRAIN
# ==========================================

print(f"\nStarting PPO training — {TOTAL_TIMESTEPS:,} timesteps")
print("=" * 50)

model.learn(
    total_timesteps = TOTAL_TIMESTEPS,
    callback        = [reward_logger, eval_callback, checkpoint_callback],
    progress_bar    = True
)

# ==========================================
# SAVE FINAL MODEL
# ==========================================

model.save(MODEL_SAVE_PATH)
print(f"\nModel saved: {MODEL_SAVE_PATH}")

reward_logger.save_log()

# ==========================================
# EVALUATE TRAINED AGENT
# ==========================================

print("\n===== EVALUATING TRAINED AGENT =====\n")

eval_env_single = TwinEVEnv(mode='val')
obs, _          = eval_env_single.reset()

results = {
    'step'        : [],
    'actual_power': [],
    'setpoint'    : [],
    'reward'      : [],
    'r_grid'      : [],
    'r_solar'     : [],
    'r_queue'     : [],
    'r_price'     : [],
}

total_reward = 0

for step in range(168):   # evaluate on 1 week

    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = eval_env_single.step(action)

    total_reward += reward

    results['step'].append(step)
    results['actual_power'].append(info['actual_power'])
    results['setpoint'].append(info['setpoint_kw'])
    results['reward'].append(info['total_reward'])
    results['r_grid'].append(info['r_grid'])
    results['r_solar'].append(info['r_solar'])
    results['r_queue'].append(info['r_queue'])
    results['r_price'].append(info['r_price'])

    if step % 24 == 0:
        print(f"Day {step//24+1:2d} | "
              f"Power: {info['actual_power']:6.1f} kW | "
              f"Reward: {info['total_reward']:+.3f} | "
              f"Grid: {info['r_grid']:+.2f} | "
              f"Solar: {info['r_solar']:+.2f} | "
              f"Queue: {info['r_queue']:+.2f}")

    if done:
        break

results_df = pd.DataFrame(results)
results_df.to_csv('rl_eval_results.csv', index=False)

ep_summary = eval_env_single.episode_summary()

print("\n===== EPISODE SUMMARY =====")
print(f"Total Reward  : {total_reward:.2f}")
print(f"Mean Reward   : {ep_summary['mean_reward']}")
print(f"Mean Power    : {ep_summary['mean_power']} kW")
print(f"Max Power     : {ep_summary['max_power']} kW")
print(f"Overloads     : {ep_summary['overloads']}")

# ==========================================
# PLOTS
# ==========================================

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('TwinEV v2 — PPO + PI Controller Evaluation', fontsize=14)

# 1. Power profile: setpoint vs actual
axes[0,0].plot(results_df['setpoint'],     label='RL Setpoint',
               linestyle='--', alpha=0.7)
axes[0,0].plot(results_df['actual_power'], label='PI Output (Actual)',
               linewidth=2)
axes[0,0].axhline(130, color='orange', linestyle=':', label='Soft Limit')
axes[0,0].axhline(150, color='red',    linestyle=':', label='Hard Limit')
axes[0,0].set_title('Power: RL Setpoint vs PI Actual Output')
axes[0,0].set_ylabel('Power (kW)')
axes[0,0].legend(fontsize=8)
axes[0,0].grid(True, alpha=0.3)

# 2. Reward breakdown
axes[0,1].fill_between(results_df['step'], results_df['reward'],
                        alpha=0.4, label='Total Reward')
axes[0,1].plot(results_df['reward'], linewidth=1.5)
axes[0,1].axhline(0, color='gray', linestyle='--')
axes[0,1].set_title('Total Reward per Step')
axes[0,1].set_ylabel('Reward')
axes[0,1].grid(True, alpha=0.3)

# 3. Reward components
axes[1,0].plot(results_df['r_grid'],  label='Grid',  linewidth=1.5)
axes[1,0].plot(results_df['r_solar'], label='Solar', linewidth=1.5)
axes[1,0].plot(results_df['r_queue'], label='Queue', linewidth=1.5)
axes[1,0].plot(results_df['r_price'], label='Price', linewidth=1.5)
axes[1,0].axhline(0, color='gray', linestyle='--')
axes[1,0].set_title('Reward Component Breakdown')
axes[1,0].set_ylabel('Component Reward')
axes[1,0].legend(fontsize=8)
axes[1,0].grid(True, alpha=0.3)

# 4. Training reward curve
if os.path.exists('training_reward_log.csv'):
    train_log = pd.read_csv('training_reward_log.csv')
    axes[1,1].plot(
        train_log['episode_reward'].rolling(20).mean(),
        label='20-ep moving avg'
    )
    axes[1,1].set_title('Training Reward (Moving Average)')
    axes[1,1].set_xlabel('Episode')
    axes[1,1].set_ylabel('Episode Reward')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
else:
    axes[1,1].text(0.5, 0.5, 'Training log\nnot found',
                   ha='center', va='center', transform=axes[1,1].transAxes)

plt.tight_layout()
plt.savefig('plot_rl_evaluation.png', dpi=150)
plt.savefig(
    "plots/03_training/training_reward_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nTraining complete. All outputs saved.")
print("Files created:")
print("  twinev_ppo_agent.zip   — trained RL model")
print("  rl_eval_results.csv    — evaluation data")
print("  plot_rl_evaluation.png — evaluation plots")