# ==========================================
# TwinEV v2 — LSTM Load Forecaster
# Run this SECOND after generate_data.py
# ==========================================
# pip install tensorflow scikit-learn pandas numpy matplotlib
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from plot_style import set_style

set_style()

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(
    'ev_synthetic_hourly_load.csv',
    index_col=0,
    parse_dates=True
)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# ==========================================
# FEATURE ENGINEERING
# ==========================================

# Lag features
df['lag1']   = df['power_kw'].shift(1)
df['lag24']  = df['power_kw'].shift(24)
df['lag168'] = df['power_kw'].shift(168)   # 1 week

# Rolling statistics
df['rolling_mean_24'] = df['power_kw'].rolling(24).mean()
df['rolling_std_24']  = df['power_kw'].rolling(24).std()
df['rolling_mean_48'] = df['power_kw'].rolling(48).mean()

# Cyclic time encoding (better than raw integers)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['day_sin']  = np.sin(2 * np.pi * df['dayofweek'] / 7)
df['day_cos']  = np.cos(2 * np.pi * df['dayofweek'] / 7)

df.dropna(inplace=True)

# ==========================================
# FEATURE LIST
# ==========================================

features = [
    'power_kw',           # target (index 0)
    'solar_kw',           # solar generation
    'price_eur_kwh',      # energy price
    'ev_count',           # number of EVs per hour
    'soc_arrival',        # avg battery SoC on arrival
    'lag1',
    'lag24',
    'lag168',
    'rolling_mean_24',
    'rolling_std_24',
    'rolling_mean_48',
    'hour_sin',
    'hour_cos',
    'day_sin',
    'day_cos',
]

print(f"\nUsing {len(features)} features: {features}")

# ==========================================
# NORMALIZATION
# ==========================================

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features])

# Save scaler for use in RL environment
import joblib
joblib.dump(scaler, 'twinev_scaler.pkl')
print("Scaler saved: twinev_scaler.pkl")

# ==========================================
# SEQUENCE CREATION
# ==========================================

SEQ_LEN = 24   # use last 24 hours to predict next hour

def build_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len, 0])   # predict power_kw
    return np.array(X), np.array(y)

X, y = build_sequences(scaled_data, SEQ_LEN)
print(f"\nSequence shape — X: {X.shape}, y: {y.shape}")

# ==========================================
# TRAIN / VAL / TEST SPLIT
# ==========================================

n         = len(X)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

X_train, y_train = X[:train_end],       y[:train_end]
X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
X_test,  y_test  = X[val_end:],          y[val_end:]

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ==========================================
# BIDIRECTIONAL LSTM MODEL
# (Bidirectional captures both past and
#  future context within the sequence window)
# ==========================================

model = Sequential([

    Bidirectional(
        LSTM(128, return_sequences=True),
        input_shape=(SEQ_LEN, len(features))
    ),
    Dropout(0.2),

    LSTM(64, return_sequences=True),
    Dropout(0.2),

    LSTM(32, return_sequences=False),
    Dropout(0.1),

    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])

model.summary()

# ==========================================
# COMPILE
# ==========================================

model.compile(
    optimizer='adam',
    loss='huber',        # Huber loss: robust to outliers vs pure MSE
    metrics=['mae']
)

# ==========================================
# CALLBACKS
# ==========================================

callbacks = [
    EarlyStopping(
        patience=15,
        restore_best_weights=True,
        monitor='val_loss'
    ),
    ModelCheckpoint(
        'best_lstm_model.keras',
        save_best_only=True,
        monitor='val_loss'
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

# ==========================================
# TRAIN
# ==========================================

print("\nTraining LSTM...")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

# ==========================================
# PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# INVERSE SCALING
# ==========================================

def inverse_target(scaled_vals, scaler, n_features):
    dummy = np.zeros((len(scaled_vals), n_features))
    dummy[:, 0] = scaled_vals.flatten()
    return scaler.inverse_transform(dummy)[:, 0]

y_test_inv = inverse_target(y_test, scaler, len(features))
y_pred_inv = inverse_target(y_pred, scaler, len(features))

# ==========================================
# METRICS
# ==========================================

rmse = math.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
mae  = mean_absolute_error(y_test_inv, y_pred_inv)
mape = np.mean(
    np.abs((y_test_inv - y_pred_inv) / (y_test_inv + 1e-5))
) * 100

print("\n===== LSTM RESULTS =====")
print(f"RMSE : {rmse:.2f} kW")
print(f"MAE  : {mae:.2f} kW")
print(f"MAPE : {mape:.2f}%")

# ==========================================
# SAVE PREDICTIONS
# ==========================================

results_df = pd.DataFrame({
    'actual'    : y_test_inv,
    'predicted' : y_pred_inv
})
results_df.to_csv('lstm_predictions.csv', index=False)
print("Saved: lstm_predictions.csv")

# Also save full-length predictions for RL environment
full_pred = model.predict(X)
full_pred_inv = inverse_target(full_pred, scaler, len(features))

full_df = pd.DataFrame({
    'actual'    : inverse_target(y, scaler, len(features)),
    'predicted' : full_pred_inv
})
full_df.to_csv('lstm_full_predictions.csv', index=False)
print("Saved: lstm_full_predictions.csv")

# ==========================================
# SAVE MODEL
# ==========================================

model.save('twinev_lstm_model.keras')
print("Saved: twinev_lstm_model.keras")

# ==========================================
# PLOTS
# ==========================================

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Forecast comparison
axes[0].plot(y_test_inv[:168], label='Actual', linewidth=1.5)
axes[0].plot(y_pred_inv[:168], linestyle='--',
             label='Predicted', linewidth=1.5)
axes[0].set_title(f'LSTM Forecast — 7 Day Window\nRMSE={rmse:.2f} kW  MAE={mae:.2f} kW')
axes[0].set_xlabel('Hour')
axes[0].set_ylabel('Power Load (kW)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Training loss
axes[1].plot(history.history['loss'],     label='Train Loss')
axes[1].plot(history.history['val_loss'], label='Val Loss')
axes[1].set_title('LSTM Training Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Huber Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot_lstm_results.png', dpi=150)
plt.show()

print("\nLSTM training complete. Run twinev_env.py next.")
