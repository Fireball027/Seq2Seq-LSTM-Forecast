# IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Model
from keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from keras.callbacks import EarlyStopping
import joblib
import math

# CONFIGURATION & SEED
np.random.seed(42)    # Make results reproducible

# Input sequence length (months)
N_STEPS_IN = 12
# Number of months to predict ahead
N_STEPS_OUT = 6

print(f"\nConfigured Encoder–Decoder LSTM with input steps: {N_STEPS_IN} and output steps: {N_STEPS_OUT}\n")

# LOAD & SCALE DATA
print("Loading dataset...")
df = pd.read_csv('airline-passengers.csv', usecols=[1])
dataset = df.values.astype('float32')

print(f"Original dataset shape: {dataset.shape}")

# Normalize data to [0, 1] range for better LSTM performance
scaler = MinMaxScaler()
dataset_scaled = scaler.fit_transform(dataset)

# Split into 67% train, 33% test
train_size = int(len(dataset_scaled) * 0.67)
train, test = dataset_scaled[:train_size], dataset_scaled[train_size:]

print(f"Train samples: {len(train)}, Test samples: {len(test)}\n")


# CREATE SUPERVISED SEQUENCES
def split_sequence(sequence, n_steps_in, n_steps_out):
    """
    Transform univariate sequence into supervised learning format.
    """
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(sequence):
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


print("Creating supervised input-output pairs...")
X_train, y_train = split_sequence(train, N_STEPS_IN, N_STEPS_OUT)
X_test, y_test = split_sequence(test, N_STEPS_IN, N_STEPS_OUT)

print(f"X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")

# Reshape to [samples, timesteps, features] for LSTM
X_train = X_train.reshape((X_train.shape[0], N_STEPS_IN, 1))
y_train = y_train.reshape((y_train.shape[0], N_STEPS_OUT, 1))
X_test = X_test.reshape((X_test.shape[0], N_STEPS_IN, 1))
y_test = y_test.reshape((y_test.shape[0], N_STEPS_OUT, 1))

print("Data reshaped for LSTM: 3D tensors.\n")

# DEFINE ENCODER–DECODER ARCHITECTURE
print("Building Encoder–Decoder LSTM model...")

# Encoder
encoder_inputs = Input(shape=(N_STEPS_IN, 1))
encoder_lstm = LSTM(100, activation='relu')(encoder_inputs)
repeat_vector = RepeatVector(N_STEPS_OUT)(encoder_lstm)
# Decoder
decoder_lstm = LSTM(100, activation='relu', return_sequences=True)(repeat_vector)
decoder_outputs = TimeDistributed(Dense(1))(decoder_lstm)

# Compile model
model = Model(inputs=encoder_inputs, outputs=decoder_outputs)
model.compile(optimizer='adam', loss='mse')

print(model.summary())

# TRAIN MODEL WITH EARLY STOPPING
print("Starting training...")
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=300,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

print("Training completed!\n")

# MAKE PREDICTIONS & INVERSE SCALE
print("Making predictions and inverting scaling...")

yhat_train = model.predict(X_train)
yhat_test = model.predict(X_test)

# Inverse transform back to original scale
yhat_train_inv = scaler.inverse_transform(yhat_train.reshape(-1, 1)).reshape(yhat_train.shape)
yhat_test_inv = scaler.inverse_transform(yhat_test.reshape(-1, 1)).reshape(yhat_test.shape)
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

print("Predictions ready.\n")


# EVALUATE PERFORMANCE
def evaluate_forecast(true_seq, pred_seq, label="Dataset"):
    """
    Calculate RMSE for each forecast horizon step.
    """
    print(f"\nRMSE for {label}:")
    for step in range(true_seq.shape[1]):
        rmse = math.sqrt(mean_squared_error(true_seq[:, step, 0], pred_seq[:, step, 0]))
        print(f"  t+{step+1} RMSE: {rmse:.2f}")


evaluate_forecast(y_train_inv, yhat_train_inv, "Train")
evaluate_forecast(y_test_inv, yhat_test_inv, "Test")

# DIAGNOSTIC PLOTS
print("\nGenerating diagnostic plots...")

# Plot Loss Curve
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Overlay forecast vs actual for first few test windows
N_EXAMPLES = 5
plt.figure(figsize=(14, 6))
for i in range(N_EXAMPLES):
    plt.plot(y_test_inv[i, :, 0], marker='o', label=f'Actual #{i+1}')
    plt.plot(yhat_test_inv[i, :, 0], marker='x', linestyle='--', label=f'Predicted #{i+1}')
plt.title(f'Overlay of {N_EXAMPLES} Test Forecast Windows')
plt.xlabel('Forecast Horizon Step')
plt.ylabel('Passengers')
plt.legend()
plt.show()

# Residuals distribution for each forecast step
residuals = y_test_inv - yhat_test_inv

for step in range(N_STEPS_OUT):
    plt.figure(figsize=(8, 4))
    sns.histplot(residuals[:, step, 0], bins=20, kde=True)
    plt.title(f'Residuals Distribution for t+{step+1}')
    plt.xlabel('Prediction Error')
    plt.show()

# Scatter plot: Actual vs Predicted
plt.figure(figsize=(8, 8))
plt.scatter(y_test_inv.flatten(), yhat_test_inv.flatten(), alpha=0.6)
plt.plot([min(y_test_inv.flatten()), max(y_test_inv.flatten())],
         [min(y_test_inv.flatten()), max(y_test_inv.flatten())],
         'r--')
plt.title('Predicted vs Actual (All Forecast Steps)')
plt.xlabel('Actual Passengers')
plt.ylabel('Predicted Passengers')
plt.show()

# RMSE by forecast step
rmse_per_step = []
for step in range(N_STEPS_OUT):
    rmse = math.sqrt(mean_squared_error(y_test_inv[:, step, 0], yhat_test_inv[:, step, 0]))
    rmse_per_step.append(rmse)

plt.figure(figsize=(10, 4))
plt.plot(range(1, N_STEPS_OUT + 1), rmse_per_step, marker='o')
plt.title('RMSE vs Forecast Horizon')
plt.xlabel('Forecast Step')
plt.ylabel('RMSE')
plt.xticks(range(1, N_STEPS_OUT + 1))
plt.show()

print("All plots generated.\n")

# SAVE MODEL & SCALER
model.save('encoder_decoder_lstm.h5')
joblib.dump(scaler, 'scaler.save')

print("Saved trained Encoder–Decoder model to 'encoder_decoder_lstm.h5'")
print("Saved scaler to 'scaler.save'\n")

print("Pipeline finished successfully! Ready for deployment.")
