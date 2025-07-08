## Overview

This project applies a powerful **Encoder–Decoder LSTM (Long Short-Term Memory)** architecture for **multi-step forecasting** on time series data. Using the classic **Airline Passengers** dataset, the model predicts future passenger counts based on historical trends.

---

## 🚀 Key Features

✅ **Multi-Step Forecasting** — Predicts multiple future time steps (e.g., 6 months ahead).
✅ **Data Normalization** — Scales data for better neural network performance.
✅ **Encoder–Decoder LSTM** — Implements sequence-to-sequence learning for time series.
✅ **Early Stopping** — Stops training when the model stops improving.
✅ **Evaluation** — Calculates RMSE for each forecast step.
✅ **Diagnostic Plots** — Visualizes actual vs. predicted, residuals, forecast horizons, and more.

---

## 📂 Project Files

### `airline-passengers.csv`

This dataset contains monthly international airline passenger numbers (1949–1960).

* **Column:**

  * `#Passengers` — number of passengers per month.

---

### `encoder_decoder_lstm.py`

This script:

* Loads and **normalizes data**.
* Creates **supervised learning sequences**.
* Defines and trains an **Encoder–Decoder LSTM**.
* **Evaluates performance**.
* Generates **diagnostic plots**.
* Saves the **trained model and scaler**.

---

## 🧩 Example Code Snippet

```python
# Create supervised sequences
X_train, y_train = split_sequence(train, N_STEPS_IN, N_STEPS_OUT)

# Define Encoder–Decoder model
encoder_inputs = Input(shape=(N_STEPS_IN, 1))
encoder_lstm = LSTM(100, activation='relu')(encoder_inputs)
repeat_vector = RepeatVector(N_STEPS_OUT)(encoder_lstm)
decoder_lstm = LSTM(100, activation='relu', return_sequences=True)(repeat_vector)
decoder_outputs = TimeDistributed(Dense(1))(decoder_lstm)

model = Model(inputs=encoder_inputs, outputs=decoder_outputs)
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, ...)
```

---

## ⚙️ How to Run

1️⃣ **Install Dependencies**

```bash
pip install numpy pandas matplotlib seaborn scikit-learn keras joblib
```

2️⃣ **Run the Script**

```bash
python encoder_decoder_lstm.py
```

3️⃣ **View Results**

* Inspect **training logs**, **RMSE scores**, and **plots**.
* Saved files:

  * `encoder_decoder_lstm.h5` — **trained model**.
  * `scaler.save` — **saved scaler** for future inference.

---

## 🔮 Future Enhancements

* **Attention Mechanism:** Improve context learning for long sequences.
* **Bidirectional LSTM:** Enhance understanding of sequential patterns.
* **Walk-Forward Validation:** Implement live forecasting.
* **Streamlit App:** Deploy as an interactive forecasting dashboard.

---

## ✅ Conclusion

This **Encoder–Decoder LSTM** pipeline is a solid baseline for **multi-step time series forecasting**.
It’s fully extensible for other **univariate** or even **multivariate time series tasks**.

## Happy Forecasting! 🚀📈
