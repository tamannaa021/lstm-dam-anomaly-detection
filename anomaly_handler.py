import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt

# Constants
DATA_PATH = "dam_discharge_data"
MODEL_FILE = os.path.join(DATA_PATH, "model.keras")
SCALER_FILE = os.path.join(DATA_PATH, "scaler.pkl")
LOG_FILE = os.path.join(DATA_PATH, "logged_values.csv")
ANOMALY_LOG = os.path.join(DATA_PATH, "anomalies.csv")
SEQUENCE_LENGTH = 10
ANOMALY_THRESHOLD = 0.15  # 15%

# Load model and scaler
def load_model_and_scaler():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        print("❌ Trained model or scaler not found. Please train the model first.")
        exit()
    model = load_model(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    print("✅ Anomaly detection system ready.\n")
    return model, scaler

# Predict next value
def predict_next(model, scaler, recent_qs):
    scaled_input = scaler.transform(np.array(recent_qs).reshape(-1, 1))
    sequence = np.array([scaled_input]).reshape((1, SEQUENCE_LENGTH, 1))
    predicted_scaled = model.predict(sequence, verbose=0)[0][-1]
    predicted_q = scaler.inverse_transform(predicted_scaled.reshape(-1, 1))[0][0]
    return predicted_q

# Log value
def log_value(q_value):
    mode = 'a' if os.path.exists(LOG_FILE) else 'w'
    df = pd.DataFrame([[q_value]], columns=["Q"])
    df.to_csv(LOG_FILE, mode=mode, index=False, header=not os.path.exists(LOG_FILE))

# Log anomaly
def log_anomaly(actual_q, predicted_q, deviation):
    mode = 'a' if os.path.exists(ANOMALY_LOG) else 'w'
    df = pd.DataFrame([[actual_q, predicted_q, deviation * 100]], columns=["Input Q", "Predicted Q", "Deviation %"])
    df.to_csv(ANOMALY_LOG, mode=mode, index=False, header=not os.path.exists(ANOMALY_LOG))

# Plot predictions
def plot_predictions(history):
    inputs = [h[0] for h in history]
    preds = [h[1] for h in history]

    plt.figure(figsize=(10, 5))
    plt.plot(inputs, label="Input Q", marker='o')
    plt.plot(preds, label="Predicted Q", marker='x')
    plt.title("Anomaly Detection Trend")
    plt.xlabel("Timestep")
    plt.ylabel("Q Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    model, scaler = load_model_and_scaler()

    # Step 1: Accept 10 new values to form initial input
    print("🔰 Please enter 10 Q values to initialize recent trend:")
    recent_qs = []
    for i in range(SEQUENCE_LENGTH):
        while True:
            try:
                val = float(input(f"Q value {i+1}/10: "))
                recent_qs.append(val)
                log_value(val)
                break
            except ValueError:
                print("❌ Please enter a valid number.")

    print("\n🚦 Initialization complete. Anomaly detection has started.\n")
    prediction_history = []
    stable_buffer = []

    # Step 2: Start anomaly detection loop
    while True:
        user_input = input("🔢 Enter new Q value (or type -1 to exit, 'v' to view graph): ").strip()

        if user_input == "-1":
            print("👋 Exiting anomaly detection.")
            break
        elif user_input.lower() == "v":
            plot_predictions(prediction_history)
            continue
        try:
            q_input = float(user_input)
        except ValueError:
            print("❌ Invalid input. Please enter a number, -1, or 'v'.")
            continue

        # Hard anomaly limits
        if q_input > 11000 or q_input < 400:
            predicted_q = predict_next(model, scaler, recent_qs)
            deviation = abs(predicted_q - q_input) / q_input
            print(f"\n📊 Predicted Q: {predicted_q:.2f}, Input Q: {q_input:.2f}, Deviation: {deviation*100:.2f}%")
            print("🚨 Anomaly detected!")
            log_anomaly(q_input, predicted_q, deviation)
            prediction_history.append((q_input, predicted_q))
            log_value(q_input)
            continue

        predicted_q = predict_next(model, scaler, recent_qs)
        deviation = abs(predicted_q - q_input) / q_input

        print(f"\n📊 Predicted Q: {predicted_q:.2f}, Input Q: {q_input:.2f}, Deviation: {deviation*100:.2f}%")
        if deviation > ANOMALY_THRESHOLD:
            print("🚨 Anomaly detected!")
            log_anomaly(q_input, predicted_q, deviation)
        else:
            print("✅ Q value is within normal range.")

        prediction_history.append((q_input, predicted_q))
        log_value(q_input)

        # Maintain a buffer of the last 5 inputs
        stable_buffer.append(q_input)
        stable_buffer = stable_buffer[-5:]

        # Condition to update trend
        if deviation <= ANOMALY_THRESHOLD:
            recent_qs.append(q_input)
            if len(recent_qs) > SEQUENCE_LENGTH:
                recent_qs.pop(0)
        else:
            trend_is_consistent = all(
                abs(stable_buffer[i] - stable_buffer[i - 1]) < 0.15 * stable_buffer[i - 1]
                for i in range(1, len(stable_buffer))
            )
            if len(stable_buffer) == 5 and trend_is_consistent:
                print("📈 Detected stable new trend. Updating memory.")
                recent_qs.append(q_input)
                if len(recent_qs) > SEQUENCE_LENGTH:
                    recent_qs.pop(0)
            else:
                print("⛔ Anomaly ignored for trend learning.")

if __name__ == "__main__":
    main()