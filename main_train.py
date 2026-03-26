import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Input
import matplotlib.pyplot as plt
import joblib

# Constants
DATA_PATH = "dam_discharge_data"
SEQUENCE_LENGTH = 10
EPOCHS = 15

model_path = os.path.join(DATA_PATH, "model.keras")
scaler_path = os.path.join(DATA_PATH, "scaler.pkl")
excel_file_path = "dam_discharge_data.xlsx"


# Ensure data directory exists
os.makedirs(DATA_PATH, exist_ok=True)

def train_model_from_excel():
    # Load Excel file
    if not os.path.exists(excel_file_path):
        print(f"❌ Excel file not found at {excel_file_path}")
        return

    df = pd.read_excel(excel_file_path)

    # Check that 'Q' column exists and is valid
    if "Q" not in df.columns:
        print("❌ 'Q' column not found in Excel file.")
        return

    df = df.dropna(subset=["Q"])
    if not np.issubdtype(df["Q"].dtype, np.number):
        print("❌ 'Q' column must contain numeric values only.")
        return

    q_values = df["Q"].values.reshape(-1, 1)

    if len(q_values) < SEQUENCE_LENGTH:
        print(f"❌ Not enough data. At least {SEQUENCE_LENGTH+1} Q values are required.")
        return

    # Normalize the data
    scaler = MinMaxScaler()
    scaled_q = scaler.fit_transform(q_values)

    # Prepare sequences
    X = []
    for i in range(SEQUENCE_LENGTH, len(scaled_q)):
        X.append(scaled_q[i - SEQUENCE_LENGTH:i])
    X = np.array(X).reshape((-1, SEQUENCE_LENGTH, 1))

    print(f"✅ Prepared {len(X)} training sequences.")

    # Define LSTM Autoencoder
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, 1)),
        LSTM(64, activation="relu", return_sequences=False),
        RepeatVector(SEQUENCE_LENGTH),
        LSTM(64, activation="relu", return_sequences=True),
        TimeDistributed(Dense(1))
    ])
    model.compile(optimizer="adam", loss="mae")

    # Train the model
    print("🚀 Training started...")
    history = model.fit(X, X, epochs=EPOCHS, batch_size=16, verbose=1)

    # Save model and scaler
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    print(f"✅ Model saved to {model_path}")
    print(f"✅ Scaler saved to {scaler_path}")

    # Plot training loss
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss", color="blue", marker='o')
    plt.title("LSTM Autoencoder Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MAE)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt_path = os.path.join(DATA_PATH, "training_loss.png")
    plt.savefig(plt_path)
    plt.show()
    print(f"📈 Loss plot saved to {plt_path}")

def main():
    train_model_from_excel()

if __name__ == "__main__":
    main()
