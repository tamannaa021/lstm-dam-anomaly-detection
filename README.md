# 🌊 LSTM-Based Anomaly Detection in Dam Discharge Data
### Real-time intelligent monitoring system built during industrial internship at a national hydroelectric power corporation

---

## 📌 Overview

Dams are critical national infrastructure — even small irregularities in discharge can cause flooding, agricultural damage, or worse. Traditional fixed-threshold systems miss subtle deviations and generate too many false alarms.

This project builds a **hybrid anomaly detection framework** that combines an **LSTM Autoencoder** with **domain-specific safety rules** to detect both sudden spikes and gradual trend shifts in real-time dam discharge data — reducing false positives by **85%** compared to conventional thresholding.

> Built during a one-month industrial internship at a major national hydroelectric power corporation in India | July 2025

---

## 🎯 What It Does

| Capability | Details |
|---|---|
| **Predicts** expected discharge values using the last 10 readings | LSTM Autoencoder trained on historical Q data |
| **Flags anomalies** when deviation > 15% from prediction | Real-time, per-reading detection |
| **Enforces safety hard limits** | Instantly flags Q < 400 m³/s or Q > 11,000 m³/s |
| **Adapts to new trends** | Updates memory only after 5 consecutive stable readings |
| **Logs everything** | CSV audit trail of all readings and anomalies |
| **Visualizes** predictions vs actuals | Matplotlib trend plots on demand |

---

## 🧠 Model Architecture
```
Input (10 timesteps)
       ↓
  LSTM Encoder (64 units, ReLU)
       ↓
  Latent Vector (compressed representation)
       ↓
  RepeatVector
       ↓
  LSTM Decoder (64 units, ReLU)
       ↓
  TimeDistributed Dense → Reconstructed Sequence
       ↓
  MAE Loss → Anomaly Score
```

The model is trained as an **autoencoder** — it learns what "normal" discharge looks like. At inference time, high reconstruction error = anomaly.

---

## 📊 Results

### Test Cases

| Scenario | Input Q | Predicted Q | Deviation | Result |
|---|---|---|---|---|
| Normal operation | 1,100 | 1,201 | 9.18% | ✅ Normal |
| Sudden spike | 2,300 | 1,136 | 50.59% | 🚨 Anomaly |
| Gradual trend shift | 2100 → 2200 (5 values) | — | Stable band | 📈 Trend adapted |

### Key Metrics
- **False positive reduction**: ~85% vs fixed-threshold baseline
- **Training**: 15 epochs, MAE loss, converges cleanly
- **Detection latency**: Per-reading (real-time loop)

---

## 🗂️ Project Structure
```
lstm-dam-anomaly-detection/
│
├── main_train.py           # Data loading, preprocessing, model training
├── anomaly_handler.py      # Real-time detection loop, logging, visualization
├── sample_data.xlsx        # Sample dataset for demonstration (replace with your own)
│
├── dam_discharge_data/
│   ├── model.keras         # Saved trained model (generated after training)
│   ├── scaler.pkl          # Saved MinMaxScaler (generated after training)
│   ├── logged_values.csv   # All discharge inputs (generated at runtime)
│   ├── anomalies.csv       # Anomaly log with deviation % (generated at runtime)
│   └── training_loss.png   # Loss curve from training (generated after training)
│
└── README.md
```

> **Note**: `model.keras`, `scaler.pkl`, and log files are generated automatically when you run the scripts. Only `sample_data.xlsx` is provided for demonstration — replace it with your own discharge dataset containing a column named `Q`.

---

## ⚙️ How to Run

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn tensorflow joblib matplotlib openpyxl
```

### 2. Prepare your data

Your Excel file must have a column named `Q` containing numeric discharge values (at least 11 rows).

A `sample_data.xlsx` with realistic demo values is included to test the pipeline immediately.

### 3. Train the model
```bash
python main_train.py
```

This will:
- Load and preprocess discharge data
- Train the LSTM Autoencoder for 15 epochs
- Save `model.keras` and `scaler.pkl` to `dam_discharge_data/`
- Save a training loss plot at `dam_discharge_data/training_loss.png`

### 4. Run real-time anomaly detection
```bash
python anomaly_handler.py
```

- Enter 10 initial Q values to prime the model
- Feed new readings one by one
- Type `v` to view the prediction vs actual graph
- Type `-1` to exit

---

## 🔍 How Anomaly Detection Works
```
New Q value received
        ↓
Hard limit check (Q < 400 or Q > 11,000)?
   YES → 🚨 Flag immediately
   NO  ↓
Predict expected Q from last 10 values
        ↓
Compute deviation = |predicted - actual| / actual
        ↓
Deviation > 15%?
   YES → 🚨 Log anomaly, check stable buffer
   NO  → ✅ Normal, update trend memory
```

**Adaptive trend learning**: The system maintains a 5-reading stable buffer. If 5 consecutive readings show consistent internal change, it recognizes this as a genuine trend shift and updates its memory — avoiding repeated false alarms during gradual operational changes.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| TensorFlow / Keras | LSTM Autoencoder |
| Scikit-learn | MinMaxScaler preprocessing |
| Pandas / NumPy | Data handling |
| Matplotlib | Visualization |
| Joblib | Scaler serialization |

---

## ⚠️ Limitations

- Requires manual entry of 10 initial values before detection begins
- Currently uses only discharge (Q); adding rainfall, reservoir level, and temperature could improve accuracy
- Designed for offline/semi-real-time use; IoT/SCADA integration would enable fully automated deployment


## 👩‍💻 Author

**Tamanna Khetrapal**
B.Tech in AI & Data Science | VIPS-TC, GGSIPU
[LinkedIn](https://www.linkedin.com/in/tamanna-khetrapal) | tamannakhetrapal21@gmail.com
