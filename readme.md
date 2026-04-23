# Fish Tank Controller

An automated aquarium management system that combines **Bluetooth Low Energy (BLE) light control**, a **Python web server dashboard**, and a **machine learning model** trained on manually logged water parameter data to predict future tank conditions.

## How It Works

```
Manual parameter readings
        │
        ▼
   Web UI (main.html)
        │
        ▼
   webServer.py ──────────────► data.csv
        │                           │
        │                           ▼
        │                  FishTankAITrain.py
        │                           │
        │                           ▼
        │                   Prediction model
        │                           │
        ◄───────────────────────────┘
        │
        ▼
  Display results on frontend
        │
        ▼
     BLE.py ──► Fluval Plant 3.0 Light
```

The user logs water parameters (e.g. pH, temperature, nitrates) manually through the web UI. These readings are stored in `data.csv` and fed into a machine learning model that learns trends over time and predicts future water conditions. Results are displayed back on the dashboard, and the system controls the **Fluval Plant 3.0 LED light** via BLE based on the current state or schedule.

## Features

- **Reverse-engineered BLE light control** — communicates with a Fluval Plant 3.0 aquarium light over BLE using a protocol reverse-engineered through decompilation of the official Fluval Android app; implemented in `BLE.py`
- **Web dashboard** — data entry form and results display served via `webServer.py` and `main.html`
- **ML predictor** — `FishTankAITrain.py` trains a TensorFlow model on historical readings in `data.csv` to forecast water parameter trends
- **Persistent state** — configuration and model state saved to `save.json`
- **Startup orchestration** — `startScripts.py` initializes all services in the correct order

## Tech Stack

| Component | Technology |
|-----------|------------|
| BLE light control | Python, Bluetooth Low Energy (BLE) |
| Web server | Python (HTTP server) |
| Frontend | HTML, CSS, JavaScript |
| ML model | Python, TensorFlow |
| Data storage | CSV, JSON |

## Setup & Usage

### Requirements

- Raspberry Pi or any machine with Bluetooth support
- Python 3.8+
- Fluval Plant 3.0 aquarium light
- Required Python packages:

```bash
pip install bleak numpy tensorflow
```

### Running the System

```bash
python startScripts.py
```

Then open `main.html` in a browser (or navigate to the host machine's local IP) to access the dashboard.

### Training the ML Model

```bash
python FishTankAITrain.py
```

Reads `data.csv` (historical manual water parameter entries) and trains a predictive model. Retrain periodically as more readings accumulate for better accuracy.

## Project Background

Built to reduce the manual overhead of aquarium maintenance. Rather than relying on fixed schedules or reactive adjustments, the system learns trends from logged water parameters and surfaces predictions before conditions drift outside safe ranges.

The BLE integration required reverse-engineering the Fluval Plant 3.0 communication protocol by decompiling the official Fluval Android APK — no public API or documentation exists for this device. The current implementation supports core lighting control; full feature parity with the official app and a more robust BLE connection are planned next.

**Planned future work:**
- More stable and complete Fluval BLE connection (full feature parity with the official app)
- Integration of automated sensor hardware (pH probe, temperature sensor, etc.) to replace manual data entry with real-time continuous readings

## Skills Demonstrated

- BLE protocol reverse engineering (Android APK decompilation)
- IoT hardware control via undocumented BLE API
- Python web server and REST-style backend development
- Machine learning on time-series tabular data with TensorFlow
- Full-stack development (Python backend + HTML/JS frontend)
- System design: multi-process startup orchestration and persistent state management
