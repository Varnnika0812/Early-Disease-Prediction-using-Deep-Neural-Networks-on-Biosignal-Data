import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, flash, jsonify, redirect, render_template_string, request, session, url_for
import joblib
from scipy.fft import fft
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from werkzeug.security import check_password_hash, generate_password_hash


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "mitbih_train.csv"
DATABASE_PATH = BASE_DIR / "users.db"
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "ecg_cnn.keras"
SCALER_PATH = ARTIFACT_DIR / "scaler.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.joblib"
CONFUSION_PATH = ARTIFACT_DIR / "confusion_matrix.npy"

CLASS_LABELS = {
    0: "Normal beat",
    1: "Supraventricular premature beat",
    2: "Premature ventricular contraction",
    3: "Fusion of ventricular and normal beat",
    4: "Unclassifiable beat",
}

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "early-disease-prediction-demo-key")

model = None
scaler = None
history_summary = None
confusion_mat = None
training_error = None
training_in_progress = False
model_ready = False
model_loading = False
model_lock = threading.Lock()
live_training_metrics = {
    "accuracy": [],
    "val_accuracy": [],
    "loss": [],
    "val_loss": [],
}
upload_eval_jobs = {}
upload_eval_lock = threading.Lock()


def init_db():
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL
            )
            """
        )
        existing = conn.execute("SELECT id FROM users WHERE username = ?", ("admin",)).fetchone()
        if existing is None:
            conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                ("admin", generate_password_hash("admin123")),
            )


init_db()


def load_model_artifacts():
    global model, scaler, history_summary, confusion_mat, model_ready, model_loading, live_training_metrics

    if not (MODEL_PATH.exists() and SCALER_PATH.exists() and METRICS_PATH.exists() and CONFUSION_PATH.exists()):
        return False

    try:
        from tensorflow.keras.models import load_model

        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        history_summary = joblib.load(METRICS_PATH)
        confusion_mat = np.load(CONFUSION_PATH)
        live_training_metrics = {
            "accuracy": list(history_summary.get("accuracy_series", [])),
            "val_accuracy": list(history_summary.get("val_accuracy_series", [])),
            "loss": list(history_summary.get("loss_series", [])),
            "val_loss": list(history_summary.get("val_loss_series", [])),
        }
        model_ready = True
        model_loading = False
        return True
    except Exception:
        model_ready = False
        model_loading = False
        return False


def save_model_artifacts(local_model, local_scaler, local_history_summary, local_confusion_mat):
    ARTIFACT_DIR.mkdir(exist_ok=True)
    local_model.save(MODEL_PATH)
    joblib.dump(local_scaler, SCALER_PATH)
    joblib.dump(local_history_summary, METRICS_PATH)
    np.save(CONFUSION_PATH, local_confusion_mat)


def current_user():
    return session.get("username")


def start_model_load_thread():
    global model_loading

    with model_lock:
        if model_ready or model_loading or training_in_progress:
            return
        if not (MODEL_PATH.exists() and SCALER_PATH.exists() and METRICS_PATH.exists() and CONFUSION_PATH.exists()):
            return
        model_loading = True

    thread = threading.Thread(target=load_model_artifacts, daemon=True)
    thread.start()


def history_has_series():
    if not history_summary:
        return False

    required_keys = ("accuracy_series", "val_accuracy_series", "loss_series", "val_loss_series")
    return all(isinstance(history_summary.get(key), list) and len(history_summary.get(key)) > 1 for key in required_keys)


def reset_live_metrics():
    global live_training_metrics

    live_training_metrics = {
        "accuracy": [],
        "val_accuracy": [],
        "loss": [],
        "val_loss": [],
    }


def current_metric_series():
    if any(live_training_metrics.values()):
        return live_training_metrics

    if history_summary:
        return {
            "accuracy": list(history_summary.get("accuracy_series", [])),
            "val_accuracy": list(history_summary.get("val_accuracy_series", [])),
            "loss": list(history_summary.get("loss_series", [])),
            "val_loss": list(history_summary.get("val_loss_series", [])),
        }

    return {
        "accuracy": [],
        "val_accuracy": [],
        "loss": [],
        "val_loss": [],
    }


def evaluate_uploaded_dataset(job_id, features, labels):
    try:
        chunk_count = min(30, max(1, len(labels)))
        chunks = [indexes for indexes in np.array_split(np.arange(len(labels)), chunk_count) if len(indexes) > 0]
        total_correct = 0
        total_loss = 0.0
        total_seen = 0

        for indexes in chunks:
            scaled = scaler.transform(features[indexes]).reshape(len(indexes), 187, 1)
            probabilities = model.predict(scaled, verbose=0)
            predicted = np.argmax(probabilities, axis=1)
            true_labels = labels[indexes]
            clipped = np.clip(probabilities[np.arange(len(true_labels)), true_labels], 1e-9, 1.0)
            losses = -np.log(clipped)

            total_correct += int(np.sum(predicted == true_labels))
            total_loss += float(np.sum(losses))
            total_seen += len(indexes)

            with upload_eval_lock:
                job = upload_eval_jobs[job_id]
                job["sample_axis"].append(int(total_seen))
                job["accuracy"].append(round(total_correct / total_seen, 5))
                job["loss"].append(round(total_loss / total_seen, 5))
                job["processed"] = int(total_seen)

            time.sleep(0.15)

        with upload_eval_lock:
            job = upload_eval_jobs[job_id]
            job["complete"] = True
            job["message"] = "Uploaded dataset evaluation complete."
    except Exception as exc:
        with upload_eval_lock:
            job = upload_eval_jobs.get(job_id)
            if job is not None:
                job["complete"] = True
                job["error"] = str(exc)


def start_uploaded_evaluation(features, labels):
    job_id = uuid.uuid4().hex
    with upload_eval_lock:
        upload_eval_jobs[job_id] = {
            "sample_count": int(len(labels)),
            "processed": 0,
            "accuracy": [],
            "loss": [],
            "sample_axis": [],
            "complete": False,
            "error": None,
            "message": "Evaluating uploaded dataset...",
        }

    thread = threading.Thread(target=evaluate_uploaded_dataset, args=(job_id, features, labels), daemon=True)
    thread.start()
    return job_id


def train_model(force=False):
    global model, scaler, history_summary, confusion_mat, training_error, training_in_progress, model_ready, live_training_metrics

    with model_lock:
        if training_in_progress:
            return
        if model_ready and not force:
            return
        training_in_progress = True
        training_error = None
        if force or not history_has_series():
            reset_live_metrics()

    try:
        from tensorflow.keras.callbacks import Callback
        from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D
        from tensorflow.keras.models import Sequential

        class LiveMetricsCallback(Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                with model_lock:
                    live_training_metrics["accuracy"].append(float(logs.get("accuracy", 0)))
                    live_training_metrics["val_accuracy"].append(float(logs.get("val_accuracy", 0)))
                    live_training_metrics["loss"].append(float(logs.get("loss", 0)))
                    live_training_metrics["val_loss"].append(float(logs.get("val_loss", 0)))

        data = pd.read_csv(DATASET_PATH, header=None)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        local_scaler = StandardScaler()
        X = local_scaler.fit_transform(X)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        local_model = Sequential(
            [
                Conv1D(32, 3, activation="relu", input_shape=(187, 1)),
                MaxPooling1D(2),
                Conv1D(64, 3, activation="relu"),
                MaxPooling1D(2),
                Flatten(),
                Dense(64, activation="relu"),
                Dropout(0.5),
                Dense(5, activation="softmax"),
            ]
        )
        local_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        local_history = local_model.fit(
            X_train,
            y_train,
            epochs=5,
            validation_data=(X_test, y_test),
            callbacks=[LiveMetricsCallback()],
            verbose=0,
        )

        y_pred = np.argmax(local_model.predict(X_test, verbose=0), axis=1)

        local_confusion_mat = confusion_matrix(y_test, y_pred)
        local_history_summary = {
            "accuracy": float(local_history.history["accuracy"][-1]),
            "val_accuracy": float(local_history.history["val_accuracy"][-1]),
            "loss": float(local_history.history["loss"][-1]),
            "val_loss": float(local_history.history["val_loss"][-1]),
            "accuracy_series": [float(value) for value in local_history.history["accuracy"]],
            "val_accuracy_series": [float(value) for value in local_history.history["val_accuracy"]],
            "loss_series": [float(value) for value in local_history.history["loss"]],
            "val_loss_series": [float(value) for value in local_history.history["val_loss"]],
        }
        save_model_artifacts(local_model, local_scaler, local_history_summary, local_confusion_mat)

        with model_lock:
            model = local_model
            scaler = local_scaler
            confusion_mat = local_confusion_mat
            history_summary = local_history_summary
            model_ready = True
            training_in_progress = False
    except Exception as exc:
        with model_lock:
            training_error = str(exc)
            training_in_progress = False
            model_ready = False


def start_training_thread(force=False):
    thread = threading.Thread(target=train_model, kwargs={"force": force}, daemon=True)
    thread.start()


def wait_for_model_ready(timeout_seconds=900):
    global training_error

    if model_ready:
        return True

    start_model_load_thread()

    if not model_loading and not training_in_progress:
        train_model()
        return model_ready

    started = time.time()
    while time.time() - started < timeout_seconds:
        if model_ready:
            return True
        if training_error:
            return False
        time.sleep(2)

    return False


def prediction_from_csv(file_storage):
    if not wait_for_model_ready():
        raise RuntimeError("Model training has not completed yet. Refresh the page in a few minutes and try again.")

    data = pd.read_csv(file_storage, header=None)
    if data.empty:
        raise ValueError("Uploaded CSV is empty.")

    evaluation_job_id = None
    if data.shape[1] == 188:
        features = data.iloc[:, :-1].apply(pd.to_numeric, errors="raise").values.astype(float)
        labels = data.iloc[:, -1].astype(int).values
        signal = features[0]
    elif data.shape[1] == 187:
        features = data.apply(pd.to_numeric, errors="raise").values.astype(float)
        labels = None
        signal = features[0]
    else:
        raise ValueError(f"Expected 187 ECG features, or 187 features plus label. Found {data.shape[1]} columns.")

    signal_scaled = scaler.transform([signal]).reshape(1, 187, 1)
    probabilities = model.predict(signal_scaled, verbose=0)[0]
    class_id = int(np.argmax(probabilities))
    frequency = np.abs(fft(signal))

    if labels is not None:
        evaluation_job_id = start_uploaded_evaluation(features, labels)

    return {
        "class_id": class_id,
        "label": CLASS_LABELS[class_id],
        "confidence": float(np.max(probabilities)),
        "probabilities": [(CLASS_LABELS[index], float(value)) for index, value in enumerate(probabilities)],
        "signal_preview": [round(float(value), 5) for value in signal[:80]],
        "frequency_preview": [round(float(value), 5) for value in frequency[:80]],
        "evaluation_job_id": evaluation_job_id,
        "has_labels": labels is not None,
    }


PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Early Disease Prediction System</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {
      --page: #edf3fb;
      --panel: #ffffff;
      --ink: #17324d;
      --muted: #59708a;
      --navy: #12304a;
      --blue: #1e88e5;
      --green: #00a98f;
      --border: #d8e4f2;
      --warn: #b86b00;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      background: var(--page);
      color: var(--ink);
    }
    .shell { max-width: 1420px; margin: 0 auto; padding: 18px; }
    .topbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      margin-bottom: 18px;
    }
    .brand { font-size: 22px; font-weight: 800; }
    .nav { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
    .nav a, .button {
      border: 0;
      background: var(--blue);
      color: #fff;
      padding: 10px 14px;
      text-decoration: none;
      font-weight: 700;
      font-size: 14px;
      cursor: pointer;
      display: inline-block;
    }
    .nav a.secondary, .button.secondary { background: #fff; color: var(--ink); border: 1px solid var(--border); }
    .hero {
      display: grid;
      grid-template-columns: minmax(0, 1.5fr) minmax(320px, 0.9fr);
      min-height: 540px;
      gap: 18px;
    }
    .hero-left {
      background: var(--navy);
      color: #fff;
      padding: 52px 46px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }
    h1 { font-size: 42px; line-height: 1.28; margin: 0 0 22px; letter-spacing: 0; }
    .hero-left p { color: #c8dbef; font-size: 18px; line-height: 1.5; max-width: 680px; }
    .workflow { background: #173a59; padding: 18px 22px; font-weight: 800; }
    .card, .panel {
      background: var(--panel);
      border: 1px solid var(--border);
    }
    .card { padding: 38px; }
    .card h2, .panel h2 { margin: 0 0 14px; font-size: 28px; }
    label { display: block; font-weight: 800; margin: 18px 0 8px; }
    input[type="text"], input[type="password"], input[type="file"] {
      width: 100%;
      padding: 14px;
      border: 1px solid var(--border);
      background: #f6faff;
      color: var(--ink);
      font-size: 16px;
    }
    .muted { color: var(--muted); line-height: 1.55; }
    .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-top: 18px; }
    .panel { padding: 24px; margin-top: 18px; }
    .panel h3 { margin: 0 0 10px; }
    .steps { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
    .step { background: #f8fbff; border: 1px solid var(--border); padding: 18px; min-height: 126px; }
    .badge { color: var(--green); font-weight: 900; margin-bottom: 8px; }
    .flash { background: #fff8f0; border: 1px solid #ffe0ba; color: var(--warn); padding: 12px 14px; margin-bottom: 14px; }
    .result { background: #f5fbf8; border-color: #caecd9; }
    .matrix { white-space: pre-wrap; font-family: Consolas, monospace; background: #f6faff; padding: 14px; overflow: auto; }
    .dashboard {
      width: 100%;
      overflow: hidden;
    }
    .dashboard-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 2px solid #9aaabc;
      padding: 12px 0 18px;
      margin-bottom: 14px;
    }
    .dashboard-head h1 { font-size: 32px; margin: 0 0 10px; }
    .dashboard-layout {
      display: grid;
      grid-template-columns: 240px minmax(0, 1fr);
      gap: 14px;
      align-items: start;
      width: 100%;
    }
    .side-stack { display: grid; gap: 14px; align-content: start; }
    .side-card, .output-card { background: #fff; border: 1px solid var(--border); padding: 16px; min-width: 0; }
    .side-card h2, .output-card h2 { font-size: 20px; margin: 0 0 8px; }
    .side-card .button { width: 100%; text-align: center; margin: 8px 0; }
    .status-box { background: #f5fbf8; border: 1px solid #caecd9; padding: 14px; margin-top: 12px; }
    .stage-box { background: #eef6ff; border: 1px solid #cfe2fb; padding: 14px; margin-top: 12px; }
    .outputs { background: #fff; border: 1px solid var(--border); padding: 16px; min-width: 0; }
    .outputs h2 { font-size: 22px; margin: 0 0 6px; }
    .result-row { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 10px; margin: 12px 0; }
    .prediction-box { background: #f8f7ff; border: 1px solid #ddd8ff; padding: 16px; min-height: 118px; min-width: 0; overflow-wrap: anywhere; }
    .confidence-box { background: #fff8f0; border: 1px solid #ffe0ba; padding: 16px; min-height: 118px; min-width: 0; overflow-wrap: anywhere; }
    .confidence-box p { margin: 5px 0; font-family: Consolas, monospace; font-size: 13px; }
    .charts-grid { display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(300px, .85fr); gap: 12px; align-items: start; }
    .chart-panel { background: #fff; border: 1px solid var(--border); padding: 14px; margin-bottom: 12px; min-width: 0; overflow: hidden; }
    .chart-panel h3 { margin: 0 0 10px; font-size: 17px; }
    .chart-box { position: relative; height: 230px; width: 100%; }
    .chart-box.small { height: 250px; }
    canvas { width: 100% !important; height: 100% !important; display: block; }
    .matrix { max-height: 180px; font-size: 13px; }
    input[type="file"] { font-size: 13px; padding: 10px; }
    @media (max-width: 860px) {
      .hero, .grid, .steps, .dashboard-layout, .result-row, .charts-grid { grid-template-columns: 1fr; }
      h1 { font-size: 32px; }
      .hero-left, .card { padding: 28px; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header class="topbar">
      <div class="brand">Early Disease Prediction System</div>
      <nav class="nav">
        <a class="secondary" href="#overview">Overview</a>
        <a class="secondary" href="#process">Process</a>
        <a class="secondary" href="#predictor">Predictor</a>
        {% if user %}
          <span class="muted">Signed in as {{ user }}</span>
          <a href="{{ url_for('logout') }}">Logout</a>
        {% endif %}
      </nav>
    </header>

    {% with messages = get_flashed_messages() %}
      {% if messages %}
        {% for message in messages %}
          <div class="flash">{{ message }}</div>
        {% endfor %}
      {% endif %}
    {% endwith %}

    {% if user %}
    <section id="predictor" class="dashboard">
      <div class="dashboard-head">
        <div>
          <h1>Early Disease Prediction System</h1>
          <p class="muted">Early Disease Prediction using Deep Neural Networks on Biosignal Data</p>
        </div>
        <a class="button secondary" href="{{ url_for('logout') }}">Logout</a>
      </div>

      <div class="dashboard-layout">
        <aside class="side-stack">
          <div class="side-card">
            <h2>Actions</h2>
            <p class="muted">Run prediction and inspect embedded training outputs.</p>
            <form method="post" action="{{ url_for('predict') }}" enctype="multipart/form-data">
              <label>Dataset</label>
              <input type="file" name="dataset" accept=".csv" required>
              <p><button class="button" type="submit">Upload the Dataset</button></p>
            </form>
            <form method="post" action="{{ url_for('refresh_metrics') }}">
              <button class="button secondary" type="submit">Refresh Metrics</button>
            </form>
            <p class="muted">Use refresh after training if you want to redraw metrics manually.</p>
          </div>

          <div class="side-card">
            <h2>Model Overview</h2>
            <p class="muted">Status, dataset path, and current output.</p>
            <div class="status-box">
              <strong>System Status</strong>
              <p>{% if model_ready %}Upload the dataset{% elif model_loading %}Loading model{% elif training_in_progress %}Training model{% elif training_error %}Training failed{% else %}Preparing model{% endif %}</p>
              <p class="muted">Time left: {% if training_in_progress or model_loading %}processing{% else %}00:00{% endif %}</p>
            </div>
            <div class="stage-box">
              <strong>Current Stage</strong>
              {% if history %}
              <p class="muted">Final accuracy={{ "%.4f"|format(history.accuracy) }} | val_accuracy={{ "%.4f"|format(history.val_accuracy) }} | loss={{ "%.4f"|format(history.loss) }} | val_loss={{ "%.4f"|format(history.val_loss) }}</p>
              {% else %}
              <p class="muted">Training metrics are loading.</p>
              {% endif %}
            </div>
            <p class="muted">Selected File: {% if prediction %}uploaded CSV{% else %}No uploaded file yet{% endif %}</p>
          </div>
        </aside>

        <section class="outputs">
          <h2>Outputs And Results</h2>
          <p class="muted">Prediction result, probabilities, confusion matrix, and uploaded signal analysis</p>
          <div class="result-row">
            <div class="prediction-box">
              <h3>Prediction Result</h3>
              {% if prediction %}
              <p><strong>Predicted Class: {{ prediction.label }} ({{ prediction.class_id }}) | Confidence: {{ "%.2f"|format(prediction.confidence * 100) }}%</strong></p>
              {% else %}
              <p class="muted"><strong>Prediction will appear here after dataset upload.</strong></p>
              {% endif %}
            </div>
            <div class="confidence-box">
              <h3>Prediction Confidence</h3>
              {% if prediction %}
                {% for label, value in prediction.probabilities %}
                  <p class="muted">{{ label }}: {{ "%.4f"|format(value) }}</p>
                {% endfor %}
              {% else %}
                <p class="muted">Class probabilities will appear here after prediction.</p>
              {% endif %}
            </div>
          </div>

          <div class="charts-grid">
            <div>
              <div class="chart-panel">
                <h3>ECG Signal (Time Domain)</h3>
                <div class="chart-box"><canvas id="signalChart"></canvas></div>
              </div>
              <div class="chart-panel">
                <h3>Frequency Domain</h3>
                <div class="chart-box"><canvas id="freqChart"></canvas></div>
              </div>
            </div>
            <div>
              <div class="chart-panel">
                <h3>Confusion Matrix</h3>
                {% if matrix %}<div class="matrix">{{ matrix }}</div>{% else %}<p class="muted">Confusion matrix will appear after training.</p>{% endif %}
              </div>
              <div class="chart-panel">
                <h3>Training Performance</h3>
                <p class="muted" id="trainingStatus">Loading training metrics...</p>
                <div class="chart-box small"><canvas id="trainingChart"></canvas></div>
              </div>
            </div>
          </div>
        </section>
      </div>
    </section>
    {% endif %}

    {% if not user %}
    <section class="hero">
      <div class="hero-left">
        <div>
          <h1>Early Disease Prediction System</h1>
          <p>Secure biosignal-based screening dashboard powered by a convolutional neural network. The system uses ECG signal samples from the MIT-BIH dataset, trains a 1D CNN, and predicts the arrhythmia class from uploaded biosignal CSV data.</p>
        </div>
        <div class="workflow">Workflow: login -> train model -> upload dataset -> review results on one screen</div>
      </div>

      <div class="card">
        {% if user %}
          <h2>Model Access</h2>
          <p class="muted">The prediction workspace is available below. Upload a compatible CSV after the model status shows ready.</p>
          <a class="button" href="#predictor">Go To Predictor</a>
        {% else %}
          <h2>Login</h2>
          <p class="muted">Use your user ID and password to access the prediction workspace.</p>
          <form method="post" action="{{ url_for('login') }}">
            <label>User ID</label>
            <input type="text" name="username" value="admin" required>
            <label>Password</label>
            <input type="password" name="password" value="admin123" required>
            <p><button class="button" type="submit">Access Dashboard</button></p>
          </form>
          <form method="post" action="{{ url_for('register') }}">
            <label>Register New User</label>
            <input type="text" name="username" placeholder="Choose a user ID" required>
            <label>Password</label>
            <input type="password" name="password" placeholder="Choose a password" required>
            <p><button class="button secondary" type="submit">Register New User</button></p>
          </form>
        {% endif %}
      </div>
    </section>
    {% endif %}

    <section id="overview" class="panel">
      <h2>What This Project Does</h2>
      <div class="grid">
        <div>
          <h3>Biosignal Input</h3>
          <p class="muted">Reads ECG signal data with 187 numeric features per sample. Files may include an additional label column.</p>
        </div>
        <div>
          <h3>Deep Learning Model</h3>
          <p class="muted">Applies standard scaling and trains a 1D convolutional neural network for five-class ECG arrhythmia classification.</p>
        </div>
        <div>
          <h3>Output Review</h3>
          <p class="muted">Shows predicted class, confidence, class probabilities, training metrics, confusion matrix, and signal previews.</p>
        </div>
      </div>
    </section>

    <section id="process" class="panel">
      <h2>Step By Step Process</h2>
      <div class="steps">
        <div class="step"><div class="badge">Step 1</div><strong>Register or login</strong><p class="muted">Create a user account or use the demo credentials.</p></div>
        <div class="step"><div class="badge">Step 2</div><strong>Train model</strong><p class="muted">The server trains the CNN on the default MIT-BIH training dataset.</p></div>
        <div class="step"><div class="badge">Step 3</div><strong>Upload CSV</strong><p class="muted">Upload ECG data with 187 features, with or without the label column.</p></div>
        <div class="step"><div class="badge">Step 4</div><strong>Review result</strong><p class="muted">Inspect prediction confidence, probabilities, signal shape, and model performance.</p></div>
      </div>
    </section>

    {% if not user %}
    <section id="predictor" class="panel">
      <h2>Prediction Workspace</h2>
      <p class="muted">
        Model status:
        {% if model_ready %}<strong>Ready</strong>{% elif model_loading %}<strong>Loading model</strong>{% elif training_in_progress %}<strong>Training in progress</strong>{% elif training_error %}<strong>Training failed</strong>{% else %}<strong>Not started</strong>{% endif %}
      </p>
      {% if training_error %}<p class="flash">{{ training_error }}</p>{% endif %}
      {% if history %}
        <p class="muted">Accuracy {{ "%.4f"|format(history.accuracy) }} | Validation accuracy {{ "%.4f"|format(history.val_accuracy) }} | Loss {{ "%.4f"|format(history.loss) }} | Validation loss {{ "%.4f"|format(history.val_loss) }}</p>
      {% endif %}
      {% if matrix %}
        <div class="matrix">{{ matrix }}</div>
      {% endif %}

      {% if user %}
        <form method="post" action="{{ url_for('predict') }}" enctype="multipart/form-data">
          <label>Upload Biosignal CSV</label>
          <input type="file" name="dataset" accept=".csv" required>
          <p><button class="button" type="submit">Run Prediction</button></p>
        </form>
      {% else %}
        <p class="muted">Login or register to use the model.</p>
      {% endif %}

      {% if prediction %}
        <div class="panel result">
          <h2>Prediction Result</h2>
          <p><strong>Predicted Class:</strong> {{ prediction.class_id }} - {{ prediction.label }}</p>
          <p><strong>Confidence:</strong> {{ "%.2f"|format(prediction.confidence * 100) }}%</p>
          <h3>Class Probabilities</h3>
          {% for label, value in prediction.probabilities %}
            <p class="muted">{{ label }}: {{ "%.4f"|format(value) }}</p>
          {% endfor %}
          <canvas id="signalChart"></canvas>
          <canvas id="freqChart"></canvas>
        </div>
        <script>
          const signalData = {{ prediction.signal_preview | tojson }};
          const freqData = {{ prediction.frequency_preview | tojson }};
          const labels = signalData.map((_, i) => i + 1);
          new Chart(document.getElementById("signalChart"), {
            type: "line",
            data: { labels, datasets: [{ label: "ECG Signal", data: signalData, borderColor: "#1e88e5", pointRadius: 0 }] },
            options: { responsive: true, plugins: { legend: { display: true } } }
          });
          new Chart(document.getElementById("freqChart"), {
            type: "line",
            data: { labels, datasets: [{ label: "Frequency Domain", data: freqData, borderColor: "#00a98f", pointRadius: 0 }] },
            options: { responsive: true, plugins: { legend: { display: true } } }
          });
        </script>
      {% endif %}
    </section>
    {% endif %}

    {% if user %}
    <script>
      const signalData = {% if prediction %}{{ prediction.signal_preview | tojson }}{% else %}[]{% endif %};
      const freqData = {% if prediction %}{{ prediction.frequency_preview | tojson }}{% else %}[]{% endif %};
      const signalLabels = signalData.map((_, i) => i + 1);
      const emptyChartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: { legend: { display: true } },
        scales: { x: { grid: { color: "#edf3fb" } }, y: { grid: { color: "#edf3fb" } } }
      };

      new Chart(document.getElementById("signalChart"), {
        type: "line",
        data: { labels: signalLabels, datasets: [{ label: "ECG Signal", data: signalData, borderColor: "#1e88e5", pointRadius: 0, tension: 0.2 }] },
        options: emptyChartOptions
      });
      new Chart(document.getElementById("freqChart"), {
        type: "line",
        data: { labels: signalLabels, datasets: [{ label: "Frequency Domain", data: freqData, borderColor: "#00b8a9", pointRadius: 0, tension: 0.2 }] },
        options: emptyChartOptions
      });

      const trainAcc = {% if history and history.accuracy_series is defined %}{{ history.accuracy_series | tojson }}{% elif history %}[{{ history.accuracy }}]{% else %}[]{% endif %};
      const valAcc = {% if history and history.val_accuracy_series is defined %}{{ history.val_accuracy_series | tojson }}{% elif history %}[{{ history.val_accuracy }}]{% else %}[]{% endif %};
      const trainLoss = {% if history and history.loss_series is defined %}{{ history.loss_series | tojson }}{% elif history %}[{{ history.loss }}]{% else %}[]{% endif %};
      const valLoss = {% if history and history.val_loss_series is defined %}{{ history.val_loss_series | tojson }}{% elif history %}[{{ history.val_loss }}]{% else %}[]{% endif %};
      const epochs = Array.from({ length: Math.max(trainAcc.length, valAcc.length, trainLoss.length, valLoss.length) }, (_, i) => i + 1);
      const uploadMetricsUrl = {% if prediction and prediction.evaluation_job_id %}"{{ url_for('upload_metrics_data', job_id=prediction.evaluation_job_id) }}"{% else %}null{% endif %};
      const trainingStatus = document.getElementById("trainingStatus");
      const trainingChart = new Chart(document.getElementById("trainingChart"), {
        type: "line",
        data: {
          labels: epochs,
          datasets: [
            { label: "Accuracy", data: trainAcc, borderColor: "#1e88e5", pointRadius: 2 },
            { label: "Loss", data: trainLoss, borderColor: "#ff8a65", pointRadius: 2 },
            { label: "Val Acc", data: valAcc, borderColor: "#00b8a9", borderDash: [6, 4], pointRadius: 2 },
            { label: "Val Loss", data: valLoss, borderColor: "#1565c0", borderDash: [6, 4], pointRadius: 2 }
          ]
        },
        options: emptyChartOptions
      });

      function updateUploadedDatasetChart(payload) {
        const accuracy = payload.accuracy || [];
        const loss = payload.loss || [];
        const sampleAxis = payload.sample_axis || accuracy.map((_, i) => i + 1);

        trainingChart.data.labels = sampleAxis;
        trainingChart.data.datasets[0].label = "Uploaded Accuracy";
        trainingChart.data.datasets[0].data = accuracy;
        trainingChart.data.datasets[1].label = "Uploaded Loss";
        trainingChart.data.datasets[1].data = loss;
        trainingChart.data.datasets[2].data = [];
        trainingChart.data.datasets[3].data = [];
        trainingChart.update();

        if (payload.error) {
          trainingStatus.textContent = `Uploaded dataset evaluation failed: ${payload.error}`;
        } else if (payload.complete) {
          const finalAccuracy = accuracy.length ? accuracy[accuracy.length - 1] : 0;
          const finalLoss = loss.length ? loss[loss.length - 1] : 0;
          trainingStatus.textContent = `Uploaded dataset complete: ${payload.processed}/${payload.sample_count} samples | accuracy=${finalAccuracy.toFixed(4)} | loss=${finalLoss.toFixed(4)}`;
        } else {
          trainingStatus.textContent = `Evaluating uploaded dataset: ${payload.processed}/${payload.sample_count} samples`;
        }
      }

      async function pollUploadedDatasetMetrics() {
        if (!uploadMetricsUrl) {
          return false;
        }

        try {
          const response = await fetch(uploadMetricsUrl, { cache: "no-store" });
          if (response.ok) {
            const payload = await response.json();
            updateUploadedDatasetChart(payload);
            return payload.complete || Boolean(payload.error);
          }
        } catch (error) {
          trainingStatus.textContent = "Unable to load uploaded dataset metrics.";
          return true;
        }

        return false;
      }

      function updateTrainingChart(payload) {
        const metrics = payload.metrics || {};
        const accuracy = metrics.accuracy || [];
        const valAccuracy = metrics.val_accuracy || [];
        const loss = metrics.loss || [];
        const valLoss = metrics.val_loss || [];
        const maxLength = Math.max(accuracy.length, valAccuracy.length, loss.length, valLoss.length);

        trainingChart.data.labels = Array.from({ length: maxLength }, (_, i) => i + 1);
        trainingChart.data.datasets[0].data = accuracy;
        trainingChart.data.datasets[1].data = loss;
        trainingChart.data.datasets[2].data = valAccuracy;
        trainingChart.data.datasets[3].data = valLoss;
        trainingChart.update();

        if (payload.training_in_progress) {
          trainingStatus.textContent = `Training in progress. Epochs completed: ${maxLength}`;
        } else if (payload.model_loading) {
          trainingStatus.textContent = "Model is loading. Metrics will appear shortly.";
        } else if (payload.model_ready && maxLength > 0) {
          trainingStatus.textContent = `Training complete. Epochs shown: ${maxLength}`;
        } else if (payload.training_error) {
          trainingStatus.textContent = `Training failed: ${payload.training_error}`;
        } else {
          trainingStatus.textContent = "Training metrics are not available yet.";
        }
      }

      async function pollTrainingMetrics() {
        try {
          const response = await fetch("{{ url_for('metrics_data') }}", { cache: "no-store" });
          if (response.ok) {
            const payload = await response.json();
            updateTrainingChart(payload);
          }
        } catch (error) {
          trainingStatus.textContent = "Unable to load live training metrics.";
        }
      }

      if (uploadMetricsUrl) {
        const uploadTimer = setInterval(async () => {
          const done = await pollUploadedDatasetMetrics();
          if (done) {
            clearInterval(uploadTimer);
          }
        }, 1000);
        pollUploadedDatasetMetrics();
      } else {
        pollTrainingMetrics();
        setInterval(pollTrainingMetrics, 2000);
      }
    </script>
    {% endif %}
  </main>
</body>
</html>
"""


@app.route("/")
def index():
    if current_user() and not model_ready and not model_loading and not training_in_progress:
        start_model_load_thread()

    matrix = np.array2string(confusion_mat) if confusion_mat is not None else None
    return render_template_string(
        PAGE,
        user=current_user(),
        model_ready=model_ready,
        model_loading=model_loading,
        training_in_progress=training_in_progress,
        training_error=training_error,
        history=history_summary,
        matrix=matrix,
        prediction=session.pop("prediction", None),
    )


@app.get("/metrics-data")
def metrics_data():
    metrics = current_metric_series()
    return jsonify(
        {
            "model_ready": model_ready,
            "model_loading": model_loading,
            "training_in_progress": training_in_progress,
            "training_error": training_error,
            "metrics": metrics,
            "summary": history_summary or {},
        }
    )


@app.get("/upload-metrics/<job_id>")
def upload_metrics_data(job_id):
    with upload_eval_lock:
        job = upload_eval_jobs.get(job_id)
        if job is None:
            return jsonify({"error": "Uploaded dataset evaluation was not found.", "complete": True}), 404

        return jsonify(dict(job))


@app.post("/register")
def register():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()
    if not username or not password:
        flash("Enter a user ID and password to register.")
        return redirect(url_for("index"))

    try:
        with sqlite3.connect(DATABASE_PATH) as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, generate_password_hash(password)),
            )
        session["username"] = username
        flash("Registration successful. You are signed in.")
    except sqlite3.IntegrityError:
        flash("That user ID already exists. Please login or choose another one.")
    return redirect(url_for("index"))


@app.post("/login")
def login():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()
    with sqlite3.connect(DATABASE_PATH) as conn:
        row = conn.execute("SELECT password_hash FROM users WHERE username = ?", (username,)).fetchone()

    if row and check_password_hash(row[0], password):
        session["username"] = username
        flash("Login successful.")
    else:
        flash("Invalid user ID or password.")
    return redirect(url_for("index"))


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.")
    return redirect(url_for("index"))


@app.post("/refresh-metrics")
def refresh_metrics():
    if not current_user():
        flash("Login or register before refreshing metrics.")
        return redirect(url_for("index"))

    if not model_ready and not model_loading:
        start_model_load_thread()
        flash("Model loading started. Refresh again in a few seconds.")
    elif model_loading:
        flash("Model is still loading. Refresh again in a few seconds.")
    elif model_ready:
        flash("Training metrics refreshed.")
    elif training_in_progress:
        flash("Training is still running. Metrics will update when it finishes.")
    elif training_error:
        flash(f"Training failed: {training_error}")
    else:
        start_training_thread()
        flash("Training started. Refresh again after it completes.")

    return redirect(url_for("index") + "#predictor")


@app.post("/predict")
def predict():
    if not current_user():
        flash("Login or register before using the prediction model.")
        return redirect(url_for("index"))

    uploaded = request.files.get("dataset")
    if uploaded is None or uploaded.filename == "":
        flash("Please upload a CSV dataset.")
        return redirect(url_for("index"))

    try:
        session["prediction"] = prediction_from_csv(uploaded)
    except Exception as exc:
        flash(str(exc))
    return redirect(url_for("index") + "#predictor")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
