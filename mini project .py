import queue
import sys
import os
import threading
import tkinter as tk
import time
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext

APP_DIR = Path(__file__).resolve().parent
PROJECT_PYTHON = APP_DIR / ".venv311" / "Scripts" / "python.exe"

if PROJECT_PYTHON.exists() and Path(sys.executable).resolve() != PROJECT_PYTHON.resolve():
    os.execv(str(PROJECT_PYTHON), [str(PROJECT_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.fft import fft
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ======================
# CUSTOMIZABLE SETTINGS
# ======================
APP_TITLE = "Early Disease Prediction System"
VALID_USER_ID = "admin"
VALID_PASSWORD = "admin123"
WINDOW_SIZE = "1480x920"
PROCESSING_COUNTDOWN_SECONDS = 300

PAGE_BG = "#edf3fb"
PANEL_BG = "#ffffff"
HERO_BG = "#12304a"
ACCENT = "#1e88e5"
ACCENT_DARK = "#1565c0"
HIGHLIGHT = "#00b8a9"
TEXT_PRIMARY = "#17324d"
TEXT_SECONDARY = "#59708a"
BORDER = "#d8e4f2"
LOG_BG = "#0f2235"
LOG_TEXT = "#d8e9fb"
SUCCESS = "#1f9d72"
ERROR = "#d9534f"


# ======================
# GLOBALS
# ======================
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE_DIR / "dataset" / "mitbih_train.csv"

scaler = None
model = None
history = None
confusion_mat = None
training_error = None
model_ready = False
training_in_progress = False
tensorflow_loaded = False
live_training_metrics = {
    "accuracy": [],
    "val_accuracy": [],
    "loss": [],
    "val_loss": [],
}
app = None
ui_queue = queue.Queue()


class QueueStream:
    def __init__(self, stream_name):
        self.stream_name = stream_name
        self._buffer = ""

    def write(self, message):
        if not message:
            return

        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip()
            if line:
                ui_queue.put(("log", f"[{self.stream_name}] {line}"))

    def flush(self):
        if self._buffer.strip():
            ui_queue.put(("log", f"[{self.stream_name}] {self._buffer.strip()}"))
        self._buffer = ""

Sequential = None
Conv1D = None
MaxPooling1D = None
Flatten = None
Dense = None
Dropout = None
Callback = None


def ensure_tensorflow_loaded():
    global tensorflow_loaded, Sequential, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Callback

    if tensorflow_loaded:
        return

    from tensorflow.keras.callbacks import Callback as KerasCallback
    from tensorflow.keras.layers import Conv1D as KerasConv1D
    from tensorflow.keras.layers import Dense as KerasDense
    from tensorflow.keras.layers import Dropout as KerasDropout
    from tensorflow.keras.layers import Flatten as KerasFlatten
    from tensorflow.keras.layers import MaxPooling1D as KerasMaxPooling1D
    from tensorflow.keras.models import Sequential as KerasSequential

    Callback = KerasCallback
    Conv1D = KerasConv1D
    Dense = KerasDense
    Dropout = KerasDropout
    Flatten = KerasFlatten
    MaxPooling1D = KerasMaxPooling1D
    Sequential = KerasSequential
    tensorflow_loaded = True


def create_queue_logger_callback():
    class QueueLoggerCallback(Callback):
        def on_train_begin(self, logs=None):
            ui_queue.put(("log", "Training started on the default biosignal dataset."))
            ui_queue.put(("log", "TensorFlow runtime loaded successfully."))
            ui_queue.put(("status", "Model training started. Please wait..."))
            ui_queue.put(("training_metrics_reset", None))

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            message = (
                f"Epoch {epoch + 1}: "
                f"accuracy={logs.get('accuracy', 0):.4f}, "
                f"val_accuracy={logs.get('val_accuracy', 0):.4f}, "
                f"loss={logs.get('loss', 0):.4f}, "
                f"val_loss={logs.get('val_loss', 0):.4f}"
            )
            ui_queue.put(("log", message))
            ui_queue.put(("training_metrics_update", {
                "epoch": epoch + 1,
                "accuracy": float(logs.get("accuracy", 0)),
                "val_accuracy": float(logs.get("val_accuracy", 0)),
                "loss": float(logs.get("loss", 0)),
                "val_loss": float(logs.get("val_loss", 0)),
            }))

    return QueueLoggerCallback()


# ======================
# MODEL PIPELINE
# ======================
def train_model(dataset_path=DEFAULT_DATASET):
    global scaler, model, history, confusion_mat, model_ready, training_error, training_in_progress, live_training_metrics

    model_ready = False
    training_in_progress = True
    training_error = None
    live_training_metrics = {
        "accuracy": [],
        "val_accuracy": [],
        "loss": [],
        "val_loss": [],
    }

    ui_queue.put(("log", "Loading TensorFlow runtime..."))
    ensure_tensorflow_loaded()

    # ======================
    # LOAD DATASET
    # ======================
    data = pd.read_csv(dataset_path, header=None)
    ui_queue.put(("log", f"Loaded training dataset: {dataset_path}"))
    ui_queue.put(("log", f"Dataset shape: {data.shape[0]} rows x {data.shape[1]} columns"))

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # ======================
    # PREPROCESSING
    # ======================
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    ui_queue.put(("log", "Standard scaling completed for training features."))

    # Reshape for CNN
    X = X.reshape(X.shape[0], X.shape[1], 1)
    ui_queue.put(("log", f"Input reshaped for CNN: {X.shape}"))

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    ui_queue.put(("log", f"Train split: {X_train.shape[0]} samples | Test split: {X_test.shape[0]} samples"))

    # ======================
    # CNN MODEL
    # ======================
    model = Sequential([
        Conv1D(32, 3, activation="relu", input_shape=(187, 1)),
        MaxPooling1D(2),
        Conv1D(64, 3, activation="relu"),
        MaxPooling1D(2),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(5, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    ui_queue.put(("log", "CNN model compiled successfully."))

    # ======================
    # TRAIN MODEL
    # ======================
    print("Training model...")
    history = model.fit(
        X_train,
        y_train,
        epochs=5,
        validation_data=(X_test, y_test),
        callbacks=[create_queue_logger_callback()],
        verbose=0,
    )

    # ======================
    # PERFORMANCE METRICS
    # ======================
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    confusion_mat = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", confusion_mat)
    ui_queue.put(("log", "Confusion matrix generated."))

    model_ready = True
    training_in_progress = False
    ui_queue.put(("training_complete", None))


def train_model_in_background():
    global training_error, training_in_progress
    try:
        train_model()
    except Exception as exc:
        training_error = str(exc)
        training_in_progress = False
        ui_queue.put(("training_failed", training_error))


# ======================
# GUI FUNCTION
# ======================
def upload():
    if not model_ready or model is None or scaler is None:
        messagebox.showwarning("Model Not Ready", "Please wait until model training is completed.")
        return

    file = filedialog.askopenfilename(
        title="Select Biosignal Dataset",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
    )
    if not file:
        return

    try:
        data = pd.read_csv(file, header=None)
    except Exception as exc:
        message = "Unsupported dataset format. Please upload a valid CSV biosignal dataset."
        ui_queue.put(("log", f"Dataset read failed: {exc}"))
        ui_queue.put(("popup_error", message))
        return

    if data.empty:
        message = "The uploaded dataset is empty. Please upload a valid biosignal dataset."
        ui_queue.put(("log", "Dataset validation failed: file is empty."))
        ui_queue.put(("popup_error", message))
        return

    if data.shape[1] < 2:
        message = "Unsupported dataset structure. Please upload a CSV with biosignal features and a label column."
        ui_queue.put(("log", f"Dataset validation failed: expected at least 2 columns, found {data.shape[1]}."))
        ui_queue.put(("popup_error", message))
        return

    expected_feature_count = 187
    actual_feature_count = data.shape[1] - 1
    if actual_feature_count != expected_feature_count:
        message = (
            f"Unsupported dataset format. Expected {expected_feature_count} ECG features, "
            f"but found {actual_feature_count}."
        )
        ui_queue.put(("log", f"Dataset validation failed: expected {expected_feature_count} features, found {actual_feature_count}."))
        ui_queue.put(("popup_error", message))
        return

    try:
        signal = pd.to_numeric(data.iloc[0, :-1], errors="raise").values.astype(float)
    except Exception as exc:
        message = "Unsupported dataset content. Please upload a numeric biosignal CSV."
        ui_queue.put(("log", f"Dataset validation failed: non-numeric values detected. {exc}"))
        ui_queue.put(("popup_error", message))
        return

    try:
        ui_queue.put(("log", f"Uploaded dataset for prediction: {file}"))
        ui_queue.put(("log", f"Using first biosignal record with {len(signal)} features."))

        freq = np.abs(fft(signal))

        signal_scaled = scaler.transform([signal])
        signal_scaled = signal_scaled.reshape(1, signal_scaled.shape[1], 1)

        pred = model.predict(signal_scaled, verbose=0)
        result = int(np.argmax(pred))
        confidence = float(np.max(pred))
    except Exception as exc:
        message = "An error occurred during processing. Please check your dataset or try again."
        ui_queue.put(("log", f"Prediction failed: {exc}"))
        ui_queue.put(("popup_error", message))
        return

    ui_queue.put(("prediction", {
        "file": file,
        "signal": signal,
        "freq": freq,
        "pred": pred[0],
        "result": result,
        "confidence": confidence,
    }))


class DiseasePredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.configure(bg=PAGE_BG)
        self.configure_window_size()

        self.file_var = tk.StringVar(value=f"Training Dataset: {DEFAULT_DATASET}")
        self.result_var = tk.StringVar(value="Prediction will appear here after dataset upload.")
        self.status_var = tk.StringVar(value="Please login to continue")
        self.timer_var = tk.StringVar(value=f"Time left: {PROCESSING_COUNTDOWN_SECONDS // 60:02d}:{PROCESSING_COUNTDOWN_SECONDS % 60:02d}")
        self.metrics_var = tk.StringVar(value="Awaiting training results")
        self.live_training_metrics = {
            "accuracy": [],
            "val_accuracy": [],
            "loss": [],
            "val_loss": [],
        }

        self.login_frame = None
        self.dashboard_frame = None
        self.user_id_entry = None
        self.password_entry = None
        self.upload_button = None
        self.training_canvas = None
        self.training_preview_canvas = None
        self.signal_canvas = None
        self.cm_text = None
        self.logs_widgets = []
        self.prediction_text = None
        self.processing_popup = None
        self.processing_popup_message = None
        self.processing_start_time = None
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.stdout_stream = QueueStream("INFO")
        self.stderr_stream = QueueStream("ERROR")

        sys.stdout = self.stdout_stream
        sys.stderr = self.stderr_stream

        self.root.after(150, self.process_ui_queue)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.build_login_page()

    def configure_window_size(self):
        desired_width, desired_height = [int(value) for value in WINDOW_SIZE.split("x")]
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        width = min(desired_width, max(screen_width - 60, 1000))
        height = min(desired_height, max(screen_height - 80, 700))
        x_pos = max((screen_width - width) // 2, 0)
        y_pos = max((screen_height - height) // 2, 0)

        self.root.geometry(f"{width}x{height}+{x_pos}+{y_pos}")
        self.root.minsize(min(1000, screen_width), min(700, screen_height))

    def process_ui_queue(self):
        while True:
            try:
                item_type, payload = ui_queue.get_nowait()
            except queue.Empty:
                break

            if item_type == "log":
                self.append_log(payload)
            elif item_type == "status":
                self.status_var.set(payload)
            elif item_type == "popup_error":
                messagebox.showerror("Error", payload)
            elif item_type == "training_metrics_reset":
                self.reset_live_training_metrics()
            elif item_type == "training_metrics_update":
                self.update_live_training_metrics(payload)
            elif item_type == "training_complete":
                self.on_training_finished()
            elif item_type == "training_failed":
                self.on_training_failed(payload)
            elif item_type == "prediction":
                self.update_prediction_view(payload)

        self.root.after(150, self.process_ui_queue)

    def clear_frames(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def on_close(self):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.root.destroy()

    def append_log(self, message):
        self.logs_widgets = [widget for widget in self.logs_widgets if widget is not None and widget.winfo_exists()]
        if not self.logs_widgets:
            return

        for widget in self.logs_widgets:
            widget.configure(state="normal")
            widget.insert("end", f"{message}\n")
            widget.see("end")
            widget.configure(state="disabled")

    def show_processing_popup(self, message):
        self.close_processing_popup()

        popup = tk.Toplevel(self.root)
        popup.title("Processing")
        popup.configure(bg=PANEL_BG)
        popup.transient(self.root)
        popup.resizable(False, False)

        popup.update_idletasks()
        width = 360
        height = 150
        x_pos = self.root.winfo_x() + max((self.root.winfo_width() // 2) - (width // 2), 0)
        y_pos = self.root.winfo_y() + max((self.root.winfo_height() // 2) - (height // 2), 0)
        popup.geometry(f"{width}x{height}+{x_pos}+{y_pos}")

        tk.Label(
            popup,
            text="Processing",
            font=("Segoe UI", 16, "bold"),
            bg=PANEL_BG,
            fg=TEXT_PRIMARY,
        ).pack(pady=(24, 8))

        self.processing_popup_message = tk.Label(
            popup,
            text=message,
            font=("Segoe UI", 10),
            bg=PANEL_BG,
            fg=TEXT_SECONDARY,
            wraplength=300,
            justify="center",
        )
        self.processing_popup_message.pack(padx=24)

        self.processing_popup = popup

    def close_processing_popup(self):
        if self.processing_popup is None:
            return

        if self.processing_popup.winfo_exists():
            self.processing_popup.destroy()

        self.processing_popup = None
        self.processing_popup_message = None

    def start_processing_timer(self):
        self.processing_start_time = time.time()
        self.update_processing_timer()

    def stop_processing_timer(self):
        self.processing_start_time = None
        self.timer_var.set("Time left: 00:00")

    def reset_live_training_metrics(self):
        self.live_training_metrics = {
            "accuracy": [],
            "val_accuracy": [],
            "loss": [],
            "val_loss": [],
        }
        self.render_training_graphs(self.live_training_metrics, show_placeholder=False)

    def update_live_training_metrics(self, payload):
        self.live_training_metrics["accuracy"].append(payload["accuracy"])
        self.live_training_metrics["val_accuracy"].append(payload["val_accuracy"])
        self.live_training_metrics["loss"].append(payload["loss"])
        self.live_training_metrics["val_loss"].append(payload["val_loss"])
        self.render_training_graphs(self.live_training_metrics, show_placeholder=False)

    def update_processing_timer(self):
        if self.processing_start_time is None:
            return

        elapsed_seconds = int(time.time() - self.processing_start_time)
        remaining_seconds = max(PROCESSING_COUNTDOWN_SECONDS - elapsed_seconds, 0)
        minutes, seconds = divmod(remaining_seconds, 60)
        detail_text = f"Model training is in progress. Please wait. Time left: {minutes:02d}:{seconds:02d}"

        self.status_var.set("Processing...")
        self.timer_var.set(f"Time left: {minutes:02d}:{seconds:02d}")
        self.metrics_var.set(detail_text)

        if self.processing_popup_message is not None and self.processing_popup_message.winfo_exists():
            self.processing_popup_message.config(text=detail_text)

        self.root.after(1000, self.update_processing_timer)

    def create_panel(self, parent, title, subtitle=None):
        panel = tk.Frame(parent, bg=PANEL_BG, highlightbackground=BORDER, highlightthickness=1)

        header = tk.Frame(panel, bg=PANEL_BG)
        header.pack(fill="x", padx=18, pady=(16, 6))

        tk.Label(
            header,
            text=title,
            font=("Segoe UI", 14, "bold"),
            bg=PANEL_BG,
            fg=TEXT_PRIMARY,
        ).pack(anchor="w")

        if subtitle:
            tk.Label(
                header,
                text=subtitle,
                font=("Segoe UI", 9),
                bg=PANEL_BG,
                fg=TEXT_SECONDARY,
            ).pack(anchor="w", pady=(2, 0))

        return panel

    def style_button(self, parent, text, command, width=18, state="normal", secondary=False):
        bg = PANEL_BG if secondary else ACCENT
        fg = TEXT_PRIMARY if secondary else "#ffffff"
        active_bg = "#e9f1fb" if secondary else ACCENT_DARK
        highlight = BORDER if secondary else ACCENT_DARK

        return tk.Button(
            parent,
            text=text,
            command=command,
            width=width,
            state=state,
            font=("Segoe UI", 10, "bold"),
            bg=bg,
            fg=fg,
            activebackground=active_bg,
            activeforeground=fg,
            relief="flat",
            bd=0,
            padx=12,
            pady=10,
            highlightthickness=1,
            highlightbackground=highlight,
        )

    def build_login_page(self):
        self.clear_frames()
        self.logs_widgets = []
        self.status_var.set("Please login to continue")

        shell = tk.Frame(self.root, bg=PAGE_BG)
        shell.pack(fill="both", expand=True, padx=32, pady=32)
        shell.grid_columnconfigure(0, weight=7)
        shell.grid_columnconfigure(1, weight=5)
        shell.grid_rowconfigure(0, weight=1)

        hero = tk.Frame(shell, bg=HERO_BG)
        hero.grid(row=0, column=0, sticky="nsew", padx=(0, 18))

        tk.Label(
            hero,
            text=APP_TITLE,
            font=("Segoe UI", 28, "bold"),
            bg=HERO_BG,
            fg="#ffffff",
            wraplength=500,
            justify="left",
        ).pack(anchor="nw", padx=40, pady=(56, 12))

        tk.Label(
            hero,
            text="Secure biosignal-based screening dashboard powered by a convolutional neural network.",
            font=("Segoe UI", 13),
            bg=HERO_BG,
            fg="#c8dbef",
            wraplength=520,
            justify="left",
        ).pack(anchor="nw", padx=40)

        info_strip = tk.Frame(hero, bg="#173a59")
        info_strip.pack(anchor="sw", fill="x", side="bottom", padx=40, pady=40)

        tk.Label(
            info_strip,
            text="Workflow: login  ->  train model  ->  upload dataset  ->  review results on one screen",
            font=("Segoe UI", 11, "bold"),
            bg="#173a59",
            fg="#ffffff",
            pady=16,
        ).pack(anchor="w", padx=20)

        card = tk.Frame(shell, bg=PANEL_BG, highlightbackground=BORDER, highlightthickness=1)
        card.grid(row=0, column=1, sticky="nsew")
        card.grid_columnconfigure(0, weight=1)

        tk.Label(
            card,
            text="Login",
            font=("Segoe UI", 24, "bold"),
            bg=PANEL_BG,
            fg=TEXT_PRIMARY,
        ).grid(row=0, column=0, sticky="w", padx=36, pady=(50, 8))

        tk.Label(
            card,
            text="Use your user ID and password to access the prediction workspace.",
            font=("Segoe UI", 10),
            bg=PANEL_BG,
            fg=TEXT_SECONDARY,
            wraplength=360,
            justify="left",
        ).grid(row=1, column=0, sticky="w", padx=36, pady=(0, 24))

        tk.Label(card, text="User ID", font=("Segoe UI", 10, "bold"), bg=PANEL_BG, fg=TEXT_PRIMARY).grid(
            row=2, column=0, sticky="w", padx=36
        )
        self.user_id_entry = tk.Entry(
            card,
            font=("Segoe UI", 12),
            relief="flat",
            bg="#f6faff",
            fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=ACCENT,
        )
        self.user_id_entry.grid(row=3, column=0, sticky="ew", padx=36, pady=(8, 18), ipady=10)

        tk.Label(card, text="Password", font=("Segoe UI", 10, "bold"), bg=PANEL_BG, fg=TEXT_PRIMARY).grid(
            row=4, column=0, sticky="w", padx=36
        )
        self.password_entry = tk.Entry(
            card,
            font=("Segoe UI", 12),
            relief="flat",
            bg="#f6faff",
            fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=ACCENT,
            show="*",
        )
        self.password_entry.grid(row=5, column=0, sticky="ew", padx=36, pady=(8, 26), ipady=10)

        self.style_button(card, "Access Dashboard", self.handle_login, width=20).grid(
            row=6, column=0, sticky="w", padx=36
        )

        tk.Button(
            card,
            text="Register New User",
            command=self.show_register_message,
            font=("Segoe UI", 10, "bold"),
            bg=PANEL_BG,
            fg=TEXT_PRIMARY,
            activebackground=PANEL_BG,
            activeforeground=TEXT_PRIMARY,
            relief="flat",
            bd=0,
            cursor="hand2",
        ).grid(row=7, column=0, sticky="w", padx=72, pady=(18, 0))

        tk.Label(
            card,
            textvariable=self.status_var,
            font=("Segoe UI", 10),
            bg=PANEL_BG,
            fg=TEXT_SECONDARY,
            wraplength=360,
            justify="left",
        ).grid(row=8, column=0, sticky="w", padx=36, pady=(38, 0))

    def show_register_message(self):
        messagebox.showinfo("Register New User", "Registration is not enabled for this demo. Use admin / admin123.")

    def build_dashboard_page(self):
        self.clear_frames()

        self.dashboard_frame = tk.Frame(self.root, bg=PAGE_BG)
        self.dashboard_frame.pack(fill="both", expand=True, padx=18, pady=18)
        self.dashboard_frame.grid_columnconfigure(0, weight=1)
        self.dashboard_frame.grid_rowconfigure(1, weight=1)

        header = tk.Frame(self.dashboard_frame, bg=PAGE_BG)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 18))
        header.grid_columnconfigure(0, weight=1)

        title_box = tk.Frame(header, bg=PAGE_BG)
        title_box.grid(row=0, column=0, sticky="w")

        tk.Label(
            title_box,
            text=APP_TITLE,
            font=("Segoe UI", 24, "bold"),
            bg=PAGE_BG,
            fg=TEXT_PRIMARY,
        ).pack(anchor="w")

        tk.Label(
            title_box,
            text="Early Disease Prediction using Deep Neural Networks on Biosignal Data",
            font=("Segoe UI", 11),
            bg=PAGE_BG,
            fg=TEXT_SECONDARY,
        ).pack(anchor="w", pady=(4, 0))

        self.style_button(header, "Logout", self.build_login_page, width=12, secondary=True).grid(
            row=0, column=1, sticky="e"
        )

        body = tk.Frame(self.dashboard_frame, bg=PAGE_BG)
        body.grid(row=1, column=0, sticky="nsew")
        body.grid_columnconfigure(0, weight=0)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        left_column = tk.Frame(body, bg=PAGE_BG, width=210)
        left_column.grid(row=0, column=0, sticky="nsw", padx=(0, 16))
        left_column.grid_propagate(False)
        left_column.grid_columnconfigure(0, weight=1)
        left_column.grid_rowconfigure(3, weight=1)

        right_column = tk.Frame(body, bg=PAGE_BG)
        right_column.grid(row=0, column=1, sticky="nsew")
        right_column.grid_columnconfigure(0, weight=1)
        right_column.grid_rowconfigure(0, weight=1)

        action_panel = self.create_panel(left_column, "Actions", "Run prediction and inspect embedded training outputs")
        action_panel.grid(row=0, column=0, sticky="ew", pady=(0, 14))

        action_body = tk.Frame(action_panel, bg=PANEL_BG)
        action_body.pack(fill="x", padx=16, pady=(0, 16))

        self.upload_button = self.style_button(
            action_body,
            "Upload the Dataset",
            upload,
            width=18,
            state="disabled",
        )
        self.upload_button.pack(anchor="w")

        self.style_button(
            action_body,
            "Refresh Metrics",
            self.refresh_training_outputs,
            width=18,
            secondary=True,
        ).pack(anchor="w", pady=(10, 0))

        tk.Label(
            action_body,
            text="Use refresh after training if you want to redraw metrics manually.",
            font=("Segoe UI", 9),
            bg=PANEL_BG,
            fg=TEXT_SECONDARY,
            wraplength=165,
            justify="left",
        ).pack(anchor="w", pady=(14, 0))

        summary_panel = self.create_panel(left_column, "Model Overview", "Status, dataset path, and current output")
        summary_panel.grid(row=1, column=0, sticky="ew", pady=(0, 14))

        summary_body = tk.Frame(summary_panel, bg=PANEL_BG)
        summary_body.pack(fill="x", padx=16, pady=(0, 16))

        status_card = tk.Frame(summary_body, bg="#f5fbf8", highlightbackground="#caecd9", highlightthickness=1)
        status_card.pack(fill="x")
        tk.Label(
            status_card,
            text="System Status",
            font=("Segoe UI", 10, "bold"),
            bg="#f5fbf8",
            fg=SUCCESS,
        ).pack(anchor="w", padx=12, pady=(12, 4))
        tk.Label(
            status_card,
            textvariable=self.status_var,
            font=("Segoe UI", 12, "bold"),
            bg="#f5fbf8",
            fg=TEXT_PRIMARY,
            wraplength=160,
            justify="left",
        ).pack(anchor="w", padx=12)
        tk.Label(
            status_card,
            textvariable=self.timer_var,
            font=("Segoe UI", 9),
            bg="#f5fbf8",
            fg=TEXT_SECONDARY,
            wraplength=160,
            justify="left",
        ).pack(anchor="w", padx=12, pady=(4, 12))

        stage_card = tk.Frame(summary_body, bg="#eef6ff", highlightbackground="#cfe2fb", highlightthickness=1)
        stage_card.pack(fill="x", pady=(12, 0))
        tk.Label(
            stage_card,
            text="Current Stage",
            font=("Segoe UI", 10, "bold"),
            bg="#eef6ff",
            fg=ACCENT_DARK,
        ).pack(anchor="w", padx=12, pady=(12, 4))
        tk.Label(
            stage_card,
            textvariable=self.metrics_var,
            font=("Segoe UI", 10),
            bg="#eef6ff",
            fg=TEXT_PRIMARY,
            wraplength=160,
            justify="left",
        ).pack(anchor="w", padx=12, pady=(0, 12))

        tk.Label(
            summary_body,
            textvariable=self.file_var,
            font=("Segoe UI", 9),
            bg=PANEL_BG,
            fg=TEXT_SECONDARY,
            wraplength=170,
            justify="left",
        ).pack(anchor="w", pady=(12, 0))

        performance_panel = self.create_panel(left_column, "Training Performance", "Accuracy and loss rendered inside the dashboard")
        performance_panel.grid(row=2, column=0, sticky="ew", pady=(0, 14))
        training_body = tk.Frame(performance_panel, bg=PANEL_BG)
        training_body.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.training_canvas = self.create_embedded_figure(
            training_body,
            facecolor="#fbfdff",
            message="Training graphs will appear here after model training finishes.",
        )

        logs_panel = self.create_panel(left_column, "Live Logs", "Training, upload, and prediction events")
        logs_panel.grid(row=3, column=0, sticky="nsew")
        sidebar_logs = scrolledtext.ScrolledText(
            logs_panel,
            font=("Consolas", 9),
            bg=LOG_BG,
            fg=LOG_TEXT,
            insertbackground="#ffffff",
            relief="flat",
            wrap="word",
            padx=12,
            pady=12,
            height=12,
        )
        sidebar_logs.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        sidebar_logs.configure(state="disabled")
        self.logs_widgets.append(sidebar_logs)

        results_panel = self.create_panel(
            right_column,
            "Outputs And Results",
            "Prediction result, probabilities, confusion matrix, and uploaded signal analysis",
        )
        results_panel.grid(row=0, column=0, sticky="nsew")

        results_body = tk.Frame(results_panel, bg=PANEL_BG)
        results_body.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        results_body.grid_columnconfigure(0, weight=3)
        results_body.grid_columnconfigure(1, weight=2)
        results_body.grid_rowconfigure(0, weight=0)
        results_body.grid_rowconfigure(1, weight=1)

        top_cards = tk.Frame(results_body, bg=PANEL_BG)
        top_cards.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        top_cards.grid_columnconfigure(0, weight=1)
        top_cards.grid_columnconfigure(1, weight=1)

        prediction_card = tk.Frame(top_cards, bg="#f8f7ff", highlightbackground="#ddd8ff", highlightthickness=1, height=150)
        prediction_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        prediction_card.grid_propagate(False)
        tk.Label(
            prediction_card,
            text="Prediction Result",
            font=("Segoe UI", 11, "bold"),
            bg="#f8f7ff",
            fg="#7154d9",
        ).pack(anchor="w", padx=14, pady=(14, 6))
        tk.Label(
            prediction_card,
            textvariable=self.result_var,
            font=("Segoe UI", 12, "bold"),
            bg="#f8f7ff",
            fg=TEXT_PRIMARY,
            wraplength=420,
            justify="left",
        ).pack(anchor="w", padx=14, pady=(0, 14))

        probability_card = tk.Frame(top_cards, bg="#fff8f0", highlightbackground="#ffe0ba", highlightthickness=1, height=150)
        probability_card.grid(row=0, column=1, sticky="nsew")
        probability_card.grid_propagate(False)
        tk.Label(
            probability_card,
            text="Prediction Confidence",
            font=("Segoe UI", 11, "bold"),
            bg="#fff8f0",
            fg="#b86b00",
        ).pack(anchor="w", padx=14, pady=(14, 6))
        self.prediction_text = tk.Text(
            probability_card,
            height=5,
            font=("Consolas", 10),
            bg="#fff8f0",
            fg=TEXT_PRIMARY,
            relief="flat",
            highlightthickness=0,
            wrap="word",
        )
        self.prediction_text.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        charts_panel = tk.Frame(results_body, bg=PANEL_BG)
        charts_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        charts_panel.grid_columnconfigure(0, weight=1)
        charts_panel.grid_rowconfigure(0, weight=1)
        self.signal_canvas = self.create_embedded_figure(
            charts_panel,
            facecolor="#fbfdff",
            message="Upload a dataset to render time-domain and frequency-domain plots here.",
        )

        right_stack = tk.Frame(results_body, bg=PANEL_BG)
        right_stack.grid(row=1, column=1, sticky="nsew")
        right_stack.grid_columnconfigure(0, weight=1)
        right_stack.grid_rowconfigure(1, weight=1)

        matrix_panel = tk.Frame(right_stack, bg=PANEL_BG, highlightbackground=BORDER, highlightthickness=1)
        matrix_panel.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        tk.Label(
            matrix_panel,
            text="Confusion Matrix",
            font=("Segoe UI", 12, "bold"),
            bg=PANEL_BG,
            fg=TEXT_PRIMARY,
        ).pack(anchor="w", padx=14, pady=(14, 8))
        self.cm_text = tk.Text(
            matrix_panel,
            font=("Consolas", 10),
            bg="#f6faff",
            fg=TEXT_PRIMARY,
            relief="flat",
            highlightthickness=1,
            highlightbackground=BORDER,
            wrap="word",
            height=8,
        )
        self.cm_text.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        lower_stack = tk.Frame(right_stack, bg=PANEL_BG)
        lower_stack.grid(row=1, column=0, sticky="nsew")
        lower_stack.grid_columnconfigure(0, weight=1)
        lower_stack.grid_rowconfigure(0, weight=0)
        lower_stack.grid_rowconfigure(1, weight=1)

        mini_training = tk.Frame(lower_stack, bg=PANEL_BG, highlightbackground=BORDER, highlightthickness=1)
        mini_training.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        tk.Label(
            mini_training,
            text="Training Performance",
            font=("Segoe UI", 11, "bold"),
            bg=PANEL_BG,
            fg=TEXT_PRIMARY,
        ).pack(anchor="w", padx=14, pady=(14, 4))
        preview_body = tk.Frame(mini_training, bg=PANEL_BG)
        preview_body.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.training_preview_canvas = self.create_embedded_figure(
            preview_body,
            facecolor="#fbfdff",
            message="Training graph will update here during model training.",
        )

        mini_logs = tk.Frame(lower_stack, bg=PANEL_BG, highlightbackground=BORDER, highlightthickness=1)
        mini_logs.grid(row=1, column=0, sticky="nsew")
        tk.Label(
            mini_logs,
            text="Live Logs",
            font=("Segoe UI", 11, "bold"),
            bg=PANEL_BG,
            fg=TEXT_PRIMARY,
        ).pack(anchor="w", padx=14, pady=(14, 4))
        inline_logs = scrolledtext.ScrolledText(
            mini_logs,
            font=("Consolas", 9),
            bg="#f8fbff",
            fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY,
            relief="flat",
            wrap="word",
            padx=10,
            pady=10,
            height=8,
        )
        inline_logs.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        inline_logs.configure(state="disabled")
        self.logs_widgets.append(inline_logs)

        self.append_log("Dashboard initialized.")
        self.append_log(f"Training dataset configured: {DEFAULT_DATASET}")
        self.cm_text.insert("1.0", "Confusion matrix will appear here after training.")
        self.prediction_text.insert("1.0", "Class probabilities will appear here after prediction.")
        self.apply_processing_state()

    def create_embedded_figure(self, parent, facecolor="#ffffff", message="", pack=True):
        figure = Figure(figsize=(7, 3.4), dpi=100, facecolor=facecolor)
        axis = figure.add_subplot(111)
        axis.set_facecolor(facecolor)
        axis.text(0.5, 0.5, message, ha="center", va="center", color=TEXT_SECONDARY, wrap=True)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color(BORDER)

        canvas = FigureCanvasTkAgg(figure, master=parent)
        canvas.draw()

        if pack:
            canvas.get_tk_widget().pack(fill="both", expand=True)

        return canvas

    def handle_login(self):
        user_id = self.user_id_entry.get().strip()
        password = self.password_entry.get().strip()

        if user_id != VALID_USER_ID or password != VALID_PASSWORD:
            self.status_var.set("Invalid user ID or password.")
            messagebox.showerror("Login Failed", "Please enter valid login credentials.")
            return

        self.status_var.set("Processing...")
        self.metrics_var.set("Model training is in progress. Please wait.")
        self.result_var.set("Prediction will appear here after dataset upload.")
        self.build_dashboard_page()

        if model_ready:
            self.status_var.set("Upload the dataset")
            self.enable_dashboard_actions()
            self.refresh_training_outputs()
            self.append_log("Existing trained model detected. Dashboard is ready for prediction.")
            self.close_processing_popup()
            self.stop_processing_timer()
            messagebox.showinfo("Upload the Dataset", "Processing is complete. You can now upload the dataset.")
            return

        if training_in_progress:
            self.start_processing_timer()
            self.disable_dashboard_actions()
            self.append_log("Training is already running in the background.")
            self.show_processing_popup("Model training is in progress. Please wait.")
            return

        self.start_processing_timer()
        self.disable_dashboard_actions()
        self.append_log("Login successful.")
        self.show_processing_popup("Model training is in progress. Please wait.")
        training_thread = threading.Thread(target=train_model_in_background, daemon=True)
        training_thread.start()

    def on_training_finished(self):
        if self.upload_button is None or not self.upload_button.winfo_exists():
            return

        self.close_processing_popup()
        self.stop_processing_timer()
        self.status_var.set("Upload the dataset")
        self.enable_dashboard_actions()
        self.refresh_training_outputs()
        self.append_log("Training completed successfully.")
        messagebox.showinfo("Upload the Dataset", "Processing is done. Please upload the dataset.")

    def on_training_failed(self, error_message):
        if self.upload_button is None or not self.upload_button.winfo_exists():
            return

        self.close_processing_popup()
        self.stop_processing_timer()
        self.status_var.set(f"Training failed: {error_message}")
        self.metrics_var.set("Training failed. Check the logs and error message.")
        self.disable_dashboard_actions()
        self.append_log(f"Training failed: {error_message}")
        messagebox.showerror("Training Failed", error_message)

    def enable_dashboard_actions(self):
        if self.upload_button is not None and self.upload_button.winfo_exists():
            self.upload_button.config(state="normal", text="Upload the Dataset", bg=ACCENT, fg="#ffffff")

    def disable_dashboard_actions(self):
        if self.upload_button is not None and self.upload_button.winfo_exists():
            self.upload_button.config(state="disabled", text="Processing...", bg="#9db7d5", fg="#eef5fc")

    def apply_processing_state(self):
        if model_ready:
            self.status_var.set("Upload the dataset")
            self.enable_dashboard_actions()
            return

        self.disable_dashboard_actions()

    def refresh_training_outputs(self):
        if history is None:
            self.metrics_var.set("Training metrics are not available yet.")
            self.append_log("Refresh requested before training completion.")
            return

        self.render_training_graphs(history.history)
        self.render_confusion_matrix()

        train_acc = history.history["accuracy"][-1]
        val_acc = history.history["val_accuracy"][-1]
        train_loss = history.history["loss"][-1]
        val_loss = history.history["val_loss"][-1]
        self.metrics_var.set(
            f"Final accuracy={train_acc:.4f} | val_accuracy={val_acc:.4f} | "
            f"loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )
        self.append_log("Embedded training metrics refreshed.")

    def render_training_graphs(self, metrics=None, show_placeholder=True):
        metrics = metrics or {}
        accuracy = metrics.get("accuracy", [])
        val_accuracy = metrics.get("val_accuracy", [])
        loss = metrics.get("loss", [])
        val_loss = metrics.get("val_loss", [])

        if not accuracy and show_placeholder:
            figure = Figure(figsize=(8, 3.6), dpi=100, facecolor="#fbfdff")
            axis = figure.add_subplot(111)
            axis.set_facecolor("#fbfdff")
            axis.text(
                0.5,
                0.5,
                "Training graphs will appear here after model training starts.",
                ha="center",
                va="center",
                color=TEXT_SECONDARY,
                wrap=True,
            )
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color(BORDER)
            self.training_canvas = self.replace_canvas_figure(self.training_canvas, figure)
            if self.training_preview_canvas is not None:
                self.training_preview_canvas = self.replace_canvas_figure(self.training_preview_canvas, figure)
            return

        figure = Figure(figsize=(8, 3.6), dpi=100, facecolor="#fbfdff")
        acc_ax = figure.add_subplot(121)
        loss_ax = figure.add_subplot(122)

        acc_ax.plot(accuracy, color=ACCENT, linewidth=2.2, label="Train")
        acc_ax.plot(val_accuracy, color=HIGHLIGHT, linewidth=2.2, label="Validation")
        acc_ax.set_title("Accuracy", color=TEXT_PRIMARY, fontsize=12, fontweight="bold")
        acc_ax.grid(alpha=0.25, color="#9eb7cf")
        acc_ax.legend(frameon=False)
        acc_ax.set_xlabel("Epoch", color=TEXT_SECONDARY)

        loss_ax.plot(loss, color=ACCENT_DARK, linewidth=2.2, label="Train")
        loss_ax.plot(val_loss, color="#ff8a65", linewidth=2.2, label="Validation")
        loss_ax.set_title("Loss", color=TEXT_PRIMARY, fontsize=12, fontweight="bold")
        loss_ax.grid(alpha=0.25, color="#9eb7cf")
        loss_ax.legend(frameon=False)
        loss_ax.set_xlabel("Epoch", color=TEXT_SECONDARY)

        for axis in (acc_ax, loss_ax):
            axis.set_facecolor("#fbfdff")
            axis.tick_params(colors=TEXT_SECONDARY)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["left"].set_color(BORDER)
            axis.spines["bottom"].set_color(BORDER)

        figure.tight_layout(pad=2.0)
        self.training_canvas = self.replace_canvas_figure(self.training_canvas, figure)
        if self.training_preview_canvas is not None:
            preview_figure = Figure(figsize=(4.2, 2.2), dpi=100, facecolor="#fbfdff")
            preview_ax = preview_figure.add_subplot(111)
            preview_ax.plot(accuracy, color=ACCENT, linewidth=2.0, label="Accuracy")
            preview_ax.plot(loss, color="#ff8a65", linewidth=2.0, label="Loss")
            if val_accuracy:
                preview_ax.plot(val_accuracy, color=HIGHLIGHT, linewidth=1.6, linestyle="--", label="Val Acc")
            if val_loss:
                preview_ax.plot(val_loss, color=ACCENT_DARK, linewidth=1.6, linestyle="--", label="Val Loss")
            preview_ax.set_facecolor("#fbfdff")
            preview_ax.grid(alpha=0.22, color="#9eb7cf")
            preview_ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
            preview_ax.set_xlabel("Epoch", color=TEXT_SECONDARY, fontsize=8)
            preview_ax.legend(frameon=False, fontsize=7, loc="best")
            preview_ax.spines["top"].set_visible(False)
            preview_ax.spines["right"].set_visible(False)
            preview_ax.spines["left"].set_color(BORDER)
            preview_ax.spines["bottom"].set_color(BORDER)
            preview_figure.tight_layout(pad=1.0)
            self.training_preview_canvas = self.replace_canvas_figure(self.training_preview_canvas, preview_figure)

    def render_confusion_matrix(self):
        self.cm_text.delete("1.0", "end")
        self.cm_text.insert("1.0", np.array2string(confusion_mat))

    def update_prediction_view(self, payload):
        result = payload["result"]
        confidence = payload["confidence"]
        signal = payload["signal"]
        freq = payload["freq"]
        pred = payload["pred"]
        file = payload["file"]

        self.file_var.set(f"Selected File: {file}")
        self.result_var.set(f"Predicted Class: {result} | Confidence: {confidence:.2%}")

        self.render_signal_graphs(signal, freq)
        self.render_prediction_probabilities(pred)

        self.append_log(f"Prediction completed. Predicted class: {result} with confidence {confidence:.2%}.")

    def render_signal_graphs(self, signal, freq):
        figure = Figure(figsize=(8, 3.6), dpi=100, facecolor="#fbfdff")
        time_ax = figure.add_subplot(211)
        freq_ax = figure.add_subplot(212)

        time_ax.plot(signal, color=ACCENT, linewidth=1.8)
        time_ax.set_title("ECG Signal (Time Domain)", color=TEXT_PRIMARY, fontsize=11, fontweight="bold")
        time_ax.grid(alpha=0.22, color="#a7bfd7")

        freq_ax.plot(freq, color=HIGHLIGHT, linewidth=1.8)
        freq_ax.set_title("Frequency Domain", color=TEXT_PRIMARY, fontsize=11, fontweight="bold")
        freq_ax.grid(alpha=0.22, color="#a7bfd7")

        for axis in (time_ax, freq_ax):
            axis.set_facecolor("#fbfdff")
            axis.tick_params(colors=TEXT_SECONDARY)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["left"].set_color(BORDER)
            axis.spines["bottom"].set_color(BORDER)

        figure.tight_layout(pad=1.8)
        self.signal_canvas = self.replace_canvas_figure(self.signal_canvas, figure)

    def render_prediction_probabilities(self, pred):
        self.prediction_text.delete("1.0", "end")
        lines = [f"Class {index}: {probability:.4f}" for index, probability in enumerate(pred)]
        self.prediction_text.insert("1.0", "\n".join(lines))

    def replace_canvas_figure(self, canvas, figure):
        widget = canvas.get_tk_widget()
        parent = widget.master
        manager = widget.winfo_manager()

        if manager == "grid":
            geometry = widget.grid_info()
            widget.destroy()
            new_canvas = FigureCanvasTkAgg(figure, master=parent)
            new_canvas.draw()
            new_canvas.get_tk_widget().grid(**geometry)
            return new_canvas

        geometry = widget.pack_info()
        widget.destroy()
        new_canvas = FigureCanvasTkAgg(figure, master=parent)
        new_canvas.draw()
        new_canvas.get_tk_widget().pack(**geometry)
        return new_canvas


# ======================
# GUI
# ======================
root = tk.Tk()
app = DiseasePredictionApp(root)
root.mainloop()
