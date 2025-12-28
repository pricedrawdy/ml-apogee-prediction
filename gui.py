import importlib.util
import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Determine repository root
domain_root = Path(__file__).resolve().parent

# Global references for GUI elements initialized in main()
root: tk.Tk | None = None
ork_path: tk.StringVar | None = None
jdk_path: tk.StringVar | None = None
flight_index_var: tk.StringVar | None = None
plot_time_var: tk.StringVar | None = None
status_var: tk.StringVar | None = None
log_text: tk.Text | None = None
run_tests_button: tk.Button | None = None

test_module = None
figure_frames: dict[str, tk.Frame] = {}
figure_canvases: dict[str, FigureCanvasTkAgg] = {}

# Order here controls layout and labels in the GUI
MODEL_DISPLAY = [
    ("random_forest", "Random Forest"),
    ("linear_regression", "Regression"),
    ("mlp", "MLP"),
]

def browse_ork():
    """Select the OpenRocket .ork file."""
    path = filedialog.askopenfilename(
        title="Select .ork file", filetypes=[("OpenRocket files", "*.ork"), ("All files", "*.*")]
    )
    if path:
        ork_path.set(path)


def browse_jdk():
    """Select the JDK or OpenRocket JAR file."""
    path = filedialog.askopenfilename(
        title="Select JDK/OpenRocket JAR", filetypes=[("Jar files", "*.jar"), ("All files", "*.*")]
    )
    if path:
        jdk_path.set(path)


def append_log(message: str):
    if not log_text:
        return
    log_text.configure(state="normal")
    log_text.insert("end", message + "\n")
    log_text.see("end")
    log_text.configure(state="disabled")


def run_command(cmd, description):
    """Run a subprocess command in a separate thread."""

    def task():
        try:
            result = subprocess.run(cmd, cwd=domain_root, capture_output=True, text=True)
            if result.returncode == 0:
                root.after(0, lambda: append_log(f"✅ {description} completed."))
            else:
                root.after(
                    0,
                    lambda: messagebox.showerror("Error", f"{description} failed:\n{result.stderr}"),
                )
        except Exception as exc:
            root.after(0, lambda: messagebox.showerror("Error", f"{description} failed: {exc}"))

    threading.Thread(target=task, daemon=True).start()


def run_rocketserializer():
    """Execute the rocket serializer to convert .ork to parameters.json."""
    ork = ork_path.get()
    jdk = jdk_path.get()
    if not ork or not jdk:
        messagebox.showwarning("Missing Paths", "Please select both the .ork file and the JDK/OpenRocket JAR path.")
        return
    output_dir = domain_root / "rocket-info"
    output_dir.mkdir(exist_ok=True)
    cmd = [
        "ork2json",
        "--filepath",
        ork,
        "--output",
        str(output_dir),
        "--ork_jar",
        jdk,
    ]
    run_command(cmd, "RocketSerializer")


def run_batch_simulation():
    cmd = [sys.executable, str(domain_root / "scripts" / "batch_simulation_creation.py")]
    run_command(cmd, "Batch Simulation Creation")


def run_sliding_window():
    cmd = [sys.executable, str(domain_root / "scripts" / "sliding_window_generator_v2.py")]
    run_command(cmd, "Sliding Window Generator")


def run_model_creation():
    cmd = [sys.executable, str(domain_root / "models" / "apogee_prediction_model_v1.py")]
    run_command(cmd, "Model Creation")


def _load_test_module():
    global test_module
    if test_module is not None:
        return test_module

    module_path = domain_root / "scripts" / "apogee_prediction_test_v1.1.py"
    spec = importlib.util.spec_from_file_location("apogee_prediction_test_v1_1", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load apogee_prediction_test_v1.1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    test_module = module
    return module


def _render_figure(model_key: str, figure):
    frame = figure_frames.get(model_key)
    if not frame:
        return

    # Clear existing content
    for widget in frame.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(figure, master=frame)
    canvas.draw()
    widget = canvas.get_tk_widget()
    widget.pack(fill="both", expand=True)

    if model_key in figure_canvases:
        old_canvas = figure_canvases[model_key]
        try:
            old_canvas.get_tk_widget().destroy()
        except Exception:
            pass
    figure_canvases[model_key] = canvas


def run_testing_script():
    """Run the prediction evaluation and embed plots for all models."""
    if not status_var:
        return

    try:
        flight_idx = int(flight_index_var.get() or 0)
    except ValueError:
        messagebox.showerror("Invalid Input", "Flight index must be an integer.")
        return

    try:
        max_plot_time = float(plot_time_var.get() or 10.0)
    except ValueError:
        messagebox.showerror("Invalid Input", "Plot max time must be a number.")
        return

    status_var.set("Running predictions...")
    append_log(f"Starting evaluations for flight {flight_idx}...")
    if run_tests_button:
        run_tests_button.config(state="disabled")

    def task():
        try:
            module = _load_test_module()
            ctx = module._build_context(
                timestep=module.DEFAULT_TIMESTEP,
                window_duration=module.DEFAULT_WINDOW_DURATION,
                stride_duration=module.DEFAULT_STRIDE_DURATION,
                total_flight_time=module.DEFAULT_TOTAL_FLIGHT_TIME,
            )

            results = {}
            for model_key, _ in MODEL_DISPLAY:
                if model_key not in module.MODEL_FILENAMES:
                    continue
                res = module.evaluate_model(
                    model_key,
                    flight_index=flight_idx,
                    plot_max_time=max_plot_time,
                    context=ctx,
                    show_plot=False,
                )
                results[model_key] = res
        except Exception as exc:
            root.after(0, lambda: messagebox.showerror("Error", f"Prediction run failed: {exc}"))
            root.after(0, lambda: status_var.set("Run failed."))
            root.after(
                0, lambda: run_tests_button.config(state="normal") if run_tests_button else None  # type: ignore[arg-type]
            )
            return

        def update_ui():
            for model_key, result in results.items():
                _render_figure(model_key, result["figure"])
                model_label = result["model_type"].replace("_", " ").title()
                append_log(
                    f"{model_label}: RMSE {result['rmse']:.2f} m | "
                    f"MAE {result['mean_abs_error']:.2f} m | Max {result['max_error']:.2f} m"
                )
            status_var.set("Finished. Plots updated.")
            if run_tests_button:
                run_tests_button.config(state="normal")

        root.after(0, update_ui)

    threading.Thread(target=task, daemon=True).start()


def _build_plot_grid(parent: tk.Widget):
    plots_frame = tk.LabelFrame(parent, text="Apogee Prediction Plots")
    plots_frame.grid(row=5, column=0, columnspan=3, padx=8, pady=8, sticky="nsew")
    plots_frame.columnconfigure((0, 1, 2), weight=1)
    plots_frame.rowconfigure(0, weight=1)

    for idx, (model_key, label) in enumerate(MODEL_DISPLAY):
        frame = tk.LabelFrame(plots_frame, text=label)
        frame.grid(row=0, column=idx, padx=5, pady=5, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        placeholder = tk.Label(frame, text="Run predictions to render plot", fg="gray")
        placeholder.pack(expand=True, fill="both", padx=10, pady=10)
        figure_frames[model_key] = frame


def main() -> None:
    """Create the GUI and start the Tkinter main loop."""
    global root, ork_path, jdk_path, flight_index_var, plot_time_var, status_var, log_text, run_tests_button

    root = tk.Tk()
    root.title("Apogee Prediction Toolkit")
    root.rowconfigure(5, weight=1)
    root.rowconfigure(6, weight=1)
    root.columnconfigure((0, 1, 2), weight=1)

    # Initialize variables after creating root window
    ork_path = tk.StringVar()
    jdk_path = tk.StringVar()
    status_var = tk.StringVar(value="Ready")

    # Attempt to load defaults from the test module
    try:
        module = _load_test_module()
        flight_index_var = tk.StringVar(value=str(module.DEFAULT_FLIGHT_INDEX))
        plot_time_var = tk.StringVar(value=str(module.DEFAULT_PLOT_MAX_TIME))
    except Exception:
        flight_index_var = tk.StringVar(value="0")
        plot_time_var = tk.StringVar(value="10.0")

    # Build GUI widgets
    tk.Label(root, text=".ork File").grid(row=0, column=0, padx=5, pady=5, sticky="e")
    tk.Entry(root, textvariable=ork_path, width=50).grid(row=0, column=1, padx=5, pady=5, sticky="we")
    tk.Button(root, text="Browse", command=browse_ork).grid(row=0, column=2, padx=5, pady=5, sticky="w")

    tk.Label(root, text="JDK / OpenRocket JAR").grid(row=1, column=0, padx=5, pady=5, sticky="e")
    tk.Entry(root, textvariable=jdk_path, width=50).grid(row=1, column=1, padx=5, pady=5, sticky="we")
    tk.Button(root, text="Browse", command=browse_jdk).grid(row=1, column=2, padx=5, pady=5, sticky="w")

    # Action buttons
    tk.Button(root, text="Run RocketSerializer", command=run_rocketserializer).grid(
        row=2, column=0, columnspan=1, padx=5, pady=10, sticky="we"
    )
    tk.Button(root, text="Batch Simulation Creation", command=run_batch_simulation).grid(
        row=2, column=1, padx=5, pady=10, sticky="we"
    )
    tk.Button(root, text="Sliding Window Generator", command=run_sliding_window).grid(
        row=2, column=2, padx=5, pady=10, sticky="we"
    )
    tk.Button(root, text="Train ML Models", command=run_model_creation).grid(
        row=3, column=0, padx=5, pady=5, sticky="we"
    )

    # Prediction controls
    tk.Label(root, text="Flight Index").grid(row=3, column=1, padx=5, pady=5, sticky="e")
    tk.Entry(root, textvariable=flight_index_var, width=8).grid(row=3, column=2, padx=5, pady=5, sticky="w")

    tk.Label(root, text="Plot Max Time (s)").grid(row=4, column=1, padx=5, pady=5, sticky="e")
    tk.Entry(root, textvariable=plot_time_var, width=8).grid(row=4, column=2, padx=5, pady=5, sticky="w")

    run_tests_button = tk.Button(root, text="Run Apogee Tests (MLP / RF / Regression)", command=run_testing_script)
    run_tests_button.grid(row=4, column=0, padx=5, pady=5, sticky="we")

    _build_plot_grid(root)

    # Status/log area
    status_frame = tk.LabelFrame(root, text="Status")
    status_frame.grid(row=6, column=0, columnspan=3, padx=8, pady=5, sticky="nsew")
    status_frame.columnconfigure(0, weight=1)
    status_frame.rowconfigure(1, weight=1)

    tk.Label(status_frame, textvariable=status_var).grid(row=0, column=0, sticky="w", padx=5, pady=2)
    log_text = tk.Text(status_frame, height=6, state="disabled")
    log_text.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

    root.mainloop()


if __name__ == "__main__":
    main()
