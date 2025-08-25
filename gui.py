import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import threading
from pathlib import Path
import os

# Determine repository root
domain_root = Path(__file__).resolve().parent

# Variables to hold file paths
ork_path = tk.StringVar()
jdk_path = tk.StringVar()


def browse_ork():
    """Select the OpenRocket .ork file."""
    path = filedialog.askopenfilename(title="Select .ork file", filetypes=[("OpenRocket files", "*.ork"), ("All files", "*.*")])
    if path:
        ork_path.set(path)


def browse_jdk():
    """Select the JDK or OpenRocket JAR file."""
    path = filedialog.askopenfilename(title="Select JDK/OpenRocket JAR", filetypes=[("Jar files", "*.jar"), ("All files", "*.*")])
    if path:
        jdk_path.set(path)


def run_command(cmd, description):
    """Run a subprocess command in a separate thread."""
    def task():
        try:
            result = subprocess.run(cmd, cwd=domain_root, capture_output=True, text=True)
            if result.returncode == 0:
                root.after(0, lambda: messagebox.showinfo("Success", f"{description} completed successfully."))
            else:
                root.after(0, lambda: messagebox.showerror("Error", f"{description} failed:\n{result.stderr}"))
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
    cmd = ["python", str(domain_root / "scripts" / "batch_simulation_creation.py")]
    run_command(cmd, "Batch Simulation Creation")


def run_sliding_window():
    cmd = ["python", str(domain_root / "scripts" / "sliding_window_generator_v2.py")]
    run_command(cmd, "Sliding Window Generator")


def run_model_creation():
    cmd = ["python", str(domain_root / "models" / "apogee_prediction_model_v1.py")]
    run_command(cmd, "Model Creation")


def run_testing_script():
    cmd = ["python", str(domain_root / "scripts" / "apogee_prediction_test_v1.1.py")]
    run_command(cmd, "Testing Script")


# Build GUI
root = tk.Tk()
root.title("Apogee Prediction Toolkit")

# File selection widgets
tk.Label(root, text=".ork File").grid(row=0, column=0, padx=5, pady=5, sticky="e")
tk.Entry(root, textvariable=ork_path, width=50).grid(row=0, column=1, padx=5, pady=5)
tk.Button(root, text="Browse", command=browse_ork).grid(row=0, column=2, padx=5, pady=5)

tk.Label(root, text="JDK / OpenRocket JAR").grid(row=1, column=0, padx=5, pady=5, sticky="e")
tk.Entry(root, textvariable=jdk_path, width=50).grid(row=1, column=1, padx=5, pady=5)
tk.Button(root, text="Browse", command=browse_jdk).grid(row=1, column=2, padx=5, pady=5)

# Action buttons
tk.Button(root, text="Run RocketSerializer", command=run_rocketserializer).grid(row=2, column=0, columnspan=3, padx=5, pady=10, sticky="we")
tk.Button(root, text="Run Batch Simulation Creation", command=run_batch_simulation).grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky="we")
tk.Button(root, text="Run Sliding Window Generator", command=run_sliding_window).grid(row=4, column=0, columnspan=3, padx=5, pady=5, sticky="we")
tk.Button(root, text="Run ML Model Creation", command=run_model_creation).grid(row=5, column=0, columnspan=3, padx=5, pady=5, sticky="we")
tk.Button(root, text="Run Testing Script", command=run_testing_script).grid(row=6, column=0, columnspan=3, padx=5, pady=5, sticky="we")

root.mainloop()
