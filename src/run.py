"""Interactive launcher for the UVL + YOLOv11 project.

Run:  python src/run.py
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
PY = sys.executable


def launch(script, streamlit=False):
    target = str(SRC / script)
    cmd = [PY, "-m", "streamlit", "run", target] if streamlit else [PY, target]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 Stopped by user.")


MENU = {
    "1": ("Analyze UVL model (FlamaPy/BDD)", lambda: launch("analyze_uvl.py")),
    "2": ("Validate + evaluate a sample configuration", lambda: launch("parse_uvl.py")),
    "3": ("Launch Streamlit dashboard", lambda: launch("app.py", streamlit=True)),
    "4": ("Train the detector", lambda: launch("train.py")),
    "5": ("Run the configuration sweep (experiments)", lambda: launch("finish_experiments.py")),
}

if __name__ == "__main__":
    print("=" * 58)
    print("🛠️  YOLOv11 + UVL Variability Launcher")
    print("=" * 58)
    for k, (label, _) in MENU.items():
        print(f"  {k}) {label}")
    print("  q) Quit")
    choice = input("Enter choice: ").strip().lower()
    if choice in MENU:
        MENU[choice][1]()
    elif choice == "q":
        print("Goodbye!")
    else:
        print("Invalid choice; launching the dashboard.")
        launch("app.py", streamlit=True)
