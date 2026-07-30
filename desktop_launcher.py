#!/usr/bin/env python3
"""Desktop launcher for the Voice Assistant Streamlit app.

Runs the Streamlit server as a subprocess and opens the default web
browser to the local app URL. When packaged with PyInstaller the
resulting binary can be distributed as a macOS/Windows desktop
application.

Prerequisites (for developers):
- The project's virtual environment must have all dependencies installed
  (`pip install -r requirements.txt`).
- The `streamlit` command is available (installed via the requirements).

Usage (development):
    python desktop_launcher.py

Packaging (one‑file executable):
    pyinstaller --onefile --name voice_assistant desktop_launcher.py

The generated binary can be distributed; it will start the app and open
the browser automatically.
"""

import subprocess
import sys
import time
import webbrowser
import os

def main() -> None:
    # Determine the directory containing this script – the project root.
    project_dir = os.path.abspath(os.path.dirname(__file__))
    # Build the command to run the Streamlit app.
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501"]
    # Launch Streamlit in a subprocess.
    proc = subprocess.Popen(
        cmd,
        cwd=project_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        # Give Streamlit a moment to start up.
        time.sleep(3)
        webbrowser.open("http://localhost:8501")
        # Wait for the subprocess to finish (Ctrl+C will terminate).
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        sys.exit(0)

if __name__ == "__main__":
    main()
