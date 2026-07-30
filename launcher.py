"""
Desktop app entrypoint.

Starts the Streamlit server in the background and opens it in a native
window via pywebview -- so this behaves like a real desktop app: no
terminal window, no manually running `streamlit run app.py`, no copying a
localhost URL into a browser. Launched by double-clicking
"NCAIR Voice Assistant.app" (see that folder's Contents/MacOS/launcher,
which just activates the venv and runs this file).

If you'd rather run it the original way (in a terminal, in a normal
browser tab), that still works too: `streamlit run app.py`. This file is
an additional way to launch the same app, not a replacement for app.py.
"""

import os
import subprocess
import sys
import time
import urllib.request

import env_config  # loads secrets.env into the environment automatically

env_config.load()

PORT = 8501
URL = f"http://localhost:{PORT}"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def _wait_for_server(timeout=60) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(URL, timeout=1)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", str(PORT),
            "--server.headless", "true",
        ],
        cwd=PROJECT_DIR,
    )

    try:
        if not _wait_for_server():
            print(
                "Streamlit didn't start within 60 seconds. Run "
                "'streamlit run app.py' directly in a terminal from this "
                "folder to see the actual error."
            )
            return

        import webview

        webview.create_window("NCAIR Voice Assistant", URL, width=1150, height=820)
        webview.start()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
