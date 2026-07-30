"""
Loads local secrets (API keys) from secrets.env -- a plain KEY=VALUE file
that lives on your machine only.

This exists because `export GEMINI_API_KEY=...` only applies to the one
terminal session where you typed it -- open a new terminal or double-click
the app and it's gone again, which is exactly the kind of thing that's
caused confusing "it worked yesterday" bugs earlier in this project.
secrets.env is read automatically every time, regardless of how the app
is launched.

Setup: copy secrets.env.example to secrets.env and fill in your real
key(s). secrets.env is for your machine only -- don't share it or include
it when zipping this project to send to your group (it's listed in the
.gitignore-style note in secrets.env.example as a reminder).
"""

import os

_loaded = False


def load(path=None):
    global _loaded
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "secrets.env")

    if not os.path.exists(path):
        return

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if not key or not value:
                continue
            # Don't override a real env var if one was already exported --
            # that should win if the person deliberately set it.
            os.environ.setdefault(key, value)

    _loaded = True


# Load immediately on import, so any module that imports this (app.py,
# launcher.py, download_models.py) gets secrets available right away
# without needing to remember to call load() themselves.
load()
