"""
Logs every completed task to completed_tasks/ -- one JSON file per task,
plus a running tasks_log.jsonl for easy iteration.
"""

import json
import os
import uuid
from datetime import datetime

TASKS_DIR = "completed_tasks"
LOG_FILE = os.path.join(TASKS_DIR, "tasks_log.jsonl")


def save_task(task: dict) -> str:
    """
    Save a completed task. Adds an id and timestamp if not already present.
    Returns the task id.
    """
    os.makedirs(TASKS_DIR, exist_ok=True)

    task = dict(task)  # don't mutate the caller's dict
    task.setdefault("id", str(uuid.uuid4())[:8])
    task.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))

    # One JSON file per task, named by id.
    task_path = os.path.join(TASKS_DIR, f"task_{task['id']}.json")
    with open(task_path, "w", encoding="utf-8") as f:
        json.dump(task, f, indent=2, ensure_ascii=False)

    # Also append to the running log for easy scanning.
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(task, ensure_ascii=False) + "\n")

    return task["id"]


def load_all_tasks() -> list:
    """Load every completed task, in the order they were logged."""
    if not os.path.exists(LOG_FILE):
        return []

    tasks = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks
