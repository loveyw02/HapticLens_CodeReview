import datetime
import atexit
from pathlib import Path

class EventLogger:
    """Simple event logger that writes timestamped lines to a file."""

    def __init__(self, path: str | Path = "gui_events.log"):
        self.file = open(path, "a", encoding="utf-8")
        atexit.register(self.close)  # Ensure file is closed on exit

    def log(self, event_type: str, data: str = "") -> None:
        timestamp = datetime.datetime.now().isoformat()
        self.file.write(f"{timestamp} | {event_type} | {data}\n")
        self.file.flush()

    def close(self) -> None:
        if not self.file.closed:
            self.file.close()
