import os
from datetime import datetime

_logfile = None


def init(log_dir):
    """Initiera loggfilen."""
    global _logfile
    _logfile = os.path.join(log_dir, "pipeline_log.txt")


def _write(level, msg, loud=False):
    """Intern funktion för att skriva loggrader."""
    if _logfile is None:
        raise RuntimeError("Logger not initialized")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [{level}] {msg}"

    # Terminalutskrift
    print(line)
    if loud:
        print("!!! " + msg.upper() + " !!!")

    # Skriv till fil
    with open(_logfile, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def log(msg):
    """INFO-nivå."""
    _write("INFO", msg)


def warn(msg):
    """WARNING-nivå."""
    _write("WARNING", msg)


def error(msg):
    """ERROR-nivå — högt skrik."""
    _write("ERROR", msg, loud=True)


def critical(msg):
    """CRITICAL-nivå — MAX skrik."""
    _write("CRITICAL", msg, loud=True)
