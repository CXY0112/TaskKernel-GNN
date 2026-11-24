"""Stop only this project's active reference runners and their descendants.

Completed NPZ exports and unfinished working directories are preserved. Used
to relocate reference generation from a slow WSL Windows mount to Linux disk.
"""
import os
from pathlib import Path
import signal
import time

processes = {}
for entry in Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    try:
        args = (entry / "cmdline").read_bytes().split(b"\0")
        status = (entry / "status").read_text()
        ppid = int(next(line.split()[1] for line in status.splitlines() if line.startswith("PPid:")))
        processes[int(entry.name)] = (ppid, args)
    except (FileNotFoundError, PermissionError, StopIteration):
        continue
project_root = Path(__file__).resolve().parents[1]
roots = []
for pid, (_, args) in processes.items():
    if not (args and Path(os.fsdecode(args[0])).name.startswith("python")
            and b"tools/run_opencarp_plan.py" in args):
        continue
    try:
        if (Path("/proc") / str(pid) / "cwd").resolve() == project_root:
            roots.append(pid)
    except FileNotFoundError:
        pass
for root in roots:
    os.kill(root, signal.SIGSTOP)
    children = {root}
    while True:
        expanded = children | {pid for pid,(parent,_) in processes.items() if parent in children}
        if expanded == children:
            break
        children = expanded
    for pid in sorted(children - {root}, reverse=True):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    os.kill(root, signal.SIGTERM)
    os.kill(root, signal.SIGCONT)
    print(f"Stopped project reference runner {root}; preserved exports and working directories", flush=True)
