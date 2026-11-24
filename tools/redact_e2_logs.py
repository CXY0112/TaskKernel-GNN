"""Remove embedded URL credentials from solver metadata before displaying logs."""
import re
import sys
from pathlib import Path

for name in sys.argv[1:]:
    path = Path(name)
    paths = [path] if path.is_file() else sorted({file for pattern in ("*.log", "*.txt", "*.out")
                                               for file in path.rglob(pattern) if file.is_file()})
    for file in paths:
        text = file.read_text(encoding="utf-8", errors="replace")
        safe = re.sub(r"(https?://)[^/\s]*@", r"\1[redacted]@", text)
        if safe != text:
            file.write_text(safe, encoding="utf-8")
