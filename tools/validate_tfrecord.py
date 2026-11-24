"""Validate TFRecord framing, count complete records, and compute MD5."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


def inspect(path: Path) -> dict[str, object]:
    byte_size = path.stat().st_size
    count = 0
    with path.open("rb") as stream:
        while stream.tell() < byte_size:
            header = stream.read(12)
            if len(header) != 12:
                raise ValueError(f"truncated TFRecord header at byte {stream.tell() - len(header)}")
            length = struct.unpack("<Q", header[:8])[0]
            stream.seek(length + 4, 1)
            if stream.tell() > byte_size:
                raise ValueError(f"record {count} extends beyond end of file")
            count += 1
        if stream.tell() != byte_size:
            raise ValueError("file does not end on a TFRecord boundary")
    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "md5").hexdigest()
    return {"file": path.name, "records": count, "bytes": byte_size, "md5": checksum}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=Path, nargs="+")
    args = parser.parse_args()
    results = [inspect(path) for path in args.files]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
