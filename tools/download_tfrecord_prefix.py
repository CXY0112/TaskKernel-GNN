"""Stream a prefix of complete records from a remote TFRecord file."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import urllib.request
from pathlib import Path


def read_exact(stream, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError(f"unexpected end of stream with {remaining} bytes missing")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--records", type=int, required=True)
    args = parser.parse_args()
    if args.records < 1:
        raise ValueError("--records must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    digest = hashlib.md5()
    written = 0
    request = urllib.request.Request(args.url, headers={"User-Agent": "TaskKernel-GNN/0.1"})
    with urllib.request.urlopen(request, timeout=60) as source, temporary.open("wb") as target:
        for index in range(args.records):
            header = read_exact(source, 12)
            length = struct.unpack("<Q", header[:8])[0]
            payload_and_crc = read_exact(source, length + 4)
            record = header + payload_and_crc
            target.write(record)
            digest.update(record)
            written += len(record)
            print(f"record {index + 1}/{args.records}: {written} bytes", flush=True)
    temporary.replace(args.output)
    manifest = {
        "source_url": args.url,
        "records": args.records,
        "bytes": written,
        "md5": digest.hexdigest(),
        "selection": "first complete TFRecord records in official file order",
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
