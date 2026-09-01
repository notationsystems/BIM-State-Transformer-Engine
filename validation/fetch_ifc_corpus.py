"""Fetch the CI-sized, commit-pinned IFC validation corpus and verify its bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


MANIFEST = Path(__file__).with_name("ifc-corpus-v1.json")


def fetch(output_directory: str | Path, *, include_large: bool = False) -> tuple[Path, ...]:
    document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if document.get("format") != "gat-ifc-validation-corpus-v1":
        raise ValueError("unsupported IFC validation corpus manifest")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for model in document["models"]:
        if not model["ci"] and not include_large:
            continue
        destination = model["destination"]
        if Path(destination).name != destination:
            raise ValueError(f"unsafe corpus destination {destination!r}")
        request = urllib.request.Request(
            model["url"],
            headers={"User-Agent": "GAT-IFC-validation/1"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read(int(model["size_bytes"]) + 1)
        actual_digest = hashlib.sha256(payload).hexdigest()
        if len(payload) != model["size_bytes"]:
            raise ValueError(
                f"{model['id']} size mismatch: {len(payload)} != {model['size_bytes']}"
            )
        if actual_digest != model["sha256"]:
            raise ValueError(
                f"{model['id']} digest mismatch: {actual_digest} != {model['sha256']}"
            )
        path = output / destination
        path.write_bytes(payload)
        written.append(path)
        print(f"verified {model['id']}: {len(payload)} bytes, sha256={actual_digest}")
    return tuple(written)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory")
    parser.add_argument(
        "--include-large",
        action="store_true",
        help="also retrieve large models such as the 65 MB Schependomlaan artifact",
    )
    args = parser.parse_args()
    fetch(args.output_directory, include_large=args.include_large)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
