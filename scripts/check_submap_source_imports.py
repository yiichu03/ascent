#!/usr/bin/env python3
"""Fail closed unless runtime imports resolve to the candidate checkout.

The PBS worker intentionally keeps its current working directory at the frozen
resource checkout because ASCENT model paths are relative to that directory.
Candidate code must nevertheless come from ``SOURCE_ROOT``.  This check is
executed from the same working directory and with the same ``PYTHONPATH`` as
the model services and navigation process.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def resolved_path(entry: str) -> Path:
    return Path(entry or ".").resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--resource-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("modules", nargs="+")
    args = parser.parse_args()

    source_root = args.source_root.resolve(strict=True)
    resource_root = args.resource_root.resolve(strict=True)
    if source_root == resource_root:
        raise SystemExit("candidate source and resource roots must differ")

    search_path = [resolved_path(entry) for entry in sys.path]
    if source_root not in search_path:
        raise SystemExit(f"candidate source missing from sys.path: {source_root}")
    source_index = search_path.index(source_root)
    resource_indices = [
        index
        for index, path in enumerate(search_path)
        if path == resource_root
    ]
    if resource_indices and min(resource_indices) < source_index:
        raise SystemExit(
            "frozen resource checkout precedes candidate source on "
            f"sys.path: resource={min(resource_indices)} "
            f"source={source_index}"
        )

    origins: dict[str, str] = {}
    for module in args.modules:
        spec = importlib.util.find_spec(module)
        if spec is None or spec.origin is None:
            raise SystemExit(f"module has no import origin: {module}")
        origin = Path(spec.origin).resolve(strict=True)
        if source_root not in origin.parents:
            raise SystemExit(
                f"module does not resolve inside candidate source: "
                f"{module} -> {origin}"
            )
        if resource_root in origin.parents:
            raise SystemExit(
                f"module resolves inside frozen resource checkout: "
                f"{module} -> {origin}"
            )
        origins[module] = str(origin)

    output = {
        "schema": "ascent_vo_submap_source_import_check_v1",
        "status": "PASS",
        "source_root": str(source_root),
        "resource_root": str(resource_root),
        "cwd": str(Path.cwd().resolve()),
        "sys_path": [str(path) for path in search_path],
        "source_path_index": source_index,
        "resource_path_indices": resource_indices,
        "module_origins": origins,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
