#!/usr/bin/env python3
"""Prove that post-capture code changes are offline-only and hash-bind them."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Sequence


ALLOWED_CHANGED_PATHS = frozenset(
    {
        "pbs/run_vpr_shadow_offline.pbs",
        "pbs/submit_vpr_shadow_offline.sh",
        "experiments/vpr_shadow/model_registry.json",
        "scripts/check_vpr_shadow_models.py",
        "scripts/check_vpr_shadow_offline_compatibility.py",
        "scripts/run_vpr_shadow_association.py",
        "scripts/run_vpr_shadow_offline_batch.py",
        "scripts/rescore_vpr_shadow_batch.py",
        "scripts/validate_vpr_shadow_capture.py",
        "scripts/vpr_shadow_capture_gate.py",
        "scripts/vpr_shadow_data.py",
        "scripts/vpr_shadow_models.py",
        "tests/test_vpr_shadow_capture_gate.py",
        "tests/test_vpr_shadow_data.py",
        "tests/test_vpr_shadow_offline_batch.py",
    }
)
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def classify_changed_paths(paths: Sequence[str]) -> list[str]:
    """Return paths that exceed the frozen offline-only compatibility scope."""

    return sorted(set(str(path) for path in paths) - ALLOWED_CHANGED_PATHS)


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build_report(
    *, source_root: Path, capture_commit: str, offline_commit: str
) -> dict[str, object]:
    source_root = source_root.resolve()
    errors: list[str] = []
    if not COMMIT_RE.fullmatch(capture_commit):
        errors.append("invalid_capture_commit")
    if not COMMIT_RE.fullmatch(offline_commit):
        errors.append("invalid_offline_commit")
    if errors:
        return {
            "schema": "ascent_v1_4_vpr_shadow_offline_compatibility_v1",
            "technical_status": "FAIL",
            "capture_source_commit": capture_commit,
            "offline_source_commit": offline_commit,
            "changed_paths": [],
            "forbidden_changed_paths": [],
            "errors": errors,
        }

    for label, commit in (
        ("capture", capture_commit),
        ("offline", offline_commit),
    ):
        if _git(source_root, "cat-file", "-e", f"{commit}^{{commit}}", check=False).returncode:
            errors.append(f"unknown_{label}_commit")
    ancestor = False
    if not errors:
        ancestor = (
            _git(
                source_root,
                "merge-base",
                "--is-ancestor",
                capture_commit,
                offline_commit,
                check=False,
            ).returncode
            == 0
        )
        if not ancestor:
            errors.append("capture_commit_not_ancestor")

    changed_paths: list[str] = []
    forbidden: list[str] = []
    diff_bytes = b""
    offline_tree = None
    if not errors:
        changed_paths = sorted(
            value
            for value in _git(
                source_root,
                "diff",
                "--name-only",
                "--diff-filter=ACDMRTUXB",
                f"{capture_commit}..{offline_commit}",
            ).stdout.splitlines()
            if value
        )
        forbidden = classify_changed_paths(changed_paths)
        if forbidden:
            errors.append("changed_path_outside_offline_allowlist")
        diff_bytes = _git(
            source_root,
            "diff",
            "--binary",
            f"{capture_commit}..{offline_commit}",
            "--",
            *changed_paths,
        ).stdout.encode("utf-8")
        offline_tree = _git(
            source_root, "rev-parse", f"{offline_commit}^{{tree}}"
        ).stdout.strip()

    paths_payload = json.dumps(
        changed_paths, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return {
        "schema": "ascent_v1_4_vpr_shadow_offline_compatibility_v1",
        "technical_status": "PASS" if not errors else "FAIL",
        "capture_source_commit": capture_commit,
        "offline_source_commit": offline_commit,
        "capture_is_ancestor": ancestor,
        "offline_tree": offline_tree,
        "changed_paths": changed_paths,
        "changed_paths_sha256": _sha256_bytes(paths_payload),
        "diff_sha256": _sha256_bytes(diff_bytes),
        "forbidden_changed_paths": forbidden,
        "capture_or_planner_paths_changed": any(
            path.startswith(("ascent/", "configs/", "experiments/vpr_shadow/manifests/"))
            for path in changed_paths
        ),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--capture-commit", required=True)
    parser.add_argument("--offline-commit", required=True)
    args = parser.parse_args()
    report = build_report(
        source_root=args.source_root,
        capture_commit=args.capture_commit,
        offline_commit=args.offline_commit,
    )
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0 if report["technical_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
