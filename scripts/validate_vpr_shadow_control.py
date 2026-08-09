#!/usr/bin/env python3
"""Gate a capture-off sentinel and emit only its action-hash reference."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

try:
    from summarize_submap_screen import load_manifest, parse_attempt, read_csv
    from vpr_shadow_capture_gate import (
        ACTION_HASH_CONTRACT,
        canonical_action_hash,
        load_action_sequences,
    )
    from vpr_shadow_data import sha256
except ImportError:
    from scripts.summarize_submap_screen import load_manifest, parse_attempt, read_csv
    from scripts.vpr_shadow_capture_gate import (
        ACTION_HASH_CONTRACT,
        canonical_action_hash,
        load_action_sequences,
    )
    from scripts.vpr_shadow_data import sha256


FORWARD_SHA256 = "6b571bb717366f7d80f61e919b33a45ac2f45925201c4e3011b3239a2c42e586"
TURN_SHA256 = "c469643f9ab35c9e1058f31fbb672a5fa3adf582987a4388bdd020dd89faf1d9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--expected-dataset", choices=("hm3d", "mp3d"), required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--historical-action-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load_historical(path: Path, split_hash: str) -> dict[str, Mapping[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema") != "ascent_v1_4_vpr_shadow_action_reference_v1"
        or value.get("action_hash_contract") != ACTION_HASH_CONTRACT
        or value.get("split_manifest_sha256") != split_hash
        or value.get("behavior_reference") != "frozen_submap_v1.2"
    ):
        raise ValueError("invalid historical v1.2 action reference")
    return {str(row["logical_case_id"]): row for row in value["episodes"]}


def main() -> int:
    args = parse_args()
    chunks, logical_order, strict_errors = load_manifest(
        args.manifest, args.expected_episodes, args.expected_dataset
    )
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    split_hash = sha256(args.split_manifest)
    if (
        split.get("role") != "engineering_sentinel"
        or split.get("dataset") != args.expected_dataset
        or [str(row["logical_case_id"]) for row in split.get("episodes", [])]
        != logical_order
    ):
        strict_errors.append("control_split_contract")
    try:
        historical = _load_historical(
            args.historical_action_reference, split_hash
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        historical = {}
        strict_errors.append(f"historical_reference:{type(exc).__name__}:{exc}")

    attempts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ignored_failed_attempt_count = 0
    for registry_order, entry in enumerate(read_csv(args.registry)):
        inventory = Path(entry["inventory"]).resolve()
        if not inventory.is_file():
            strict_errors.append(f"missing_inventory:{inventory}")
            continue
        for row in read_csv(inventory):
            chunk_id = row.get("chunk_id", "")
            if (
                row.get("stage") != "screen"
                or row.get("condition") != "B2"
                or chunk_id not in chunks
            ):
                strict_errors.append(f"unexpected_inventory:{chunk_id}")
                continue
            if row.get("terminal_class") != "complete":
                ignored_failed_attempt_count += 1
                continue
            if any(
                int(row.get(key, -1)) != 0
                for key in (
                    "vpr_metadata_count",
                    "vpr_reset_count",
                    "vpr_keyframe_count",
                    "vpr_parse_error_count",
                    "vpr_unknown_count",
                )
            ) or row.get("vpr_shadow_manifest"):
                strict_errors.append(f"control_contains_capture:{chunk_id}")
            accepted, parse_errors = parse_attempt(
                row,
                chunk=chunks[chunk_id],
                mode="full",
                source_commit=args.source_commit,
                forward_sha256=FORWARD_SHA256,
                turn_sha256=TURN_SHA256,
                calibration=None,
            )
            strict_errors.extend(parse_errors)
            try:
                actions, _ = load_action_sequences(Path(row["vo_diagnostics"]))
            except (OSError, KeyError, TypeError, ValueError) as exc:
                strict_errors.append(f"{chunk_id}:actions:{type(exc).__name__}:{exc}")
                continue
            for logical_id, episode in accepted.items():
                runtime_id = str(episode["runtime_episode_id"])
                sequence = actions.get(runtime_id)
                if sequence is None:
                    strict_errors.append(f"{logical_id}:missing_actions")
                    continue
                attempts[logical_id].append(
                    {
                        "logical_case_id": logical_id,
                        "chunk_id": chunk_id,
                        "priority": int(row["priority"]),
                        "registry_order": registry_order,
                        "action_count": len(sequence),
                        "action_hash": canonical_action_hash(sequence),
                    }
                )

    selected: dict[str, dict[str, Any]] = {}
    for logical_id, values in attempts.items():
        values.sort(key=lambda row: (row["priority"], row["registry_order"]))
        priorities = Counter(row["priority"] for row in values)
        if any(count > 1 for count in priorities.values()):
            strict_errors.append(f"duplicate_episode_priority:{logical_id}")
        selected[logical_id] = values[0]
    missing = [logical_id for logical_id in logical_order if logical_id not in selected]
    if missing:
        strict_errors.append(f"missing_control_episodes:{len(missing)}")
    if set(historical) != set(logical_order):
        strict_errors.append("historical_reference_coverage")
    mismatches = [
        logical_id
        for logical_id in logical_order
        if logical_id in selected
        and logical_id in historical
        and (
            selected[logical_id]["action_count"]
            != int(historical[logical_id]["action_count"])
            or selected[logical_id]["action_hash"]
            != str(historical[logical_id]["action_hash"])
        )
    ]
    if mismatches:
        strict_errors.append(f"historical_action_equivalence:{len(mismatches)}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    reference_entries = [
        {
            "logical_case_id": logical_id,
            "chunk_id": selected[logical_id]["chunk_id"],
            "action_count": selected[logical_id]["action_count"],
            "action_hash": selected[logical_id]["action_hash"],
        }
        for logical_id in logical_order
        if logical_id in selected
    ]
    reference = {
        "schema": "ascent_v1_4_vpr_shadow_action_reference_v1",
        "role": "engineering_sentinel",
        "behavior_reference": "same_commit_capture_off_control",
        "action_hash_contract": ACTION_HASH_CONTRACT,
        "source_commit": args.source_commit,
        "split_manifest_sha256": split_hash,
        "episode_count": len(reference_entries),
        "episodes": reference_entries,
    }
    reference_path = output_dir / "action_reference.json"
    with reference_path.open("x", encoding="utf-8") as handle:
        json.dump(reference, handle, indent=2, sort_keys=True)
        handle.write("\n")
    technical_status = (
        "PASS"
        if not strict_errors and len(selected) == args.expected_episodes
        else "FAIL"
    )
    summary = {
        "schema": "ascent_v1_4_vpr_shadow_control_gate_v1",
        "technical_status": technical_status,
        "dataset": args.expected_dataset,
        "role": "engineering_sentinel",
        "expected_episodes": args.expected_episodes,
        "valid_episodes": len(selected),
        "capture_enabled": False,
        "historical_reference_required": True,
        "historical_action_equivalent_episodes": (
            len(historical) - len(mismatches) if historical else 0
        ),
        "historical_action_mismatch_count": len(mismatches),
        "navigation_metrics_emitted": False,
        "association_scores_emitted": False,
        "ignored_failed_attempt_count": ignored_failed_attempt_count,
        "strict_error_count": len(strict_errors),
        "strict_errors": strict_errors,
        "provenance": {
            "source_commit": args.source_commit,
            "manifest_sha256": sha256(args.manifest),
            "split_manifest_sha256": split_hash,
            "registry_sha256": sha256(args.registry),
            "historical_action_reference_sha256": sha256(
                args.historical_action_reference
            ),
            "action_reference_sha256": sha256(reference_path),
        },
    }
    with (output_dir / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
