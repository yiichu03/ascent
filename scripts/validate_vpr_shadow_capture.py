#!/usr/bin/env python3
"""Strict technical gate for passive capture; never emit navigation metrics."""

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
        validate_capture_attempt,
    )
    from vpr_shadow_data import sha256
except ImportError:
    from scripts.summarize_submap_screen import (
        load_manifest,
        parse_attempt,
        read_csv,
    )
    from scripts.vpr_shadow_capture_gate import (
        ACTION_HASH_CONTRACT,
        canonical_action_hash,
        load_action_sequences,
        validate_capture_attempt,
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
    parser.add_argument("--action-reference", type=Path)
    parser.add_argument("--control-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _reference(path: Path | None, split_hash: str) -> dict[str, Mapping[str, Any]]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema") != "ascent_v1_4_vpr_shadow_action_reference_v1"
        or value.get("action_hash_contract") != ACTION_HASH_CONTRACT
        or value.get("split_manifest_sha256") != split_hash
    ):
        raise ValueError("invalid sentinel action reference")
    return {str(row["logical_case_id"]): row for row in value["episodes"]}


def main() -> int:
    args = parse_args()
    chunks, logical_order, strict_errors = load_manifest(
        args.manifest, args.expected_episodes, args.expected_dataset
    )
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    split_order = [str(row["logical_case_id"]) for row in split.get("episodes", [])]
    if (
        split.get("dataset") != args.expected_dataset
        or split_order != logical_order
        or int(split.get("episode_count", -1)) != args.expected_episodes
    ):
        strict_errors.append("split_materialized_identity_contract")
    split_hash = sha256(args.split_manifest)
    try:
        reference = _reference(args.action_reference, split_hash)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        reference = {}
        strict_errors.append(f"action_reference:{type(exc).__name__}:{exc}")
    control_summary: dict[str, Any] | None = None
    if args.control_summary is not None:
        try:
            control_summary = json.loads(
                args.control_summary.read_text(encoding="utf-8")
            )
            if (
                control_summary.get("schema")
                != "ascent_v1_4_vpr_shadow_control_gate_v1"
                or control_summary.get("technical_status") != "PASS"
                or control_summary.get("capture_enabled") is not False
                or int(control_summary.get("strict_error_count", -1)) != 0
                or int(control_summary.get("valid_episodes", -1))
                != args.expected_episodes
                or control_summary.get(
                    "historical_action_equivalence_required"
                ) is not False
                or control_summary.get(
                    "same_commit_capture_action_equivalence_required"
                ) is not True
                or control_summary.get("provenance", {}).get("source_commit")
                != args.source_commit
                or control_summary.get("provenance", {}).get(
                    "split_manifest_sha256"
                )
                != split_hash
            ):
                raise ValueError("capture-off control did not pass")
        except (OSError, KeyError, TypeError, ValueError) as exc:
            control_summary = None
            strict_errors.append(f"control_summary:{type(exc).__name__}:{exc}")
    if control_summary is not None:
        if args.action_reference is None:
            strict_errors.append("control_action_reference_missing")
        elif (
            sha256(args.action_reference)
            != control_summary.get("provenance", {}).get(
                "action_reference_sha256"
            )
        ):
            strict_errors.append("control_action_reference_hash")
        else:
            try:
                reference_payload = json.loads(
                    args.action_reference.read_text(encoding="utf-8")
                )
                if (
                    reference_payload.get("behavior_reference")
                    != "same_commit_capture_off_control"
                    or reference_payload.get("source_commit")
                    != args.source_commit
                ):
                    strict_errors.append("control_action_reference_contract")
            except (OSError, TypeError, ValueError) as exc:
                strict_errors.append(
                    f"control_action_reference:{type(exc).__name__}:{exc}"
                )

    attempts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    capture_attempt_count = 0
    ignored_failed_attempt_count = 0
    for registry_order, entry in enumerate(read_csv(args.registry)):
        inventory = Path(entry["inventory"]).resolve()
        if not inventory.is_file():
            strict_errors.append(f"missing_inventory:{inventory}")
            continue
        for row in read_csv(inventory):
            if row.get("stage") != "screen" or row.get("condition") != "B2":
                strict_errors.append(f"unexpected_inventory:{row.get('stage')}:{row.get('condition')}")
                continue
            chunk_id = row.get("chunk_id", "")
            if chunk_id not in chunks:
                strict_errors.append(f"unexpected_chunk:{chunk_id}")
                continue
            if row.get("terminal_class") != "complete":
                ignored_failed_attempt_count += 1
                continue
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
            capture_manifest = Path(row.get("vpr_shadow_manifest", "")).resolve()
            report, capture_errors = validate_capture_attempt(
                capture_manifest=capture_manifest,
                submap_diagnostics=Path(row["submap_diagnostics"]),
                vo_diagnostics=Path(row["vo_diagnostics"]),
                source_commit=args.source_commit,
                dataset=args.expected_dataset,
            )
            strict_errors.extend(
                f"{chunk_id}:{error}" for error in capture_errors
            )
            capture_attempt_count += 1
            try:
                action_sequences, runtime_order = load_action_sequences(
                    Path(row["vo_diagnostics"])
                )
            except (OSError, KeyError, TypeError, ValueError) as exc:
                strict_errors.append(f"{chunk_id}:actions:{type(exc).__name__}:{exc}")
                continue
            sequence_by_runtime = {
                runtime_id: sequence
                for sequence, runtime_id in enumerate(runtime_order)
            }
            episode_frame_counts = report.get("episode_frame_counts", {})
            for logical_id, episode in accepted.items():
                runtime_id = str(episode["runtime_episode_id"])
                actions = action_sequences.get(runtime_id)
                sequence = sequence_by_runtime.get(runtime_id)
                if actions is None or sequence is None:
                    strict_errors.append(f"{logical_id}:missing_action_sequence")
                    continue
                attempts[logical_id].append(
                    {
                        "logical_case_id": logical_id,
                        "chunk_id": chunk_id,
                        "priority": int(row["priority"]),
                        "registry_order": registry_order,
                        "action_count": len(actions),
                        "action_hash": canonical_action_hash(actions),
                        "episode_sequence": sequence,
                        "keyframe_count": int(
                            episode_frame_counts.get(str(sequence), 0)
                        ),
                        "capture_manifest": report.get("capture_manifest"),
                        "capture_manifest_sha256": report.get(
                            "capture_manifest_sha256"
                        ),
                        "vo_diagnostics": str(Path(row["vo_diagnostics"]).resolve()),
                        "submap_diagnostics": str(
                            Path(row["submap_diagnostics"]).resolve()
                        ),
                        "identity_path": str(Path(row["identity_path"]).resolve()),
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
        strict_errors.append(f"missing_capture_episodes:{len(missing)}")
    if reference and set(reference) != set(logical_order):
        strict_errors.append("action_reference_coverage")
    action_mismatches = []
    for logical_id, expected in reference.items():
        observed = selected.get(logical_id)
        if observed is None:
            continue
        if (
            int(expected["action_count"]) != observed["action_count"]
            or str(expected["action_hash"]) != observed["action_hash"]
        ):
            action_mismatches.append(logical_id)
    if action_mismatches:
        strict_errors.append(f"action_equivalence:{len(action_mismatches)}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    episodes_path = output_dir / "episodes.jsonl"
    with episodes_path.open("x", encoding="utf-8") as handle:
        for logical_id in logical_order:
            if logical_id in selected:
                handle.write(
                    json.dumps(selected[logical_id], sort_keys=True) + "\n"
                )
    technical_status = (
        "PASS"
        if not strict_errors and len(selected) == args.expected_episodes
        else "FAIL"
    )
    summary = {
        "schema": "ascent_v1_4_vpr_shadow_capture_gate_v1",
        "technical_status": technical_status,
        "dataset": args.expected_dataset,
        "role": split.get("role"),
        "expected_episodes": args.expected_episodes,
        "valid_episodes": len(selected),
        "capture_attempt_count": capture_attempt_count,
        "ignored_failed_attempt_count": ignored_failed_attempt_count,
        "action_reference_required": args.action_reference is not None,
        "capture_off_control_required": args.control_summary is not None,
        "action_equivalent_episodes": (
            len(reference) - len(action_mismatches) if reference else None
        ),
        "action_mismatch_count": len(action_mismatches),
        "action_hash_contract": ACTION_HASH_CONTRACT,
        "navigation_metrics_emitted": False,
        "association_scores_emitted": False,
        "strict_error_count": len(strict_errors),
        "strict_errors": strict_errors,
        "provenance": {
            "source_commit": args.source_commit,
            "manifest_sha256": sha256(args.manifest),
            "split_manifest_sha256": split_hash,
            "registry_sha256": sha256(args.registry),
            "action_reference_sha256": (
                None if args.action_reference is None else sha256(args.action_reference)
            ),
            "control_summary_sha256": (
                None if args.control_summary is None else sha256(args.control_summary)
            ),
            "historical_action_reference_sha256": (
                None
                if control_summary is None
                else control_summary["provenance"][
                    "historical_action_reference_sha256"
                ]
            ),
            "episodes_sha256": sha256(episodes_path),
        },
    }
    with (output_dir / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
