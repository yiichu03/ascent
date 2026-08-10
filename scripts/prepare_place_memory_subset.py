#!/usr/bin/env python3
"""Prepare an outcome-blind, high-association HM3D mechanism subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


AUDIT_REQUIRED_FIELDS = {
    "dataset",
    "arm",
    "case_id",
    "first_graph_nonadjacent_vo_consistent_return_step",
}
SELECTION_FORBIDDEN_FIELDS = frozenset(
    {"success", "spl", "action_steps", "success_delta", "spl_delta"}
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[List[Dict[str, str]], List[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        return [dict(row) for row in reader], fields


def write_csv(
    path: Path, rows: Iterable[Mapping[str, object]], fields: Sequence[str]
) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def select_ids(
    audit_rows: Sequence[Mapping[str, str]],
    canonical_by_id: Mapping[str, Mapping[str, str]],
    *,
    count: int,
    max_event_step: int,
) -> tuple[List[str], Dict[str, int]]:
    """Select using identity, event timing, and scene only—never outcomes."""

    candidates: List[tuple[int, str, str]] = []
    for row in audit_rows:
        # Keep the exact access list explicit.  Outcome columns may coexist in
        # the source CSV but cannot participate in this function.
        dataset = row.get("dataset", "")
        arm = row.get("arm", "")
        logical_id = row.get("case_id", "")
        event_text = row.get(
            "first_graph_nonadjacent_vo_consistent_return_step", ""
        )
        if dataset != "hm3d" or arm != "submap_v1_2" or not event_text:
            continue
        canonical = canonical_by_id.get(logical_id)
        if canonical is None:
            continue
        event_step = int(float(event_text))
        if event_step > max_event_step:
            continue
        candidates.append((event_step, logical_id, canonical["scene_id"]))
    candidates.sort(key=lambda item: (item[0], item[1]))

    by_scene: Dict[str, List[tuple[int, str]]] = defaultdict(list)
    for event_step, logical_id, scene_id in candidates:
        by_scene[scene_id].append((event_step, logical_id))
    scene_order = sorted(
        by_scene,
        key=lambda scene: (
            by_scene[scene][0][0],
            by_scene[scene][0][1],
            scene,
        ),
    )
    selected: List[str] = []
    round_index = 0
    while len(selected) < count:
        added = False
        for scene in scene_order:
            if round_index < len(by_scene[scene]):
                selected.append(by_scene[scene][round_index][1])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        round_index += 1
    if len(selected) != count:
        raise ValueError(
            f"only {len(selected)} eligible episodes for requested {count}"
        )
    event_by_id = {logical_id: step for step, logical_id, _ in candidates}
    return selected, event_by_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-episodes", type=Path, required=True)
    parser.add_argument("--canonical-selection", type=Path, required=True)
    parser.add_argument("--v12-episodes", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--max-event-step", type=int, default=300)
    parser.add_argument("--chunk-size", type=int, default=10)
    args = parser.parse_args()
    if args.count < 1 or args.max_event_step < 1 or args.chunk_size < 1:
        raise SystemExit("count, max-event-step, and chunk-size must be positive")

    audit_path = args.audit_episodes.resolve()
    canonical_path = args.canonical_selection.resolve()
    baseline_path = args.v12_episodes.resolve()
    for path in (audit_path, canonical_path, baseline_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    audit_rows, audit_fields = read_csv(audit_path)
    missing = AUDIT_REQUIRED_FIELDS.difference(audit_fields)
    if missing:
        raise ValueError(f"audit missing fields: {sorted(missing)}")
    canonical_rows, canonical_fields = read_csv(canonical_path)
    canonical_by_id = {
        row["logical_case_id"]: row for row in canonical_rows
    }
    if len(canonical_by_id) != len(canonical_rows):
        raise ValueError("canonical logical_case_id is not unique")

    # The selected IDs are frozen before the historical score file is opened.
    selected_ids, event_by_id = select_ids(
        audit_rows,
        canonical_by_id,
        count=args.count,
        max_event_step=args.max_event_step,
    )
    selected_rows = []
    selection_rule = (
        "hm3d_val1000_submap_v1_2_graph_nondirect_vo_consistent_"
        f"by_step_{args.max_event_step}_scene_round_robin_no_outcome"
    )
    for index, logical_id in enumerate(selected_ids):
        row = dict(canonical_by_id[logical_id])
        row.update(
            {
                "manifest_version": (
                    "ascent_vo_submap_v1_4_oracle_mechanism_subset_v1"
                ),
                "selection_rule": selection_rule,
                "dataset_case_index": str(index),
                "chunk_id": (
                    f"hm3d_v14_oracle_mechanism_c"
                    f"{index // args.chunk_size:03d}"
                ),
                "selection_event_step": str(event_by_id[logical_id]),
            }
        )
        selected_rows.append(row)

    baseline_rows, baseline_fields = read_csv(baseline_path)
    baseline_by_id = {
        row["logical_case_id"]: row for row in baseline_rows
    }
    if len(baseline_by_id) != len(baseline_rows):
        raise ValueError("v1.2 baseline logical_case_id is not unique")
    missing_baseline = [
        logical_id for logical_id in selected_ids if logical_id not in baseline_by_id
    ]
    if missing_baseline:
        raise ValueError(f"v1.2 baseline missing {len(missing_baseline)} IDs")
    selected_baseline = [baseline_by_id[item] for item in selected_ids]

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    selection_output = output_root / "selection.csv"
    baseline_output = output_root / "v1_2_baseline.csv"
    audit_output = output_root / "preparation_audit.json"
    output_selection_fields = list(canonical_fields)
    if "selection_event_step" not in output_selection_fields:
        output_selection_fields.append("selection_event_step")
    write_csv(selection_output, selected_rows, output_selection_fields)
    write_csv(baseline_output, selected_baseline, baseline_fields)

    scene_count = len({row["scene_id"] for row in selected_rows})
    selected_steps = [event_by_id[item] for item in selected_ids]
    audit = {
        "schema": "ascent_vo_submap_v1_4_oracle_subset_preparation_v1",
        "status": "PASS",
        "dataset": "hm3d",
        "source_arm": "submap_v1_2",
        "selection_rule": selection_rule,
        "metrics_used_for_selection": False,
        "forbidden_selection_fields": sorted(SELECTION_FORBIDDEN_FIELDS),
        "selected_episode_count": len(selected_ids),
        "selected_scene_count": scene_count,
        "max_event_step": args.max_event_step,
        "selected_event_step_min": min(selected_steps),
        "selected_event_step_max": max(selected_steps),
        "chunk_size": args.chunk_size,
        "expected_chunk_count": (
            len(selected_ids) + args.chunk_size - 1
        )
        // args.chunk_size,
        "audit_episodes": str(audit_path),
        "audit_episodes_sha256": sha256(audit_path),
        "canonical_selection": str(canonical_path),
        "canonical_selection_sha256": sha256(canonical_path),
        "v1_2_baseline_source": str(baseline_path),
        "v1_2_baseline_source_sha256": sha256(baseline_path),
        "selection": str(selection_output),
        "selection_sha256": sha256(selection_output),
        "v1_2_baseline": str(baseline_output),
        "v1_2_baseline_sha256": sha256(baseline_output),
        "logical_case_ids": selected_ids,
    }
    with audit_output.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
