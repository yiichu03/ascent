#!/usr/bin/env python3
"""Local no-Habitat inference gate for all pinned VPR shadow models."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import cv2
import numpy as np

try:
    from vpr_shadow_data import ShadowFrame, sha256
    from vpr_shadow_geometry import (
        lift_matched_keypoints,
        verify_rigid_correspondences,
    )
    from vpr_shadow_models import (
        GlobalRetriever,
        LocalGeometryMatcher,
        load_and_validate_registry,
    )
except ImportError:
    from scripts.vpr_shadow_data import ShadowFrame, sha256
    from scripts.vpr_shadow_geometry import (
        lift_matched_keypoints,
        verify_rigid_correspondences,
    )
    from scripts.vpr_shadow_models import (
        GlobalRetriever,
        LocalGeometryMatcher,
        load_and_validate_registry,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _fixture(path: Path) -> None:
    rng = np.random.default_rng(20260809)
    image = rng.integers(0, 50, size=(240, 320, 3), dtype=np.uint8)
    for y in range(20, 230, 30):
        for x in range(20, 310, 30):
            color = tuple(int(value) for value in rng.integers(80, 255, size=3))
            cv2.circle(image, (x, y), 6, color, thickness=-1)
            cv2.line(image, (x - 8, y), (x + 8, y), (255, 255, 255), 1)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"cannot write fixture {path}")


def _frame(path: Path, frame_id_step: int) -> ShadowFrame:
    return ShadowFrame(
        capture_root=path.parent,
        env=0,
        episode_sequence=0,
        action_step=frame_id_step,
        admission_index=frame_id_step,
        submap_id=f"fixture_{frame_id_step}",
        floor_id=0,
        world_pose_vo=np.zeros(3),
        local_pose=np.zeros(3),
        tf_camera_to_submap=np.eye(4),
        rgb_path=path,
        depth_path=path.parent / "unused.depth.png",
        min_depth=0.5,
        max_depth=5.0,
        fx=160.0,
        fy=160.0,
    )


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    fixture_path = output_dir / "fixture.jpg"
    _fixture(fixture_path)
    frames = [_frame(fixture_path, 0), _frame(fixture_path, 1)]
    registry = load_and_validate_registry(project_root, args.registry)
    global_results = {}
    for name in ("mixvpr", "megaloc"):
        model = GlobalRetriever(
            name=name,
            project_root=project_root,
            registry=registry,
            device=args.device,
        )
        descriptors = model.encode(frames, batch_size=2)
        values = np.stack([descriptors[frame.frame_id] for frame in frames])
        expected_dimension = int(
            registry["global_retrievers"][name]["descriptor_dimension"]
        )
        if values.shape != (2, expected_dimension):
            raise ValueError(f"unexpected {name} descriptor shape {values.shape}")
        norms = np.linalg.norm(values, axis=1)
        similarity = float(values[0] @ values[1])
        if not np.allclose(norms, 1.0, atol=1e-4) or similarity < 0.999:
            raise ValueError(f"{name} identical-image inference mismatch")
        global_results[name] = {
            "descriptor_dimension": expected_dimension,
            "norms": norms.tolist(),
            "identical_similarity": similarity,
        }
        del model, descriptors, values
        gc.collect()

    local = LocalGeometryMatcher(
        project_root=project_root,
        registry=registry,
        device=args.device,
    )
    keypoints0, keypoints1, matches = local.match(frames[0], frames[1])
    depth = np.full((240, 320), 0.5, dtype=np.float32)
    source, target = lift_matched_keypoints(
        keypoints_source=keypoints0,
        keypoints_target=keypoints1,
        matches=matches,
        depth_source=depth,
        depth_target=depth,
        source_calibration=frames[0].calibration,
        target_calibration=frames[1].calibration,
    )
    verification = verify_rigid_correspondences(
        source,
        target,
        match_count=len(matches),
        seed_key="local_model_gate",
    )
    if len(matches) < 100 or not verification.pair_supported:
        raise ValueError("ALIKED+LightGlue identical-image geometry gate failed")
    payload = {
        "schema": "ascent_v1_4_vpr_shadow_model_gate_v1",
        "technical_status": "PASS",
        "device": args.device,
        "registry_sha256": sha256(args.registry),
        "fixture_sha256": sha256(fixture_path),
        "global_retrievers": global_results,
        "local_verifier": verification.as_dict(),
    }
    with (output_dir / "model_gate.json").open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
