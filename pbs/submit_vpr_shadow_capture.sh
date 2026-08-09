#!/bin/bash
# Submit one hash-bound, policy-passive v1.4 VPR shadow capture role.

set -euo pipefail

USAGE='submit_vpr_shadow_capture.sh hm3d_sentinel5|hm3d_cal50|hm3d_test100|mp3d_test150 [auto|autox] [stamp]'
ROLE=${1:?usage: $USAGE}
QUEUE=${2:-auto}
STAMP=${3:-20260809_vpr_shadow_${ROLE}_first_attempt}
NO_QSUB=${ASCENT_VPR_SHADOW_NO_QSUB:-0}

PROJECT=/scratch/e1538633/liuyi/drift-aware-submap-exploration
SOURCE_ROOT=$PROJECT/external/ascent_vo_submap_v1_4
FROZEN_V12_ROOT=$PROJECT/external/ascent_vo_submap_v1_2
RESOURCE_ROOT=$PROJECT/external/ascent
POINTNAV_VO_ROOT=$PROJECT/external/PointNav-VO
ARTIFACT_ROOT=$PROJECT/artifacts/objectnav/vpr_shadow
INPUT_ROOT=$ARTIFACT_ROOT/inputs
SPLIT_ROOT=$SOURCE_ROOT/experiments/vpr_shadow/manifests
ACTION_REFERENCE=$SOURCE_ROOT/experiments/vpr_shadow/hm3d_sentinel5_action_reference.json
CONTROLLER=$SOURCE_ROOT/pbs/run_submap_v1_2_3shared.pbs
WORKER=$SOURCE_ROOT/pbs/run_submap_v1_2_lane.sh
ASCENT_PYTHON=/scratch/e1538633/liuyi/micromamba/envs/ascent_nav/bin/python
FROZEN_PARENT=f238fc5dd8863f190ad9c1c39422dcc066ccda28

case "$QUEUE" in auto|autox) ;; *) echo queue_must_be_auto_or_autox; exit 20 ;; esac
[[ "$STAMP" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo invalid_stamp; exit 20; }
case "$ROLE" in
  hm3d_sentinel5)
    DATASET=hm3d
    EXPECTED_EPISODES=5
    EXPECTED_CHUNKS=1
    INPUT_NAME=hm3d_sentinel5_20260809_v1
    SPLIT_NAME=hm3d_sentinel5.json
    REQUIRED_SPLIT_ROLE=engineering_sentinel
    JOB_NAME=asv14s
    WALLTIME=24:00:00
    SENTINEL_SUMMARY=
    ;;
  hm3d_cal50)
    DATASET=hm3d
    EXPECTED_EPISODES=50
    EXPECTED_CHUNKS=5
    INPUT_NAME=hm3d_cal50_20260809_v1
    SPLIT_NAME=hm3d_cal50.json
    REQUIRED_SPLIT_ROLE=threshold_calibration_only
    JOB_NAME=asv14c
    WALLTIME=72:00:00
    SENTINEL_SUMMARY=${ASCENT_VPR_SHADOW_SENTINEL_SUMMARY:?required after sentinel PASS}
    ;;
  hm3d_test100)
    DATASET=hm3d
    EXPECTED_EPISODES=100
    EXPECTED_CHUNKS=5
    INPUT_NAME=hm3d_test100_20260809_v1
    SPLIT_NAME=hm3d_test100.json
    REQUIRED_SPLIT_ROLE=locked_in_distribution_test
    JOB_NAME=asv14t
    WALLTIME=96:00:00
    SENTINEL_SUMMARY=${ASCENT_VPR_SHADOW_SENTINEL_SUMMARY:?required after sentinel PASS}
    ;;
  mp3d_test150)
    DATASET=mp3d
    EXPECTED_EPISODES=150
    EXPECTED_CHUNKS=5
    INPUT_NAME=mp3d_test150_20260809_v1
    SPLIT_NAME=mp3d_test150.json
    REQUIRED_SPLIT_ROLE=locked_cross_dataset_transfer_test
    JOB_NAME=asv14m
    WALLTIME=96:00:00
    SENTINEL_SUMMARY=${ASCENT_VPR_SHADOW_SENTINEL_SUMMARY:?required after sentinel PASS}
    ;;
  *) echo invalid_vpr_shadow_role; exit 20 ;;
esac

PREPARED_ROOT=$INPUT_ROOT/$INPUT_NAME
PREPARATION_AUDIT=$PREPARED_ROOT/preparation_audit.json
MANIFEST=$PREPARED_ROOT/materialized/chunk_manifest.json
SPLIT_MANIFEST=$SPLIT_ROOT/$SPLIT_NAME
SENTINEL_MANIFEST=$INPUT_ROOT/hm3d_sentinel5_20260809_v1/materialized/chunk_manifest.json
for path in "$SOURCE_ROOT" "$FROZEN_V12_ROOT" "$CONTROLLER" "$WORKER" \
  "$PREPARATION_AUDIT" "$MANIFEST" "$SPLIT_MANIFEST" "$SENTINEL_MANIFEST" \
  "$ACTION_REFERENCE"; do
  [ -e "$path" ] || { echo "missing_required_path=$path"; exit 21; }
done
[ "$(git -C "$FROZEN_V12_ROOT" rev-parse HEAD)" = "$FROZEN_PARENT" ] || {
  echo frozen_v1_2_head_mismatch; exit 22;
}
[ -z "$(git -C "$FROZEN_V12_ROOT" status --porcelain --untracked-files=all)" ] || {
  echo frozen_v1_2_not_clean; exit 23;
}
[ -z "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=all)" ] || {
  echo source_worktree_not_clean
  git -C "$SOURCE_ROOT" status --short
  exit 23
}
SOURCE_COMMIT=$(git -C "$SOURCE_ROOT" rev-parse HEAD)
git -C "$SOURCE_ROOT" merge-base --is-ancestor "$FROZEN_PARENT" "$SOURCE_COMMIT" || {
  echo v1_4_not_descended_from_frozen_v1_2; exit 23;
}
UPSTREAM=$(git -C "$SOURCE_ROOT" rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)
[ -n "$UPSTREAM" ] || { echo source_branch_has_no_upstream; exit 23; }
git -C "$SOURCE_ROOT" merge-base --is-ancestor "$SOURCE_COMMIT" "$UPSTREAM" || {
  echo source_commit_not_pushed; exit 23;
}

MANIFEST_SHA256=$(sha256sum "$MANIFEST" | awk '{print $1}')
SPLIT_MANIFEST_SHA256=$(sha256sum "$SPLIT_MANIFEST" | awk '{print $1}')
ACTION_REFERENCE_SHA256=$(sha256sum "$ACTION_REFERENCE" | awk '{print $1}')
SENTINEL_MANIFEST_SHA256=$(sha256sum "$SENTINEL_MANIFEST" | awk '{print $1}')
CONTROLLER_SHA256=$(sha256sum "$CONTROLLER" | awk '{print $1}')
WORKER_SHA256=$(sha256sum "$WORKER" | awk '{print $1}')
RESOURCE_COMMIT=$(git -C "$RESOURCE_ROOT" rev-parse HEAD)
POINTNAV_VO_COMMIT=$(git -C "$POINTNAV_VO_ROOT" rev-parse HEAD)

"$ASCENT_PYTHON" - "$PREPARATION_AUDIT" "$MANIFEST" "$SPLIT_MANIFEST" \
  "$MANIFEST_SHA256" "$SPLIT_MANIFEST_SHA256" "$DATASET" \
  "$REQUIRED_SPLIT_ROLE" "$EXPECTED_EPISODES" "$EXPECTED_CHUNKS" <<'PY'
import hashlib, json, sys
from pathlib import Path
audit_path, manifest_path, split_path = map(Path, sys.argv[1:4])
manifest_hash, split_hash, dataset, role = sys.argv[4:8]
episodes, chunks = map(int, sys.argv[8:10])
audit = json.load(audit_path.open(encoding="utf-8"))
manifest = json.load(manifest_path.open(encoding="utf-8"))
split = json.load(split_path.open(encoding="utf-8"))
assert audit["schema"] == "ascent_v1_4_vpr_shadow_prepared_input_v1"
assert audit["outcome_fields_read"] is False
assert audit["evaluation_gt_trajectory_read"] is False
assert audit["materialized_manifest_sha256"] == manifest_hash
assert audit["split_manifest_sha256"] == split_hash
assert (audit["dataset"], audit["role"]) == (dataset, role)
assert (manifest["dataset"], manifest["split"]) == (dataset, "train")
assert (manifest["episode_count"], manifest["chunk_count"]) == (episodes, chunks)
assert (split["dataset"], split["role"], split["episode_count"]) == (dataset, role, episodes)
assert manifest["logical_case_ids"] == [row["logical_case_id"] for row in split["episodes"]]
assert audit["logical_case_ids"] == manifest["logical_case_ids"]
for chunk in manifest["chunks"]:
    for path_key, hash_key in (("data_path", "data_path_sha256"),
                               ("content_file", "content_file_sha256"),
                               ("identity_path", "identity_sha256")):
        path = Path(chunk[path_key])
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == chunk[hash_key]
PY

if [ -n "$SENTINEL_SUMMARY" ]; then
  [ -f "$SENTINEL_SUMMARY" ] || { echo missing_sentinel_summary; exit 21; }
  "$ASCENT_PYTHON" - "$SENTINEL_SUMMARY" "$SOURCE_COMMIT" \
    "$ACTION_REFERENCE_SHA256" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["schema"] == "ascent_v1_4_vpr_shadow_capture_gate_v1"
assert value["technical_status"] == "PASS"
assert value["role"] == "engineering_sentinel"
assert value["valid_episodes"] == 5
assert value["action_reference_required"] is True
assert value["capture_off_control_required"] is True
assert value["action_equivalent_episodes"] == 5
assert value["action_mismatch_count"] == 0
assert value["navigation_metrics_emitted"] is False
assert value["association_scores_emitted"] is False
assert value["provenance"]["source_commit"] == sys.argv[2]
assert value["provenance"]["control_summary_sha256"] is not None
assert value["provenance"]["historical_action_reference_sha256"] == sys.argv[3]
PY
fi

DRY_RUN_JSON=$(
  "$ASCENT_PYTHON" - <<PY
import json
print(json.dumps({
    "status": "PASS",
    "schema": "ascent_v1_4_vpr_shadow_capture_submission_preflight_v1",
    "role": "$ROLE",
    "dataset": "$DATASET",
    "queue_request": "$QUEUE",
    "walltime": "$WALLTIME",
    "stamp": "$STAMP",
    "source_commit": "$SOURCE_COMMIT",
    "frozen_v1_2_parent": "$FROZEN_PARENT",
    "resource_commit": "$RESOURCE_COMMIT",
    "pointnav_vo_commit": "$POINTNAV_VO_COMMIT",
    "artifact_root": "$ARTIFACT_ROOT",
    "manifest": "$MANIFEST",
    "manifest_sha256": "$MANIFEST_SHA256",
    "split_manifest": "$SPLIT_MANIFEST",
    "split_manifest_sha256": "$SPLIT_MANIFEST_SHA256",
    "expected_chunks": $EXPECTED_CHUNKS,
    "expected_episodes": $EXPECTED_EPISODES,
    "controller_sha256": "$CONTROLLER_SHA256",
    "worker_sha256": "$WORKER_SHA256",
    "passive_capture_only": True,
    "vpr_runtime_association": False,
    "planner_action_change": False,
    "navigation_metrics_emitted_by_gate": False,
    "association_scores_emitted_by_capture": False,
    "sentinel_unlock": "$SENTINEL_SUMMARY" or None,
    "qsub_executed": False,
}, indent=2, sort_keys=True))
PY
)
if [ "$NO_QSUB" = 1 ]; then
  printf '%s\n' "$DRY_RUN_JSON"
  exit 0
fi

LOG_ROOT=$ARTIFACT_ROOT/pbs_logs
SUBMISSION_ROOT=$ARTIFACT_ROOT/submissions
mkdir -p "$LOG_ROOT" "$SUBMISSION_ROOT"
PBS_LOG=$LOG_ROOT/vpr_shadow_${ROLE}_${STAMP}.pbs.log
SUBMISSION_JSON=$SUBMISSION_ROOT/vpr_shadow_${ROLE}_${STAMP}.json
[ ! -e "$PBS_LOG" ] || { echo "pbs_log_exists=$PBS_LOG"; exit 24; }
[ ! -e "$SUBMISSION_JSON" ] || { echo "submission_record_exists=$SUBMISSION_JSON"; exit 24; }

ACTION_REFERENCE_FOR_JOB=
ACTION_REFERENCE_SHA256_FOR_JOB=
if [ "$ROLE" = hm3d_sentinel5 ]; then
  ACTION_REFERENCE_FOR_JOB=$ACTION_REFERENCE
  ACTION_REFERENCE_SHA256_FOR_JOB=$ACTION_REFERENCE_SHA256
fi
ENVIRONMENT=$(IFS=,; echo \
"ASCENT_SUBMAP_V12_SOURCE_ROOT=$SOURCE_ROOT,\
ASCENT_SUBMAP_V12_DATASET=$DATASET,\
ASCENT_SUBMAP_V12_MODE=shadow,\
ASCENT_SUBMAP_V12_ARTIFACT_ROOT=$ARTIFACT_ROOT,\
ASCENT_SUBMAP_V12_MANIFEST=$MANIFEST,\
ASCENT_SUBMAP_V12_MANIFEST_SHA256=$MANIFEST_SHA256,\
ASCENT_SUBMAP_V12_EXPECTED_CHUNKS=$EXPECTED_CHUNKS,\
ASCENT_SUBMAP_V12_EXPECTED_EPISODES=$EXPECTED_EPISODES,\
ASCENT_SUBMAP_V12_GATE_MANIFEST=$SENTINEL_MANIFEST,\
ASCENT_SUBMAP_V12_GATE_MANIFEST_SHA256=$SENTINEL_MANIFEST_SHA256,\
ASCENT_SUBMAP_V12_SOURCE_COMMIT=$SOURCE_COMMIT,\
ASCENT_SUBMAP_V12_RESOURCE_COMMIT=$RESOURCE_COMMIT,\
ASCENT_SUBMAP_V12_POINTNAV_VO_COMMIT=$POINTNAV_VO_COMMIT,\
ASCENT_SUBMAP_V12_CONTROLLER_SHA256=$CONTROLLER_SHA256,\
ASCENT_SUBMAP_V12_WORKER_SHA256=$WORKER_SHA256,\
ASCENT_SUBMAP_V12_STAMP=$STAMP,\
ASCENT_VPR_SHADOW_ROLE=$ROLE,\
ASCENT_VPR_SHADOW_SPLIT_MANIFEST=$SPLIT_MANIFEST,\
ASCENT_VPR_SHADOW_SPLIT_MANIFEST_SHA256=$SPLIT_MANIFEST_SHA256,\
ASCENT_VPR_SHADOW_ACTION_REFERENCE=$ACTION_REFERENCE_FOR_JOB,\
ASCENT_VPR_SHADOW_ACTION_REFERENCE_SHA256=$ACTION_REFERENCE_SHA256_FOR_JOB")

JOB_ID=$(
  qsub -q "$QUEUE" -N "$JOB_NAME" -l "walltime=$WALLTIME" \
    -o "$PBS_LOG" -v "$ENVIRONMENT" "$CONTROLLER"
)
printf '%s\n' "$JOB_ID"
"$ASCENT_PYTHON" - "$SUBMISSION_JSON" "$JOB_ID" "$PBS_LOG" <<PY
import json, sys
output = json.loads('''$DRY_RUN_JSON''')
output.update({
    "schema": "ascent_v1_4_vpr_shadow_capture_submission_v1",
    "job_id": sys.argv[2],
    "pbs_log": sys.argv[3],
    "qsub_executed": True,
})
with open(sys.argv[1], "x", encoding="utf-8") as handle:
    json.dump(output, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
