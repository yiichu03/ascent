#!/bin/bash
# Submit one hash-bound HM3D v1.4 oracle task-memory experiment.

set -euo pipefail

USAGE='submit_place_memory.sh train150|mechanism_subset auto|autox [stamp]'
EXPERIMENT=${1:?usage: $USAGE}
QUEUE=${2:?usage: $USAGE}
STAMP=${3:-20260811_v1_4_oracle_task_memory_${EXPERIMENT}_first_attempt}
DATASET=hm3d
MODE=full
NO_QSUB=${ASCENT_PLACE_MEMORY_V14_NO_QSUB:-0}

PROJECT=/scratch/e1538633/liuyi/drift-aware-submap-exploration
SOURCE_ROOT=$PROJECT/external/ascent_vo_submap_v1_4_oracle_task_memory
RESOURCE_ROOT=$PROJECT/external/ascent
POINTNAV_VO_ROOT=$PROJECT/external/PointNav-VO
ARTIFACT_ROOT=${ASCENT_PLACE_MEMORY_V14_ARTIFACT_ROOT:-$PROJECT/artifacts/objectnav/submap_v1_4_oracle_task_memory}
MANIFEST_ROOT=$PROJECT/artifacts/objectnav/submap_v1/manifests
MECHANISM_INPUT_ROOT=$ARTIFACT_ROOT/inputs
CONTROLLER=$SOURCE_ROOT/pbs/run_place_memory_3shared.pbs
WORKER=$SOURCE_ROOT/pbs/run_place_memory_lane.sh
SCRATCH_ROOT=/scratch/e1538633/liuyi
ASCENT_PYTHON=$SCRATCH_ROOT/micromamba/envs/ascent_nav/bin/python

TRAIN150_BASELINE=$PROJECT/artifacts/objectnav/submap_v1_2/runs/submap_v1_2_hm3d_full_550927.hopper-m-02_20260801_submap_v1_2_hm3d_train150_exact_retry1_after_gate_native_abort/final_gate/episodes.csv
case "$EXPERIMENT" in
  train150)
    SMOKE_MANIFEST=$MANIFEST_ROOT/hm3d_smoke5_relocated.json
    SMOKE_MANIFEST_SHA256=5166c2ab2000799a05c30aaa56e11ccf828b20c66ae2f4fc8a01065f53d49538
    MANIFEST=$MANIFEST_ROOT/hm3d150_relocated.json
    MANIFEST_SHA256=23297001bb08e22c48e073d4438918b74973712e5cbe99b06a9e62b4cd3ccc02
    EXPECTED_CHUNKS=5
    EXPECTED_EPISODES=150
    BASELINE_CSVS=$TRAIN150_BASELINE
    BASELINE_SHA256=38ddf6efa753fd643b4a06d7a0fef9e16e55a7cde446569c10f0f4fc534215f6
    JOB_NAME=asv14t
    ;;
  mechanism_subset)
    SMOKE_MANIFEST=$PROJECT/artifacts/objectnav/submap_v1_2/inputs/hm3d_val5_materialized_20260802_v3/chunk_manifest.json
    SMOKE_MANIFEST_SHA256=f26ff346bf82d32e619c868d518fc5f75f06767ea64ec61656f951995cdc07d1
    MANIFEST=$MECHANISM_INPUT_ROOT/hm3d_val60_mechanism_materialized_20260810_v1/chunk_manifest.json
    MANIFEST_SHA256=2061ad76b54ce54f0b371fd28e290938b83200f2ec84d908e2498e1687bbc2fd
    EXPECTED_CHUNKS=6
    EXPECTED_EPISODES=60
    BASELINE_CSVS=$MECHANISM_INPUT_ROOT/hm3d_val60_mechanism_20260810_v1/v1_2_baseline.csv
    BASELINE_SHA256=d36be28d21372fe46c5750c309c774d430c0565e8db91f0c431960e167249d34
    SUBSET_AUDIT=$MECHANISM_INPUT_ROOT/hm3d_val60_mechanism_20260810_v1/preparation_audit.json
    SUBSET_AUDIT_SHA256=d22d47e863ddd5e29c6c77e5803a3b5a0e9f07465adea2cffe07f7a350410b97
    JOB_NAME=asv14m
    ;;
  *) echo "invalid_experiment=$EXPERIMENT"; exit 20 ;;
esac
case "$QUEUE" in auto|autox) ;; *) echo queue_must_be_auto_or_autox; exit 20 ;; esac
WALLTIME=24:00:00

for path in "$CONTROLLER" "$WORKER" "$MANIFEST" "$SMOKE_MANIFEST"; do
  [ -f "$path" ] || { echo "missing_file=$path"; exit 21; }
done
[ "$(sha256sum "$MANIFEST" | awk '{print $1}')" = "$MANIFEST_SHA256" ] || {
  echo manifest_hash_mismatch; exit 22;
}
[ "$(sha256sum "$SMOKE_MANIFEST" | awk '{print $1}')" = "$SMOKE_MANIFEST_SHA256" ] || {
  echo smoke_manifest_hash_mismatch; exit 22;
}
IFS=: read -r -a BASELINE_PATHS <<< "$BASELINE_CSVS"
for baseline in "${BASELINE_PATHS[@]}"; do
  [ -f "$baseline" ] || { echo "missing_baseline=$baseline"; exit 21; }
done
[ "$(sha256sum "$BASELINE_CSVS" | awk '{print $1}')" = "$BASELINE_SHA256" ] || {
  echo baseline_hash_mismatch; exit 22;
}
if [ "$EXPERIMENT" = mechanism_subset ]; then
  [ -f "$SUBSET_AUDIT" ] || { echo missing_subset_audit; exit 21; }
  [ "$(sha256sum "$SUBSET_AUDIT" | awk '{print $1}')" = "$SUBSET_AUDIT_SHA256" ] || {
    echo subset_audit_hash_mismatch; exit 22;
  }
  "$ASCENT_PYTHON" - "$SUBSET_AUDIT" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["status"] == "PASS"
assert value["dataset"] == "hm3d"
assert value["source_arm"] == "submap_v1_2"
assert value["metrics_used_for_selection"] is False
assert value["selected_episode_count"] == 60
assert value["selected_scene_count"] == 11
assert value["max_event_step"] == 300
assert value["expected_chunk_count"] == 6
PY
fi
[ -z "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=all)" ] || {
  echo source_worktree_not_clean
  git -C "$SOURCE_ROOT" status --short
  exit 23
}
SOURCE_COMMIT=$(git -C "$SOURCE_ROOT" rev-parse HEAD)
UPSTREAM=$(git -C "$SOURCE_ROOT" rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)
[ -n "$UPSTREAM" ] || { echo source_branch_has_no_upstream; exit 23; }
git -C "$SOURCE_ROOT" merge-base --is-ancestor "$SOURCE_COMMIT" "$UPSTREAM" || {
  echo source_commit_not_pushed; exit 23;
}
RESOURCE_COMMIT=$(git -C "$RESOURCE_ROOT" rev-parse HEAD)
POINTNAV_VO_COMMIT=$(git -C "$POINTNAV_VO_ROOT" rev-parse HEAD)
CONTROLLER_SHA256=$(sha256sum "$CONTROLLER" | awk '{print $1}')
WORKER_SHA256=$(sha256sum "$WORKER" | awk '{print $1}')

"$ASCENT_PYTHON" - "$MANIFEST" "$EXPECTED_CHUNKS" "$EXPECTED_EPISODES" "$DATASET" <<'PY'
import hashlib, json, sys
from pathlib import Path
value = json.load(open(sys.argv[1], encoding="utf-8"))
expected_chunks, expected_episodes = int(sys.argv[2]), int(sys.argv[3])
assert value["schema"] == "ascent_vo_submap_screen_materialized_v1"
assert value["dataset"] == sys.argv[4] and value["split"] == "train"
assert len(value["chunks"]) == expected_chunks
assert value["episode_count"] == expected_episodes
assert sum(int(chunk["episode_count"]) for chunk in value["chunks"]) == expected_episodes
scene_config = Path(value["scene_dataset_config"])
assert scene_config.is_absolute() and scene_config.is_file()
assert hashlib.sha256(scene_config.read_bytes()).hexdigest() == value["scene_dataset_config_sha256"]
PY
"$ASCENT_PYTHON" - "$SOURCE_ROOT/scripts" "$MANIFEST" \
  "$EXPECTED_EPISODES" "$BASELINE_CSVS" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from summarize_place_memory_screen import load_baseline
from summarize_submap_screen import load_manifest
chunks, order, errors = load_manifest(
    Path(sys.argv[2]), int(sys.argv[3]), "hm3d"
)
baseline, baseline_errors, _ = load_baseline(
    [Path(sys.argv[4])], dataset="hm3d", logical_order=order
)
assert not errors and not baseline_errors
identities = {
    value["logical_case_id"]: value["scene_id"]
    for chunk in chunks.values()
    for value in chunk["identity_by_runtime"].values()
}
assert all(identities[key] == value["scene_id"] for key, value in baseline.items())
PY

DRY_RUN_JSON=$(
  "$ASCENT_PYTHON" - <<PY
import json
print(json.dumps({
    "status": "PASS",
    "schema": "ascent_vo_submap_v1_4_oracle_task_memory_submission_preflight_v1",
    "dataset": "$DATASET",
    "experiment": "$EXPERIMENT",
    "mode": "$MODE",
    "queue_request": "$QUEUE",
    "walltime": "$WALLTIME",
    "stamp": "$STAMP",
    "source_commit": "$SOURCE_COMMIT",
    "resource_commit": "$RESOURCE_COMMIT",
    "pointnav_vo_commit": "$POINTNAV_VO_COMMIT",
    "artifact_root": "$ARTIFACT_ROOT",
    "manifest": "$MANIFEST",
    "manifest_sha256": "$MANIFEST_SHA256",
    "gate_manifest": "$SMOKE_MANIFEST",
    "gate_manifest_sha256": "$SMOKE_MANIFEST_SHA256",
    "expected_chunks": $EXPECTED_CHUNKS,
    "expected_episodes": $EXPECTED_EPISODES,
    "baseline_csvs": "$BASELINE_CSVS".split(":") if "$BASELINE_CSVS" else [],
    "baseline_sha256": "$BASELINE_SHA256",
    "controller_sha256": "$CONTROLLER_SHA256",
    "worker_sha256": "$WORKER_SHA256",
    "oracle_identity_only": True,
    "gt_policy_output_fields": ["event_sequence", "reference_submap_id"],
    "oracle_scope": "vo_consistent_graph_nondirect_fragmentation_only",
    "same_place_identity_grants_consumption": False,
    "place_consumption_authority": "matched_repeat_low_gain_excursion",
    "place_repeat_settlement": "online_when_evidence_complete_and_residual_live",
    "method_version": "submap_v1.4_oracle_task_memory",
    "handoff_enabled": True,
    "exhaustion_recovery_enabled": True,
    "five_episode_gate_before_full": "$MODE" == "full",
    "fixed_technical_retry_per_failed_unit": 1,
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
PBS_LOG=$LOG_ROOT/${EXPERIMENT}_${STAMP}.pbs.log
SUBMISSION_JSON=$SUBMISSION_ROOT/${EXPERIMENT}_${STAMP}.json
[ ! -e "$PBS_LOG" ] || { echo "pbs_log_exists=$PBS_LOG"; exit 24; }
[ ! -e "$SUBMISSION_JSON" ] || { echo "submission_record_exists=$SUBMISSION_JSON"; exit 24; }

ENVIRONMENT=$(IFS=,; echo \
"ASCENT_PLACE_MEMORY_V14_DATASET=$DATASET,\
ASCENT_PLACE_MEMORY_V14_MODE=$MODE,\
ASCENT_PLACE_MEMORY_V14_EXPERIMENT=$EXPERIMENT,\
ASCENT_PLACE_MEMORY_V14_ARTIFACT_ROOT=$ARTIFACT_ROOT,\
ASCENT_PLACE_MEMORY_V14_MANIFEST=$MANIFEST,\
ASCENT_PLACE_MEMORY_V14_MANIFEST_SHA256=$MANIFEST_SHA256,\
ASCENT_PLACE_MEMORY_V14_EXPECTED_CHUNKS=$EXPECTED_CHUNKS,\
ASCENT_PLACE_MEMORY_V14_EXPECTED_EPISODES=$EXPECTED_EPISODES,\
ASCENT_PLACE_MEMORY_V14_GATE_MANIFEST=$SMOKE_MANIFEST,\
ASCENT_PLACE_MEMORY_V14_GATE_MANIFEST_SHA256=$SMOKE_MANIFEST_SHA256,\
ASCENT_PLACE_MEMORY_V14_SOURCE_COMMIT=$SOURCE_COMMIT,\
ASCENT_PLACE_MEMORY_V14_RESOURCE_COMMIT=$RESOURCE_COMMIT,\
ASCENT_PLACE_MEMORY_V14_POINTNAV_VO_COMMIT=$POINTNAV_VO_COMMIT,\
ASCENT_PLACE_MEMORY_V14_CONTROLLER_SHA256=$CONTROLLER_SHA256,\
ASCENT_PLACE_MEMORY_V14_WORKER_SHA256=$WORKER_SHA256,\
ASCENT_PLACE_MEMORY_V14_STAMP=$STAMP,\
ASCENT_PLACE_MEMORY_V14_BASELINE_CSVS=$BASELINE_CSVS")

JOB_ID=$(
  qsub -q "$QUEUE" -N "$JOB_NAME" -l "walltime=$WALLTIME" \
    -o "$PBS_LOG" -v "$ENVIRONMENT" "$CONTROLLER"
)
printf '%s\n' "$JOB_ID"
"$ASCENT_PYTHON" - "$SUBMISSION_JSON" "$JOB_ID" "$PBS_LOG" <<PY
import json, sys
from pathlib import Path
output = json.loads('''$DRY_RUN_JSON''')
output.update({
    "schema": "ascent_vo_submap_v1_4_oracle_task_memory_submission_v1",
    "job_id": sys.argv[2],
    "pbs_log": sys.argv[3],
    "qsub_executed": True,
})
path = Path(sys.argv[1])
with path.open("x", encoding="utf-8") as handle:
    json.dump(output, handle, indent=2, sort_keys=True)
    handle.write("\\n")
PY
