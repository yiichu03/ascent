#!/bin/bash
# Submit one hash-bound B2-only ASCENT-VO submap-v1.2 screen.

set -euo pipefail

USAGE='submit_submap_v1_2.sh hm3d|mp3d smoke|full [auto|autox] [stamp]'
DATASET=${1:?usage: $USAGE}
MODE=${2:?usage: $USAGE}
QUEUE=${3:-autox}
STAMP=${4:-20260731_submap_v1_2_${DATASET}_${MODE}_first_attempt}
NO_QSUB=${ASCENT_SUBMAP_V12_NO_QSUB:-0}

PROJECT=/scratch/e1538633/liuyi/drift-aware-submap-exploration
SOURCE_ROOT=$PROJECT/external/ascent_vo_submap_v1_2
RESOURCE_ROOT=$PROJECT/external/ascent
POINTNAV_VO_ROOT=$PROJECT/external/PointNav-VO
ARTIFACT_ROOT=${ASCENT_SUBMAP_V12_ARTIFACT_ROOT:-$PROJECT/artifacts/objectnav/submap_v1_2}
MANIFEST_ROOT=$PROJECT/artifacts/objectnav/submap_v1/manifests
CONTROLLER=$SOURCE_ROOT/pbs/run_submap_v1_2_3shared.pbs
WORKER=$SOURCE_ROOT/pbs/run_submap_v1_2_lane.sh
SCRATCH_ROOT=/scratch/e1538633/liuyi
ASCENT_PYTHON=$SCRATCH_ROOT/micromamba/envs/ascent_nav/bin/python

case "$DATASET" in
  hm3d)
    SMOKE_MANIFEST=$MANIFEST_ROOT/hm3d_smoke5_relocated.json
    SMOKE_MANIFEST_SHA256=5166c2ab2000799a05c30aaa56e11ccf828b20c66ae2f4fc8a01065f53d49538
    FULL_MANIFEST=$MANIFEST_ROOT/hm3d150_relocated.json
    FULL_MANIFEST_SHA256=23297001bb08e22c48e073d4438918b74973712e5cbe99b06a9e62b4cd3ccc02
    BASELINE_CSVS=$PROJECT/artifacts/objectnav/pose_factorial/results/pose_factorial_hm3d_completed_gate_544332_20260728_v1/episodes.csv
    JOB_NAME=asv12h
    ;;
  mp3d)
    SMOKE_MANIFEST=$MANIFEST_ROOT/mp3d_smoke5_relocated.json
    SMOKE_MANIFEST_SHA256=7b31e958a74641c986d373b4693bc691c2698b2ca186012f918d56f57f3f015a
    FULL_MANIFEST=$MANIFEST_ROOT/mp3d150_relocated.json
    FULL_MANIFEST_SHA256=12632eee47c8274a6df8088f4423ca321789ebc495098fba8839460c5f98cc79
    BASELINE_CSVS=$PROJECT/artifacts/objectnav/pose_factorial/runs/pose_factorial_shard_547621.hopper-m-02_20260728_pose_factorial_mp3d_smallx_c00_c01_c02_first_attempt/final_gate/episodes.csv:$PROJECT/artifacts/objectnav/pose_factorial/runs/pose_factorial_shard_547624.hopper-m-02_20260728_pose_factorial_mp3d_small_c03_c04_first_attempt/final_gate/episodes.csv
    JOB_NAME=asv12m
    ;;
  *) echo "invalid_dataset=$DATASET"; exit 20 ;;
esac
case "$QUEUE" in auto|autox) ;; *) echo queue_must_be_auto_or_autox; exit 20 ;; esac
case "$MODE" in
  smoke)
    MANIFEST=$SMOKE_MANIFEST
    MANIFEST_SHA256=$SMOKE_MANIFEST_SHA256
    EXPECTED_CHUNKS=1
    EXPECTED_EPISODES=5
    WALLTIME=24:00:00
    BASELINE_CSVS=""
    ;;
  full)
    MANIFEST=$FULL_MANIFEST
    MANIFEST_SHA256=$FULL_MANIFEST_SHA256
    EXPECTED_CHUNKS=5
    EXPECTED_EPISODES=150
    WALLTIME=96:00:00
    ;;
  *) echo "invalid_mode=$MODE"; exit 20 ;;
esac

for path in "$CONTROLLER" "$WORKER" "$MANIFEST" "$SMOKE_MANIFEST"; do
  [ -f "$path" ] || { echo "missing_file=$path"; exit 21; }
done
[ "$(sha256sum "$MANIFEST" | awk '{print $1}')" = "$MANIFEST_SHA256" ] || {
  echo manifest_hash_mismatch; exit 22;
}
[ "$(sha256sum "$SMOKE_MANIFEST" | awk '{print $1}')" = "$SMOKE_MANIFEST_SHA256" ] || {
  echo smoke_manifest_hash_mismatch; exit 22;
}
if [ "$MODE" = full ]; then
  IFS=: read -r -a BASELINE_PATHS <<< "$BASELINE_CSVS"
  for baseline in "${BASELINE_PATHS[@]}"; do
    [ -f "$baseline" ] || { echo "missing_baseline=$baseline"; exit 21; }
  done
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

DRY_RUN_JSON=$(
  "$ASCENT_PYTHON" - <<PY
import json
print(json.dumps({
    "status": "PASS",
    "schema": "ascent_vo_submap_v1_2_submission_preflight_v1",
    "dataset": "$DATASET",
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
    "controller_sha256": "$CONTROLLER_SHA256",
    "worker_sha256": "$WORKER_SHA256",
    "b2_only": True,
    "method_version": "submap_v1.2",
    "handoff_enabled": True,
    "exhaustion_recovery_enabled": True,
    "five_episode_gate_before_full": "$MODE" == "full",
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
PBS_LOG=$LOG_ROOT/submap_v1_2_${DATASET}_${MODE}_${STAMP}.pbs.log
SUBMISSION_JSON=$SUBMISSION_ROOT/submap_v1_2_${DATASET}_${MODE}_${STAMP}.json
[ ! -e "$PBS_LOG" ] || { echo "pbs_log_exists=$PBS_LOG"; exit 24; }
[ ! -e "$SUBMISSION_JSON" ] || { echo "submission_record_exists=$SUBMISSION_JSON"; exit 24; }

ENVIRONMENT=$(IFS=,; echo \
"ASCENT_SUBMAP_V12_DATASET=$DATASET,\
ASCENT_SUBMAP_V12_MODE=$MODE,\
ASCENT_SUBMAP_V12_ARTIFACT_ROOT=$ARTIFACT_ROOT,\
ASCENT_SUBMAP_V12_MANIFEST=$MANIFEST,\
ASCENT_SUBMAP_V12_MANIFEST_SHA256=$MANIFEST_SHA256,\
ASCENT_SUBMAP_V12_EXPECTED_CHUNKS=$EXPECTED_CHUNKS,\
ASCENT_SUBMAP_V12_EXPECTED_EPISODES=$EXPECTED_EPISODES,\
ASCENT_SUBMAP_V12_GATE_MANIFEST=$SMOKE_MANIFEST,\
ASCENT_SUBMAP_V12_GATE_MANIFEST_SHA256=$SMOKE_MANIFEST_SHA256,\
ASCENT_SUBMAP_V12_SOURCE_COMMIT=$SOURCE_COMMIT,\
ASCENT_SUBMAP_V12_RESOURCE_COMMIT=$RESOURCE_COMMIT,\
ASCENT_SUBMAP_V12_POINTNAV_VO_COMMIT=$POINTNAV_VO_COMMIT,\
ASCENT_SUBMAP_V12_CONTROLLER_SHA256=$CONTROLLER_SHA256,\
ASCENT_SUBMAP_V12_WORKER_SHA256=$WORKER_SHA256,\
ASCENT_SUBMAP_V12_STAMP=$STAMP,\
ASCENT_SUBMAP_V12_BASELINE_CSVS=$BASELINE_CSVS")

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
    "schema": "ascent_vo_submap_v1_2_submission_v1",
    "job_id": sys.argv[2],
    "pbs_log": sys.argv[3],
    "qsub_executed": True,
})
path = Path(sys.argv[1])
with path.open("x", encoding="utf-8") as handle:
    json.dump(output, handle, indent=2, sort_keys=True)
    handle.write("\\n")
PY
