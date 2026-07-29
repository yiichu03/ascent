#!/bin/bash
# Submit one hash-bound ASCENT-VO submap smoke or HM3D-150 screen.

set -euo pipefail

MODE=${1:?usage: submit_submap_screen.sh smoke|full [calibration_json] [auto|autox] [stamp]}
NO_QSUB=${ASCENT_SUBMAP_NO_QSUB:-0}
PROJECT=/scratch/e1538633/liuyi/drift-aware-submap-exploration
SOURCE_ROOT=$PROJECT/external/ascent_vo_submap_v1
RESOURCE_ROOT=$PROJECT/external/ascent
POINTNAV_VO_ROOT=$PROJECT/external/PointNav-VO
CONTROLLER=$SOURCE_ROOT/pbs/run_submap_screen_3shared.pbs
WORKER=$SOURCE_ROOT/pbs/run_submap_screen_lane.sh
SMOKE_MANIFEST=/scratch/e1538633/liuyi/submap_v1_hm3d_smoke5_materialized_20260729_v3/chunk_manifest.json
SMOKE_MANIFEST_SHA256=bd94492db0a3fb6ed542745e592cefdfb0e748025e8fd037764967c077521358
FULL_MANIFEST=/scratch/e1538633/liuyi/submap_v1_hm3d150_materialized_20260729_v3/chunk_manifest.json
FULL_MANIFEST_SHA256=cfbb66998ac56c4fb574e9573844828bc45ee8d25db139ea1af6a0c26ce3192d
CALIBRATION_MANIFEST=/scratch/e1538633/liuyi/submap_v1_hm3d_calibration30_materialized_20260729_v3/chunk_manifest.json
SCRATCH_ROOT=/scratch/e1538633/liuyi
ASCENT_PYTHON=$SCRATCH_ROOT/micromamba/envs/ascent_nav/bin/python
LOG_ROOT=$SCRATCH_ROOT/submap_v1_pbs_logs
SUBMISSION_ROOT=$SCRATCH_ROOT/submap_v1_submissions

case "$MODE" in
  smoke)
    CALIBRATION_JSON=""
    QUEUE=${2:-auto}
    STAMP=${3:-20260729_submap_v1_smoke_first_attempt}
    MANIFEST=$SMOKE_MANIFEST
    MANIFEST_SHA256=$SMOKE_MANIFEST_SHA256
    EXPECTED_CHUNKS=1
    EXPECTED_EPISODES=5
    WALLTIME=36:00:00
    [ -f "$CALIBRATION_MANIFEST" ] || {
      echo "missing_calibration_manifest=$CALIBRATION_MANIFEST"
      exit 21
    }
    CALIBRATION_MANIFEST_SHA256=$(
      sha256sum "$CALIBRATION_MANIFEST" | awk '{print $1}'
    )
    CALIBRATION_JSON_SHA256=""
    ;;
  full)
    CALIBRATION_JSON=${2:?full mode requires calibration_json}
    QUEUE=${3:-auto}
    STAMP=${4:-20260729_submap_v1_hm3d150_first_attempt}
    MANIFEST=$FULL_MANIFEST
    MANIFEST_SHA256=$FULL_MANIFEST_SHA256
    EXPECTED_CHUNKS=5
    EXPECTED_EPISODES=150
    WALLTIME=96:00:00
    CALIBRATION_MANIFEST=""
    CALIBRATION_MANIFEST_SHA256=""
    [ -f "$CALIBRATION_JSON" ] || {
      echo "missing_calibration_json=$CALIBRATION_JSON"
      exit 21
    }
    [ "$("$ASCENT_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$CALIBRATION_JSON")" = PASS ] || {
      echo calibration_status_not_pass
      exit 25
    }
    CALIBRATION_JSON_SHA256=$(
      sha256sum "$CALIBRATION_JSON" | awk '{print $1}'
    )
    ;;
  *) echo "invalid_mode=$MODE"; exit 20 ;;
esac
case "$QUEUE" in auto|autox) ;; *) echo "queue_must_be_auto_or_autox"; exit 20 ;; esac

for path in "$CONTROLLER" "$WORKER" "$MANIFEST"; do
  [ -f "$path" ] || { echo "missing_file=$path"; exit 21; }
done
[ "$(sha256sum "$MANIFEST" | awk '{print $1}')" = "$MANIFEST_SHA256" ] || {
  echo manifest_hash_mismatch
  exit 22
}
[ -z "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=all)" ] || {
  echo source_worktree_not_clean
  git -C "$SOURCE_ROOT" status --short
  exit 23
}
SOURCE_COMMIT=$(git -C "$SOURCE_ROOT" rev-parse HEAD)
UPSTREAM=$(git -C "$SOURCE_ROOT" rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)
[ -n "$UPSTREAM" ] || { echo source_branch_has_no_upstream; exit 23; }
git -C "$SOURCE_ROOT" merge-base --is-ancestor "$SOURCE_COMMIT" "$UPSTREAM" || {
  echo source_commit_not_pushed
  exit 23
}
RESOURCE_COMMIT=$(git -C "$RESOURCE_ROOT" rev-parse HEAD)
POINTNAV_VO_COMMIT=$(git -C "$POINTNAV_VO_ROOT" rev-parse HEAD)
CONTROLLER_SHA256=$(sha256sum "$CONTROLLER" | awk '{print $1}')
WORKER_SHA256=$(sha256sum "$WORKER" | awk '{print $1}')

"$ASCENT_PYTHON" - "$MANIFEST" "$EXPECTED_CHUNKS" "$EXPECTED_EPISODES" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
value = json.load(open(sys.argv[1], encoding="utf-8"))
expected_chunks, expected_episodes = int(sys.argv[2]), int(sys.argv[3])
assert value["schema"] == "ascent_vo_submap_screen_materialized_v1"
assert len(value["chunks"]) == expected_chunks
assert value["episode_count"] == expected_episodes
assert sum(chunk["episode_count"] for chunk in value["chunks"]) == expected_episodes
scene_config = Path(value["scene_dataset_config"])
assert scene_config.is_absolute() and scene_config.is_file()
digest = hashlib.sha256(scene_config.read_bytes()).hexdigest()
assert digest == value["scene_dataset_config_sha256"]
PY

if [ "$NO_QSUB" = 1 ]; then
  "$ASCENT_PYTHON" - <<PY
import json
print(json.dumps({
    "status": "PASS",
    "mode": "$MODE",
    "queue_request": "$QUEUE",
    "walltime": "$WALLTIME",
    "stamp": "$STAMP",
    "source_commit": "$SOURCE_COMMIT",
    "resource_commit": "$RESOURCE_COMMIT",
    "pointnav_vo_commit": "$POINTNAV_VO_COMMIT",
    "manifest": "$MANIFEST",
    "manifest_sha256": "$MANIFEST_SHA256",
    "expected_chunks": $EXPECTED_CHUNKS,
    "expected_episodes": $EXPECTED_EPISODES,
    "calibration_manifest": "$CALIBRATION_MANIFEST" or None,
    "calibration_manifest_sha256": "$CALIBRATION_MANIFEST_SHA256" or None,
    "calibration_json": "$CALIBRATION_JSON" or None,
    "calibration_json_sha256": "$CALIBRATION_JSON_SHA256" or None,
    "controller_sha256": "$CONTROLLER_SHA256",
    "worker_sha256": "$WORKER_SHA256",
    "qsub_executed": False,
}, indent=2, sort_keys=True))
PY
  exit 0
fi

mkdir -p "$LOG_ROOT" "$SUBMISSION_ROOT"
PBS_LOG=$LOG_ROOT/submap_${MODE}_${STAMP}.pbs.log
SUBMISSION_JSON=$SUBMISSION_ROOT/submap_${MODE}_${STAMP}.json
[ ! -e "$PBS_LOG" ] || { echo "pbs_log_exists=$PBS_LOG"; exit 24; }
[ ! -e "$SUBMISSION_JSON" ] || {
  echo "submission_record_exists=$SUBMISSION_JSON"
  exit 24
}

ENVIRONMENT=$(IFS=,; echo \
"ASCENT_SUBMAP_MODE=$MODE,\
ASCENT_SUBMAP_MANIFEST=$MANIFEST,\
ASCENT_SUBMAP_MANIFEST_SHA256=$MANIFEST_SHA256,\
ASCENT_SUBMAP_EXPECTED_CHUNKS=$EXPECTED_CHUNKS,\
ASCENT_SUBMAP_EXPECTED_EPISODES=$EXPECTED_EPISODES,\
ASCENT_SUBMAP_SOURCE_COMMIT=$SOURCE_COMMIT,\
ASCENT_SUBMAP_RESOURCE_COMMIT=$RESOURCE_COMMIT,\
ASCENT_SUBMAP_POINTNAV_VO_COMMIT=$POINTNAV_VO_COMMIT,\
ASCENT_SUBMAP_CONTROLLER_SHA256=$CONTROLLER_SHA256,\
ASCENT_SUBMAP_WORKER_SHA256=$WORKER_SHA256,\
ASCENT_SUBMAP_STAMP=$STAMP,\
ASCENT_SUBMAP_CALIBRATION_MANIFEST=$CALIBRATION_MANIFEST,\
ASCENT_SUBMAP_CALIBRATION_MANIFEST_SHA256=$CALIBRATION_MANIFEST_SHA256,\
ASCENT_SUBMAP_CALIBRATION_JSON=$CALIBRATION_JSON,\
ASCENT_SUBMAP_CALIBRATION_JSON_SHA256=$CALIBRATION_JSON_SHA256")

JOB_ID=$(
  qsub -q "$QUEUE" -l "walltime=$WALLTIME" \
    -o "$PBS_LOG" -v "$ENVIRONMENT" "$CONTROLLER"
)
printf '%s\n' "$JOB_ID"
"$ASCENT_PYTHON" - "$SUBMISSION_JSON" <<PY
import json
from pathlib import Path
output = {
    "schema": "ascent_vo_submap_submission_v1",
    "job_id": "$JOB_ID",
    "mode": "$MODE",
    "queue_request": "$QUEUE",
    "walltime": "$WALLTIME",
    "stamp": "$STAMP",
    "source_commit": "$SOURCE_COMMIT",
    "resource_commit": "$RESOURCE_COMMIT",
    "pointnav_vo_commit": "$POINTNAV_VO_COMMIT",
    "manifest": "$MANIFEST",
    "manifest_sha256": "$MANIFEST_SHA256",
    "expected_chunks": $EXPECTED_CHUNKS,
    "expected_episodes": $EXPECTED_EPISODES,
    "calibration_manifest": "$CALIBRATION_MANIFEST" or None,
    "calibration_manifest_sha256": "$CALIBRATION_MANIFEST_SHA256" or None,
    "calibration_json": "$CALIBRATION_JSON" or None,
    "calibration_json_sha256": "$CALIBRATION_JSON_SHA256" or None,
    "controller_sha256": "$CONTROLLER_SHA256",
    "worker_sha256": "$WORKER_SHA256",
    "pbs_log": "$PBS_LOG",
}
path = Path("$SUBMISSION_JSON")
with path.open("x", encoding="utf-8") as handle:
    json.dump(output, handle, indent=2, sort_keys=True)
    handle.write("\\n")
PY
