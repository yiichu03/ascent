#!/bin/bash
# Submit one hash-bound HM3D or MP3D ASCENT-VO submap screen.

set -euo pipefail

USAGE='submit_submap_screen.sh hm3d|mp3d smoke|full calibration_json_or_dash [auto|autox] [stamp] [shard_index] [shard_count]'
DATASET=${1:?usage: $USAGE}
MODE=${2:?usage: $USAGE}
CALIBRATION_ARG=${3:--}
SHARD_INDEX=${6:-}
SHARD_COUNT=${7:-}
NO_QSUB=${ASCENT_SUBMAP_NO_QSUB:-0}
PROJECT=/scratch/e1538633/liuyi/drift-aware-submap-exploration
SOURCE_ROOT=$PROJECT/external/ascent_vo_submap_v1
RESOURCE_ROOT=$PROJECT/external/ascent
POINTNAV_VO_ROOT=$PROJECT/external/PointNav-VO
ARTIFACT_ROOT=${ASCENT_SUBMAP_ARTIFACT_ROOT:-$PROJECT/artifacts/objectnav/submap_v1}
MANIFEST_ROOT=$ARTIFACT_ROOT/manifests
CONTROLLER=$SOURCE_ROOT/pbs/run_submap_screen_3shared.pbs
WORKER=$SOURCE_ROOT/pbs/run_submap_screen_lane.sh
CALIBRATION_MANIFEST=$MANIFEST_ROOT/hm3d_calibration30_relocated.json
CALIBRATION_MANIFEST_EXPECTED_SHA256=54c51cf8632aaffe25deb9a41e0b42594a1dc6b8204a8b56afd87fcf38561c10
SCRATCH_ROOT=/scratch/e1538633/liuyi
ASCENT_PYTHON=$SCRATCH_ROOT/micromamba/envs/ascent_nav/bin/python

case "$DATASET" in
  hm3d)
    SMOKE_MANIFEST=$MANIFEST_ROOT/hm3d_smoke5_relocated.json
    SMOKE_MANIFEST_SHA256=5166c2ab2000799a05c30aaa56e11ccf828b20c66ae2f4fc8a01065f53d49538
    FULL_MANIFEST=$MANIFEST_ROOT/hm3d150_relocated.json
    FULL_MANIFEST_SHA256=23297001bb08e22c48e073d4438918b74973712e5cbe99b06a9e62b4cd3ccc02
    ;;
  mp3d)
    SMOKE_MANIFEST=$MANIFEST_ROOT/mp3d_smoke5_relocated.json
    SMOKE_MANIFEST_SHA256=7b31e958a74641c986d373b4693bc691c2698b2ca186012f918d56f57f3f015a
    FULL_MANIFEST=$MANIFEST_ROOT/mp3d150_relocated.json
    FULL_MANIFEST_SHA256=12632eee47c8274a6df8088f4423ca321789ebc495098fba8839460c5f98cc79
    ;;
  *) echo "invalid_dataset=$DATASET"; exit 20 ;;
esac
LOG_ROOT=$ARTIFACT_ROOT/pbs_logs
SUBMISSION_ROOT=$ARTIFACT_ROOT/submissions
CALIBRATION_JSON=""
CALIBRATION_JSON_SHA256=""
CALIBRATION_MANIFEST_SHA256=""

case "$MODE" in
  smoke)
    [ -z "$SHARD_INDEX" ] && [ -z "$SHARD_COUNT" ] || {
      echo smoke_does_not_accept_shard_arguments
      exit 20
    }
    QUEUE=${4:-autox}
    STAMP=${5:-20260729_submap_v1_${DATASET}_unified_smoke_first_attempt}
    MANIFEST=$SMOKE_MANIFEST
    MANIFEST_SHA256=$SMOKE_MANIFEST_SHA256
    EXPECTED_CHUNKS=1
    EXPECTED_EPISODES=5
    WALLTIME=36:00:00
    if [ "$DATASET" = hm3d ]; then
      [ "$CALIBRATION_ARG" = - ] || {
        echo hm3d_smoke_calibration_argument_must_be_dash
        exit 20
      }
      [ -f "$CALIBRATION_MANIFEST" ] || {
        echo "missing_calibration_manifest=$CALIBRATION_MANIFEST"
        exit 21
      }
      CALIBRATION_MANIFEST_SHA256=$(
        sha256sum "$CALIBRATION_MANIFEST" | awk '{print $1}'
      )
      [ "$CALIBRATION_MANIFEST_SHA256" = \
        "$CALIBRATION_MANIFEST_EXPECTED_SHA256" ] || {
        echo calibration_manifest_hash_mismatch
        exit 22
      }
    else
      CALIBRATION_JSON=$CALIBRATION_ARG
      CALIBRATION_MANIFEST=""
    fi
    ;;
  full)
    QUEUE=${4:-autox}
    STAMP=${5:-20260729_submap_v1_${DATASET}150_first_attempt}
    MANIFEST=$FULL_MANIFEST
    MANIFEST_SHA256=$FULL_MANIFEST_SHA256
    EXPECTED_CHUNKS=5
    EXPECTED_EPISODES=150
    if [ -n "$SHARD_INDEX" ] || [ -n "$SHARD_COUNT" ]; then
      [ "$SHARD_COUNT" = 4 ] || {
        echo shard_count_must_be_4
        exit 20
      }
      case "$DATASET:$SHARD_INDEX" in
        hm3d:0)
          MANIFEST=$MANIFEST_ROOT/hm3d150_shard0of4.json
          MANIFEST_SHA256=b6e8a4ac3c006358fb2dd04b90726f32e5bfe771319350443d7668446f0f41d6
          ;;
        hm3d:1)
          MANIFEST=$MANIFEST_ROOT/hm3d150_shard1of4.json
          MANIFEST_SHA256=d0b0b9145e8195d071085e9470f136d7420871d552d7ce8f71a2df3c67dca56c
          ;;
        hm3d:2)
          MANIFEST=$MANIFEST_ROOT/hm3d150_shard2of4.json
          MANIFEST_SHA256=9cb69163e4b77e942b8c47ae4e491c2fe0ce805024d27861dc9626299c102f17
          ;;
        hm3d:3)
          MANIFEST=$MANIFEST_ROOT/hm3d150_shard3of4.json
          MANIFEST_SHA256=17186739f6dfa865614b0565cabdb4ad3d7aded4ca3320b0a6c210e61cdc9bb2
          ;;
        mp3d:0)
          MANIFEST=$MANIFEST_ROOT/mp3d150_shard0of4.json
          MANIFEST_SHA256=4c3c549be9515d486fe9f17317ce34915559f411703f44256bd116f414caa7b4
          ;;
        mp3d:1)
          MANIFEST=$MANIFEST_ROOT/mp3d150_shard1of4.json
          MANIFEST_SHA256=4f2c9c60aaa8b35f92e7fb33b77a6405d1beaab4555c7f109ccc8632a0ea53bc
          ;;
        mp3d:2)
          MANIFEST=$MANIFEST_ROOT/mp3d150_shard2of4.json
          MANIFEST_SHA256=29ab581d29c141ad70b40660494fdd76e6065bf7f5ff2ccc3c2785996810ed80
          ;;
        mp3d:3)
          MANIFEST=$MANIFEST_ROOT/mp3d150_shard3of4.json
          MANIFEST_SHA256=b02dfc71d26c3046a38be0472a72fdb90774a42d080e6c33ba7015d3684ab156
          ;;
        *) echo shard_index_must_be_0_to_3; exit 20 ;;
      esac
      if [ "$SHARD_INDEX" = 0 ]; then
        EXPECTED_CHUNKS=2
        EXPECTED_EPISODES=60
      else
        EXPECTED_CHUNKS=1
        EXPECTED_EPISODES=30
      fi
    else
      SHARD_INDEX=""
      SHARD_COUNT=""
    fi
    WALLTIME=96:00:00
    CALIBRATION_JSON=$CALIBRATION_ARG
    CALIBRATION_MANIFEST=""
    ;;
  *) echo "invalid_mode=$MODE"; exit 20 ;;
esac
case "$QUEUE" in auto|autox) ;; *) echo "queue_must_be_auto_or_autox"; exit 20 ;; esac
if [ -n "$CALIBRATION_JSON" ]; then
  [ -f "$CALIBRATION_JSON" ] || {
    echo "missing_calibration_json=$CALIBRATION_JSON"
    exit 21
  }
  "$ASCENT_PYTHON" - "$CALIBRATION_JSON" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["status"] == "PASS"
assert value["navigation_metrics_used"] is False
assert value["config"]["provisional_thresholds"] is False
PY
  CALIBRATION_JSON_SHA256=$(
    sha256sum "$CALIBRATION_JSON" | awk '{print $1}'
  )
fi

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

"$ASCENT_PYTHON" - "$MANIFEST" "$EXPECTED_CHUNKS" "$EXPECTED_EPISODES" "$DATASET" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
value = json.load(open(sys.argv[1], encoding="utf-8"))
expected_chunks, expected_episodes = int(sys.argv[2]), int(sys.argv[3])
expected_dataset = sys.argv[4]
assert value["schema"] == "ascent_vo_submap_screen_materialized_v1"
assert value["dataset"] == expected_dataset
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
    "dataset": "$DATASET",
    "mode": "$MODE",
    "shard_index": "$SHARD_INDEX" or None,
    "shard_count": "$SHARD_COUNT" or None,
    "queue_request": "$QUEUE",
    "walltime": "$WALLTIME",
    "stamp": "$STAMP",
    "source_commit": "$SOURCE_COMMIT",
    "resource_commit": "$RESOURCE_COMMIT",
    "pointnav_vo_commit": "$POINTNAV_VO_COMMIT",
    "artifact_root": "$ARTIFACT_ROOT",
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
PBS_LOG=$LOG_ROOT/submap_${DATASET}_${MODE}_${STAMP}.pbs.log
SUBMISSION_JSON=$SUBMISSION_ROOT/submap_${DATASET}_${MODE}_${STAMP}.json
[ ! -e "$PBS_LOG" ] || { echo "pbs_log_exists=$PBS_LOG"; exit 24; }
[ ! -e "$SUBMISSION_JSON" ] || {
  echo "submission_record_exists=$SUBMISSION_JSON"
  exit 24
}

ENVIRONMENT=$(IFS=,; echo \
"ASCENT_SUBMAP_DATASET=$DATASET,\
ASCENT_SUBMAP_MODE=$MODE,\
ASCENT_SUBMAP_ARTIFACT_ROOT=$ARTIFACT_ROOT,\
ASCENT_SUBMAP_SHARD_INDEX=$SHARD_INDEX,\
ASCENT_SUBMAP_SHARD_COUNT=$SHARD_COUNT,\
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
    "dataset": "$DATASET",
    "mode": "$MODE",
    "shard_index": "$SHARD_INDEX" or None,
    "shard_count": "$SHARD_COUNT" or None,
    "queue_request": "$QUEUE",
    "walltime": "$WALLTIME",
    "stamp": "$STAMP",
    "source_commit": "$SOURCE_COMMIT",
    "resource_commit": "$RESOURCE_COMMIT",
    "pointnav_vo_commit": "$POINTNAV_VO_COMMIT",
    "artifact_root": "$ARTIFACT_ROOT",
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
