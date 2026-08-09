#!/bin/bash
# Submit one hash-bound offline VPR retrieval/verification/scoring job.

set -euo pipefail

USAGE='submit_vpr_shadow_offline.sh ROLE RETRIEVER CAPTURE_GATE_DIR [auto|autox] [CALIBRATION_FILE] [STAMP]'
ROLE=${1:?usage: $USAGE}
RETRIEVER=${2:?usage: $USAGE}
CAPTURE_GATE_DIR=${3:?usage: $USAGE}
QUEUE=${4:-auto}
CALIBRATION_FILE=${5:-}
STAMP=${6:-20260809_vpr_offline_${ROLE}_${RETRIEVER}_first_attempt}
NO_QSUB=${ASCENT_VPR_OFFLINE_NO_QSUB:-0}

PROJECT=/scratch/e1538633/liuyi/drift-aware-submap-exploration
SOURCE_ROOT=$PROJECT/external/ascent_vo_submap_v1_4
ARTIFACT_ROOT=$PROJECT/artifacts/objectnav/vpr_shadow
MODEL_REGISTRY=$SOURCE_ROOT/experiments/vpr_shadow/model_registry.json
MODEL_GATE=$ARTIFACT_ROOT/local_model_gate_20260809_v2/model_gate.json
RUNNER=$SOURCE_ROOT/pbs/run_vpr_shadow_offline.pbs
ASCENT_PYTHON=/scratch/e1538633/liuyi/micromamba/envs/ascent_nav/bin/python

case "$QUEUE" in auto|autox) ;; *) echo queue_must_be_auto_or_autox; exit 20 ;; esac
case "$RETRIEVER" in mixvpr|megaloc) ;; *) echo invalid_retriever; exit 20 ;; esac
[[ "$STAMP" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo invalid_stamp; exit 20; }
case "$ROLE" in
  hm3d_cal50)
    SPLIT_NAME=hm3d_cal50.json
    EXPECTED_ROLE=threshold_calibration_only
    JOB_NAME=av14c${RETRIEVER:0:1}
    WALLTIME=48:00:00
    [ -z "$CALIBRATION_FILE" ] || { echo calibration_forbidden_for_cal50; exit 20; }
    ;;
  hm3d_test100)
    SPLIT_NAME=hm3d_test100.json
    EXPECTED_ROLE=locked_in_distribution_test
    JOB_NAME=av14h${RETRIEVER:0:1}
    WALLTIME=72:00:00
    [ -n "$CALIBRATION_FILE" ] || { echo calibration_required; exit 20; }
    ;;
  mp3d_test150)
    SPLIT_NAME=mp3d_test150.json
    EXPECTED_ROLE=locked_cross_dataset_transfer_test
    JOB_NAME=av14m${RETRIEVER:0:1}
    WALLTIME=96:00:00
    [ -n "$CALIBRATION_FILE" ] || { echo calibration_required; exit 20; }
    ;;
  *) echo invalid_role; exit 20 ;;
esac

CAPTURE_GATE_DIR=$(realpath "$CAPTURE_GATE_DIR")
CAPTURE_SUMMARY=$CAPTURE_GATE_DIR/summary.json
CAPTURE_EPISODES=$CAPTURE_GATE_DIR/episodes.jsonl
SPLIT_MANIFEST=$SOURCE_ROOT/experiments/vpr_shadow/manifests/$SPLIT_NAME
for path in "$SOURCE_ROOT" "$RUNNER" "$MODEL_REGISTRY" "$MODEL_GATE" \
  "$CAPTURE_SUMMARY" "$CAPTURE_EPISODES" "$SPLIT_MANIFEST"; do
  [ -e "$path" ] || { echo "missing_required_path=$path"; exit 21; }
done
[ -z "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=all)" ] || {
  echo source_worktree_not_clean; exit 23;
}
SOURCE_COMMIT=$(git -C "$SOURCE_ROOT" rev-parse HEAD)
UPSTREAM=$(git -C "$SOURCE_ROOT" rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)
[ -n "$UPSTREAM" ] || { echo source_branch_has_no_upstream; exit 23; }
git -C "$SOURCE_ROOT" merge-base --is-ancestor "$SOURCE_COMMIT" "$UPSTREAM" || {
  echo source_commit_not_pushed; exit 23;
}

RUNNER_SHA256=$(sha256sum "$RUNNER" | awk '{print $1}')
MODEL_REGISTRY_SHA256=$(sha256sum "$MODEL_REGISTRY" | awk '{print $1}')
CAPTURE_SUMMARY_SHA256=$(sha256sum "$CAPTURE_SUMMARY" | awk '{print $1}')
CAPTURE_EPISODES_SHA256=$(sha256sum "$CAPTURE_EPISODES" | awk '{print $1}')
SPLIT_MANIFEST_SHA256=$(sha256sum "$SPLIT_MANIFEST" | awk '{print $1}')
CALIBRATION_FILE_SHA256=
if [ -n "$CALIBRATION_FILE" ]; then
  CALIBRATION_FILE=$(realpath "$CALIBRATION_FILE")
  [ -f "$CALIBRATION_FILE" ] || { echo missing_calibration_file; exit 21; }
  CALIBRATION_FILE_SHA256=$(sha256sum "$CALIBRATION_FILE" | awk '{print $1}')
fi

"$ASCENT_PYTHON" - "$MODEL_GATE" "$MODEL_REGISTRY_SHA256" \
  "$CAPTURE_SUMMARY" "$CAPTURE_EPISODES_SHA256" "$SPLIT_MANIFEST_SHA256" \
  "$SOURCE_COMMIT" "$EXPECTED_ROLE" "$RETRIEVER" "$CALIBRATION_FILE" <<'PY'
import json, sys
model_gate = json.load(open(sys.argv[1], encoding="utf-8"))
assert model_gate["schema"] == "ascent_v1_4_vpr_shadow_model_gate_v1"
assert model_gate["technical_status"] == "PASS"
assert model_gate["registry_sha256"] == sys.argv[2]
capture = json.load(open(sys.argv[3], encoding="utf-8"))
assert capture["schema"] == "ascent_v1_4_vpr_shadow_capture_gate_v1"
assert capture["technical_status"] == "PASS"
assert capture["strict_error_count"] == 0
assert capture["provenance"]["episodes_sha256"] == sys.argv[4]
assert capture["provenance"]["split_manifest_sha256"] == sys.argv[5]
assert capture["provenance"]["source_commit"] == sys.argv[6]
assert capture["role"] == sys.argv[7]
assert capture["navigation_metrics_emitted"] is False
assert capture["association_scores_emitted"] is False
if sys.argv[9]:
    calibration = json.load(open(sys.argv[9], encoding="utf-8"))
    assert calibration["schema"] == "ascent_v1_4_vpr_shadow_calibration_v1"
    assert calibration["calibration_gate"] == "PASS"
    assert calibration["retriever"] == sys.argv[8]
PY

DRY_RUN_JSON=$(
  "$ASCENT_PYTHON" - <<PY
import json
print(json.dumps({
    "status": "PASS",
    "schema": "ascent_v1_4_vpr_shadow_offline_submission_preflight_v1",
    "role": "$ROLE",
    "retriever": "$RETRIEVER",
    "queue_request": "$QUEUE",
    "walltime": "$WALLTIME",
    "stamp": "$STAMP",
    "source_commit": "$SOURCE_COMMIT",
    "capture_summary": "$CAPTURE_SUMMARY",
    "capture_summary_sha256": "$CAPTURE_SUMMARY_SHA256",
    "capture_episodes_sha256": "$CAPTURE_EPISODES_SHA256",
    "split_manifest_sha256": "$SPLIT_MANIFEST_SHA256",
    "model_registry_sha256": "$MODEL_REGISTRY_SHA256",
    "calibration_file": "$CALIBRATION_FILE" or None,
    "calibration_file_sha256": "$CALIBRATION_FILE_SHA256" or None,
    "raw_all_units_before_gt_scoring": True,
    "planner_action_change": False,
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
PBS_LOG=$LOG_ROOT/vpr_offline_${ROLE}_${RETRIEVER}_${STAMP}.pbs.log
SUBMISSION_JSON=$SUBMISSION_ROOT/vpr_offline_${ROLE}_${RETRIEVER}_${STAMP}.json
[ ! -e "$PBS_LOG" ] || { echo "pbs_log_exists=$PBS_LOG"; exit 24; }
[ ! -e "$SUBMISSION_JSON" ] || { echo "submission_record_exists=$SUBMISSION_JSON"; exit 24; }

ENVIRONMENT=$(IFS=,; echo \
"ASCENT_VPR_OFFLINE_ROLE=$ROLE,\
ASCENT_VPR_OFFLINE_RETRIEVER=$RETRIEVER,\
ASCENT_VPR_OFFLINE_CAPTURE_SUMMARY=$CAPTURE_SUMMARY,\
ASCENT_VPR_OFFLINE_CAPTURE_EPISODES=$CAPTURE_EPISODES,\
ASCENT_VPR_OFFLINE_SPLIT_MANIFEST=$SPLIT_MANIFEST,\
ASCENT_VPR_OFFLINE_CALIBRATION_FILE=$CALIBRATION_FILE,\
ASCENT_VPR_OFFLINE_SOURCE_COMMIT=$SOURCE_COMMIT,\
ASCENT_VPR_OFFLINE_RUNNER_SHA256=$RUNNER_SHA256,\
ASCENT_VPR_OFFLINE_MODEL_REGISTRY_SHA256=$MODEL_REGISTRY_SHA256,\
ASCENT_VPR_OFFLINE_CAPTURE_SUMMARY_SHA256=$CAPTURE_SUMMARY_SHA256,\
ASCENT_VPR_OFFLINE_CAPTURE_EPISODES_SHA256=$CAPTURE_EPISODES_SHA256,\
ASCENT_VPR_OFFLINE_SPLIT_MANIFEST_SHA256=$SPLIT_MANIFEST_SHA256,\
ASCENT_VPR_OFFLINE_CALIBRATION_FILE_SHA256=$CALIBRATION_FILE_SHA256,\
ASCENT_VPR_OFFLINE_STAMP=$STAMP")
JOB_ID=$(
  qsub -q "$QUEUE" -N "$JOB_NAME" -l "walltime=$WALLTIME" \
    -o "$PBS_LOG" -v "$ENVIRONMENT" "$RUNNER"
)
printf '%s\n' "$JOB_ID"
"$ASCENT_PYTHON" - "$SUBMISSION_JSON" "$JOB_ID" "$PBS_LOG" <<PY
import json, sys
output = json.loads('''$DRY_RUN_JSON''')
output.update({
    "schema": "ascent_v1_4_vpr_shadow_offline_submission_v1",
    "job_id": sys.argv[2],
    "pbs_log": sys.argv[3],
    "qsub_executed": True,
})
with open(sys.argv[1], "x", encoding="utf-8") as handle:
    json.dump(output, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
