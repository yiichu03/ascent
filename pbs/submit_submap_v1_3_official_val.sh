#!/bin/bash
# Submit one hash-bound official-val shard for frozen submap-v1.3.

set -euo pipefail

USAGE='submit_submap_v1_3_official_val.sh JOB_CONFIG auto|autox STAMP smoke|full [SMOKE_SUMMARY]'
CONFIG=${1:?usage: $USAGE}
QUEUE=${2:?usage: $USAGE}
STAMP=${3:?usage: $USAGE}
RUN_STAGE=${4:?usage: $USAGE}
SMOKE_SUMMARY=${5:-}
NO_QSUB=${ASCENT_SUBMAP_V13_OFFVAL_NO_QSUB:-0}

PROJECT=/scratch/e1538633/liuyi/drift-aware-submap-exploration
SOURCE_ROOT=$PROJECT/external/ascent_vo_submap_v1_3
RESOURCE_ROOT=$PROJECT/external/ascent
POINTNAV_VO_ROOT=$PROJECT/external/PointNav-VO
CONTROLLER=$SOURCE_ROOT/pbs/run_submap_v1_3_official_val_3shared.pbs
WORKER=$SOURCE_ROOT/pbs/run_submap_v1_3_lane.sh
SCRATCH_ROOT=/scratch/e1538633/liuyi
ASCENT_PYTHON=$SCRATCH_ROOT/micromamba/envs/ascent_nav/bin/python
EXPECTED_ARTIFACT_ROOT=$PROJECT/artifacts/objectnav/submap_v1_3

case "$QUEUE" in auto|autox) ;; *) echo queue_must_be_auto_or_autox; exit 20 ;; esac
case "$RUN_STAGE" in smoke|full) ;; *) echo stage_must_be_smoke_or_full; exit 20 ;; esac
[[ "$STAMP" =~ ^[A-Za-z0-9._-]+$ ]] || { echo invalid_stamp; exit 20; }
CONFIG=$(readlink -f "$CONFIG")
for path in "$CONFIG" "$CONTROLLER" "$WORKER"; do
  [ -f "$path" ] || { echo "missing_file=$path"; exit 21; }
done

mapfile -t CONFIG_VALUES < <(
  "$ASCENT_PYTHON" - "$CONFIG" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["schema"] == "ascent_vo_submap_v1_3_official_val_job_v1"
for field in (
    "job_key", "job_name", "dataset", "artifact_root", "manifest",
    "manifest_sha256", "gate_manifest", "gate_manifest_sha256",
    "selection", "selection_sha256", "baseline_csv", "baseline_sha256",
    "preparation_audit", "preparation_audit_sha256", "expected_episodes",
    "expected_chunks", "scientific_start_index",
    "scientific_stop_index_exclusive", "expected_official_total",
):
    print(value[field])
PY
)
[ "${#CONFIG_VALUES[@]}" = 19 ] || { echo config_field_count; exit 22; }
JOB_KEY=${CONFIG_VALUES[0]}
JOB_NAME=${CONFIG_VALUES[1]}
DATASET=${CONFIG_VALUES[2]}
ARTIFACT_ROOT=${CONFIG_VALUES[3]}
MANIFEST=${CONFIG_VALUES[4]}
MANIFEST_SHA256=${CONFIG_VALUES[5]}
GATE_MANIFEST=${CONFIG_VALUES[6]}
GATE_MANIFEST_SHA256=${CONFIG_VALUES[7]}
SELECTION=${CONFIG_VALUES[8]}
SELECTION_SHA256=${CONFIG_VALUES[9]}
BASELINE_CSV=${CONFIG_VALUES[10]}
BASELINE_SHA256=${CONFIG_VALUES[11]}
PREPARATION_AUDIT=${CONFIG_VALUES[12]}
PREPARATION_AUDIT_SHA256=${CONFIG_VALUES[13]}
EXPECTED_EPISODES=${CONFIG_VALUES[14]}
EXPECTED_CHUNKS=${CONFIG_VALUES[15]}
SCIENTIFIC_START=${CONFIG_VALUES[16]}
SCIENTIFIC_STOP=${CONFIG_VALUES[17]}
EXPECTED_OFFICIAL_TOTAL=${CONFIG_VALUES[18]}
case "$DATASET" in hm3d|mp3d) ;; *) echo invalid_dataset; exit 20 ;; esac
[[ "$JOB_NAME" =~ ^[A-Za-z][A-Za-z0-9_-]{0,14}$ ]] || {
  echo invalid_job_name; exit 20;
}
[ "$(readlink -f "$ARTIFACT_ROOT")" = "$EXPECTED_ARTIFACT_ROOT" ] || {
  echo invalid_artifact_root; exit 22;
}

for path in "$MANIFEST" "$GATE_MANIFEST" "$SELECTION" "$BASELINE_CSV" \
  "$PREPARATION_AUDIT"; do
  [ -f "$path" ] || { echo "missing_input=$path"; exit 21; }
done
check_hash() {
  local path=$1 expected=$2 label=$3
  [ "$(sha256sum "$path" | awk '{print $1}')" = "$expected" ] || {
    echo "${label}_hash_mismatch"; exit 22;
  }
}
CONFIG_SHA256=$(sha256sum "$CONFIG" | awk '{print $1}')
check_hash "$MANIFEST" "$MANIFEST_SHA256" manifest
check_hash "$GATE_MANIFEST" "$GATE_MANIFEST_SHA256" gate_manifest
check_hash "$SELECTION" "$SELECTION_SHA256" selection
check_hash "$BASELINE_CSV" "$BASELINE_SHA256" baseline
check_hash "$PREPARATION_AUDIT" "$PREPARATION_AUDIT_SHA256" preparation_audit

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

"$ASCENT_PYTHON" - "$CONFIG" "$EXPECTED_ARTIFACT_ROOT" <<'PY'
import csv, gzip, hashlib, json, sys
from pathlib import Path

config_path, expected_artifact_root = map(Path, sys.argv[1:])
value = json.load(config_path.open(encoding="utf-8"))

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

assert value["schema"] == "ascent_vo_submap_v1_3_official_val_job_v1"
assert value["scientific_split"] == "val"
assert value["transport_split"] == "train"
assert value["selection_rule"] == "canonical_official_val_contiguous_index_no_metric_filter"
assert value["metrics_used_for_selection"] is False
assert value["pose_source"] == "zhao_rgbd_2021"
assert value["policy_gt_isolation"] is True
assert value["evaluation_gt_only"] is True
assert value["method_version"] == "submap_v1.3"
assert value["automatic_technical_retry_per_failed_unit"] == 0
assert value["metric_driven_retry"] is False
assert value["lane_count"] == 3
assert value["gate_episodes"] == 5 and value["gate_chunks"] == 1
assert Path(value["artifact_root"]).resolve() == expected_artifact_root.resolve()
start = int(value["scientific_start_index"])
stop = int(value["scientific_stop_index_exclusive"])
expected = int(value["expected_episodes"])
assert stop - start == expected and 0 <= start < stop
assert stop <= int(value["expected_official_total"])

selection_path = Path(value["selection"])
with selection_path.open(newline="", encoding="utf-8") as handle:
    selection = list(csv.DictReader(handle))
dataset = value["dataset"]
ids = [row["logical_case_id"] for row in selection]
assert ids == [f"{dataset}_val_{index:04d}" for index in range(start, stop)]
assert [int(row["dataset_case_index"]) for row in selection] == list(range(start, stop))
assert all(row["selection_rule"] == value["selection_rule"] for row in selection)

for key, hash_key in (
    ("manifest", "manifest_sha256"),
    ("gate_manifest", "gate_manifest_sha256"),
    ("selection", "selection_sha256"),
    ("baseline_csv", "baseline_sha256"),
    ("preparation_audit", "preparation_audit_sha256"),
):
    assert sha256(Path(value[key])) == value[hash_key]
manifest = json.load(open(value["manifest"], encoding="utf-8"))
gate = json.load(open(value["gate_manifest"], encoding="utf-8"))
assert manifest["schema"] == "ascent_vo_submap_screen_materialized_v1"
assert manifest["dataset"] == dataset and manifest["split"] == "train"
assert manifest["logical_case_ids"] == ids
assert manifest["episode_count"] == expected
assert len(manifest["chunks"]) == int(value["expected_chunks"])
assert sum(int(chunk["episode_count"]) for chunk in manifest["chunks"]) == expected
assert gate["dataset"] == dataset and gate["split"] == "train"
assert gate["logical_case_ids"] == value["smoke_logical_case_ids"]
assert gate["episode_count"] == 5 and len(gate["chunks"]) == 1

audit = json.load(open(value["preparation_audit"], encoding="utf-8"))
assert audit["status"] == "PASS" and audit["metrics_used_for_selection"] is False
assert audit["selected_start_index"] == start
assert audit["selected_stop_index_exclusive"] == stop
transport = Path(audit["transport_root_file"])
assert sha256(transport) == audit["transport_root_file_sha256"]
with gzip.open(transport, "rt", encoding="utf-8") as handle:
    transport_payload = json.load(handle)
assert "content_scenes_path" not in transport_payload
assert transport_payload["episodes"] == []
PY

SMOKE_SUMMARY_SHA256=
if [ "$RUN_STAGE" = full ]; then
  [ -n "$SMOKE_SUMMARY" ] || { echo full_requires_smoke_summary; exit 20; }
  SMOKE_SUMMARY=$(readlink -f "$SMOKE_SUMMARY")
  [ -f "$SMOKE_SUMMARY" ] || { echo missing_smoke_summary; exit 21; }
  SMOKE_SUMMARY_SHA256=$(sha256sum "$SMOKE_SUMMARY" | awk '{print $1}')
fi

DRY_RUN_JSON=$(
  "$ASCENT_PYTHON" - "$CONFIG" "$CONFIG_SHA256" "$QUEUE" "$STAMP" \
    "$SOURCE_COMMIT" "$RESOURCE_COMMIT" "$POINTNAV_VO_COMMIT" \
    "$CONTROLLER_SHA256" "$WORKER_SHA256" "$RUN_STAGE" \
    "$SMOKE_SUMMARY" "$SMOKE_SUMMARY_SHA256" <<'PY'
import json, sys
config = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps({
    "schema": "ascent_vo_submap_v1_3_official_val_submission_preflight_v1",
    "status": "PASS",
    "job_key": config["job_key"],
    "job_name": config["job_name"],
    "dataset": config["dataset"],
    "scientific_split": "val",
    "transport_split": "train",
    "scientific_start_index": config["scientific_start_index"],
    "scientific_stop_index_exclusive": config["scientific_stop_index_exclusive"],
    "expected_episodes": config["expected_episodes"],
    "expected_chunks": config["expected_chunks"],
    "queue_request": sys.argv[3],
    "stamp": sys.argv[4],
    "source_commit": sys.argv[5],
    "resource_commit": sys.argv[6],
    "pointnav_vo_commit": sys.argv[7],
    "config": sys.argv[1],
    "config_sha256": sys.argv[2],
    "controller_sha256": sys.argv[8],
    "worker_sha256": sys.argv[9],
    "run_stage": sys.argv[10],
    "smoke_summary": sys.argv[11] or None,
    "smoke_summary_sha256": sys.argv[12] or None,
    "artifact_root": config["artifact_root"],
    "selection_rule": config["selection_rule"],
    "metrics_used_for_selection": False,
    "pose_source": "zhao_rgbd_2021",
    "policy_gt_isolation": True,
    "evaluation_gt_only": True,
    "method_version": "submap_v1.3",
    "automatic_technical_retry_per_failed_unit": 0,
    "metric_driven_retry": False,
    "single_five_episode_gate_before_all_shards": True,
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
PBS_LOG=$LOG_ROOT/submap_v1_3_official_val_${RUN_STAGE}_${JOB_KEY}_${STAMP}.pbs.log
SUBMISSION_JSON=$SUBMISSION_ROOT/submap_v1_3_official_val_${RUN_STAGE}_${JOB_KEY}_${STAMP}.json
[ ! -e "$PBS_LOG" ] || { echo "pbs_log_exists=$PBS_LOG"; exit 24; }
[ ! -e "$SUBMISSION_JSON" ] || {
  echo "submission_record_exists=$SUBMISSION_JSON"; exit 24;
}

ENVIRONMENT=$(IFS=,; echo \
"ASCENT_SUBMAP_V13_OFFVAL_CONFIG=$CONFIG,\
ASCENT_SUBMAP_V13_OFFVAL_CONFIG_SHA256=$CONFIG_SHA256,\
ASCENT_SUBMAP_V13_OFFVAL_SOURCE_COMMIT=$SOURCE_COMMIT,\
ASCENT_SUBMAP_V13_OFFVAL_RESOURCE_COMMIT=$RESOURCE_COMMIT,\
ASCENT_SUBMAP_V13_OFFVAL_POINTNAV_VO_COMMIT=$POINTNAV_VO_COMMIT,\
ASCENT_SUBMAP_V13_OFFVAL_CONTROLLER_SHA256=$CONTROLLER_SHA256,\
ASCENT_SUBMAP_V13_OFFVAL_WORKER_SHA256=$WORKER_SHA256,\
ASCENT_SUBMAP_V13_OFFVAL_STAMP=$STAMP,\
ASCENT_SUBMAP_V13_OFFVAL_STAGE=$RUN_STAGE,\
ASCENT_SUBMAP_V13_OFFVAL_SMOKE_SUMMARY=$SMOKE_SUMMARY,\
ASCENT_SUBMAP_V13_OFFVAL_SMOKE_SUMMARY_SHA256=$SMOKE_SUMMARY_SHA256")

SUBMIT_JOB_NAME=$JOB_NAME
if [ "$RUN_STAGE" = smoke ]; then SUBMIT_JOB_NAME=v13smoke; fi

JOB_ID=$(
  qsub -q "$QUEUE" -N "$SUBMIT_JOB_NAME" -l walltime=96:00:00 \
    -o "$PBS_LOG" -v "$ENVIRONMENT" "$CONTROLLER"
)
printf '%s\n' "$JOB_ID"
"$ASCENT_PYTHON" - "$SUBMISSION_JSON" "$JOB_ID" "$PBS_LOG" <<PY
import json, sys
output = json.loads('''$DRY_RUN_JSON''')
output.update({
    "schema": "ascent_vo_submap_v1_3_official_val_submission_v1",
    "job_id": sys.argv[2],
    "pbs_log": sys.argv[3],
    "qsub_executed": True,
})
with open(sys.argv[1], "x", encoding="utf-8") as handle:
    json.dump(output, handle, indent=2, sort_keys=True)
    handle.write("\\n")
PY
