#!/bin/bash
# Hash-bound submission for one of three HM3D v1.4 case-audit shards.

set -euo pipefail

USAGE='submit_case_audit.sh 0|1|2 auto|autox [stamp]'
SHARD=${1:?usage: $USAGE}
QUEUE=${2:?usage: $USAGE}
STAMP=${3:-20260812_v1_4_case_audit_shadow_s${SHARD}_first_attempt}
NO_QSUB=${ASCENT_V14_CASE_AUDIT_NO_QSUB:-0}

case "$SHARD" in 0|1|2) ;; *) echo invalid_shard; exit 20 ;; esac
case "$QUEUE" in auto|autox) ;; *) echo queue_must_be_auto_or_autox; exit 20 ;; esac
case "$NO_QSUB" in 0|1) ;; *) echo invalid_no_qsub; exit 20 ;; esac

PROJECT=/scratch/e1538633/liuyi/drift-aware-submap-exploration
SOURCE_ROOT=$PROJECT/external/ascent_vo_submap_v1_4_case_audit
RESOURCE_ROOT=$PROJECT/external/ascent
POINTNAV_VO_ROOT=$PROJECT/external/PointNav-VO
ARTIFACT_ROOT=$PROJECT/artifacts/objectnav/submap_v1_4_case_audit
INPUT_ROOT=$ARTIFACT_ROOT/inputs/hm3d_train180_case_audit_20260812_v1
MANIFEST=$INPUT_ROOT/shard${SHARD}_materialized/chunk_manifest.json
ORIGINAL_BASELINE=$INPUT_ROOT/shard${SHARD}/original_ascent_baseline.csv
PREPARATION_AUDIT=$INPUT_ROOT/preparation_audit.json
GATE_MANIFEST=$PROJECT/artifacts/objectnav/submap_v1/manifests/hm3d_smoke5_relocated.json
CONTROLLER=$SOURCE_ROOT/pbs/run_place_memory_3shared.pbs
WORKER=$SOURCE_ROOT/pbs/run_place_memory_lane.sh
ASCENT_PYTHON=/scratch/e1538633/liuyi/micromamba/envs/ascent_nav/bin/python
WALLTIME=24:00:00
EXPECTED_CHUNKS=6
EXPECTED_EPISODES=60
DATASET=hm3d
EXPERIMENT=case_shard
MODE=full

MANIFEST_HASHES=(
  91ad35c1a68b03da03e6defbf2e64494cd8a412e8914c050d8cedea71540811b
  d34109c425c4771d668082bb866daa7b5d57b046356ae37bce017905fec67242
  4793c758d3c3d2f58f03fb22807ec381cb356f59bba4c6dd44b5ca58cbf4b6a8
)
BASELINE_HASHES=(
  30a050e7768a42a7a153c5d1799987517dae25d92fa9cb0949a487d1c27fcb9a
  d397c93d263480b1f95931f0536a4419d48c62b69a010a2764eb8bed9e6077a6
  ce13cd2ffb80646d92e0a84d9773ffdac705cea55a483dd30f957e122924405d
)
MANIFEST_SHA256=${MANIFEST_HASHES[$SHARD]}
ORIGINAL_BASELINE_SHA256=${BASELINE_HASHES[$SHARD]}
PREPARATION_AUDIT_SHA256=78d9272ae54017a17a892b704a33275740a3293b35e01b3f2512c61fe1983b4a
GATE_MANIFEST_SHA256=5166c2ab2000799a05c30aaa56e11ccf828b20c66ae2f4fc8a01065f53d49538

for path in "$MANIFEST" "$ORIGINAL_BASELINE" "$PREPARATION_AUDIT" \
  "$GATE_MANIFEST" "$CONTROLLER" "$WORKER"; do
  [ -f "$path" ] || { echo "missing_file=$path"; exit 21; }
done
[ "$(sha256sum "$MANIFEST" | awk '{print $1}')" = "$MANIFEST_SHA256" ] || {
  echo manifest_hash_mismatch; exit 22;
}
[ "$(sha256sum "$ORIGINAL_BASELINE" | awk '{print $1}')" = "$ORIGINAL_BASELINE_SHA256" ] || {
  echo original_baseline_hash_mismatch; exit 22;
}
[ "$(sha256sum "$PREPARATION_AUDIT" | awk '{print $1}')" = "$PREPARATION_AUDIT_SHA256" ] || {
  echo preparation_audit_hash_mismatch; exit 22;
}
[ "$(sha256sum "$GATE_MANIFEST" | awk '{print $1}')" = "$GATE_MANIFEST_SHA256" ] || {
  echo gate_manifest_hash_mismatch; exit 22;
}

"$ASCENT_PYTHON" - "$MANIFEST" "$ORIGINAL_BASELINE" \
  "$PREPARATION_AUDIT" "$SHARD" <<'PY'
import csv, json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
baseline = list(csv.DictReader(open(sys.argv[2], encoding="utf-8")))
audit = json.load(open(sys.argv[3], encoding="utf-8"))
shard = int(sys.argv[4])
assert manifest["schema"] == "ascent_vo_submap_screen_materialized_v1"
assert manifest["dataset"] == "hm3d" and manifest["split"] == "train"
assert manifest["episode_count"] == 60 and manifest["chunk_count"] == 6
assert len(manifest["logical_case_ids"]) == 60
assert len(baseline) == 60
assert [row["logical_case_id"] for row in baseline] == manifest["logical_case_ids"]
assert all(float(row["original_success"]) >= 0.5 for row in baseline)
assert audit["status"] == "PASS" and audit["total_episode_count"] == 180
entry = audit["shards"][shard]
assert entry["shard"] == shard and entry["episode_count"] == 60
assert entry["target_counts"] == {
    "bed": 10, "chair": 10, "plant": 10,
    "sofa": 10, "toilet": 10, "tv_monitor": 10,
}
PY

[ -z "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=all)" ] || {
  echo source_worktree_not_clean; git -C "$SOURCE_ROOT" status --short; exit 23;
}
SOURCE_COMMIT=$(git -C "$SOURCE_ROOT" rev-parse HEAD)
[ "$(git -C "$SOURCE_ROOT" merge-base HEAD f238fc5)" = \
  f238fc5dd8863f190ad9c1c39422dcc066ccda28 ] || {
  echo wrong_v1_2_ancestry; exit 23;
}
UPSTREAM=$(git -C "$SOURCE_ROOT" rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)
[ -n "$UPSTREAM" ] || { echo source_branch_has_no_upstream; exit 23; }
git -C "$SOURCE_ROOT" merge-base --is-ancestor "$SOURCE_COMMIT" "$UPSTREAM" || {
  echo source_commit_not_pushed; exit 23;
}
RESOURCE_COMMIT=$(git -C "$RESOURCE_ROOT" rev-parse HEAD)
POINTNAV_VO_COMMIT=$(git -C "$POINTNAV_VO_ROOT" rev-parse HEAD)
CONTROLLER_SHA256=$(sha256sum "$CONTROLLER" | awk '{print $1}')
WORKER_SHA256=$(sha256sum "$WORKER" | awk '{print $1}')

DRY_RUN_JSON=$(
  "$ASCENT_PYTHON" - <<PY
import json
print(json.dumps({
  "status": "PASS",
  "schema": "ascent_vo_submap_v1_4_case_audit_submission_preflight_v1",
  "scientific_role": "case_mining_only_not_method_improvement",
  "shard": $SHARD,
  "queue_request": "$QUEUE",
  "walltime": "$WALLTIME",
  "stamp": "$STAMP",
  "source_commit": "$SOURCE_COMMIT",
  "resource_commit": "$RESOURCE_COMMIT",
  "pointnav_vo_commit": "$POINTNAV_VO_COMMIT",
  "manifest": "$MANIFEST",
  "manifest_sha256": "$MANIFEST_SHA256",
  "original_baseline": "$ORIGINAL_BASELINE",
  "original_baseline_sha256": "$ORIGINAL_BASELINE_SHA256",
  "preparation_audit": "$PREPARATION_AUDIT",
  "preparation_audit_sha256": "$PREPARATION_AUDIT_SHA256",
  "gate_manifest": "$GATE_MANIFEST",
  "gate_manifest_sha256": "$GATE_MANIFEST_SHA256",
  "controller_sha256": "$CONTROLLER_SHA256",
  "worker_sha256": "$WORKER_SHA256",
  "expected_chunks": 6,
  "expected_episodes": 60,
  "paired_conditions": ["B1_ascent_vo", "B2_v1_2_oracle_shadow"],
  "oracle_identity_only": True,
  "gt_policy_output_fields": ["event_sequence", "reference_submap_id"],
  "shadow_only": True,
  "actual_task_memory_action_changes": 0,
  "no_metric_driven_retry": True,
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
PBS_LOG=$LOG_ROOT/case_shard${SHARD}_${STAMP}.pbs.log
SUBMISSION_JSON=$SUBMISSION_ROOT/case_shard${SHARD}_${STAMP}.json
[ ! -e "$PBS_LOG" ] || { echo "pbs_log_exists=$PBS_LOG"; exit 24; }
[ ! -e "$SUBMISSION_JSON" ] || { echo "submission_exists=$SUBMISSION_JSON"; exit 24; }

ENVIRONMENT=$(IFS=,; echo \
"ASCENT_PLACE_MEMORY_V14_DATASET=$DATASET,\
ASCENT_PLACE_MEMORY_V14_MODE=$MODE,\
ASCENT_PLACE_MEMORY_V14_EXPERIMENT=$EXPERIMENT,\
ASCENT_PLACE_MEMORY_V14_ARTIFACT_ROOT=$ARTIFACT_ROOT,\
ASCENT_PLACE_MEMORY_V14_MANIFEST=$MANIFEST,\
ASCENT_PLACE_MEMORY_V14_MANIFEST_SHA256=$MANIFEST_SHA256,\
ASCENT_PLACE_MEMORY_V14_EXPECTED_CHUNKS=$EXPECTED_CHUNKS,\
ASCENT_PLACE_MEMORY_V14_EXPECTED_EPISODES=$EXPECTED_EPISODES,\
ASCENT_PLACE_MEMORY_V14_GATE_MANIFEST=$GATE_MANIFEST,\
ASCENT_PLACE_MEMORY_V14_GATE_MANIFEST_SHA256=$GATE_MANIFEST_SHA256,\
ASCENT_PLACE_MEMORY_V14_SOURCE_COMMIT=$SOURCE_COMMIT,\
ASCENT_PLACE_MEMORY_V14_RESOURCE_COMMIT=$RESOURCE_COMMIT,\
ASCENT_PLACE_MEMORY_V14_POINTNAV_VO_COMMIT=$POINTNAV_VO_COMMIT,\
ASCENT_PLACE_MEMORY_V14_CONTROLLER_SHA256=$CONTROLLER_SHA256,\
ASCENT_PLACE_MEMORY_V14_WORKER_SHA256=$WORKER_SHA256,\
ASCENT_PLACE_MEMORY_V14_STAMP=$STAMP,\
ASCENT_PLACE_MEMORY_V14_ORIGINAL_BASELINE_CSV=$ORIGINAL_BASELINE,\
ASCENT_PLACE_MEMORY_V14_ORIGINAL_BASELINE_SHA256=$ORIGINAL_BASELINE_SHA256")

JOB_ID=$(
  qsub -q "$QUEUE" -N "asv14c${SHARD}" -l "walltime=$WALLTIME" \
    -o "$PBS_LOG" -v "$ENVIRONMENT" "$CONTROLLER"
)
printf '%s\n' "$JOB_ID"
"$ASCENT_PYTHON" - "$SUBMISSION_JSON" "$JOB_ID" "$PBS_LOG" <<PY
import json, sys
from pathlib import Path
value = json.loads('''$DRY_RUN_JSON''')
value.update({"job_id": sys.argv[2], "pbs_log": sys.argv[3], "qsub_executed": True})
with Path(sys.argv[1]).open("x", encoding="utf-8") as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
