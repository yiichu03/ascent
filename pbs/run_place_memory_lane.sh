#!/bin/bash
# One B2-only ASCENT-VO place-memory-v1.4 lane with independent services.

set -euo pipefail

PROJECT=/scratch/e1538633/liuyi/drift-aware-submap-exploration
SOURCE_ROOT=$PROJECT/external/ascent_vo_submap_v1_4_oracle_task_memory
RESOURCE_ROOT=$PROJECT/external/ascent
POINTNAV_VO_ROOT=$PROJECT/external/PointNav-VO
CHECKPOINT_DIR=$POINTNAV_VO_ROOT/pretrained_ckpts/vo
HEALTH_CHECK=$SOURCE_ROOT/scripts/check_submap_services.py

DATASET=${ASCENT_PLACE_MEMORY_V14_DATASET:?required}
MODE=${ASCENT_PLACE_MEMORY_V14_MODE:?required}
EXPERIMENT=${ASCENT_PLACE_MEMORY_V14_EXPERIMENT:?required}
SCENES_ROOT=${ASCENT_PLACE_MEMORY_V14_SCENES_ROOT:?required}
MANIFEST=${ASCENT_PLACE_MEMORY_V14_MANIFEST:?required}
MANIFEST_SHA256=${ASCENT_PLACE_MEMORY_V14_MANIFEST_SHA256:?required}
SOURCE_COMMIT=${ASCENT_PLACE_MEMORY_V14_SOURCE_COMMIT:?required}
RESOURCE_COMMIT=${ASCENT_PLACE_MEMORY_V14_RESOURCE_COMMIT:?required}
POINTNAV_VO_COMMIT=${ASCENT_PLACE_MEMORY_V14_POINTNAV_VO_COMMIT:?required}
LANE_ID=${ASCENT_PLACE_MEMORY_V14_LANE_ID:?required}
LANE_COUNT=${ASCENT_PLACE_MEMORY_V14_LANE_COUNT:?required}
EXPECTED_CHUNKS=${ASCENT_PLACE_MEMORY_V14_LANE_EXPECTED_CHUNKS:?required}
EXPECTED_EPISODES=${ASCENT_PLACE_MEMORY_V14_LANE_EXPECTED_EPISODES:?required}
STAGE_ROOT=${ASCENT_PLACE_MEMORY_V14_STAGE_ROOT:?required}
BASE_PORT=${ASCENT_PLACE_MEMORY_V14_BASE_PORT:?required}
DEADLINE_EPOCH=${ASCENT_PLACE_MEMORY_V14_DEADLINE_EPOCH:?required}

FORWARD_SHA256=6b571bb717366f7d80f61e919b33a45ac2f45925201c4e3011b3239a2c42e586
TURN_SHA256=c469643f9ab35c9e1058f31fbb672a5fa3adf582987a4388bdd020dd89faf1d9
SCRATCH_ROOT=/scratch/e1538633/liuyi
ASCENT_ENV=$SCRATCH_ROOT/micromamba/envs/ascent_nav
ASCENT_PYTHON=$ASCENT_ENV/bin/python
RUN_ROOT=$STAGE_ROOT/lane_$LANE_ID
MAX_CONSECUTIVE_PROCESS_FAILURES=3

case "$MODE" in smoke|full) ;; *) echo "invalid_mode=$MODE"; exit 20 ;; esac
case "$DATASET" in hm3d|mp3d) ;; *) echo "invalid_dataset=$DATASET"; exit 20 ;; esac
case "$EXPERIMENT" in train150|mechanism_subset) ;; *) echo "invalid_experiment=$EXPERIMENT"; exit 20 ;; esac
for value in "$LANE_ID" "$LANE_COUNT" "$EXPECTED_CHUNKS" \
  "$EXPECTED_EPISODES" "$BASE_PORT" "$DEADLINE_EPOCH"; do
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "invalid_integer=$value"; exit 20; }
done
[ "$LANE_ID" -lt "$LANE_COUNT" ] || { echo invalid_lane; exit 20; }
[ ! -e "$RUN_ROOT" ] || { echo "run_root_exists=$RUN_ROOT"; exit 24; }
mkdir -p "$RUN_ROOT"/{attempts,mpl_config,xdg_cache,service_logs}
mkdir -p "$SCRATCH_ROOT/cache"/{pip,huggingface,torch,xdg,matplotlib,conda_pkgs}

for path in \
  "$MANIFEST" "$HEALTH_CHECK" "$SOURCE_ROOT/ascent/run.py" \
  "$SOURCE_ROOT/ascent/ascent_policy.py" "$SOURCE_ROOT/ascent/submaps" \
  "$SOURCE_ROOT/ascent/vo/pose_provider.py" \
  "$SOURCE_ROOT/ascent/vo/oracle_place.py" "$SOURCE_ROOT/model_api" \
  "$SOURCE_ROOT/dummy_policy.pth" \
  "$SOURCE_ROOT/pretrained_weights/Qwen2.5-7b" \
  "$SOURCE_ROOT/pretrained_weights/mobile_sam.pt" \
  "$SOURCE_ROOT/pretrained_weights/groundingdino_swint_ogc.pth" \
  "$SOURCE_ROOT/pretrained_weights/dfine_x_obj2coco.pth" \
  "$SOURCE_ROOT/pretrained_weights/ram_plus_swin_large_14m.pth" \
  "$SOURCE_ROOT/pretrained_weights/rednet_semmap_mp3d_40.pth" \
  "$SOURCE_ROOT/pretrained_weights/resnet50_places365.pth.tar" \
  "$SOURCE_ROOT/third_party/vlfm/data/pointnav_weights.pth" \
  "$CHECKPOINT_DIR/act_forward.pth" \
  "$CHECKPOINT_DIR/act_left_right_inv_joint.pth" \
  "$SCENES_ROOT/$DATASET"; do
  [ -e "$path" ] || { echo "missing_required_resource=$path"; exit 21; }
done
command -v timeout >/dev/null || { echo missing_timeout; exit 21; }
[ "$(sha256sum "$MANIFEST" | awk '{print $1}')" = "$MANIFEST_SHA256" ] || {
  echo manifest_hash_mismatch; exit 22;
}
[ "$(sha256sum "$CHECKPOINT_DIR/act_forward.pth" | awk '{print $1}')" = "$FORWARD_SHA256" ] || {
  echo forward_checkpoint_hash_mismatch; exit 22;
}
[ "$(sha256sum "$CHECKPOINT_DIR/act_left_right_inv_joint.pth" | awk '{print $1}')" = "$TURN_SHA256" ] || {
  echo turn_checkpoint_hash_mismatch; exit 22;
}
[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" = "$SOURCE_COMMIT" ] || {
  echo source_commit_mismatch; exit 22;
}
[ -z "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=all)" ] || {
  echo source_worktree_not_clean
  git -C "$SOURCE_ROOT" status --short
  exit 23
}
[ "$(git -C "$RESOURCE_ROOT" rev-parse HEAD)" = "$RESOURCE_COMMIT" ] || {
  echo resource_commit_mismatch; exit 22;
}
[ "$(git -C "$POINTNAV_VO_ROOT" rev-parse HEAD)" = "$POINTNAV_VO_COMMIT" ] || {
  echo pointnav_vo_commit_mismatch; exit 22;
}

read -r SCENE_DATASET_CONFIG SCENE_DATASET_CONFIG_SHA256 < <(
  "$ASCENT_PYTHON" - "$MANIFEST" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
print(value["scene_dataset_config"], value["scene_dataset_config_sha256"])
PY
)
[ -f "$SCENE_DATASET_CONFIG" ] || { echo scene_dataset_config_missing; exit 21; }
[ "$(sha256sum "$SCENE_DATASET_CONFIG" | awk '{print $1}')" = "$SCENE_DATASET_CONFIG_SHA256" ] || {
  echo scene_dataset_config_hash_mismatch; exit 22;
}

if [ "$MODE" = smoke ]; then
  SCREEN_MAX_ACTIONS=80
  SUBRUN_TIMEOUT_SECONDS=10800
else
  SCREEN_MAX_ACTIONS=500
  SUBRUN_TIMEOUT_SECONDS=72000
fi

export CUDA_HOME=$ASCENT_ENV
export PATH="$SCRATCH_ROOT/tools/bin:$CUDA_HOME/bin:$PATH"
export MAMBA_ROOT_PREFIX=$SCRATCH_ROOT/micromamba
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="$ASCENT_ENV/lib:$ASCENT_ENV/lib64:$ASCENT_ENV/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export PIP_CACHE_DIR=$SCRATCH_ROOT/cache/pip
export HF_HOME=$SCRATCH_ROOT/cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TORCH_HOME=$SCRATCH_ROOT/cache/torch
export XDG_CACHE_HOME=$RUN_ROOT/xdg_cache
export MPLCONFIGDIR=$RUN_ROOT/mpl_config
export CONDA_PKGS_DIRS=$SCRATCH_ROOT/cache/conda_pkgs
export ASCENT_REQUEST_TIMEOUT_SECONDS=60
export ASCENT_SERVER_WAIT_SECONDS=1200
export HYDRA_FULL_ERROR=1 PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1 HABITAT_ENV_DEBUG=1
export ASCENT_VO_CHECKPOINT_DIR=$CHECKPOINT_DIR
export QWEN2_5_PORT=$BASE_PORT
export BLIP2ITM_PORT=$((BASE_PORT + 1))
export SAM_PORT=$((BASE_PORT + 2))
export GROUNDING_DINO_PORT=$((BASE_PORT + 3))
export RAM_PORT=$((BASE_PORT + 4))
export DFINE_PORT=$((BASE_PORT + 5))
export PYTHONPATH="$SOURCE_ROOT:$SOURCE_ROOT/third_party/vlfm:$SOURCE_ROOT/third_party/frontier_exploration:$SOURCE_ROOT/third_party/depth_camera_filtering:$SOURCE_ROOT/third_party/recognize-anything:$SOURCE_ROOT/third_party/D-FINE:$SOURCE_ROOT/third_party/places365"
unset ASCENT_OCSM_ENABLED ASCENT_STAIR_DISTANCE_ORDER_FIX
unset ASCENT_POSE_SCHEDULE_PATH ASCENT_POSE_ARM
unset ASCENT_PLACE_MEMORY_ENABLED ASCENT_PLACE_ORACLE_ENABLED
unset ASCENT_PLACE_ORACLE_DIAGNOSTICS_PATH

mapfile -t CHUNK_ROWS < <(
  "$ASCENT_PYTHON" - "$MANIFEST" "$LANE_ID" "$LANE_COUNT" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
lane, lanes = int(sys.argv[2]), int(sys.argv[3])
for index, chunk in enumerate(manifest["chunks"]):
    if index % lanes == lane:
        print("\t".join(str(value) for value in (
            index, chunk["chunk_id"], chunk["data_path"],
            chunk["identity_path"], chunk["episode_count"],
        )))
PY
)
PLANNED_EPISODES=0
for encoded in "${CHUNK_ROWS[@]}"; do
  IFS=$'\t' read -r _ _ _ _ count <<< "$encoded"
  PLANNED_EPISODES=$((PLANNED_EPISODES + count))
done
[ "${#CHUNK_ROWS[@]}" = "$EXPECTED_CHUNKS" ] || {
  echo "lane_chunk_count expected=$EXPECTED_CHUNKS actual=${#CHUNK_ROWS[@]}"; exit 25;
}
[ "$PLANNED_EPISODES" = "$EXPECTED_EPISODES" ] || {
  echo "lane_episode_count expected=$EXPECTED_EPISODES actual=$PLANNED_EPISODES"; exit 25;
}

{
  echo schema=ascent_vo_submap_v1_4_oracle_task_memory_lane_v2
  echo scientific_role=oracle_identity_only_task_memory_screen
  echo pbs_jobid=${PBS_JOBID:-manual}
  echo host=$(hostname)
  echo mode=$MODE
  echo experiment=$EXPERIMENT
  echo dataset=$DATASET
  echo lane_id=$LANE_ID
  echo lane_count=$LANE_COUNT
  echo source_commit=$SOURCE_COMMIT
  echo resource_commit=$RESOURCE_COMMIT
  echo pointnav_vo_commit=$POINTNAV_VO_COMMIT
  echo manifest=$MANIFEST
  echo manifest_sha256=$MANIFEST_SHA256
  echo expected_chunks=$EXPECTED_CHUNKS
  echo expected_episodes=$EXPECTED_EPISODES
  echo screen_max_actions=$SCREEN_MAX_ACTIONS
  echo pose_source=zhao_rgbd_2021
  echo evaluation_pose_source=habitat_ground_truth
  echo gt_policy_isolation=1
  echo gt_policy_output_fields=event_sequence,reference_submap_id
  echo oracle_scope=vo_consistent_graph_nondirect_fragmentation_only
  echo oracle_physical_planar_radius_m=0.75
  echo oracle_physical_height_radius_m=0.75
  echo oracle_min_step_separation=30
  echo oracle_min_excursion_m=2.0
  echo oracle_vo_consistent_radius_m=1.5
  echo place_low_gain_area_m2=0.5
  echo place_shadow_low_gain_area_m2=1.0
  echo place_branch_association_radius_m=0.5
  echo place_branch_match_radius_m=1.0
  echo place_branch_match_margin_m=0.25
  echo place_persistent_frontier_observations=2
  echo place_minimum_arrival_observations=2
  echo place_minimum_repeat_arrival_observations=3
  echo place_minimum_excursion_start_distance_m=1.4
  echo same_place_identity_grants_consumption=0
  echo place_consumption_authority=matched_repeat_low_gain_excursion
  echo place_repeat_settlement=online_when_evidence_complete_and_residual_live
  echo no_metric_driven_retry=1
  echo technical_retry_policy=one_fixed_retry_per_failed_unit
  echo method_version=submap_v1.4_oracle_task_memory
  echo handoff_enabled=1
  echo exhaustion_recovery_enabled=1
  echo start_time=$(date -Is)
} > "$RUN_ROOT/environment_preflight.txt"

cd "$SOURCE_ROOT"
"$ASCENT_PYTHON" -c \
  'from importlib.util import find_spec; from pathlib import Path; import sys; root=Path(sys.argv[1]).resolve(); actual=Path(find_spec("ascent.ascent_policy").origin).resolve(); print(actual); assert root in actual.parents' \
  "$SOURCE_ROOT" > "$RUN_ROOT/import_check.log" 2>&1

SERVER_PIDS=()
cleanup() {
  local status=$?
  set +e
  local pid
  for pid in "${SERVER_PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${SERVER_PIDS[@]:-}"; do wait "$pid" 2>/dev/null || true; done
  if [ ! -e "$RUN_ROOT/terminal_status.txt" ]; then
    printf 'lane_exit_status=%s stage=aborted time=%s\n' \
      "$status" "$(date -Is)" > "$RUN_ROOT/terminal_status.txt"
  fi
  return "$status"
}
trap cleanup EXIT
trap 'exit 130' HUP INT TERM

start_service() {
  local name=$1 module=$2 port=$3
  CUDA_VISIBLE_DEVICES=0 "$ASCENT_PYTHON" -u -m "$module" --port "$port" \
    > "$RUN_ROOT/service_logs/$name.log" 2>&1 &
  SERVER_PIDS+=("$!")
}
start_service qwen2_5 model_api.qwen25_out "$QWEN2_5_PORT"
start_service blip2itm model_api.blip2itm_out "$BLIP2ITM_PORT"
start_service sam model_api.sam_out "$SAM_PORT"
start_service gdino model_api.grounding_dino_out "$GROUNDING_DINO_PORT"
start_service ram model_api.ram_out "$RAM_PORT"
start_service dfine model_api.dfine_out "$DFINE_PORT"

run_health_check() {
  local stage=$1 output=$2 wait_seconds=$3
  "$ASCENT_PYTHON" "$HEALTH_CHECK" \
    --stage "$stage" --wait-seconds "$wait_seconds" \
    --interval-seconds 15 --request-timeout 3 > "$output" 2>&1
}
run_health_check initial "$RUN_ROOT/service_health_initial.log" \
  "$ASCENT_SERVER_WAIT_SECONDS" || exit 41

INVENTORY=$RUN_ROOT/inventory.csv
printf 'priority,stage,condition,chunk_id,expected_episodes,process_exit_status,post_health_status,terminal_class,vo_metadata_count,vo_step_count,episode_end_count,vo_technical_error_count,vo_parse_error_count,vo_unknown_count,submap_metadata_count,submap_reset_count,submap_endpoint_count,submap_event_count,submap_parse_error_count,submap_unknown_count,oracle_metadata_count,oracle_step_count,oracle_episode_end_count,oracle_parse_error_count,oracle_unknown_count,vo_diagnostics,submap_diagnostics,oracle_diagnostics,identity_path,run_log\n' \
  > "$INVENTORY"

diagnostic_counts() {
  local path=$1 family=$2
  "$ASCENT_PYTHON" - "$path" "$family" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path
path, family = Path(sys.argv[1]), sys.argv[2]
counts, parse = Counter(), 0
if path.is_file():
    for line in path.open(errors="replace", encoding="utf-8"):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            parse += 1
            continue
        counts[str(record.get("record_type"))] += 1
if family == "vo":
    known = {"run_metadata", "vo_step", "episode_end", "vo_technical_error"}
    values = (counts["run_metadata"], counts["vo_step"], counts["episode_end"],
              counts["vo_technical_error"], parse,
              sum(v for k, v in counts.items() if k not in known))
elif family == "submap":
    known = {"submap_run_metadata", "submap_episode_reset",
             "submap_action_endpoint", "submap_event"}
    values = (counts["submap_run_metadata"], counts["submap_episode_reset"],
              counts["submap_action_endpoint"], counts["submap_event"], parse,
              sum(v for k, v in counts.items() if k not in known))
elif family == "oracle":
    known = {"oracle_place_run_metadata", "oracle_place_step",
             "oracle_place_episode_end"}
    values = (counts["oracle_place_run_metadata"],
              counts["oracle_place_step"],
              counts["oracle_place_episode_end"], parse,
              sum(v for k, v in counts.items() if k not in known))
else:
    raise SystemExit(f"unknown diagnostic family: {family}")
print(*values)
PY
}

FAILED_UNITS=()
CONSECUTIVE_PROCESS_FAILURES=0
ATTEMPT_SEQUENCE=0
run_unit() {
  local priority=$1 encoded=$2
  local chunk_index chunk_id data_path identity_path expected
  IFS=$'\t' read -r chunk_index chunk_id data_path identity_path expected <<< "$encoded"
  local now remaining timeout_seconds
  now=$(date +%s)
  remaining=$((DEADLINE_EPOCH - now))
  [ "$remaining" -gt 900 ] || { echo "stage_deadline_before_unit=$chunk_id"; return 44; }
  timeout_seconds=$SUBRUN_TIMEOUT_SECONDS
  [ "$timeout_seconds" -le "$remaining" ] || timeout_seconds=$remaining
  ATTEMPT_SEQUENCE=$((ATTEMPT_SEQUENCE + 1))
  local attempt_name=p${priority}_${ATTEMPT_SEQUENCE}_B2
  local unit_dir=$RUN_ROOT/attempts/$attempt_name/$chunk_id
  [ ! -e "$unit_dir" ] || { echo "unit_exists=$unit_dir"; return 24; }
  mkdir -p "$unit_dir"/{video,tb,mpl_config,xdg_cache}
  run_health_check "before_${attempt_name}_${chunk_id}" \
    "$unit_dir/service_health_before.log" 0 || return 41

  local vo_diagnostics=$unit_dir/vo_diagnostics.jsonl
  local submap_diagnostics=$unit_dir/submap_diagnostics.jsonl
  local oracle_diagnostics=$unit_dir/oracle_place_diagnostics.jsonl
  local run_id=${MODE}__screen__B2__${attempt_name}__${chunk_id}
  local cmd=(
    "$ASCENT_PYTHON" -u -m ascent.run
    "--config-name=eval_ascent_${DATASET}.yaml"
    "habitat.dataset.data_path=$data_path"
    "habitat.dataset.content_scenes=[*]"
    "habitat.dataset.split=train"
    "habitat.dataset.scenes_dir=$SCENES_ROOT"
    "habitat.seed=100"
    "habitat_baselines.eval.split=train"
    "habitat_baselines.test_episode_count=-1"
    "habitat.environment.max_episode_steps=$SCREEN_MAX_ACTIONS"
    "habitat_baselines.num_environments=1"
    "habitat_baselines.video_dir=$unit_dir/video"
    "habitat_baselines.tensorboard_dir=$unit_dir/tb"
    "habitat_baselines.eval.video_option=[]"
    "ascent_submaps.provisional_thresholds=false"
    "ascent_submaps.handoff_enabled=true"
    "ascent_submaps.exhaustion_recovery_enabled=true"
    "ascent_submaps.min_action_endpoints=20"
    "ascent_submaps.min_anchor_displacement_m=1.5"
    "ascent_submaps.overlap_threshold=0.35"
    "ascent_submaps.low_overlap_consecutive=3"
    "ascent_submaps.gateway_frontier_resolution_radius_m=1.0"
    "ascent_submaps.gateway_reached_radius_m=0.9"
    "ascent_submaps.route_min_progress_m=0.30"
    "ascent_submaps.route_max_stagnation_actions=30"
    "ascent_submaps.route_max_waypoint_actions=60"
    "ascent_place_memory.enabled=true"
    "ascent_place_memory.oracle_enabled=true"
    "ascent_place_memory.low_gain_area_m2=0.5"
    "ascent_place_memory.shadow_low_gain_area_m2=1.0"
    "ascent_place_memory.branch_association_radius_m=0.5"
    "ascent_place_memory.branch_match_radius_m=1.0"
    "ascent_place_memory.branch_match_margin_m=0.25"
    "ascent_place_memory.persistent_frontier_observations=2"
    "ascent_place_memory.minimum_arrival_observations=2"
    "ascent_place_memory.minimum_repeat_arrival_observations=3"
    "ascent_place_memory.minimum_excursion_start_distance_m=1.4"
    "ascent_place_memory.oracle_physical_planar_radius_m=0.75"
    "ascent_place_memory.oracle_physical_height_radius_m=0.75"
    "ascent_place_memory.oracle_min_step_separation=30"
    "ascent_place_memory.oracle_min_excursion_m=2.0"
    "ascent_place_memory.oracle_vo_consistent_radius_m=1.5"
  )
  {
    printf 'cd %q\n' "$SOURCE_ROOT"
    printf 'ASCENT_SUBMAP_ENABLED=true ASCENT_SUBMAP_ALLOW_PROVISIONAL=false '
    printf 'ASCENT_SUBMAP_HANDOFF_ENABLED=true '
    printf 'ASCENT_SUBMAP_EXHAUSTION_RECOVERY_ENABLED=true '
    printf 'ASCENT_PLACE_MEMORY_ENABLED=true ASCENT_PLACE_ORACLE_ENABLED=true '
    printf 'ASCENT_VO_DIAGNOSTICS_PATH=%q ASCENT_SUBMAP_DIAGNOSTICS_PATH=%q ' \
      "$vo_diagnostics" "$submap_diagnostics"
    printf 'ASCENT_PLACE_ORACLE_DIAGNOSTICS_PATH=%q ' "$oracle_diagnostics"
    printf 'timeout --signal=TERM --kill-after=120s %qs ' "$timeout_seconds"
    printf '%q ' "${cmd[@]}"
    printf '\n'
  } > "$unit_dir/exact_command.txt"
  {
    echo source_commit=$SOURCE_COMMIT
    echo manifest_sha256=$MANIFEST_SHA256
    echo condition=B2
    echo method_version=submap_v1.4_oracle_task_memory
    echo handoff_enabled=1
    echo exhaustion_recovery_enabled=1
    echo oracle_identity_only=1
    echo oracle_scope=vo_consistent_graph_nondirect_fragmentation_only
    echo place_low_gain_area_m2=0.5
    echo place_shadow_low_gain_area_m2=1.0
    echo place_branch_association_radius_m=0.5
    echo place_branch_match_radius_m=1.0
    echo place_branch_match_margin_m=0.25
    echo place_persistent_frontier_observations=2
    echo place_minimum_arrival_observations=2
    echo place_minimum_repeat_arrival_observations=3
    echo place_minimum_excursion_start_distance_m=1.4
    echo oracle_physical_planar_radius_m=0.75
    echo oracle_physical_height_radius_m=0.75
    echo oracle_min_step_separation=30
    echo oracle_min_excursion_m=2.0
    echo oracle_vo_consistent_radius_m=1.5
    echo max_actions=$SCREEN_MAX_ACTIONS
    echo expected_episodes=$expected
    echo no_metric_driven_retry=1
    echo technical_retry_policy=one_fixed_retry_per_failed_unit
    echo attempt_priority=$priority
  } > "$unit_dir/code_provenance.txt"

  echo "unit_start priority=$priority condition=B2 chunk=$chunk_id time=$(date -Is)" \
    | tee -a "$RUN_ROOT/driver.log"
  local status
  if (
    export ASCENT_SUBMAP_ENABLED=true
    export ASCENT_SUBMAP_ALLOW_PROVISIONAL=false
    export ASCENT_SUBMAP_HANDOFF_ENABLED=true
    export ASCENT_SUBMAP_EXHAUSTION_RECOVERY_ENABLED=true
    export ASCENT_PLACE_MEMORY_ENABLED=true
    export ASCENT_PLACE_ORACLE_ENABLED=true
    export ASCENT_VO_DIAGNOSTICS_PATH=$vo_diagnostics
    export ASCENT_VO_RUN_ID=$run_id
    export ASCENT_VO_SOURCE_COMMIT=$SOURCE_COMMIT
    export ASCENT_SUBMAP_DIAGNOSTICS_PATH=$submap_diagnostics
    export ASCENT_SUBMAP_RUN_ID=$run_id
    export ASCENT_SUBMAP_SOURCE_COMMIT=$SOURCE_COMMIT
    export ASCENT_PLACE_ORACLE_DIAGNOSTICS_PATH=$oracle_diagnostics
    export XDG_CACHE_HOME=$unit_dir/xdg_cache
    export MPLCONFIGDIR=$unit_dir/mpl_config
    cd "$SOURCE_ROOT"
    timeout --signal=TERM --kill-after=120s "${timeout_seconds}s" \
      "${cmd[@]}" > "$unit_dir/run.log" 2>&1 < /dev/null
  ); then status=0; else status=$?; fi
  echo "$status" > "$unit_dir/exit_status.txt"

  local post_health=0
  run_health_check "after_${attempt_name}_${chunk_id}" \
    "$unit_dir/service_health_after.log" 0 || post_health=$?
  local vm vs ve vt vp vu sm sr se sv sp su om os oe op ou
  read -r vm vs ve vt vp vu <<< "$(diagnostic_counts "$vo_diagnostics" vo)"
  read -r sm sr se sv sp su <<< "$(diagnostic_counts "$submap_diagnostics" submap)"
  read -r om os oe op ou <<< "$(diagnostic_counts "$oracle_diagnostics" oracle)"
  local terminal_class=complete
  if [ "$post_health" -ne 0 ]; then
    terminal_class=infrastructure_failure
  elif [ "$status" -ne 0 ]; then
    terminal_class=process_failure
  elif [ "$vt" -ne 0 ] || [ "$vp" -ne 0 ] || [ "$vu" -ne 0 ] || [ "$vm" -ne 1 ]; then
    terminal_class=logging_failure
  elif [ "$sm" -ne 1 ] || [ "$sp" -ne 0 ] || [ "$su" -ne 0 ]; then
    terminal_class=logging_failure
  elif [ "$om" -ne 1 ] || [ "$op" -ne 0 ] || [ "$ou" -ne 0 ]; then
    terminal_class=logging_failure
  elif [ "$ve" -ne "$expected" ]; then
    terminal_class=incomplete_diagnostics
  elif [ "$oe" -ne "$expected" ]; then
    terminal_class=incomplete_diagnostics
  fi
  printf '%s,screen,B2,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$priority" "$chunk_id" "$expected" "$status" "$post_health" \
    "$terminal_class" "$vm" "$vs" "$ve" "$vt" "$vp" "$vu" \
    "$sm" "$sr" "$se" "$sv" "$sp" "$su" "$om" "$os" "$oe" "$op" "$ou" \
    "$vo_diagnostics" "$submap_diagnostics" "$oracle_diagnostics" \
    "$identity_path" "$unit_dir/run.log" >> "$INVENTORY"
  echo "unit_end priority=$priority condition=B2 chunk=$chunk_id status=$status class=$terminal_class vo_ends=$ve oracle_ends=$oe time=$(date -Is)" \
    | tee -a "$RUN_ROOT/driver.log"
  case "$terminal_class" in
    complete) return 0 ;;
    infrastructure_failure) return 41 ;;
    process_failure) return 42 ;;
    logging_failure) return 43 ;;
    incomplete_diagnostics) return 44 ;;
    *) return 45 ;;
  esac
}

for encoded in "${CHUNK_ROWS[@]}"; do
  set +e
  run_unit 0 "$encoded"
  result=$?
  set -e
  case "$result" in
    0) CONSECUTIVE_PROCESS_FAILURES=0 ;;
    42|44)
      FAILED_UNITS+=("$encoded")
      if [ "$MODE" = full ]; then
        CONSECUTIVE_PROCESS_FAILURES=$((CONSECUTIVE_PROCESS_FAILURES + 1))
        [ "$CONSECUTIVE_PROCESS_FAILURES" -lt "$MAX_CONSECUTIVE_PROCESS_FAILURES" ] || exit 42
      fi
      ;;
    *) exit "$result" ;;
  esac
done

if [ "${#FAILED_UNITS[@]}" -gt 0 ]; then
  for failed in "${FAILED_UNITS[@]}"; do
    set +e
    run_unit 1 "$failed"
    result=$?
    set -e
    case "$result" in
      0) ;;
      42|44)
        echo "technical_retry_exhausted=B2:$failed"
        if [ "$MODE" = smoke ]; then exit "$result"; fi
        ;;
      *) exit "$result" ;;
    esac
  done
fi

run_health_check final "$RUN_ROOT/service_health_final.log" 0 || exit 41
[ -z "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=all)" ] || {
  echo source_changed_during_run; exit 47;
}
printf 'lane_exit_status=0 stage=complete time=%s\n' "$(date -Is)" \
  > "$RUN_ROOT/terminal_status.txt"
exit 0
