#!/bin/bash
# One independent ASCENT-VO submap lane with its own six services.

set -euo pipefail

PROJECT=/scratch/e1538633/liuyi/drift-aware-submap-exploration
SOURCE_ROOT=$PROJECT/external/ascent_vo_submap_v1
RESOURCE_ROOT=$PROJECT/external/ascent
POINTNAV_VO_ROOT=$PROJECT/external/PointNav-VO
CHECKPOINT_DIR=$POINTNAV_VO_ROOT/pretrained_ckpts/vo
HEALTH_CHECK=$SOURCE_ROOT/scripts/check_submap_services.py

DATASET=${ASCENT_SUBMAP_DATASET:?ASCENT_SUBMAP_DATASET is required}
MODE=${ASCENT_SUBMAP_MODE:?ASCENT_SUBMAP_MODE is required}
SCENES_ROOT=${ASCENT_SUBMAP_SCENES_ROOT:?ASCENT_SUBMAP_SCENES_ROOT is required}
RUN_CALIBRATION=${ASCENT_SUBMAP_RUN_CALIBRATION:?ASCENT_SUBMAP_RUN_CALIBRATION is required}
MANIFEST=${ASCENT_SUBMAP_MANIFEST:?ASCENT_SUBMAP_MANIFEST is required}
EXPECTED_MANIFEST_SHA256=${ASCENT_SUBMAP_MANIFEST_SHA256:?ASCENT_SUBMAP_MANIFEST_SHA256 is required}
SOURCE_COMMIT=${ASCENT_SUBMAP_SOURCE_COMMIT:?ASCENT_SUBMAP_SOURCE_COMMIT is required}
RESOURCE_COMMIT=${ASCENT_SUBMAP_RESOURCE_COMMIT:?ASCENT_SUBMAP_RESOURCE_COMMIT is required}
POINTNAV_VO_COMMIT=${ASCENT_SUBMAP_POINTNAV_VO_COMMIT:?ASCENT_SUBMAP_POINTNAV_VO_COMMIT is required}
LANE_ID=${ASCENT_SUBMAP_LANE_ID:?ASCENT_SUBMAP_LANE_ID is required}
LANE_COUNT=${ASCENT_SUBMAP_LANE_COUNT:?ASCENT_SUBMAP_LANE_COUNT is required}
EXPECTED_CHUNKS=${ASCENT_SUBMAP_LANE_EXPECTED_CHUNKS:?ASCENT_SUBMAP_LANE_EXPECTED_CHUNKS is required}
EXPECTED_EPISODES=${ASCENT_SUBMAP_LANE_EXPECTED_EPISODES:?ASCENT_SUBMAP_LANE_EXPECTED_EPISODES is required}
STAGE_ROOT=${ASCENT_SUBMAP_STAGE_ROOT:?ASCENT_SUBMAP_STAGE_ROOT is required}
BASE_PORT=${ASCENT_SUBMAP_BASE_PORT:?ASCENT_SUBMAP_BASE_PORT is required}
DEADLINE_EPOCH=${ASCENT_SUBMAP_DEADLINE_EPOCH:?ASCENT_SUBMAP_DEADLINE_EPOCH is required}
CALIBRATION_MANIFEST=${ASCENT_SUBMAP_CALIBRATION_MANIFEST:-}
CALIBRATION_MANIFEST_SHA256=${ASCENT_SUBMAP_CALIBRATION_MANIFEST_SHA256:-}
CALIBRATION_JSON=${ASCENT_SUBMAP_CALIBRATION_JSON:-}
CALIBRATION_JSON_SHA256=${ASCENT_SUBMAP_CALIBRATION_JSON_SHA256:-}

FORWARD_SHA256=6b571bb717366f7d80f61e919b33a45ac2f45925201c4e3011b3239a2c42e586
TURN_SHA256=c469643f9ab35c9e1058f31fbb672a5fa3adf582987a4388bdd020dd89faf1d9
SCRATCH_ROOT=/scratch/e1538633/liuyi
ASCENT_ENV=$SCRATCH_ROOT/micromamba/envs/ascent_nav
ASCENT_PYTHON=$ASCENT_ENV/bin/python
RUN_ROOT=$STAGE_ROOT/lane_$LANE_ID
MAX_CONSECUTIVE_PROCESS_FAILURES=3

case "$MODE" in smoke|full) ;; *) echo "invalid_mode=$MODE"; exit 20 ;; esac
case "$DATASET" in hm3d|mp3d) ;; *) echo "invalid_dataset=$DATASET"; exit 20 ;; esac
case "$RUN_CALIBRATION" in true|false) ;; *) echo "invalid_run_calibration=$RUN_CALIBRATION"; exit 20 ;; esac
if [ "$RUN_CALIBRATION" = true ] \
  && { [ "$DATASET" != hm3d ] || [ "$MODE" != smoke ]; }; then
  echo invalid_calibration_mode
  exit 20
fi
for value in "$LANE_ID" "$LANE_COUNT" "$EXPECTED_CHUNKS" \
  "$EXPECTED_EPISODES" "$BASE_PORT" "$DEADLINE_EPOCH"; do
  [[ "$value" =~ ^[0-9]+$ ]] || {
    echo "invalid_integer=$value"
    exit 20
  }
done
[ "$LANE_ID" -lt "$LANE_COUNT" ] || { echo invalid_lane; exit 20; }
[ ! -e "$RUN_ROOT" ] || { echo "run_root_exists=$RUN_ROOT"; exit 24; }
mkdir -p "$RUN_ROOT"/{attempts,mpl_config,xdg_cache,service_logs}
mkdir -p "$SCRATCH_ROOT/cache"/{pip,huggingface,torch,xdg,matplotlib,conda_pkgs}

for path in \
  "$MANIFEST" "$HEALTH_CHECK" "$SOURCE_ROOT/ascent/run.py" \
  "$SOURCE_ROOT/ascent/ascent_policy.py" "$SOURCE_ROOT/ascent/submaps" \
  "$SOURCE_ROOT/ascent/vo/pose_provider.py" "$SOURCE_ROOT/model_api" \
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
[ "$(sha256sum "$MANIFEST" | awk '{print $1}')" = "$EXPECTED_MANIFEST_SHA256" ] || {
  echo manifest_hash_mismatch
  exit 22
}
read -r SCENE_DATASET_CONFIG SCENE_DATASET_CONFIG_SHA256 < <(
  "$ASCENT_PYTHON" - "$MANIFEST" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
print(
    value["scene_dataset_config"],
    value["scene_dataset_config_sha256"],
)
PY
)
[ -f "$SCENE_DATASET_CONFIG" ] || {
  echo scene_dataset_config_missing
  exit 21
}
[ "$(sha256sum "$SCENE_DATASET_CONFIG" | awk '{print $1}')" = "$SCENE_DATASET_CONFIG_SHA256" ] || {
  echo scene_dataset_config_hash_mismatch
  exit 22
}
[ "$(sha256sum "$CHECKPOINT_DIR/act_forward.pth" | awk '{print $1}')" = "$FORWARD_SHA256" ] || {
  echo forward_checkpoint_hash_mismatch
  exit 22
}
[ "$(sha256sum "$CHECKPOINT_DIR/act_left_right_inv_joint.pth" | awk '{print $1}')" = "$TURN_SHA256" ] || {
  echo turn_checkpoint_hash_mismatch
  exit 22
}
[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" = "$SOURCE_COMMIT" ] || {
  echo source_commit_mismatch
  exit 22
}
[ -z "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=all)" ] || {
  echo source_worktree_not_clean
  git -C "$SOURCE_ROOT" status --short
  exit 23
}
[ "$(git -C "$RESOURCE_ROOT" rev-parse HEAD)" = "$RESOURCE_COMMIT" ] || {
  echo resource_commit_mismatch
  exit 22
}
[ "$(git -C "$POINTNAV_VO_ROOT" rev-parse HEAD)" = "$POINTNAV_VO_COMMIT" ] || {
  echo pointnav_vo_commit_mismatch
  exit 22
}

if [ "$MODE" = smoke ]; then
  [ "$LANE_COUNT" = 1 ] || { echo smoke_lane_count_contract; exit 20; }
  SCREEN_MAX_ACTIONS=80
  SUBRUN_TIMEOUT_SECONDS=10800
else
  SCREEN_MAX_ACTIONS=500
  SUBRUN_TIMEOUT_SECONDS=72000
fi
if [ "$RUN_CALIBRATION" = true ]; then
  [ -f "$CALIBRATION_MANIFEST" ] || {
    echo missing_calibration_manifest
    exit 21
  }
  [ "$(sha256sum "$CALIBRATION_MANIFEST" | awk '{print $1}')" = "$CALIBRATION_MANIFEST_SHA256" ] || {
    echo calibration_manifest_hash_mismatch
    exit 22
  }
else
  [ -f "$CALIBRATION_JSON" ] || { echo missing_calibration_json; exit 21; }
  [ "$(sha256sum "$CALIBRATION_JSON" | awk '{print $1}')" = "$CALIBRATION_JSON_SHA256" ] || {
    echo calibration_json_hash_mismatch
    exit 22
  }
  [ "$("$ASCENT_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$CALIBRATION_JSON")" = PASS ] || {
    echo calibration_gate_not_pass
    exit 25
  }
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

mapfile -t CHUNK_ROWS < <(
  "$ASCENT_PYTHON" - "$MANIFEST" "$LANE_ID" "$LANE_COUNT" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
lane, lanes = int(sys.argv[2]), int(sys.argv[3])
for index, chunk in enumerate(manifest["chunks"]):
    if index % lanes != lane:
        continue
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
  echo "lane_chunk_count expected=$EXPECTED_CHUNKS actual=${#CHUNK_ROWS[@]}"
  exit 25
}
[ "$PLANNED_EPISODES" = "$EXPECTED_EPISODES" ] || {
  echo "lane_episode_count expected=$EXPECTED_EPISODES actual=$PLANNED_EPISODES"
  exit 25
}

CALIBRATION_ROW=""
if [ "$RUN_CALIBRATION" = true ]; then
  CALIBRATION_ROW=$(
    "$ASCENT_PYTHON" - "$CALIBRATION_MANIFEST" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
if len(manifest["chunks"]) != 1:
    raise SystemExit("calibration must have exactly one chunk")
chunk = manifest["chunks"][0]
print("\t".join(str(value) for value in (
    0, chunk["chunk_id"], chunk["data_path"],
    chunk["identity_path"], chunk["episode_count"],
)))
PY
  )
else
  mapfile -t CALIBRATION_VALUES < <(
    "$ASCENT_PYTHON" - "$CALIBRATION_JSON" <<'PY'
import json
import sys
config = json.load(open(sys.argv[1], encoding="utf-8"))["config"]
keys = (
    "min_action_endpoints", "min_path_length_m", "overlap_threshold",
    "low_overlap_consecutive", "max_motion_budget_m",
    "rotation_weight_m_per_rad", "gateway_frontier_resolution_radius_m",
    "gateway_reached_radius_m",
)
for key in keys:
    print(f"{key}\t{config[key]}")
PY
  )
  declare -A CALIBRATED
  for value in "${CALIBRATION_VALUES[@]}"; do
    IFS=$'\t' read -r key item <<< "$value"
    CALIBRATED[$key]=$item
  done
fi

{
  echo schema=ascent_vo_submap_lane_v1
  echo scientific_role=paired_representation_method_screen
  echo pbs_jobid=${PBS_JOBID:-manual}
  echo host=$(hostname)
  echo mode=$MODE
  echo dataset=$DATASET
  echo run_calibration=$RUN_CALIBRATION
  echo lane_id=$LANE_ID
  echo lane_count=$LANE_COUNT
  echo source_commit=$SOURCE_COMMIT
  echo resource_commit=$RESOURCE_COMMIT
  echo pointnav_vo_commit=$POINTNAV_VO_COMMIT
  echo manifest=$MANIFEST
  echo manifest_sha256=$EXPECTED_MANIFEST_SHA256
  echo scene_dataset_config=$SCENE_DATASET_CONFIG
  echo scene_dataset_config_sha256=$SCENE_DATASET_CONFIG_SHA256
  echo calibration_json=$CALIBRATION_JSON
  echo calibration_json_sha256=$CALIBRATION_JSON_SHA256
  echo expected_chunks=$EXPECTED_CHUNKS
  echo expected_episodes=$EXPECTED_EPISODES
  echo screen_max_actions=$SCREEN_MAX_ACTIONS
  echo pose_source=zhao_rgbd_2021
  echo evaluation_pose_source=habitat_ground_truth
  echo gt_policy_isolation=1
  echo no_metric_driven_retry=1
  echo base_port=$BASE_PORT
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
  for pid in "${SERVER_PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${SERVER_PIDS[@]:-}"; do
    wait "$pid" 2>/dev/null || true
  done
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
printf 'priority,stage,condition,chunk_id,expected_episodes,process_exit_status,post_health_status,terminal_class,vo_metadata_count,vo_step_count,episode_end_count,vo_technical_error_count,vo_parse_error_count,vo_unknown_count,submap_metadata_count,submap_reset_count,submap_endpoint_count,submap_event_count,submap_parse_error_count,submap_unknown_count,vo_diagnostics,submap_diagnostics,identity_path,run_log\n' \
  > "$INVENTORY"

diagnostic_counts() {
  local path=$1 family=$2
  "$ASCENT_PYTHON" - "$path" "$family" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path
path, family = Path(sys.argv[1]), sys.argv[2]
counts = Counter()
parse = 0
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
    values = (
        counts["run_metadata"], counts["vo_step"], counts["episode_end"],
        counts["vo_technical_error"], parse,
        sum(value for key, value in counts.items() if key not in known),
    )
else:
    known = {
        "submap_run_metadata", "submap_episode_reset",
        "submap_action_endpoint", "submap_event",
    }
    values = (
        counts["submap_run_metadata"], counts["submap_episode_reset"],
        counts["submap_action_endpoint"], counts["submap_event"], parse,
        sum(value for key, value in counts.items() if key not in known),
    )
print(*values)
PY
}

FAILED_UNITS=()
CONSECUTIVE_PROCESS_FAILURES=0
ATTEMPT_SEQUENCE=0
run_unit() {
  local priority=$1 stage=$2 condition=$3 encoded=$4
  local chunk_index chunk_id data_path identity_path expected
  IFS=$'\t' read -r chunk_index chunk_id data_path identity_path expected \
    <<< "$encoded"
  local now remaining timeout_seconds max_actions
  now=$(date +%s)
  remaining=$((DEADLINE_EPOCH - now))
  [ "$remaining" -gt 900 ] || {
    echo "stage_deadline_before_unit=$stage:$condition:$chunk_id"
    return 44
  }
  timeout_seconds=$SUBRUN_TIMEOUT_SECONDS
  [ "$timeout_seconds" -le "$remaining" ] || timeout_seconds=$remaining
  if [ "$stage" = calibration ]; then
    max_actions=120
  else
    max_actions=$SCREEN_MAX_ACTIONS
  fi
  ATTEMPT_SEQUENCE=$((ATTEMPT_SEQUENCE + 1))
  local attempt_name=p${priority}_${ATTEMPT_SEQUENCE}_${condition}
  local unit_dir=$RUN_ROOT/attempts/$attempt_name/$chunk_id
  [ ! -e "$unit_dir" ] || { echo "unit_exists=$unit_dir"; return 24; }
  mkdir -p "$unit_dir"/{video,tb,mpl_config,xdg_cache}
  run_health_check "before_${attempt_name}_${chunk_id}" \
    "$unit_dir/service_health_before.log" 0 || return 41

  local vo_diagnostics=$unit_dir/vo_diagnostics.jsonl
  local submap_diagnostics=$unit_dir/submap_diagnostics.jsonl
  local run_id=${MODE}__${stage}__${condition}__${attempt_name}__${chunk_id}
  local submap_enabled=false
  local submap_allow=false
  local overrides=()
  case "$condition" in
    CAL)
      submap_enabled=true
      submap_allow=true
      overrides+=(
        "ascent_submaps.provisional_thresholds=true"
        "ascent_submaps.min_action_endpoints=10000"
        "ascent_submaps.min_path_length_m=10000.0"
        "ascent_submaps.overlap_threshold=0.0"
        "ascent_submaps.low_overlap_consecutive=10000"
        "ascent_submaps.max_motion_budget_m=10000.0"
      )
      ;;
    B1)
      submap_enabled=false
      ;;
    B2)
      submap_enabled=true
      if [ "$RUN_CALIBRATION" = true ]; then
        submap_allow=true
        overrides+=(
          "ascent_submaps.provisional_thresholds=true"
          "ascent_submaps.min_action_endpoints=3"
          "ascent_submaps.min_path_length_m=0.25"
          "ascent_submaps.overlap_threshold=0.30"
          "ascent_submaps.low_overlap_consecutive=2"
          "ascent_submaps.max_motion_budget_m=0.75"
        )
      else
        submap_allow=false
        overrides+=(
          "ascent_submaps.provisional_thresholds=false"
          "ascent_submaps.min_action_endpoints=${CALIBRATED[min_action_endpoints]}"
          "ascent_submaps.min_path_length_m=${CALIBRATED[min_path_length_m]}"
          "ascent_submaps.overlap_threshold=${CALIBRATED[overlap_threshold]}"
          "ascent_submaps.low_overlap_consecutive=${CALIBRATED[low_overlap_consecutive]}"
          "ascent_submaps.max_motion_budget_m=${CALIBRATED[max_motion_budget_m]}"
          "ascent_submaps.rotation_weight_m_per_rad=${CALIBRATED[rotation_weight_m_per_rad]}"
          "ascent_submaps.gateway_frontier_resolution_radius_m=${CALIBRATED[gateway_frontier_resolution_radius_m]}"
          "ascent_submaps.gateway_reached_radius_m=${CALIBRATED[gateway_reached_radius_m]}"
        )
      fi
      ;;
    *) echo "invalid_condition=$condition"; return 20 ;;
  esac

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
    "habitat.environment.max_episode_steps=$max_actions"
    "habitat_baselines.num_environments=1"
    "habitat_baselines.video_dir=$unit_dir/video"
    "habitat_baselines.tensorboard_dir=$unit_dir/tb"
    "habitat_baselines.eval.video_option=[]"
    "${overrides[@]}"
  )
  {
    printf 'cd %q\n' "$SOURCE_ROOT"
    printf 'ASCENT_SUBMAP_ENABLED=%q ASCENT_SUBMAP_ALLOW_PROVISIONAL=%q ' \
      "$submap_enabled" "$submap_allow"
    printf 'ASCENT_VO_DIAGNOSTICS_PATH=%q ASCENT_SUBMAP_DIAGNOSTICS_PATH=%q ' \
      "$vo_diagnostics" "$submap_diagnostics"
    printf 'timeout --signal=TERM --kill-after=120s %qs ' "$timeout_seconds"
    printf '%q ' "${cmd[@]}"
    printf '\n'
  } > "$unit_dir/exact_command.txt"
  {
    echo source_commit=$SOURCE_COMMIT
    echo manifest_sha256=$EXPECTED_MANIFEST_SHA256
    echo condition=$condition
    echo stage=$stage
    echo max_actions=$max_actions
    echo expected_episodes=$expected
    echo submap_enabled=$submap_enabled
    echo submap_allow_provisional=$submap_allow
    if [ "$RUN_CALIBRATION" = false ]; then
      echo calibration_json=$CALIBRATION_JSON
      echo calibration_json_sha256=$CALIBRATION_JSON_SHA256
    fi
  } > "$unit_dir/code_provenance.txt"

  echo "unit_start priority=$priority stage=$stage condition=$condition chunk=$chunk_id time=$(date -Is)" \
    | tee -a "$RUN_ROOT/driver.log"
  local status
  if (
    export ASCENT_SUBMAP_ENABLED=$submap_enabled
    export ASCENT_SUBMAP_ALLOW_PROVISIONAL=$submap_allow
    export ASCENT_VO_DIAGNOSTICS_PATH=$vo_diagnostics
    export ASCENT_VO_RUN_ID=$run_id
    export ASCENT_VO_SOURCE_COMMIT=$SOURCE_COMMIT
    export ASCENT_SUBMAP_DIAGNOSTICS_PATH=$submap_diagnostics
    export ASCENT_SUBMAP_RUN_ID=$run_id
    export ASCENT_SUBMAP_SOURCE_COMMIT=$SOURCE_COMMIT
    export XDG_CACHE_HOME=$unit_dir/xdg_cache
    export MPLCONFIGDIR=$unit_dir/mpl_config
    cd "$SOURCE_ROOT"
    timeout --signal=TERM --kill-after=120s "${timeout_seconds}s" \
      "${cmd[@]}" > "$unit_dir/run.log" 2>&1 < /dev/null
  ); then
    status=0
  else
    status=$?
  fi
  echo "$status" > "$unit_dir/exit_status.txt"

  local post_health=0
  run_health_check "after_${attempt_name}_${chunk_id}" \
    "$unit_dir/service_health_after.log" 0 || post_health=$?
  local vm vs ve vt vp vu sm sr se sv sp su
  read -r vm vs ve vt vp vu <<< "$(diagnostic_counts "$vo_diagnostics" vo)"
  if [ "$submap_enabled" = true ]; then
    read -r sm sr se sv sp su <<< "$(diagnostic_counts "$submap_diagnostics" submap)"
  else
    sm=0 sr=0 se=0 sv=0 sp=0 su=0
  fi
  local terminal_class=complete
  if [ "$post_health" -ne 0 ]; then
    terminal_class=infrastructure_failure
  elif [ "$status" -ne 0 ]; then
    terminal_class=process_failure
  elif [ "$vt" -ne 0 ] || [ "$vp" -ne 0 ] || [ "$vu" -ne 0 ] \
    || [ "$vm" -ne 1 ]; then
    terminal_class=logging_failure
  elif [ "$submap_enabled" = true ] && {
    [ "$sm" -ne 1 ] || [ "$sp" -ne 0 ] || [ "$su" -ne 0 ];
  }; then
    terminal_class=logging_failure
  elif [ "$ve" -ne "$expected" ]; then
    terminal_class=incomplete_diagnostics
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$priority" "$stage" "$condition" "$chunk_id" "$expected" \
    "$status" "$post_health" "$terminal_class" "$vm" "$vs" "$ve" \
    "$vt" "$vp" "$vu" "$sm" "$sr" "$se" "$sv" "$sp" "$su" \
    "$vo_diagnostics" "$submap_diagnostics" "$identity_path" \
    "$unit_dir/run.log" >> "$INVENTORY"
  echo "unit_end priority=$priority stage=$stage condition=$condition chunk=$chunk_id status=$status class=$terminal_class ends=$ve time=$(date -Is)" \
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

if [ "$RUN_CALIBRATION" = true ]; then
  set +e
  run_unit 0 calibration CAL "$CALIBRATION_ROW"
  calibration_result=$?
  set -e
  case "$calibration_result" in
    0) ;;
    42|44)
      # One fixed technical retry is retained for the longer calibration run.
      run_unit 1 calibration CAL "$CALIBRATION_ROW" || exit $?
      ;;
    *) exit "$calibration_result" ;;
  esac
fi

for encoded in "${CHUNK_ROWS[@]}"; do
  IFS=$'\t' read -r chunk_index _ <<< "$encoded"
  if [ "$MODE" = smoke ]; then
    # Exercise the changed method first so an integration error fails fast
    # before spending time on the already-validated default-off condition.
    conditions=(B2 B1)
  elif [ "$((chunk_index % 2))" = 0 ]; then
    conditions=(B1 B2)
  else
    conditions=(B2 B1)
  fi
  for condition in "${conditions[@]}"; do
    set +e
    run_unit 0 screen "$condition" "$encoded"
    result=$?
    set -e
    case "$result" in
      0) CONSECUTIVE_PROCESS_FAILURES=0 ;;
      42|44)
        if [ "$MODE" = smoke ]; then
          exit "$result"
        fi
        FAILED_UNITS+=("$condition"$'\t'"$encoded")
        CONSECUTIVE_PROCESS_FAILURES=$((CONSECUTIVE_PROCESS_FAILURES + 1))
        [ "$CONSECUTIVE_PROCESS_FAILURES" -lt "$MAX_CONSECUTIVE_PROCESS_FAILURES" ] || exit 42
        ;;
      *) exit "$result" ;;
    esac
  done
done

if [ "$MODE" = full ] && [ "${#FAILED_UNITS[@]}" -gt 0 ]; then
  for failed in "${FAILED_UNITS[@]}"; do
    IFS=$'\t' read -r condition chunk_index chunk_id data_path identity_path expected <<< "$failed"
    encoded=$(printf '%s\t%s\t%s\t%s\t%s' \
      "$chunk_index" "$chunk_id" "$data_path" "$identity_path" "$expected")
    set +e
    run_unit 1 screen "$condition" "$encoded"
    result=$?
    set -e
    case "$result" in
      0) ;;
      42|44) echo "recovery_incomplete=$condition:$chunk_id" ;;
      *) exit "$result" ;;
    esac
  done
fi

run_health_check final "$RUN_ROOT/service_health_final.log" 0 || exit 41
[ -z "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=all)" ] || {
  echo source_changed_during_run
  exit 47
}
printf 'lane_exit_status=0 stage=complete time=%s\n' "$(date -Is)" \
  > "$RUN_ROOT/terminal_status.txt"
exit 0
