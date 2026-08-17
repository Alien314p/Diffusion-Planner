#!/usr/bin/env bash
set -euo pipefail

export HYDRA_FULL_ERROR=1
PREFERRED_STORAGE_ROOT="${DIFFUSION_PLANNER_STORAGE_ROOT:-/data/saba/parnia/simulation_runs_by_t}"
ACTIVATION_ROOT="${DIFFUSION_PLANNER_ACTIVATION_ROOT:-/data/saba/parnia/activations_by_t}"
SCENARIO_COUNT="${DIFFUSION_PLANNER_SCENARIO_COUNT:-1000}"
TARGET_TS="${DIFFUSION_PLANNER_TARGET_TS:-[0.1,0.2,0.4,0.6,0.8]}"
RAY_THREADS_PER_NODE="${DIFFUSION_PLANNER_RAY_THREADS_PER_NODE:-2}"
LOCAL_RUNTIME_ROOT="${DIFFUSION_PLANNER_RUNTIME_ROOT:-/data/saba/parnia/Project/runtime}"
export MPLCONFIGDIR="$LOCAL_RUNTIME_ROOT/matplotlib"
export RAY_TMPDIR="$LOCAL_RUNTIME_ROOT"
export TMPDIR="$LOCAL_RUNTIME_ROOT/tmp"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

mkdir -p "$MPLCONFIGDIR" "$RAY_TMPDIR" "$TMPDIR" "$ACTIVATION_ROOT"

mkdir -p "$PREFERRED_STORAGE_ROOT/simulation"

REPO_ROOT="/data/saba/parnia/Project/Diffusion-Planner"
export NUPLAN_DEVKIT_ROOT="/data/saba/parnia/nuplan-devkit"
export NUPLAN_DATA_ROOT="/data/saba/parnia/Project/data/data/cache/mini"
export NUPLAN_MAPS_ROOT="/data/saba/parnia/nuplan_maps"
# NuPlan adds its own /exp component to this root.
export NUPLAN_EXP_ROOT="$PREFERRED_STORAGE_ROOT"

PYTHON_BIN="/data/saba/miniforge3/envs/diffusion_planner/bin/python"
SPLIT="all_scenarios"
CHALLENGE="closed_loop_nonreactive_agents"
SCENARIO_BUILDER="nuplan_mini"
DB_FILES="[$NUPLAN_DATA_ROOT/2021.06.14.18.33.41_veh-35_03901_04264.db,$NUPLAN_DATA_ROOT/2021.07.16.00.51.05_veh-17_01352_01901.db,$NUPLAN_DATA_ROOT/2021.10.05.07.10.04_veh-52_01442_01802.db]"
BRANCH_NAME="diffusion_planner_release"
ARGS_FILE="$REPO_ROOT/checkpoints/args.json"
CKPT_FILE="$REPO_ROOT/checkpoints/model.pth"
PLANNER="diffusion_planner"

cd "$REPO_ROOT"
echo "Checkpoint:  $CKPT_FILE"
echo "Mini log:    $NUPLAN_DATA_ROOT"
echo "Databases:   $DB_FILES"
echo "Results:     $NUPLAN_EXP_ROOT"
echo "Activations: $ACTIVATION_ROOT"
echo "Scenarios:   $SCENARIO_COUNT"
echo "t values:    $TARGET_TS"
echo "Ray threads: $RAY_THREADS_PER_NODE"

declare -A INITIAL_COUNTS
for TARGET_T in 0.1 0.2 0.4 0.6 0.8; do
  ACTIVATION_DIR="$ACTIVATION_ROOT/t_$TARGET_T"
  mkdir -p "$ACTIVATION_DIR"
  INITIAL_COUNTS[$TARGET_T]="$(find "$ACTIVATION_DIR" -maxdepth 1 -type f -name '*.pt' | wc -l)"
done

env \
  HYDRA_FULL_ERROR="$HYDRA_FULL_ERROR" \
  MPLCONFIGDIR="$MPLCONFIGDIR" \
  RAY_TMPDIR="$RAY_TMPDIR" \
  TMPDIR="$TMPDIR" \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  OMP_NUM_THREADS="$OMP_NUM_THREADS" \
  MKL_NUM_THREADS="$MKL_NUM_THREADS" \
  NUPLAN_DEVKIT_ROOT="$NUPLAN_DEVKIT_ROOT" \
  NUPLAN_DATA_ROOT="$NUPLAN_DATA_ROOT" \
  NUPLAN_MAPS_ROOT="$NUPLAN_MAPS_ROOT" \
  NUPLAN_EXP_ROOT="$NUPLAN_EXP_ROOT" \
  "$PYTHON_BIN" -u "$NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py" \
  +simulation="$CHALLENGE" \
  ego_controller=perfect_tracking_controller \
  planner="$PLANNER" \
  planner.diffusion_planner.config.args_file="$ARGS_FILE" \
  planner.diffusion_planner.ckpt_path="$CKPT_FILE" \
  planner.diffusion_planner.activation_target_ts="$TARGET_TS" \
  planner.diffusion_planner.activation_dir="$ACTIVATION_ROOT" \
  planner.diffusion_planner.enable_sae_reconstruction=false \
  scenario_builder="$SCENARIO_BUILDER" \
  scenario_builder.data_root="$NUPLAN_DATA_ROOT" \
  scenario_builder.db_files="$DB_FILES" \
  scenario_filter="$SPLIT" \
  scenario_filter.limit_total_scenarios="$SCENARIO_COUNT" \
  scenario_filter.shuffle=true \
  group="$PREFERRED_STORAGE_ROOT" \
  experiment_uid="$PLANNER/mini/activations/multi_t" \
  verbose=true \
  worker=ray_distributed \
  worker.threads_per_node="$RAY_THREADS_PER_NODE" \
  distributed_mode=SINGLE_NODE \
  number_of_gpus_allocated_per_simulation=0.15 \
  enable_simulation_progress_bar=true \
  'hydra.searchpath=[pkg://diffusion_planner.config.scenario_filter,pkg://diffusion_planner.config,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]'

for TARGET_T in 0.1 0.2 0.4 0.6 0.8; do
  ACTIVATION_DIR="$ACTIVATION_ROOT/t_$TARGET_T"
  FINAL_COUNT="$(find "$ACTIVATION_DIR" -maxdepth 1 -type f -name '*.pt' | wc -l)"
  SAVED_COUNT="$((FINAL_COUNT - INITIAL_COUNTS[$TARGET_T]))"
  if [ "$SAVED_COUNT" -ne "$SCENARIO_COUNT" ]; then
    echo "ERROR: t_$TARGET_T added $SAVED_COUNT activations; expected $SCENARIO_COUNT" >&2
    exit 1
  fi
  echo "Completed t_$TARGET_T: added $SAVED_COUNT activations ($FINAL_COUNT total)"
done
echo "Simulation: $PREFERRED_STORAGE_ROOT/simulation/$CHALLENGE/$PLANNER/mini/activations/multi_t"
