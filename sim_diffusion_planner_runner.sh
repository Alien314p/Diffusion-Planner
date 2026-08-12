#!/usr/bin/env bash
set -euo pipefail

export HYDRA_FULL_ERROR=1
export MPLCONFIGDIR=/tmp/diffusion-planner-matplotlib
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

REPO_ROOT="/data/saba/parnia/Project/Diffusion-Planner"
export NUPLAN_DEVKIT_ROOT="/data/saba/parnia/nuplan-devkit"
export NUPLAN_DATA_ROOT="/data/saba/parnia/Project/data/data/cache/mini"
export NUPLAN_MAPS_ROOT="/data/saba/parnia/nuplan_maps"
export NUPLAN_EXP_ROOT="/data/saba/parnia/Project/exp"

PYTHON_BIN="/data/saba/miniforge3/envs/diffusion_planner/bin/python"
SPLIT="all_scenarios"
CHALLENGE="closed_loop_nonreactive_agents"
SCENARIO_BUILDER="nuplan_mini"
DB_FILES="[$NUPLAN_DATA_ROOT/2021.05.12.22.00.38_veh-35_01008_01518.db,$NUPLAN_DATA_ROOT/2021.06.14.18.33.41_veh-35_03901_04264.db,$NUPLAN_DATA_ROOT/2021.07.16.00.51.05_veh-17_01352_01901.db,$NUPLAN_DATA_ROOT/2021.10.05.07.10.04_veh-52_01442_01802.db]"
BRANCH_NAME="diffusion_planner_release"
ARGS_FILE="$REPO_ROOT/checkpoints/args.json"
CKPT_FILE="$REPO_ROOT/checkpoints/model.pth"
PLANNER="diffusion_planner"

cd "$REPO_ROOT"
echo "Checkpoint: $CKPT_FILE"
echo "Mini log:   $NUPLAN_DATA_ROOT"
echo "Databases:  $DB_FILES"
echo "Results:    $NUPLAN_EXP_ROOT"

"$PYTHON_BIN" -u "$NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py" \
  +simulation="$CHALLENGE" \
  ego_controller=perfect_tracking_controller \
  planner="$PLANNER" \
  planner.diffusion_planner.config.args_file="$ARGS_FILE" \
  planner.diffusion_planner.ckpt_path="$CKPT_FILE" \
  scenario_builder="$SCENARIO_BUILDER" \
  scenario_builder.data_root="$NUPLAN_DATA_ROOT" \
  scenario_builder.db_files="$DB_FILES" \
  scenario_filter="$SPLIT" \
  scenario_filter.limit_total_scenarios=150 \
  scenario_filter.shuffle=true \
  experiment_uid="$PLANNER/mini/$BRANCH_NAME/model_$(date +%Y-%m-%d-%H-%M-%S)" \
  verbose=true \
  worker=ray_distributed \
  worker.threads_per_node=2 \
  distributed_mode=SINGLE_NODE \
  number_of_gpus_allocated_per_simulation=0.15 \
  enable_simulation_progress_bar=true \
  'hydra.searchpath=[pkg://diffusion_planner.config.scenario_filter,pkg://diffusion_planner.config,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]'
