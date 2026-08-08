#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/data/saba/parnia/Project"
REPO_ROOT="$PROJECT_ROOT/Diffusion-Planner"
NUPLAN_DEVKIT_ROOT="/data/saba/parnia/nuplan-devkit"
NUPLAN_DATA_ROOT="$PROJECT_ROOT/data/data/cache/mini"
NUPLAN_MAPS_ROOT="/data/saba/parnia/nuplan_maps"
NUPLAN_EXP_ROOT="$PROJECT_ROOT/exp"
PYTHON_BIN="/data/saba/miniforge3/envs/diffusion_planner/bin/python"

export NUPLAN_DEVKIT_ROOT NUPLAN_DATA_ROOT NUPLAN_MAPS_ROOT NUPLAN_EXP_ROOT
export HYDRA_FULL_ERROR=1
export MPLCONFIGDIR="/tmp/diffusion-planner-matplotlib"

cd "$REPO_ROOT"
"$PYTHON_BIN" -u "$NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py" \
  +simulation=closed_loop_nonreactive_agents \
  planner=diffusion_planner \
  planner.diffusion_planner.config.args_file="$REPO_ROOT/checkpoints/args.json" \
  planner.diffusion_planner.ckpt_path="$REPO_ROOT/checkpoints/model.pth" \
  planner.diffusion_planner.device=cpu \
  scenario_builder=nuplan_mini \
  scenario_builder.data_root="$NUPLAN_DATA_ROOT" \
  scenario_filter=all_scenarios \
  scenario_filter.limit_total_scenarios=1 \
  scenario_filter.shuffle=false \
  experiment_uid="diffusion_planner/mini/cpu_$(date +%Y%m%d_%H%M%S)" \
  verbose=true \
  worker=sequential \
  distributed_mode=SINGLE_NODE \
  number_of_gpus_allocated_per_simulation=0 \
  enable_simulation_progress_bar=true \
  'hydra.searchpath=[pkg://diffusion_planner.config.scenario_filter,pkg://diffusion_planner.config,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]'
