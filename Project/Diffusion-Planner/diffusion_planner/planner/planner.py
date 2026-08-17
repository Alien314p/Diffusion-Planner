
import warnings
from pathlib import Path
import torch
import numpy as np
from typing import Deque, Dict, List, Type

warnings.filterwarnings("ignore")

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.common.utils.interpolatable_state import InterpolatableState
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.observation.observation_type import Observation, DetectionsTracks
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner, PlannerInitialization, PlannerInput
)

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.data_process.data_processor import DataProcessor
from diffusion_planner.utils.config import Config

from diffusion_planner.activation_extractor import (
    DEFAULT_ACTIVATION_DIR,
    DEFAULT_SAE_PATH,
    DiTActivationExtractor,
    SAEReconstructionHook,
)

def identity(ego_state, predictions):
    return predictions


class DiffusionPlanner(AbstractPlanner):
    # Ask nuPlan's planner builder to pass the scenario to the constructor.
    requires_scenario: bool = True

    def __init__(
            self,
            config: Config,
            ckpt_path: str,

            past_trajectory_sampling: TrajectorySampling, 
            future_trajectory_sampling: TrajectorySampling,
            scenario: AbstractScenario,

            enable_ema: bool = True,
            device: str = "cpu",
            activation_dir: str = str(DEFAULT_ACTIVATION_DIR),
            activation_target_ts=(0.1, 0.2, 0.4, 0.6, 0.8),
            enable_sae_reconstruction: bool = False,
            sae_path: str = str(DEFAULT_SAE_PATH),
            comparison_dir: str = "/data/saba/parnia/trajectory_comparisons",
        ):

        assert device in ["cpu", "cuda"], f"device {device} not supported"
        if device == "cuda":
            assert torch.cuda.is_available(), "cuda is not available"
            
        self._future_horizon = future_trajectory_sampling.time_horizon # [s] 
        self._step_interval = future_trajectory_sampling.time_horizon / future_trajectory_sampling.num_poses # [s]
        
        self._config = config
        self._ckpt_path = ckpt_path

        self._past_trajectory_sampling = past_trajectory_sampling
        self._future_trajectory_sampling = future_trajectory_sampling

        self._ema_enabled = enable_ema
        self._device = device
        self._scenario = scenario
        self._enable_sae_reconstruction = enable_sae_reconstruction
        self._comparison_dir = Path(comparison_dir)
        self._comparison_index = 0
        self._scenario_info = {
            "token": scenario.token,
            "scenario_name": scenario.scenario_name,
            "scenario_type": scenario.scenario_type,
            "log_name": scenario.log_name,
            "map_name": scenario.map_api.map_name,
        }

        

        self._planner = Diffusion_Planner(config)

        self.activation_extractor = DiTActivationExtractor(
            self._planner,
            save_dir=activation_dir,
            target_ts=activation_target_ts,
        )
        self.sae_reconstruction_hook = None
        if enable_sae_reconstruction:
            self.sae_reconstruction_hook = SAEReconstructionHook(self._planner, sae_path)
            self._comparison_dir.mkdir(parents=True, exist_ok=True)

        self.data_processor = DataProcessor(config)
        
        self.observation_normalizer = config.observation_normalizer

    def name(self) -> str:
        """
        Inherited.
        """
        return "diffusion_planner"
    
    def observation_type(self) -> Type[Observation]:
        """
        Inherited.
        """
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        """
        Inherited.
        """
        self._map_api = initialization.map_api
        self._route_roadblock_ids = initialization.route_roadblock_ids

        if self._ckpt_path is not None:
            state_dict:Dict = torch.load(self._ckpt_path, map_location=self._device)
            
            if self._ema_enabled:
                state_dict = state_dict['ema_state_dict']
            else:
                if "model" in state_dict.keys():
                    state_dict = state_dict['model']
            # use for ddp
            model_state_dict = {k[len("module."):]: v for k, v in state_dict.items() if k.startswith("module.")}
            self._planner.load_state_dict(model_state_dict)
        else:
            print("load random model")
        
        self._planner.eval()
        self._planner = self._planner.to(self._device)
        self._initialization = initialization

        self.activation_extractor.reset_simulation(self._scenario_info)


    def planner_input_to_model_inputs(self, planner_input: PlannerInput) -> Dict[str, torch.Tensor]:
        history = planner_input.history
        traffic_light_data = list(planner_input.traffic_light_data)
        model_inputs = self.data_processor.observation_adapter(history, traffic_light_data, self._map_api, self._route_roadblock_ids, self._device)

        return model_inputs

    def outputs_to_trajectory(self, outputs: Dict[str, torch.Tensor], ego_state_history: Deque[EgoState]) -> List[InterpolatableState]:    
        # Scenario metadata is available here as self._scenario_info, and the
        # complete nuPlan scenario object is available as self._scenario.
        predictions = outputs['prediction'][0, 0].detach().cpu().numpy().astype(np.float64) # T, 4
        heading = np.arctan2(predictions[:, 3], predictions[:, 2])[..., None]
        predictions = np.concatenate([predictions[..., :2], heading], axis=-1) 

        states = transform_predictions_to_states(predictions, ego_state_history, self._future_horizon, self._step_interval)

        return states
    
    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Inherited.
        """
        inputs = self.planner_input_to_model_inputs(current_input)

        inputs = self.observation_normalizer(inputs)     

        self.activation_extractor.begin_planning_call()

        if self.sae_reconstruction_hook is None:
            _, outputs = self._planner(inputs)
        else:
            # Restore RNG before the altered pass so both trajectories start from
            # exactly the same diffusion noise; their only difference is the SAE.
            cpu_rng = torch.random.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

            self.sae_reconstruction_hook.enabled = False
            _, baseline_outputs = self._planner(inputs)

            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            self.sae_reconstruction_hook.enabled = True
            try:
                _, outputs = self._planner(inputs)
            finally:
                self.sae_reconstruction_hook.enabled = False

            self._save_trajectory_comparison(baseline_outputs, outputs, current_input)

        self.activation_extractor.finish_planning_call()

        # _, outputs = self._planner(inputs)   
        

        trajectory = InterpolatedTrajectory(
            trajectory=self.outputs_to_trajectory(outputs, current_input.history.ego_states)
        )

        return trajectory

    def _save_trajectory_comparison(self, baseline, altered, current_input):
        """Save plot-ready local trajectories with stable, distinct colors."""
        iteration = current_input.iteration.index
        expert_states = list(self._scenario.get_ego_future_trajectory(
            iteration,
            self._future_horizon,
            self._future_trajectory_sampling.num_poses,
        ))
        from nuplan.planning.training.preprocessing.features.trajectory_utils import (
            convert_absolute_to_relative_poses,
        )
        anchor = current_input.history.current_state[0].rear_axle
        expert = convert_absolute_to_relative_poses(
            anchor, [state.rear_axle for state in expert_states]
        )
        record = {
            "scenario_info": self._scenario_info,
            "iteration": iteration,
            "expert": torch.as_tensor(expert),
            "model": baseline["prediction"][0, 0].detach().cpu(),
            "altered": altered["prediction"][0, 0].detach().cpu(),
            "colors": {"expert": "#FA9600", "model": "#1f77b4", "altered": "#d627a8"},
        }
        name = f"{self._scenario.token}_{self._comparison_index:04d}.pt"
        torch.save(record, self._comparison_dir / name)
        self._comparison_index += 1
