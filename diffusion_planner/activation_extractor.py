import uuid
from pathlib import Path

import torch


DEFAULT_ACTIVATION_DIR = Path("/data/saba/parnia/activation_pt_files")


class DiTActivationExtractor:
    """
    Saves ONE ego activation per simulation:
      - after the second DiT block: blocks[1]
      - at the DiT call whose diffusion time is closest to target_t (default 0.5)
      - activation shape: [hidden_dim] (normally [192])
    """

    def __init__(self, model, save_dir=DEFAULT_ACTIVATION_DIR, target_t=0.5):
        self.target_t = target_t
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Diffusion_Planner -> decoder -> Decoder -> DiT
        self.dit = model.decoder.decoder.dit

        # blocks[1] = SECOND DiT block
        self.second_block = self.dit.blocks[1]

        self.current_t = None
        self.best = None
        self.saved_for_simulation = False
        self.scenario_info = {}
        self.simulation_id = None

        # Hook 1: remember which diffusion timestep DiT is currently processing.
        self._dit_hook = self.dit.register_forward_pre_hook(self._on_dit_start)

        # Hook 2: capture output activation of the second DiT block.
        self._block_hook = self.second_block.register_forward_hook(self._on_second_block)

    def reset_simulation(self, scenario_info=None):
        """Call once when a new nuPlan simulation/scenario starts."""
        self.current_t = None
        self.best = None
        self.saved_for_simulation = False
        self.scenario_info = scenario_info or {}
        self.simulation_id = uuid.uuid4().hex[:12]

    def begin_planning_call(self):
        """
        Call immediately before model inference.
        We only use the first planning call of each simulation.
        """
        if not self.saved_for_simulation:
            self.best = None

    def _on_dit_start(self, module, inputs):
        if self.saved_for_simulation:
            return

        # DiT.forward(x, t, ...)
        t = inputs[1]
        self.current_t = float(t.detach().flatten()[0].cpu())

    def _on_second_block(self, module, inputs, output):
        if self.saved_for_simulation or self.current_t is None:
            return

        distance = abs(self.current_t - self.target_t)

        # Keep the activation from the timestep closest to t=0.5.
        if self.best is None or distance < self.best["distance"]:
            # output shape: [B, P, hidden_dim]
            # [0, 0] = first batch item, ego token
            ego_activation = output[0, 0].detach().cpu().clone()

            self.best = {
                "distance": distance,
                "actual_t": self.current_t,
                "activation": ego_activation,
            }

    def finish_planning_call(self):
        """
        Call immediately after model inference.
        Saves one .pt file, then ignores later planner calls in this simulation.
        """
        if self.saved_for_simulation or self.best is None:
            return

        record = {
            "simulation_id": self.simulation_id,
            "scenario_info": self.scenario_info,
            "target_t": self.target_t,
            "actual_t": self.best["actual_t"],
            "block_index": 1,  # second block
            "agent": "ego",
            "activation": self.best["activation"],  # normally shape [192]
        }

        path = self.save_dir / f"{self.simulation_id}.pt"
        torch.save(record, path)

        print(
            f"[ActivationExtractor] saved {path} | "
            f"t={record['actual_t']:.4f} | "
            f"shape={tuple(record['activation'].shape)}"
        )

        self.saved_for_simulation = True

    def close(self):
        self._dit_hook.remove()
        self._block_hook.remove()
