import uuid
from pathlib import Path

import torch


DEFAULT_ACTIVATION_DIR = Path("/data/saba/parnia/activation_pt_files")
DEFAULT_SAE_PATH = Path("/data/saba/parnia/Project/sae_output/ae_final.pt")


class TopKSAE(torch.nn.Module):
    """Small inference-only copy of the Top-K SAE used by ``sae_train.py``.

    Keeping this here avoids importing ``dictionary_learning`` (and its nnsight
    side effects) in every nuPlan simulation worker.
    """

    def __init__(self, activation_dim, dict_size, k):
        super().__init__()
        self.encoder = torch.nn.Linear(activation_dim, dict_size)
        self.decoder = torch.nn.Linear(dict_size, activation_dim, bias=False)
        self.b_dec = torch.nn.Parameter(torch.zeros(activation_dim))
        self.register_buffer("k", torch.tensor(k, dtype=torch.int))
        self.register_buffer("threshold", torch.tensor(-1.0))

    def forward(self, activation):
        features = torch.relu(self.encoder(activation - self.b_dec))
        topk = features.topk(int(self.k.item()), dim=-1, sorted=False)
        sparse = torch.zeros_like(features).scatter_(-1, topk.indices, topk.values)
        return self.decoder(sparse) + self.b_dec

    @classmethod
    def from_checkpoint(cls, path):
        state = torch.load(path, map_location="cpu")
        dict_size, activation_dim = state["encoder.weight"].shape
        model = cls(activation_dim, dict_size, int(state["k"].item()))
        model.load_state_dict(state)
        model.eval()
        model.requires_grad_(False)
        return model


class SAEReconstructionHook:
    """Replace the second DiT block's ego activation with its SAE reconstruction.

    A forward hook may return a replacement output. Consequently the reconstructed
    tensor becomes the input to ``blocks[2]`` without changing the other tokens.
    """

    def __init__(self, model, sae_path=DEFAULT_SAE_PATH, block_index=1):
        self.sae = TopKSAE.from_checkpoint(Path(sae_path))
        self.enabled = False
        self.block_index = block_index
        self._device = None
        block = model.decoder.decoder.dit.blocks[block_index]
        self._hook = block.register_forward_hook(self._reconstruct_ego)

    def _reconstruct_ego(self, module, inputs, output):
        if not self.enabled:
            return None
        if self._device != output.device:
            self.sae.to(output.device)
            self._device = output.device
        altered = output.clone()
        with torch.no_grad():
            altered[:, 0, :] = self.sae(output[:, 0, :])
        return altered

    def close(self):
        self._hook.remove()


class DiTActivationExtractor:
    """
    Saves one ego activation per requested diffusion timestep and simulation:
      - after the second DiT block: blocks[1]
      - at the DiT calls closest to each value in target_ts
      - each target is written to save_dir/t_<target>/
      - activation shape: [hidden_dim] (normally [192])
    """

    def __init__(
        self,
        model,
        save_dir=DEFAULT_ACTIVATION_DIR,
        target_ts=(0.1, 0.2, 0.4, 0.6, 0.8),
    ):
        self.target_ts = tuple(float(target_t) for target_t in target_ts)
        if not self.target_ts:
            raise ValueError("target_ts must contain at least one timestep")
        if len(set(self.target_ts)) != len(self.target_ts):
            raise ValueError("target_ts must not contain duplicates")

        self.save_dir = Path(save_dir)
        self.target_dirs = {
            target_t: self.save_dir / f"t_{target_t:g}" for target_t in self.target_ts
        }
        for target_dir in self.target_dirs.values():
            target_dir.mkdir(parents=True, exist_ok=True)

        # Diffusion_Planner -> decoder -> Decoder -> DiT
        self.dit = model.decoder.decoder.dit

        # blocks[1] = SECOND DiT block
        self.second_block = self.dit.blocks[1]

        self.current_t = None
        self.best_by_target = {}
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
        self.best_by_target = {}
        self.saved_for_simulation = False
        self.scenario_info = scenario_info or {}
        self.simulation_id = uuid.uuid4().hex[:12]

    def begin_planning_call(self):
        """
        Call immediately before model inference.
        We only use the first planning call of each simulation.
        """
        if not self.saved_for_simulation:
            self.best_by_target = {}

    def _on_dit_start(self, module, inputs):
        if self.saved_for_simulation:
            return

        # DiT.forward(x, t, ...)
        t = inputs[1]
        self.current_t = float(t.detach().flatten()[0].cpu())

    def _on_second_block(self, module, inputs, output):
        if self.saved_for_simulation or self.current_t is None:
            return

        # output shape: [B, P, hidden_dim]; [0, 0] is the first ego token.
        ego_activation = None
        for target_t in self.target_ts:
            distance = abs(self.current_t - target_t)
            best = self.best_by_target.get(target_t)
            if best is None or distance < best["distance"]:
                if ego_activation is None:
                    ego_activation = output[0, 0].detach().cpu().clone()
                self.best_by_target[target_t] = {
                    "distance": distance,
                    "actual_t": self.current_t,
                    "activation": ego_activation,
                }

    def finish_planning_call(self):
        """
        Call immediately after model inference.
        Saves one .pt file per requested t, then ignores later planner calls.
        """
        if self.saved_for_simulation or not self.best_by_target:
            return

        for target_t in self.target_ts:
            best = self.best_by_target.get(target_t)
            if best is None:
                continue
            record = {
                "simulation_id": self.simulation_id,
                "scenario_info": self.scenario_info,
                "target_t": target_t,
                "actual_t": best["actual_t"],
                "block_index": 1,  # second block
                "agent": "ego",
                "activation": best["activation"],  # normally shape [192]
            }
            path = self.target_dirs[target_t] / f"{self.simulation_id}.pt"
            torch.save(record, path)
            print(
                f"[ActivationExtractor] saved {path} | "
                f"target_t={target_t:g} | actual_t={record['actual_t']:.4f} | "
                f"shape={tuple(record['activation'].shape)}"
            )

        self.saved_for_simulation = True

    def close(self):
        self._dit_hook.remove()
        self._block_hook.remove()
