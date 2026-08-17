#!/usr/bin/env python3
"""Plot expert, original-model, and SAE-altered trajectories from one record."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    data = torch.load(args.record, map_location="cpu", weights_only=False)
    output = args.output or args.record.with_suffix(".png")
    labels = {"expert": "Expert", "model": "Model", "altered": "SAE altered"}

    fig, ax = plt.subplots(figsize=(7, 7))
    for key in ("expert", "model", "altered"):
        trajectory = torch.as_tensor(data[key])
        ax.plot(
            trajectory[:, 0], trajectory[:, 1],
            color=data["colors"][key], label=labels[key], linewidth=2.5,
        )
    ax.scatter([0], [0], color="black", marker="o", label="Current ego")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitudinal position (m)")
    ax.set_ylabel("Lateral position (m)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    print(output)


if __name__ == "__main__":
    main()
