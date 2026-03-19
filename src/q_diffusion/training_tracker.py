"""Loss tracking for AdaRound optimization."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class BlockTrainingLog:
    block_idx: int
    history: List[dict] = field(default_factory=list)
    mse_before: float = 0.0     # Naive rounding MSE (before AdaRound)
    mse_after: float = 0.0      # Final MSE (after AdaRound freeze)
    improvement_ratio: float = 0.0  # mse_before / mse_after

    def append(self, iteration: int, recon_loss: float, reg_loss: float,
               total_loss: float, beta: float):
        self.history.append({
            "iter": iteration,
            "recon_loss": recon_loss,
            "reg_loss": reg_loss,
            "total_loss": total_loss,
            "beta": beta,
        })

    def finalize(self, mse_before: float, mse_after: float):
        self.mse_before = mse_before
        self.mse_after = mse_after
        self.improvement_ratio = mse_before / max(mse_after, 1e-12)


class TrainingTracker:
    """Accumulates BlockTrainingLogs across all 25 blocks."""

    def __init__(self):
        self.logs: List[BlockTrainingLog] = []

    def add_block(self, log: BlockTrainingLog):
        self.logs.append(log)

    def print_block_summary(self, log: BlockTrainingLog):
        print(f"  Block {log.block_idx:2d}: "
              f"MSE before={log.mse_before:.6e}, "
              f"after={log.mse_after:.6e}, "
              f"ratio={log.improvement_ratio:.2f}x")

    def print_overall_summary(self):
        print("\n" + "=" * 70)
        print("AdaRound Optimization Summary")
        print("=" * 70)
        print(f"{'Block':>6} | {'MSE Before':>12} | {'MSE After':>12} | {'Improvement':>12}")
        print("-" * 50)
        for log in self.logs:
            print(f"{log.block_idx:6d} | {log.mse_before:12.6e} | "
                  f"{log.mse_after:12.6e} | {log.improvement_ratio:11.2f}x")
        print("=" * 70)

        if self.logs:
            avg_ratio = sum(l.improvement_ratio for l in self.logs) / len(self.logs)
            print(f"Average improvement: {avg_ratio:.2f}x")

    def save_json(self, path: str):
        """Serialize all logs to JSON."""
        data = []
        for log in self.logs:
            data.append({
                "block_idx": log.block_idx,
                "mse_before": log.mse_before,
                "mse_after": log.mse_after,
                "improvement_ratio": log.improvement_ratio,
                "history": log.history,
            })
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Training logs saved to {path}")

    def plot_loss_curves(self, output_dir: str):
        """Generate per-block loss vs iteration PNGs."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available, skipping loss curve plots")
            return

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        for log in self.logs:
            if not log.history:
                continue
            iters = [h["iter"] for h in log.history]
            recon = [h["recon_loss"] for h in log.history]
            reg = [h["reg_loss"] for h in log.history]
            total = [h["total_loss"] for h in log.history]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            ax1.plot(iters, recon, label="recon_loss", alpha=0.8)
            ax1.plot(iters, total, label="total_loss", alpha=0.8)
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("Loss")
            ax1.set_title(f"Block {log.block_idx} — Reconstruction Loss")
            ax1.set_yscale("log")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.plot(iters, reg, label="reg_loss", color="orange", alpha=0.8)
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("Regularization Loss")
            ax2.set_title(f"Block {log.block_idx} — AdaRound Regularization")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            fig.tight_layout()
            fig.savefig(out / f"block_{log.block_idx:02d}_loss.png", dpi=100)
            plt.close(fig)

        print(f"Loss curve plots saved to {output_dir}")
