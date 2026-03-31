"""MLX forward-hook mechanism for capturing per-channel activation statistics.

MLX lacks PyTorch-style register_forward_hook, so we monkey-patch each target
module's __call__ via a dynamically created subclass.  Statistics are reduced
and materialized (mx.eval) inside the hook so the lazy computation graph can
be freed immediately.
"""

from __future__ import annotations

import logging
from typing import List

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


class ChannelStatsCollector:
    """Accumulates per-channel summary statistics across denoising steps.

    Instead of storing per-prompt records (memory-heavy), this collector keeps
    running aggregates: elementwise max and Welford online mean/variance across
    all prompts+seeds for each (layer, step_idx).

    After collection, ``get_trajectory(layer_name)`` returns arrays of shape
    ``[num_steps, d_in]``.
    """

    def __init__(self):
        self._step_idx: int = 0
        self._sigma: float = 0.0
        self._prompt_id: str = ""
        self._seed: int = 0
        self._call_count: int = 0

        # Keyed by (layer_name, step_idx).
        # Values: {"max": ndarray, "mean_sum": ndarray, "var_sum": ndarray,
        #          "count": int, "n_tokens": int}
        self._agg: dict = {}

    def set_context(self, step_idx: int, sigma: float,
                    prompt_id: str, seed: int):
        self._step_idx = step_idx
        self._sigma = sigma
        self._prompt_id = prompt_id
        self._seed = seed

    def record(self, layer_name: str, X: mx.array, W: mx.array):
        X_fp32 = X.astype(mx.float32)
        X_flat = X_fp32.reshape(-1, X_fp32.shape[-1])

        act_abs = mx.abs(X_flat)
        act_max = mx.max(act_abs, axis=0)
        act_mean = mx.mean(act_abs, axis=0)
        act_var = mx.var(act_abs, axis=0)

        mx.eval(act_max, act_mean, act_var)

        act_max_np = np.array(act_max)
        act_mean_np = np.array(act_mean)
        act_var_np = np.array(act_var)
        n_tokens = int(X_flat.shape[0])

        key = (layer_name, self._step_idx)
        if key not in self._agg:
            self._agg[key] = {
                "max": act_max_np.copy(),
                "mean_sum": act_mean_np.copy(),
                "var_sum": act_var_np.copy(),
                "count": 1,
                "n_tokens": n_tokens,
                "sigma": self._sigma,
            }
        else:
            entry = self._agg[key]
            np.maximum(entry["max"], act_max_np, out=entry["max"])
            entry["mean_sum"] += act_mean_np
            entry["var_sum"] += act_var_np
            entry["count"] += 1
            entry["n_tokens"] = max(entry["n_tokens"], n_tokens)

        self._call_count += 1

    @property
    def call_count(self) -> int:
        return self._call_count

    def layer_names(self) -> set:
        return {k[0] for k in self._agg}

    def num_steps(self) -> int:
        steps = {k[1] for k in self._agg}
        return max(steps) + 1 if steps else 0

    def sigma_values(self) -> np.ndarray:
        """Return sigma values ordered by step index."""
        step_sigma = {}
        for (_, step_idx), entry in self._agg.items():
            if step_idx not in step_sigma:
                step_sigma[step_idx] = entry["sigma"]
        n = self.num_steps()
        return np.array([step_sigma[i] for i in range(n)])

    def get_trajectory(self, layer_name: str) -> np.ndarray:
        """Return aggregated activation salience of shape [num_steps, d_in].

        Salience is the worst-case (elementwise max) channel max across all
        prompts and seeds, following the PTQ4DiT convention.
        """
        n = self.num_steps()
        rows = []
        for step_idx in range(n):
            key = (layer_name, step_idx)
            if key in self._agg:
                rows.append(self._agg[key]["max"])
            else:
                logger.warning("Missing data for %s step %d", layer_name, step_idx)
                d_in = next(iter(self._agg.values()))["max"].shape[0]
                rows.append(np.zeros(d_in))
        return np.stack(rows)

    def get_mean_trajectory(self, layer_name: str) -> np.ndarray:
        """Return mean activation magnitude trajectory [num_steps, d_in]."""
        n = self.num_steps()
        rows = []
        for step_idx in range(n):
            key = (layer_name, step_idx)
            if key in self._agg:
                entry = self._agg[key]
                rows.append(entry["mean_sum"] / entry["count"])
            else:
                logger.warning("Missing mean data for %s step %d", layer_name, step_idx)
                d_in = next(iter(self._agg.values()))["max"].shape[0]
                rows.append(np.zeros(d_in))
        return np.stack(rows)


class LinearHook:
    """Monkey-patches an nn.Linear module so that every __call__ routes through
    the collector's record method before executing the original linear op."""

    def __init__(self, module, name: str, collector: ChannelStatsCollector):
        self.name = name
        self.collector = collector
        self._original_cls = module.__class__

        outer = self
        original_call = module.__class__.__call__

        def hooked_call(self_module, x):
            outer.collector.record(outer.name, x, self_module.weight)
            return original_call(self_module, x)

        module.__class__ = type(
            module.__class__.__name__ + "_Hooked",
            (module.__class__,),
            {"__call__": hooked_call},
        )
        self.module = module

    def remove(self):
        """Restore the module's original class."""
        self.module.__class__ = self._original_cls


def install_hooks(
    registry: list[dict],
    collector: ChannelStatsCollector,
) -> List[LinearHook]:
    """Install hooks on every layer in the registry. Returns the hook handles."""
    hooks = []
    for entry in registry:
        hook = LinearHook(entry["module"], entry["name"], collector)
        hooks.append(hook)
    logger.info("Installed %d hooks", len(hooks))
    return hooks


def remove_hooks(hooks: List[LinearHook]):
    """Remove all hooks and restore original classes."""
    for hook in hooks:
        hook.remove()
    logger.info("Removed %d hooks", len(hooks))
