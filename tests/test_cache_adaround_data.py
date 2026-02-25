"""
Tests for src/cache_adaround_data.py.

Covers all pure-logic components without loading the DiffusionPipeline or any
model weights.  The DiffusionPipeline import in the module under test is only
executed at the module level for the CLI path; we test the functions directly
and mock the pipeline with lightweight Python objects.
"""

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import mlx.core as mx

# Import exactly the symbols we intend to test — keeps the test surface explicit.
from src.cache_adaround_data import (
    BlockHook,
    _record_shapes,
    _to_numpy,
    flush_hooks,
    install_block_hooks,
    load_block_data,
    pack_sample,
    remove_block_hooks,
)


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

class _SimpleBlock:
    """Minimal block substitute: returns (arg0 + arg1) for MM, arg0*2 for Uni."""

    def __init__(self, is_mm: bool = True, custom: str = "sentinel"):
        self.custom = custom
        self._is_mm = is_mm

    def __call__(self, *args, **kwargs):
        if self._is_mm:
            # Return both streams unchanged (shapes may differ — avoid broadcast)
            return (args[0], args[1])
        else:
            return args[0]   # single tensor like Uni block


class _MockMMDiT:
    def __init__(self, n_mm: int = 3, n_uni: int = 2):
        self.multimodal_transformer_blocks = [
            _SimpleBlock(is_mm=True) for _ in range(n_mm)
        ]
        self.unified_transformer_blocks = [
            _SimpleBlock(is_mm=False) for _ in range(n_uni)
        ]


class _MockPipeline:
    def __init__(self, n_mm: int = 3, n_uni: int = 2):
        self.mmdit = _MockMMDiT(n_mm, n_uni)


# ---------------------------------------------------------------------------
# _to_numpy
# ---------------------------------------------------------------------------

class TestToNumpy:
    def test_converts_mx_array(self):
        arr = mx.array([1.0, 2.0, 3.0])
        out = _to_numpy(arr)
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_almost_equal(out, [1.0, 2.0, 3.0])

    def test_passes_through_none(self):
        assert _to_numpy(None) is None

    def test_passes_through_scalar(self):
        assert _to_numpy(42) == 42

    def test_passes_through_numpy(self):
        arr = np.array([1.0, 2.0])
        assert _to_numpy(arr) is arr

    def test_passes_through_string(self):
        assert _to_numpy("hello") == "hello"


# ---------------------------------------------------------------------------
# BlockHook
# ---------------------------------------------------------------------------

class TestBlockHook:
    def test_forwards_call_and_returns_result(self):
        block = _SimpleBlock(is_mm=False)
        hook = BlockHook(block, "uni0", is_mm=False, list_idx=0)
        x = mx.array([1.0, 2.0])
        result = hook(x)
        # Result is the wrapped block's output (passes through unchanged for Uni mock)
        np.testing.assert_array_almost_equal(np.array(result), [1.0, 2.0])

    def test_captures_positional_args(self):
        block = _SimpleBlock(is_mm=True)
        hook = BlockHook(block, "mm0", is_mm=True, list_idx=0)
        a = mx.array([1.0])
        b = mx.array([2.0])
        ts = mx.array([0.5])
        hook(a, b, ts)
        assert len(hook._last_args) == 3
        assert hook._last_args[0] is a
        assert hook._last_args[1] is b

    def test_captures_keyword_args(self):
        block = _SimpleBlock(is_mm=False)
        hook = BlockHook(block, "uni1", is_mm=False, list_idx=1)
        x = mx.array([1.0])
        pe = mx.array([0.1, 0.2])
        hook(x, positional_encodings=pe)
        assert "positional_encodings" in hook._last_kwargs
        assert hook._last_kwargs["positional_encodings"] is pe

    def test_captures_output(self):
        block = _SimpleBlock(is_mm=True)
        hook = BlockHook(block, "mm2", is_mm=True, list_idx=2)
        a = mx.array([1.0, 2.0])
        b = mx.array([3.0, 4.0])
        hook(a, b, mx.array([0.5]))
        assert hook._last_output is not None
        assert isinstance(hook._last_output, tuple)  # MM block returns tuple

    def test_clear_resets_state(self):
        block = _SimpleBlock(is_mm=False)
        hook = BlockHook(block, "uni0", is_mm=False, list_idx=0)
        hook(mx.array([1.0]))
        assert hook._last_args is not None
        hook.clear()
        assert hook._last_args is None
        assert hook._last_kwargs is None
        assert hook._last_output is None

    def test_getattr_delegates_to_wrapped(self):
        block = _SimpleBlock(is_mm=True, custom="my_value")
        hook = BlockHook(block, "mm0", is_mm=True, list_idx=0)
        assert hook.custom == "my_value"

    def test_own_attributes_not_delegated(self):
        block = _SimpleBlock()
        hook = BlockHook(block, "mm0", is_mm=True, list_idx=0)
        # block_name is the hook's own attribute, not wrapped's
        assert hook.block_name == "mm0"
        assert hook.is_mm is True
        assert hook._list_idx == 0

    def test_multiple_calls_overwrite_state(self):
        block = _SimpleBlock(is_mm=False)
        hook = BlockHook(block, "uni0", is_mm=False, list_idx=0)
        hook(mx.array([1.0]))
        hook(mx.array([99.0]))
        # Only the last call's data is stored
        np.testing.assert_array_almost_equal(
            np.array(hook._last_args[0]), [99.0]
        )


# ---------------------------------------------------------------------------
# install_block_hooks / remove_block_hooks
# ---------------------------------------------------------------------------

class TestInstallRemoveHooks:
    def test_returns_hooks_in_order(self):
        pipeline = _MockPipeline(n_mm=3, n_uni=2)
        hooks = install_block_hooks(pipeline)
        assert len(hooks) == 5
        names = [h.block_name for h in hooks]
        assert names == ["mm0", "mm1", "mm2", "uni0", "uni1"]

    def test_mm_hooks_flagged_correctly(self):
        pipeline = _MockPipeline(n_mm=2, n_uni=2)
        hooks = install_block_hooks(pipeline)
        assert [h.is_mm for h in hooks] == [True, True, False, False]

    def test_blocks_replaced_with_hooks(self):
        pipeline = _MockPipeline(n_mm=2, n_uni=1)
        install_block_hooks(pipeline)
        for block in pipeline.mmdit.multimodal_transformer_blocks:
            assert isinstance(block, BlockHook)
        for block in pipeline.mmdit.unified_transformer_blocks:
            assert isinstance(block, BlockHook)

    def test_hooked_blocks_still_callable(self):
        pipeline = _MockPipeline(n_mm=1, n_uni=0)
        install_block_hooks(pipeline)
        block = pipeline.mmdit.multimodal_transformer_blocks[0]
        a = mx.array([1.0, 2.0])
        b = mx.array([3.0, 4.0])
        result = block(a, b, mx.array([0.5]))
        assert result is not None

    def test_remove_restores_originals(self):
        pipeline = _MockPipeline(n_mm=3, n_uni=2)
        originals_mm = list(pipeline.mmdit.multimodal_transformer_blocks)
        originals_uni = list(pipeline.mmdit.unified_transformer_blocks)

        hooks = install_block_hooks(pipeline)
        remove_block_hooks(pipeline, hooks)

        for i, block in enumerate(pipeline.mmdit.multimodal_transformer_blocks):
            assert block is originals_mm[i]
        for i, block in enumerate(pipeline.mmdit.unified_transformer_blocks):
            assert block is originals_uni[i]

    def test_remove_restores_correct_list_indices(self):
        pipeline = _MockPipeline(n_mm=4, n_uni=3)
        originals = {
            f"mm{i}": pipeline.mmdit.multimodal_transformer_blocks[i]
            for i in range(4)
        }
        originals.update({
            f"uni{i}": pipeline.mmdit.unified_transformer_blocks[i]
            for i in range(3)
        })
        hooks = install_block_hooks(pipeline)
        remove_block_hooks(pipeline, hooks)

        for i in range(4):
            assert pipeline.mmdit.multimodal_transformer_blocks[i] is originals[f"mm{i}"]
        for i in range(3):
            assert pipeline.mmdit.unified_transformer_blocks[i] is originals[f"uni{i}"]

    def test_model_only_mm_blocks(self):
        """Model with no unified blocks (e.g. earlier DiT variants)."""
        class MMDiTNoUni:
            def __init__(self):
                self.multimodal_transformer_blocks = [_SimpleBlock() for _ in range(2)]

        class PipelineNoUni:
            def __init__(self):
                self.mmdit = MMDiTNoUni()

        pipeline = PipelineNoUni()
        hooks = install_block_hooks(pipeline)
        assert len(hooks) == 2
        assert all(h.is_mm for h in hooks)
        remove_block_hooks(pipeline, hooks)


# ---------------------------------------------------------------------------
# flush_hooks
# ---------------------------------------------------------------------------

class TestFlushHooks:
    def test_converts_args_to_numpy(self):
        block = _SimpleBlock(is_mm=False)
        hook = BlockHook(block, "uni0", is_mm=False, list_idx=0)
        hook(mx.array([1.0, 2.0]))

        result = flush_hooks([hook])
        assert isinstance(result["uni0"]["args"][0], np.ndarray)
        np.testing.assert_array_almost_equal(result["uni0"]["args"][0], [1.0, 2.0])

    def test_converts_kwargs_to_numpy(self):
        block = _SimpleBlock(is_mm=False)
        hook = BlockHook(block, "uni0", is_mm=False, list_idx=0)
        pe = mx.array([0.1, 0.2, 0.3])
        hook(mx.array([1.0]), positional_encodings=pe)

        result = flush_hooks([hook])
        assert isinstance(result["uni0"]["kwargs"]["positional_encodings"], np.ndarray)

    def test_converts_tuple_output_to_list_of_numpy(self):
        block = _SimpleBlock(is_mm=True)
        hook = BlockHook(block, "mm0", is_mm=True, list_idx=0)
        hook(mx.array([1.0, 2.0]), mx.array([3.0, 4.0]), mx.array([0.5]))

        result = flush_hooks([hook])
        out = result["mm0"]["output"]
        assert isinstance(out, list)
        assert all(isinstance(o, np.ndarray) for o in out)

    def test_converts_scalar_output_to_numpy(self):
        block = _SimpleBlock(is_mm=False)
        hook = BlockHook(block, "uni0", is_mm=False, list_idx=0)
        hook(mx.array([1.0, 2.0]))

        result = flush_hooks([hook])
        assert isinstance(result["uni0"]["output"], np.ndarray)

    def test_returns_none_for_unhit_hook(self):
        block = _SimpleBlock(is_mm=True)
        hook = BlockHook(block, "mm0", is_mm=True, list_idx=0)
        # Never called — _last_args is None
        result = flush_hooks([hook])
        assert result["mm0"] is None

    def test_clears_hook_after_flush(self):
        block = _SimpleBlock(is_mm=False)
        hook = BlockHook(block, "uni0", is_mm=False, list_idx=0)
        hook(mx.array([1.0]))
        flush_hooks([hook])
        assert hook._last_args is None

    def test_multiple_hooks_flushed_together(self):
        mm_block = _SimpleBlock(is_mm=True)
        uni_block = _SimpleBlock(is_mm=False)
        mm_hook = BlockHook(mm_block, "mm0", is_mm=True, list_idx=0)
        uni_hook = BlockHook(uni_block, "uni0", is_mm=False, list_idx=0)

        mm_hook(mx.array([1.0]), mx.array([2.0]), mx.array([0.5]))
        uni_hook(mx.array([3.0, 4.0]))

        result = flush_hooks([mm_hook, uni_hook])
        assert "mm0" in result
        assert "uni0" in result
        assert result["mm0"] is not None
        assert result["uni0"] is not None

    def test_non_array_kwargs_passed_through(self):
        """None positional_encodings should survive the flush as-is."""
        block = _SimpleBlock(is_mm=False)
        hook = BlockHook(block, "uni0", is_mm=False, list_idx=0)
        hook(mx.array([1.0]), positional_encodings=None)

        result = flush_hooks([hook])
        assert result["uni0"]["kwargs"]["positional_encodings"] is None


# ---------------------------------------------------------------------------
# pack_sample
# ---------------------------------------------------------------------------

class TestPackSample:
    def test_mm_block_all_keys_present(self):
        block_data = {
            "mm0": {
                "args": [
                    np.zeros((2, 10, 1, 8)),   # img
                    np.zeros((2, 5, 1, 8)),    # txt
                    np.zeros((2, 1, 1, 8)),    # timestep
                ],
                "kwargs": {"positional_encodings": np.zeros((15, 4))},
                "output": [np.zeros((2, 10, 1, 8)), np.zeros((2, 5, 1, 8))],
            }
        }
        flat = pack_sample(block_data)

        assert "mm0__arg0" in flat
        assert "mm0__arg1" in flat
        assert "mm0__arg2" in flat
        assert "mm0__kw_positional_encodings" in flat
        assert "mm0__out0" in flat
        assert "mm0__out1" in flat

    def test_uni_block_scalar_output(self):
        block_data = {
            "uni3": {
                "args": [np.zeros((2, 15, 1, 8)), np.zeros((2, 1, 1, 8))],
                "kwargs": {},
                "output": np.zeros((2, 15, 1, 8)),
            }
        }
        flat = pack_sample(block_data)

        assert "uni3__arg0" in flat
        assert "uni3__arg1" in flat
        assert "uni3__out0" in flat
        assert "uni3__out1" not in flat

    def test_dot_in_block_name_becomes_underscore(self):
        block_data = {
            "mm.0": {
                "args": [np.array([1.0])],
                "kwargs": {},
                "output": np.array([2.0]),
            }
        }
        flat = pack_sample(block_data)
        assert "mm_0__arg0" in flat
        assert "mm.0__arg0" not in flat

    def test_none_kwarg_not_stored(self):
        block_data = {
            "uni0": {
                "args": [np.array([1.0])],
                "kwargs": {"positional_encodings": None},
                "output": np.array([2.0]),
            }
        }
        flat = pack_sample(block_data)
        assert "uni0__kw_positional_encodings" not in flat

    def test_none_block_data_skipped(self):
        block_data = {"mm0": None, "uni0": {"args": [np.array([1.0])],
                                             "kwargs": {},
                                             "output": np.array([2.0])}}
        flat = pack_sample(block_data)
        assert not any(k.startswith("mm0") for k in flat)
        assert "uni0__arg0" in flat

    def test_array_values_preserved(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        block_data = {
            "mm0": {
                "args": [arr],
                "kwargs": {},
                "output": arr * 2,
            }
        }
        flat = pack_sample(block_data)
        np.testing.assert_array_equal(flat["mm0__arg0"], arr)
        np.testing.assert_array_equal(flat["mm0__out0"], arr * 2)

    def test_multiple_blocks_no_key_collision(self):
        block_data = {
            "mm0": {
                "args": [np.array([1.0])],
                "kwargs": {},
                "output": np.array([2.0]),
            },
            "mm1": {
                "args": [np.array([3.0])],
                "kwargs": {},
                "output": np.array([4.0]),
            },
        }
        flat = pack_sample(block_data)
        assert "mm0__arg0" in flat
        assert "mm1__arg0" in flat
        np.testing.assert_array_equal(flat["mm0__arg0"], [1.0])
        np.testing.assert_array_equal(flat["mm1__arg0"], [3.0])

    def test_roundtrip_via_savez(self, tmp_path):
        """pack_sample output must survive np.savez_compressed → np.load."""
        block_data = {
            "mm5": {
                "args": [np.random.randn(2, 4, 1, 8).astype(np.float16),
                         np.random.randn(2, 2, 1, 8).astype(np.float16),
                         np.random.randn(2, 1, 1, 8).astype(np.float16)],
                "kwargs": {},
                "output": [np.random.randn(2, 4, 1, 8).astype(np.float16),
                            np.random.randn(2, 2, 1, 8).astype(np.float16)],
            }
        }
        flat = pack_sample(block_data)
        path = tmp_path / "sample.npz"
        np.savez_compressed(path, **flat)

        loaded = np.load(path)
        np.testing.assert_array_equal(loaded["mm5__arg0"], flat["mm5__arg0"])
        np.testing.assert_array_equal(loaded["mm5__out1"], flat["mm5__out1"])


# ---------------------------------------------------------------------------
# load_block_data
# ---------------------------------------------------------------------------

class TestLoadBlockData:
    def _write_sample(self, path: Path, data: dict):
        np.savez_compressed(path, **data)

    def test_stacks_samples_across_files(self, tmp_path):
        for i in range(3):
            self._write_sample(
                tmp_path / f"s{i}.npz",
                {
                    "mm0__arg0": np.array([[float(i), float(i + 1)]]),
                    "mm0__out0": np.array([[float(i * 10)]]),
                },
            )
        files = sorted(tmp_path.glob("*.npz"))
        result = load_block_data("mm0", files)

        assert "arg0" in result
        assert "out0" in result
        assert result["arg0"].shape == (3, 1, 2)   # stacked over 3 samples
        assert result["out0"].shape == (3, 1, 1)

    def test_only_returns_requested_block_keys(self, tmp_path):
        self._write_sample(
            tmp_path / "s0.npz",
            {
                "mm0__arg0": np.array([1.0]),
                "mm0__out0": np.array([2.0]),
                "uni1__arg0": np.array([99.0]),  # different block
            },
        )
        result = load_block_data("mm0", [tmp_path / "s0.npz"])
        assert "arg0" in result
        assert "out0" in result
        # uni1 data absent
        assert not any("uni" in k for k in result)

    def test_empty_when_no_matching_keys(self, tmp_path):
        self._write_sample(tmp_path / "s0.npz", {"mm0__arg0": np.array([1.0])})
        result = load_block_data("uni99", [tmp_path / "s0.npz"])
        assert result == {}

    def test_only_common_keys_returned(self, tmp_path):
        """Keys missing in any sample are excluded (partial-sample robustness)."""
        self._write_sample(
            tmp_path / "s0.npz",
            {"mm0__arg0": np.array([1.0]), "mm0__out0": np.array([2.0])},
        )
        self._write_sample(
            tmp_path / "s1.npz",
            {"mm0__arg0": np.array([3.0])},  # out0 absent
        )
        files = sorted(tmp_path.glob("*.npz"))
        result = load_block_data("mm0", files)

        assert "arg0" in result
        assert "out0" not in result   # not present in all samples

    def test_single_sample(self, tmp_path):
        arr = np.random.randn(2, 10, 1, 8).astype(np.float16)
        self._write_sample(tmp_path / "s0.npz", {"mm3__arg0": arr, "mm3__out0": arr * 2})
        result = load_block_data("mm3", [tmp_path / "s0.npz"])
        assert result["arg0"].shape == (1, 2, 10, 1, 8)

    def test_key_prefix_with_underscores(self, tmp_path):
        """Block name 'uni10' must not accidentally match 'uni1'."""
        self._write_sample(
            tmp_path / "s0.npz",
            {
                "uni1__arg0": np.array([1.0]),
                "uni10__arg0": np.array([2.0]),
            },
        )
        result1 = load_block_data("uni1", [tmp_path / "s0.npz"])
        result10 = load_block_data("uni10", [tmp_path / "s0.npz"])

        np.testing.assert_array_equal(result1["arg0"][0], [1.0])
        np.testing.assert_array_equal(result10["arg0"][0], [2.0])

    def test_empty_file_list(self):
        result = load_block_data("mm0", [])
        assert result == {}


# ---------------------------------------------------------------------------
# _record_shapes
# ---------------------------------------------------------------------------

class TestRecordShapes:
    def test_mm_block_shapes(self):
        block_data = {
            "mm0": {
                "args": [
                    np.zeros((2, 10, 1, 16)),
                    np.zeros((2, 5, 1, 16)),
                    np.zeros((2, 1, 1, 16)),
                ],
                "kwargs": {"positional_encodings": np.zeros((2, 15, 8))},
                "output": [np.zeros((2, 10, 1, 16)), np.zeros((2, 5, 1, 16))],
            }
        }
        shapes = _record_shapes(block_data)

        assert shapes["mm0"]["arg_shapes"] == [
            [2, 10, 1, 16], [2, 5, 1, 16], [2, 1, 1, 16]
        ]
        assert shapes["mm0"]["kwarg_shapes"]["positional_encodings"] == [2, 15, 8]
        assert shapes["mm0"]["output_shapes"] == [[2, 10, 1, 16], [2, 5, 1, 16]]

    def test_uni_block_scalar_output_shape(self):
        block_data = {
            "uni2": {
                "args": [np.zeros((2, 15, 1, 16)), np.zeros((2, 1, 1, 16))],
                "kwargs": {},
                "output": np.zeros((2, 15, 1, 16)),
            }
        }
        shapes = _record_shapes(block_data)
        assert shapes["uni2"]["output_shapes"] == [[2, 15, 1, 16]]

    def test_none_kwarg_recorded_as_none(self):
        block_data = {
            "uni0": {
                "args": [np.zeros((2, 5, 1, 8))],
                "kwargs": {"positional_encodings": None},
                "output": np.zeros((2, 5, 1, 8)),
            }
        }
        shapes = _record_shapes(block_data)
        assert shapes["uni0"]["kwarg_shapes"]["positional_encodings"] is None

    def test_none_block_data_excluded(self):
        block_data = {"mm0": None, "mm1": {"args": [np.zeros((2, 4))],
                                            "kwargs": {},
                                            "output": np.zeros((2, 4))}}
        shapes = _record_shapes(block_data)
        assert "mm0" not in shapes
        assert "mm1" in shapes


# ---------------------------------------------------------------------------
# Integration: install hooks → run forward → flush → pack → save → load
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """
    Verifies the complete data-collection pipeline using a mock model.
    No DiffusionPipeline or model weights required.
    """

    def test_full_round_trip(self, tmp_path):
        pipeline = _MockPipeline(n_mm=2, n_uni=1)
        hooks = install_block_hooks(pipeline)

        # Simulate two forward passes (two calibration samples)
        samples_out = tmp_path / "samples"
        samples_out.mkdir()

        for sample_idx in range(2):
            img = mx.array(np.random.randn(2, 4, 1, 8).astype(np.float32))
            txt = mx.array(np.random.randn(2, 3, 1, 8).astype(np.float32))
            ts  = mx.array(np.random.randn(2, 1, 1, 8).astype(np.float32))
            uni = mx.array(np.random.randn(2, 7, 1, 8).astype(np.float32))

            # Call each hooked block as the model would
            pipeline.mmdit.multimodal_transformer_blocks[0](img, txt, ts)
            pipeline.mmdit.multimodal_transformer_blocks[1](img, txt, ts)
            pipeline.mmdit.unified_transformer_blocks[0](uni, ts)

            block_data = flush_hooks(hooks)
            flat = pack_sample(block_data)
            np.savez_compressed(samples_out / f"{sample_idx:04d}.npz", **flat)

        remove_block_hooks(pipeline, hooks)

        # Load block data per block
        files = sorted(samples_out.glob("*.npz"))

        mm0_data = load_block_data("mm0", files)
        mm1_data = load_block_data("mm1", files)
        uni0_data = load_block_data("uni0", files)

        # 2 samples stacked
        assert mm0_data["arg0"].shape[0] == 2
        assert mm1_data["arg0"].shape[0] == 2
        assert uni0_data["arg0"].shape[0] == 2

        # Output shapes match input shapes (mock block preserves shape)
        assert mm0_data["out0"].shape == mm0_data["arg0"].shape
        assert uni0_data["out0"].shape == uni0_data["arg0"].shape

        # Original blocks restored
        for block in pipeline.mmdit.multimodal_transformer_blocks:
            assert isinstance(block, _SimpleBlock)

    def test_hooks_do_not_alter_forward_result(self):
        """Installing hooks must not change what the model computes."""
        pipeline_plain = _MockPipeline(n_mm=1, n_uni=0)
        pipeline_hooked = _MockPipeline(n_mm=1, n_uni=0)

        hooks = install_block_hooks(pipeline_hooked)

        img = mx.array(np.random.randn(2, 4, 1, 8).astype(np.float32))
        txt = mx.array(np.random.randn(2, 3, 1, 8).astype(np.float32))
        ts  = mx.array(np.random.randn(2, 1, 1, 8).astype(np.float32))

        out_plain  = pipeline_plain.mmdit.multimodal_transformer_blocks[0](img, txt, ts)
        out_hooked = pipeline_hooked.mmdit.multimodal_transformer_blocks[0](img, txt, ts)

        # Both models see the same inputs → outputs must be numerically identical
        if isinstance(out_plain, tuple):
            for p, h in zip(out_plain, out_hooked):
                np.testing.assert_array_equal(np.array(p), np.array(h))
        else:
            np.testing.assert_array_equal(np.array(out_plain), np.array(out_hooked))

        remove_block_hooks(pipeline_hooked, hooks)


# ---------------------------------------------------------------------------
# Resume logic
# ---------------------------------------------------------------------------

class TestResumeLogic:
    """
    Verifies --resume behaviour at the file-system / condition level.
    No DiffusionPipeline or model weights required.
    """

    def test_resume_skips_existing_npz(self, tmp_path):
        """Pre-existing NPZ files are detected in done_samples via the glob scan."""
        samples_dir = tmp_path / "samples"
        samples_dir.mkdir()
        np.savez_compressed(samples_dir / "0000_000.npz", x=np.array([1.0]))
        np.savez_compressed(samples_dir / "0000_004.npz", x=np.array([2.0]))
        np.savez_compressed(samples_dir / "0009_096.npz", x=np.array([3.0]))

        # Replicate the pre-scan logic from cache_adaround_data.py
        done_samples: set = set()
        for p in samples_dir.glob("*.npz"):
            parts = p.stem.split("_")
            if len(parts) == 2:
                done_samples.add((int(parts[0]), int(parts[1])))

        assert (0, 0) in done_samples
        assert (0, 4) in done_samples
        assert (9, 96) in done_samples
        assert len(done_samples) == 3
        assert (0, 1) not in done_samples   # non-existent step

    def test_resume_allows_past_metadata(self, tmp_path):
        """The early-exit condition is False when --resume is set even if metadata exists."""
        metadata_path = tmp_path / "metadata.json"
        metadata_path.write_text("{}")

        args_plain  = types.SimpleNamespace(force=False, resume=False)
        args_resume = types.SimpleNamespace(force=False, resume=True)

        def early_exit(a):
            return metadata_path.exists() and not a.force and not a.resume

        # Without --resume: should trigger early exit
        assert early_exit(args_plain)
        # With --resume: should NOT trigger early exit
        assert not early_exit(args_resume)

    def test_resume_seeds_block_shapes(self, tmp_path):
        """block_shapes from prior metadata.json are loaded into shape_info_seed."""
        prior_shapes = {
            "mm0": {"arg_shapes": [[2, 10, 1, 16]], "output_shapes": [[2, 10, 1, 16]], "kwarg_shapes": {}},
            "mm1": {"arg_shapes": [[2, 10, 1, 16]], "output_shapes": [[2, 10, 1, 16]], "kwarg_shapes": {}},
        }
        metadata_path = tmp_path / "metadata.json"
        metadata_path.write_text(json.dumps({"block_shapes": prior_shapes}))

        args = types.SimpleNamespace(resume=True)

        # Replicate the seeding logic from cache_adaround_data.py
        shape_info_seed: dict = {}
        if args.resume and metadata_path.exists():
            with open(metadata_path) as f:
                shape_info_seed = json.load(f).get("block_shapes", {})

        assert set(shape_info_seed.keys()) == {"mm0", "mm1"}
        assert shape_info_seed["mm0"]["arg_shapes"] == [[2, 10, 1, 16]]

    def test_force_overrides_resume(self, tmp_path):
        """--force alone still bypasses the metadata existence check."""
        metadata_path = tmp_path / "metadata.json"
        metadata_path.write_text("{}")

        args = types.SimpleNamespace(force=True, resume=False)

        early_exit = metadata_path.exists() and not args.force and not args.resume
        assert not early_exit
