"""
Tests for src/generate_calibration_data.py.

Tests are grouped by function and avoid loading any model weights.
  - append_dims / to_d  : pure MLX math
  - load_prompts         : CSV file I/O (tmp_path fixtures)
  - sample_euler_with_calibration : NPZ file writing verified via a
      lightweight duck-type mock of CFGDenoiser
  - main()               : directory/manifest/resume/seed logic tested
      by patching initialize_pipeline and generate_with_calibration
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import mlx.core as mx

from src.generate_calibration_data import (
    append_dims,
    load_prompts,
    main,
    sample_euler_with_calibration,
    to_d,
)


# ---------------------------------------------------------------------------
# Mock denoiser — duck-types CFGDenoiser for sample_euler_with_calibration
# ---------------------------------------------------------------------------

class _MockDenoiser:
    """
    Minimal stand-in for CFGDenoiser.

    sample_euler_with_calibration accesses:
      model.model.sampler.timestep(sigmas)    -> timestep array
      model.model.activation_dtype            -> dtype for .astype()
      model.cache_modulation_params(p, ts)    -> no-op
      model(x, timestep, sigma, **kwargs)     -> denoised x (identity here)
      model.clear_cache()                     -> no-op
    """

    class _Sampler:
        def timestep(self, sigmas):
            return sigmas   # identity: timestep == sigma for testing

    class _Inner:
        activation_dtype = mx.float32
        sampler = None   # assigned in __init__

    def __init__(self):
        self.model = _MockDenoiser._Inner()
        self.model.sampler = _MockDenoiser._Sampler()
        self.call_count = 0

    def cache_modulation_params(self, pooled, timesteps):
        pass

    def __call__(self, x, timestep, sigma, **kwargs):
        self.call_count += 1
        return x   # identity denoising → d = 0 → x unchanged each step

    def clear_cache(self):
        pass


def _make_extra_args(n_tokens: int = 4, dim: int = 8):
    """Return a minimal extra_args dict matching generate_with_calibration."""
    return {
        "conditioning":       mx.zeros((1, n_tokens, dim)),
        "cfg_weight":         7.5,
        "pooled_conditioning": mx.zeros((1, dim)),
    }


# ---------------------------------------------------------------------------
# append_dims
# ---------------------------------------------------------------------------

class TestAppendDims:
    def test_1d_to_3d(self):
        x = mx.array([1.0, 2.0, 3.0])   # shape (3,)
        result = append_dims(x, 3)
        assert result.shape == (3, 1, 1)

    def test_scalar_to_2d(self):
        x = mx.array(2.0)               # shape ()
        result = append_dims(x, 2)
        assert result.shape == (1, 1)

    def test_already_correct_ndim_unchanged(self):
        x = mx.zeros((3, 4, 5))
        result = append_dims(x, 3)
        assert result.shape == (3, 4, 5)

    def test_2d_to_4d(self):
        x = mx.zeros((2, 8))
        result = append_dims(x, 4)
        assert result.shape == (2, 8, 1, 1)

    def test_values_preserved(self):
        x = mx.array([1.0, 2.0])
        result = append_dims(x, 2)
        np.testing.assert_array_equal(np.array(result).ravel(), [1.0, 2.0])

    def test_adds_exactly_one_dim(self):
        x = mx.zeros((5, 3))
        result = append_dims(x, 3)
        assert result.ndim == 3
        assert result.shape == (5, 3, 1)

    def test_no_dims_added_when_at_target(self):
        x = mx.zeros((4,))
        assert append_dims(x, 1).shape == (4,)


# ---------------------------------------------------------------------------
# to_d
# ---------------------------------------------------------------------------

class TestToD:
    def test_simple_1d(self):
        x        = mx.array([2.0, 3.0])
        denoised = mx.array([1.0, 1.0])
        sigma    = mx.array(0.5)
        # d = (x - denoised) / sigma = [1, 2] / 0.5 = [2, 4]
        d = to_d(x, sigma, denoised)
        np.testing.assert_array_almost_equal(np.array(d), [2.0, 4.0])

    def test_zero_residual_gives_zero_derivative(self):
        x = mx.array([3.0, 5.0])
        d = to_d(x, mx.array(1.0), x)   # denoised == x
        np.testing.assert_array_almost_equal(np.array(d), [0.0, 0.0])

    def test_sigma_scales_derivative(self):
        x        = mx.array([1.0])
        denoised = mx.array([0.0])
        # d = 1 / sigma
        assert abs(float(np.array(to_d(x, mx.array(2.0),  denoised))[0]) - 0.5) < 1e-6
        assert abs(float(np.array(to_d(x, mx.array(0.5),  denoised))[0]) - 2.0) < 1e-6
        assert abs(float(np.array(to_d(x, mx.array(1.0),  denoised))[0]) - 1.0) < 1e-6

    def test_sigma_broadcasts_over_2d_tensor(self):
        x        = mx.ones((3, 4)) * 2.0
        denoised = mx.zeros((3, 4))
        sigma    = mx.array(2.0)   # scalar broadcasts to (3, 4) after append_dims
        d = to_d(x, sigma, denoised)
        assert d.shape == (3, 4)
        np.testing.assert_array_almost_equal(np.array(d), np.ones((3, 4)))

    def test_sigma_broadcasts_over_4d_tensor(self):
        x        = mx.ones((2, 8, 8, 4)) * 3.0
        denoised = mx.ones((2, 8, 8, 4))
        sigma    = mx.array(2.0)
        d = to_d(x, sigma, denoised)
        assert d.shape == (2, 8, 8, 4)
        np.testing.assert_array_almost_equal(np.array(d), np.ones((2, 8, 8, 4)))

    def test_negative_residual(self):
        x        = mx.array([0.0])
        denoised = mx.array([2.0])
        sigma    = mx.array(1.0)
        d = to_d(x, sigma, denoised)
        np.testing.assert_array_almost_equal(np.array(d), [-2.0])


# ---------------------------------------------------------------------------
# load_prompts
# ---------------------------------------------------------------------------

class TestLoadPrompts:
    def test_returns_defaults_when_file_missing(self, tmp_path):
        result = load_prompts(tmp_path / "nonexistent.csv", 10)
        assert result == [
            "a photo of a cat",
            "abstract art with vibrant colors",
            "a landscape with mountains",
        ]

    def test_defaults_respect_max_count(self, tmp_path):
        result = load_prompts(tmp_path / "nonexistent.csv", 2)
        assert len(result) == 2

    def test_defaults_max_count_1(self, tmp_path):
        result = load_prompts(tmp_path / "nonexistent.csv", 1)
        assert result == ["a photo of a cat"]

    def test_loads_prompts_from_csv(self, tmp_path):
        csv_path = tmp_path / "prompts.csv"
        csv_path.write_text("prompt\nfirst prompt\nsecond prompt\nthird prompt\n")
        result = load_prompts(csv_path, 10)
        assert result == ["first prompt", "second prompt", "third prompt"]

    def test_respects_max_count(self, tmp_path):
        csv_path = tmp_path / "prompts.csv"
        csv_path.write_text("prompt\n" + "\n".join(f"prompt {i}" for i in range(10)))
        result = load_prompts(csv_path, 3)
        assert len(result) == 3
        assert result == ["prompt 0", "prompt 1", "prompt 2"]

    def test_skips_empty_rows(self, tmp_path):
        csv_path = tmp_path / "prompts.csv"
        csv_path.write_text("prompt\ngood prompt\n\n  \nanother prompt\n")
        result = load_prompts(csv_path, 10)
        assert len(result) == 2
        assert "good prompt" in result
        assert "another prompt" in result

    def test_strips_whitespace_from_prompts(self, tmp_path):
        csv_path = tmp_path / "prompts.csv"
        csv_path.write_text("prompt\n  spaces around  \nnormal\n")
        result = load_prompts(csv_path, 10)
        assert result[0] == "spaces around"

    def test_max_count_exceeds_available(self, tmp_path):
        csv_path = tmp_path / "prompts.csv"
        csv_path.write_text("prompt\nonly one\n")
        result = load_prompts(csv_path, 100)
        assert result == ["only one"]

    def test_empty_csv_returns_empty_list(self, tmp_path):
        csv_path = tmp_path / "prompts.csv"
        csv_path.write_text("prompt\n")   # header only
        result = load_prompts(csv_path, 10)
        assert result == []

    def test_max_count_zero_returns_empty(self, tmp_path):
        result = load_prompts(tmp_path / "nonexistent.csv", 0)
        assert result == []


# ---------------------------------------------------------------------------
# sample_euler_with_calibration
# ---------------------------------------------------------------------------

class TestSampleEulerWithCalibration:
    """
    Uses _MockDenoiser (identity denoising) to verify file-writing behaviour
    without loading any model weights.
    With identity denoising d = 0, so x is unchanged across all steps.
    """

    def _run(self, tmp_path, n_sigmas=4, img_idx=0, latent_shape=(1, 8, 8, 4)):
        """Helper: run the sampler and return (result_x, npz_files)."""
        model = _MockDenoiser()
        x = mx.zeros(latent_shape)
        sigmas = mx.array([1.0 - i * (1.0 / n_sigmas) for i in range(n_sigmas)])
        extra_args = _make_extra_args()

        result_x, iter_time = sample_euler_with_calibration(
            model, x, sigmas, extra_args,
            img_idx=img_idx, samples_dir=tmp_path
        )
        npz_files = sorted(tmp_path.glob("*.npz"))
        return result_x, iter_time, npz_files, model

    def test_writes_correct_number_of_files(self, tmp_path):
        # n_sigmas = 4 → 3 loop steps + 1 final = 4 files
        _, _, files, _ = self._run(tmp_path, n_sigmas=4)
        assert len(files) == 4

    def test_two_sigmas_writes_two_files(self, tmp_path):
        # n_sigmas = 2 → 1 loop step + 1 final = 2 files
        _, _, files, _ = self._run(tmp_path, n_sigmas=2)
        assert len(files) == 2

    def test_file_naming_uses_img_idx_and_step(self, tmp_path):
        _, _, files, _ = self._run(tmp_path, n_sigmas=3, img_idx=7)
        names = {f.name for f in files}
        assert "0007_000.npz" in names
        assert "0007_001.npz" in names
        assert "0007_002.npz" in names

    def test_file_naming_zero_pads_img_idx(self, tmp_path):
        _, _, files, _ = self._run(tmp_path, n_sigmas=2, img_idx=42)
        assert any(f.name.startswith("0042_") for f in files)

    def test_each_file_has_required_keys(self, tmp_path):
        _, _, files, _ = self._run(tmp_path, n_sigmas=3)
        required = {"x", "timestep", "sigma", "step_index", "image_id", "is_final"}
        for f in files:
            npz = np.load(f)
            assert required.issubset(set(npz.files)), f"{f.name} missing keys"

    def test_is_final_false_for_all_loop_steps(self, tmp_path):
        _, _, files, _ = self._run(tmp_path, n_sigmas=4, img_idx=0)
        # Files 000, 001, 002 are loop steps; 003 is final
        loop_files = [f for f in files if not f.name.endswith("_003.npz")]
        for f in loop_files:
            assert not bool(np.load(f)["is_final"]), f"{f.name} should have is_final=False"

    def test_is_final_true_for_last_file(self, tmp_path):
        _, _, files, _ = self._run(tmp_path, n_sigmas=4, img_idx=0)
        last = max(files, key=lambda f: f.name)
        assert bool(np.load(last)["is_final"])

    def test_step_index_matches_position(self, tmp_path):
        _, _, files, _ = self._run(tmp_path, n_sigmas=4, img_idx=0)
        for f in files:
            npz = np.load(f)
            # Filename is {img:04d}_{step:03d}.npz
            expected_step = int(f.stem.split("_")[1])
            assert int(npz["step_index"]) == expected_step

    def test_image_id_stored_correctly(self, tmp_path):
        _, _, files, _ = self._run(tmp_path, n_sigmas=3, img_idx=5)
        for f in files:
            assert int(np.load(f)["image_id"]) == 5

    def test_x_shape_matches_input(self, tmp_path):
        latent_shape = (2, 4, 4, 8)
        _, _, files, _ = self._run(tmp_path, n_sigmas=3, latent_shape=latent_shape)
        for f in files:
            assert np.load(f)["x"].shape == latent_shape

    def test_returns_mx_array(self, tmp_path):
        result_x, _, _, _ = self._run(tmp_path, n_sigmas=3)
        assert isinstance(result_x, mx.array)

    def test_returns_positive_iter_time(self, tmp_path):
        _, iter_time, _, _ = self._run(tmp_path, n_sigmas=3)
        assert iter_time >= 0.0

    def test_model_called_once_per_step(self, tmp_path):
        # n_sigmas=4 → 3 denoising calls
        _, _, _, model = self._run(tmp_path, n_sigmas=4)
        assert model.call_count == 3

    def test_pooled_conditioning_popped_from_extra_args(self, tmp_path):
        """extra_args must not contain pooled_conditioning when model is called."""
        model = _MockDenoiser()
        x = mx.zeros((1, 4, 4, 4))
        sigmas = mx.array([1.0, 0.5, 0.0])
        extra_args = _make_extra_args()

        received_kwargs = []

        original_call = model.__call__
        def capturing_call(x, timestep, sigma, **kwargs):
            received_kwargs.append(set(kwargs.keys()))
            return original_call(x, timestep, sigma, **kwargs)
        model.__call__ = capturing_call

        sample_euler_with_calibration(
            model, x, sigmas, extra_args, img_idx=0, samples_dir=tmp_path
        )

        for kw in received_kwargs:
            assert "pooled_conditioning" not in kw
            assert "conditioning" in kw
            assert "cfg_weight" in kw

    def test_works_with_multiple_images(self, tmp_path):
        for img_idx in range(3):
            model = _MockDenoiser()
            x = mx.zeros((1, 4, 4, 4))
            sigmas = mx.array([1.0, 0.5, 0.0])
            sample_euler_with_calibration(
                model, x, sigmas, _make_extra_args(),
                img_idx=img_idx, samples_dir=tmp_path
            )
        # 3 sigmas → 2 loop steps + 1 final = 3 files per image; 3 images = 9 files
        files = list(tmp_path.glob("*.npz"))
        assert len(files) == 9
        prefixes = {f.name[:4] for f in files}
        assert prefixes == {"0000", "0001", "0002"}

    def test_step_pbar_updated(self, tmp_path):
        pbar = MagicMock()
        model = _MockDenoiser()
        x = mx.zeros((1, 4, 4, 4))
        sigmas = mx.array([1.0, 0.5, 0.0])   # 2 loop steps
        sample_euler_with_calibration(
            model, x, sigmas, _make_extra_args(),
            img_idx=0, samples_dir=tmp_path, step_pbar=pbar
        )
        assert pbar.update.call_count == 2   # called once per loop step


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def _write_prompts_csv(path: Path, prompts):
    path.write_text("prompt\n" + "\n".join(prompts) + "\n")


def _mock_pipeline():
    return MagicMock()


def _mock_generate_fn(image=None, latent=None):
    """Return a generate_with_calibration mock that yields fixed (image, latent)."""
    if image is None:
        image = Image.new("RGB", (8, 8))
    if latent is None:
        latent = mx.zeros((1, 4, 4, 4))
    return MagicMock(return_value=(image, latent))


class TestMain:
    def _run_main(self, monkeypatch, argv, mock_gen=None, mock_pipeline=None):
        if mock_gen is None:
            mock_gen = _mock_generate_fn()
        if mock_pipeline is None:
            mock_pipeline = _mock_pipeline()
        monkeypatch.setattr("sys.argv", argv)
        with patch("src.generate_calibration_data.initialize_pipeline",
                   return_value=mock_pipeline), \
             patch("src.generate_calibration_data.generate_with_calibration",
                   mock_gen):
            main()

    # --- directory creation ---

    def test_creates_samples_directory(self, tmp_path, monkeypatch):
        _write_prompts_csv(tmp_path / "p.csv", ["test prompt"])
        self._run_main(monkeypatch, [
            "prog", "--calib-dir", str(tmp_path / "out"),
            "--prompt-csv", str(tmp_path / "p.csv"),
            "--num-images", "1", "--num-steps", "2",
        ])
        assert (tmp_path / "out" / "samples").is_dir()

    def test_creates_images_directory(self, tmp_path, monkeypatch):
        _write_prompts_csv(tmp_path / "p.csv", ["test prompt"])
        self._run_main(monkeypatch, [
            "prog", "--calib-dir", str(tmp_path / "out"),
            "--prompt-csv", str(tmp_path / "p.csv"),
            "--num-images", "1", "--num-steps", "2",
        ])
        assert (tmp_path / "out" / "images").is_dir()

    def test_creates_latents_directory(self, tmp_path, monkeypatch):
        _write_prompts_csv(tmp_path / "p.csv", ["test prompt"])
        self._run_main(monkeypatch, [
            "prog", "--calib-dir", str(tmp_path / "out"),
            "--prompt-csv", str(tmp_path / "p.csv"),
            "--num-images", "1", "--num-steps", "2",
        ])
        assert (tmp_path / "out" / "latents").is_dir()

    # --- manifest contents ---

    def test_manifest_written_after_run(self, tmp_path, monkeypatch):
        _write_prompts_csv(tmp_path / "p.csv", ["cat", "dog"])
        self._run_main(monkeypatch, [
            "prog", "--calib-dir", str(tmp_path / "out"),
            "--prompt-csv", str(tmp_path / "p.csv"),
            "--num-images", "2", "--num-steps", "3",
        ])
        assert (tmp_path / "out" / "manifest.json").exists()

    def test_manifest_n_completed(self, tmp_path, monkeypatch):
        _write_prompts_csv(tmp_path / "p.csv", ["cat", "dog"])
        self._run_main(monkeypatch, [
            "prog", "--calib-dir", str(tmp_path / "out"),
            "--prompt-csv", str(tmp_path / "p.csv"),
            "--num-images", "2", "--num-steps", "3",
        ])
        manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
        assert manifest["n_completed"] == 2

    def test_manifest_records_num_steps(self, tmp_path, monkeypatch):
        _write_prompts_csv(tmp_path / "p.csv", ["prompt"])
        self._run_main(monkeypatch, [
            "prog", "--calib-dir", str(tmp_path / "out"),
            "--prompt-csv", str(tmp_path / "p.csv"),
            "--num-images", "1", "--num-steps", "7",
        ])
        manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
        assert manifest["num_steps"] == 7

    def test_manifest_records_cfg_scale(self, tmp_path, monkeypatch):
        _write_prompts_csv(tmp_path / "p.csv", ["prompt"])
        self._run_main(monkeypatch, [
            "prog", "--calib-dir", str(tmp_path / "out"),
            "--prompt-csv", str(tmp_path / "p.csv"),
            "--num-images", "1", "--num-steps", "2",
            "--cfg-weight", "5.0",
        ])
        manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
        assert manifest["cfg_scale"] == 5.0

    def test_manifest_images_list_length(self, tmp_path, monkeypatch):
        _write_prompts_csv(tmp_path / "p.csv", ["a", "b", "c"])
        self._run_main(monkeypatch, [
            "prog", "--calib-dir", str(tmp_path / "out"),
            "--prompt-csv", str(tmp_path / "p.csv"),
            "--num-images", "3", "--num-steps", "2",
        ])
        manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
        assert len(manifest["images"]) == 3

    def test_manifest_images_contain_prompt(self, tmp_path, monkeypatch):
        _write_prompts_csv(tmp_path / "p.csv", ["first prompt", "second prompt"])
        self._run_main(monkeypatch, [
            "prog", "--calib-dir", str(tmp_path / "out"),
            "--prompt-csv", str(tmp_path / "p.csv"),
            "--num-images", "2", "--num-steps", "2",
        ])
        manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
        prompts_in_manifest = [img["prompt"] for img in manifest["images"]]
        assert "first prompt" in prompts_in_manifest
        assert "second prompt" in prompts_in_manifest

    def test_manifest_images_have_required_fields(self, tmp_path, monkeypatch):
        _write_prompts_csv(tmp_path / "p.csv", ["test"])
        self._run_main(monkeypatch, [
            "prog", "--calib-dir", str(tmp_path / "out"),
            "--prompt-csv", str(tmp_path / "p.csv"),
            "--num-images", "1", "--num-steps", "2",
        ])
        manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
        img = manifest["images"][0]
        for field in ("image_id", "prompt", "seed", "cfg_weight",
                      "num_steps", "filename", "latent_filename"):
            assert field in img, f"Missing field: {field}"

    # --- seed assignment ---

    def test_seed_increments_per_image(self, tmp_path, monkeypatch):
        _write_prompts_csv(tmp_path / "p.csv", ["a", "b", "c"])
        seen_seeds = []

        def capturing_gen(pipeline, prompt, seed, **kwargs):
            seen_seeds.append(seed)
            return Image.new("RGB", (8, 8)), mx.zeros((1, 4, 4, 4))

        self._run_main(monkeypatch, [
            "prog", "--calib-dir", str(tmp_path / "out"),
            "--prompt-csv", str(tmp_path / "p.csv"),
            "--num-images", "3", "--num-steps", "2",
            "--seed", "10",
        ], mock_gen=capturing_gen)

        assert seen_seeds == [10, 11, 12]

    def test_seed_base_respects_flag(self, tmp_path, monkeypatch):
        _write_prompts_csv(tmp_path / "p.csv", ["a"])
        seen_seeds = []

        def capturing_gen(pipeline, prompt, seed, **kwargs):
            seen_seeds.append(seed)
            return Image.new("RGB", (8, 8)), mx.zeros((1, 4, 4, 4))

        self._run_main(monkeypatch, [
            "prog", "--calib-dir", str(tmp_path / "out"),
            "--prompt-csv", str(tmp_path / "p.csv"),
            "--num-images", "1", "--num-steps", "2",
            "--seed", "99",
        ], mock_gen=capturing_gen)

        assert seen_seeds == [99]

    def test_manifest_seed_base_recorded(self, tmp_path, monkeypatch):
        _write_prompts_csv(tmp_path / "p.csv", ["prompt"])
        self._run_main(monkeypatch, [
            "prog", "--calib-dir", str(tmp_path / "out"),
            "--prompt-csv", str(tmp_path / "p.csv"),
            "--num-images", "1", "--num-steps", "2",
            "--seed", "77",
        ])
        manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
        assert manifest["seed_base"] == 77

    # --- output files ---

    def test_image_png_saved_per_image(self, tmp_path, monkeypatch):
        _write_prompts_csv(tmp_path / "p.csv", ["a", "b"])
        self._run_main(monkeypatch, [
            "prog", "--calib-dir", str(tmp_path / "out"),
            "--prompt-csv", str(tmp_path / "p.csv"),
            "--num-images", "2", "--num-steps", "2",
        ])
        assert (tmp_path / "out" / "images" / "0000.png").exists()
        assert (tmp_path / "out" / "images" / "0001.png").exists()

    def test_latent_npy_saved_per_image(self, tmp_path, monkeypatch):
        _write_prompts_csv(tmp_path / "p.csv", ["a", "b"])
        self._run_main(monkeypatch, [
            "prog", "--calib-dir", str(tmp_path / "out"),
            "--prompt-csv", str(tmp_path / "p.csv"),
            "--num-images", "2", "--num-steps", "2",
        ])
        assert (tmp_path / "out" / "latents" / "0000.npy").exists()
        assert (tmp_path / "out" / "latents" / "0001.npy").exists()

    def test_initialize_pipeline_called_per_image(self, tmp_path, monkeypatch):
        _write_prompts_csv(tmp_path / "p.csv", ["a", "b", "c"])
        monkeypatch.setattr("sys.argv", [
            "prog", "--calib-dir", str(tmp_path / "out"),
            "--prompt-csv", str(tmp_path / "p.csv"),
            "--num-images", "3", "--num-steps", "2",
        ])
        with patch("src.generate_calibration_data.initialize_pipeline",
                   return_value=MagicMock()) as mock_init, \
             patch("src.generate_calibration_data.generate_with_calibration",
                   return_value=(Image.new("RGB", (8, 8)), mx.zeros((1, 4, 4, 4)))):
            main()
        assert mock_init.call_count == 3

    # --- resume mode ---

    def _setup_partial_run(self, tmp_path, done_count: int, total_prompts: int):
        """Create directories and manifest as if `done_count` images are complete."""
        out = tmp_path / "out"
        (out / "samples").mkdir(parents=True)
        (out / "images").mkdir()
        (out / "latents").mkdir()

        prompts = [f"prompt {i}" for i in range(total_prompts)]
        _write_prompts_csv(tmp_path / "p.csv", prompts)

        images = []
        for i in range(done_count):
            img_path = out / "images" / f"{i:04d}.png"
            Image.new("RGB", (8, 8)).save(img_path)
            images.append({
                "image_id": i, "prompt": prompts[i], "seed": 42 + i,
                "cfg_weight": 7.5, "num_steps": 2,
                "filename": f"{i:04d}.png", "latent_filename": f"{i:04d}.npy",
            })

        manifest = {
            "n_completed": done_count, "num_steps": 2, "cfg_scale": 7.5,
            "latent_size": [64, 64], "prompt_path": str(tmp_path / "p.csv"),
            "num_images": total_prompts, "model_version": "...",
            "use_t5": True, "seed_base": 42,
            "images_saved": True, "latents_saved": True, "images": images,
        }
        (out / "manifest.json").write_text(json.dumps(manifest))
        return out

    def test_resume_skips_completed_images(self, tmp_path, monkeypatch):
        out = self._setup_partial_run(tmp_path, done_count=2, total_prompts=4)
        call_count = [0]

        def counting_gen(pipeline, prompt, seed, **kwargs):
            call_count[0] += 1
            return Image.new("RGB", (8, 8)), mx.zeros((1, 4, 4, 4))

        monkeypatch.setattr("sys.argv", [
            "prog", "--calib-dir", str(out),
            "--prompt-csv", str(tmp_path / "p.csv"),
            "--num-images", "4", "--num-steps", "2", "--resume",
        ])
        with patch("src.generate_calibration_data.initialize_pipeline",
                   return_value=MagicMock()), \
             patch("src.generate_calibration_data.generate_with_calibration",
                   side_effect=counting_gen):
            main()

        assert call_count[0] == 2   # only 2 new images generated

    def test_resume_final_manifest_has_all_images(self, tmp_path, monkeypatch):
        out = self._setup_partial_run(tmp_path, done_count=1, total_prompts=3)

        monkeypatch.setattr("sys.argv", [
            "prog", "--calib-dir", str(out),
            "--prompt-csv", str(tmp_path / "p.csv"),
            "--num-images", "3", "--num-steps", "2", "--resume",
        ])
        with patch("src.generate_calibration_data.initialize_pipeline",
                   return_value=MagicMock()), \
             patch("src.generate_calibration_data.generate_with_calibration",
                   return_value=(Image.new("RGB", (8, 8)), mx.zeros((1, 4, 4, 4)))):
            main()

        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["n_completed"] == 3
        assert len(manifest["images"]) == 3

    def test_no_resume_reruns_all_images(self, tmp_path, monkeypatch):
        """Without --resume, all images are generated even if PNGs exist."""
        out = self._setup_partial_run(tmp_path, done_count=2, total_prompts=2)
        call_count = [0]

        def counting_gen(pipeline, prompt, seed, **kwargs):
            call_count[0] += 1
            return Image.new("RGB", (8, 8)), mx.zeros((1, 4, 4, 4))

        monkeypatch.setattr("sys.argv", [
            "prog", "--calib-dir", str(out),
            "--prompt-csv", str(tmp_path / "p.csv"),
            "--num-images", "2", "--num-steps", "2",
            # no --resume flag
        ])
        with patch("src.generate_calibration_data.initialize_pipeline",
                   return_value=MagicMock()), \
             patch("src.generate_calibration_data.generate_with_calibration",
                   side_effect=counting_gen):
            main()

        assert call_count[0] == 2   # all 2 images re-generated
