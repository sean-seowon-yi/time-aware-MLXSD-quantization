# Calibration Data Collection - Checklist

Use this checklist to ensure you complete all steps correctly.

---

## ☐ Pre-Flight Checks

- [ ] Have at least 5 GB free disk space (for small test)
- [ ] Have at least 50 GB free disk space (for full 1000 images)
- [ ] Have `all_prompts.csv` file with at least 1000 prompts
- [ ] DiffusionKit is installed and working
- [ ] Can run: `python -c "from diffusionkit.mlx import DiffusionPipeline; print('OK')"`

---

## ☐ Phase 1: Generate Calibration Data

### Step 1.1: Test with Small Dataset (6 minutes)

```bash
python -m src.generate_calibration_data \
    --num-images 10 \
    --num-steps 50 \
    --calib-dir test_calibration
```

- [ ] Script completed without errors
- [ ] Images look correct (not noise)
- [ ] Check: `ls test_calibration/images/*.png`
- [ ] Check: `python -m src.verify_calibration --calib-dir test_calibration`

### Step 1.2: Generate Full Dataset (11 hours)

```bash
python -m src.generate_calibration_data \
    --num-images 1000 \
    --num-steps 50 \
    --calib-dir calibration_data \
    --prompt-csv all_prompts.csv
```

**Before starting**:
- [ ] Ensure machine won't sleep (adjust power settings)
- [ ] Consider running in `screen` or `tmux` session
- [ ] Start before going to bed or leaving for the day

**If interrupted**:
```bash
# Resume from where it left off
python -m src.generate_calibration_data \
    --num-images 1000 \
    --num-steps 50 \
    --resume
```

### Step 1.3: Verify Calibration Data

```bash
python -m src.verify_calibration --calib-dir calibration_data
```

- [ ] All samples present: `51,000/51,000`
- [ ] All latents present: `1,000/1,000`
- [ ] All images present: `1,000/1,000`
- [ ] Sample format valid
- [ ] No suspicious values

**If issues found**:
- [ ] Check error messages
- [ ] If manifest corrupted: `python -m src.rebuild_manifest --calib-dir calibration_data`
- [ ] If samples missing: Run generation with `--resume`

---

## ☐ Phase 2: Collect Activation Statistics

### Step 2.1: Collect Activations (30 minutes)

```bash
python -m src.collect_layer_activations \
    --calib-dir calibration_data \
    --num-images 100
```

- [ ] Script completed without errors
- [ ] Check: `ls calibration_data/activations/layer_statistics.json`
- [ ] File size > 10 KB

### Step 2.2: Verify Activation Statistics

```bash
python -m src.verify_calibration --calib-dir calibration_data
```

- [ ] ✓ Activation statistics: PASS
- [ ] Statistics for 20+ layers
- [ ] No suspicious values (NaN, inf, zero range)

**If issues found**:
```bash
# Force regenerate
python -m src.collect_layer_activations \
    --calib-dir calibration_data \
    --num-images 100 \
    --force
```

---

## ☐ Phase 3: Analyze Statistics

### Step 3.1: View Analysis

```bash
python -m src.analyze_activations \
    --stats calibration_data/activations/layer_statistics.json
```

**Check output**:
- [ ] Shows layer statistics summary
- [ ] Shows bit-width distribution
- [ ] Identifies sensitive layers
- [ ] Lists layers for aggressive quantization

### Step 3.2: Export Quantization Config

```bash
python -m src.analyze_activations \
    --stats calibration_data/activations/layer_statistics.json \
    --export-config quantization_config.json
```

- [ ] File created: `quantization_config.json`
- [ ] Contains `default_config`
- [ ] Contains `layer_configs` with per-layer settings

---

## ☐ Phase 4: Final Verification

### Step 4.1: Complete Verification

```bash
python -m src.verify_calibration --calib-dir calibration_data
```

**Expected output**:
```
✓ Calibration samples: PASS
✓ Activation statistics: PASS

✓ All checks passed! Data ready for quantization.
```

### Step 4.2: Storage Check

- [ ] Total storage: ~8 GB for 1000 images + activations
- [ ] Check: `du -sh calibration_data`

---

## ☐ What You Should Have Now

### Files

- [ ] `calibration_data/samples/` (51,000 .npz files)
- [ ] `calibration_data/latents/` (1,000 .npy files)
- [ ] `calibration_data/images/` (1,000 .png files)
- [ ] `calibration_data/manifest.json`
- [ ] `calibration_data/activations/layer_statistics.json`
- [ ] `calibration_data/activations/collection_metadata.json`
- [ ] `quantization_config.json`

### Verification

- [ ] Can load a calibration sample:
  ```python
  import numpy as np
  data = np.load('calibration_data/samples/0000_000.npz')
  print(data['x'].shape)  # Should be (1, 16, 64, 64)
  ```

- [ ] Can load statistics:
  ```python
  import json
  with open('calibration_data/activations/layer_statistics.json') as f:
      stats = json.load(f)
  print(f"Layers: {len(stats)}")  # Should be 20+
  ```

- [ ] Can load quantization config:
  ```python
  import json
  with open('quantization_config.json') as f:
      config = json.load(f)
  print(config.keys())  # Should have 'default_config', 'layer_configs'
  ```

---

## ☐ Next Steps

- [ ] **Implement TaQ-DiT quantization** using `quantization_config.json`
- [ ] **Test quantized model** on sample images
- [ ] **Evaluate with FID** using `calibration_data/images/`
- [ ] **Iterate** on quantization strategy based on results

---

## ☐ Troubleshooting

If anything goes wrong, check:

1. **Error logs**:
   - Look at terminal output for error messages
   - Common issues documented in `README_CALIBRATION.md`

2. **Verification**:
   ```bash
   python -m src.verify_calibration --calib-dir calibration_data
   ```

3. **Rebuild if needed**:
   ```bash
   # Rebuild manifest
   python -m src.rebuild_manifest --calib-dir calibration_data
   
   # Regenerate activations
   python -m src.collect_layer_activations \
       --calib-dir calibration_data \
       --num-images 100 \
       --force
   ```

4. **Check disk space**:
   ```bash
   df -h
   ```

5. **Check memory**:
   ```bash
   htop  # or top
   ```

---

## ☐ Success Criteria

You're ready to proceed when:

- [x] ✓ 1000 images generated successfully
- [x] ✓ All samples, latents, and images present
- [x] ✓ Activation statistics collected
- [x] ✓ Quantization config exported
- [x] ✓ All verification checks pass
- [x] ✓ Total storage ~8 GB
- [x] ✓ No error messages in verification

**If all checks pass: Congratulations! You're ready to implement TaQ-DiT quantization!** 🎉

---

## Quick Reference

```bash
# Generate test (6 min)
python -m src.generate_calibration_data --num-images 10 --num-steps 50 --calib-dir test

# Generate full (11 hours)
python -m src.generate_calibration_data --num-images 1000 --num-steps 50

# Collect activations (30 min)
python -m src.collect_layer_activations --calib-dir calibration_data --num-images 100

# Analyze
python -m src.analyze_activations --stats calibration_data/activations/layer_statistics.json --export-config quantization_config.json

# Verify
python -m src.verify_calibration --calib-dir calibration_data
```
