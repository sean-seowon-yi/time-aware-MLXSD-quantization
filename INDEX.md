# Calibration Data Collection - Documentation Index

Complete system for generating calibration data and collecting activation statistics for diffusion transformer quantization.

---

## 📚 Documentation

### Quick Start
- **[CALIBRATION_CHECKLIST.md](CALIBRATION_CHECKLIST.md)** - Step-by-step checklist with verification ⭐ **Start here!**
- **[QUICKSTART_CALIBRATION.md](QUICKSTART_CALIBRATION.md)** - Command reference and quick examples

### Complete Guide
- **[README_CALIBRATION.md](README_CALIBRATION.md)** - Comprehensive documentation with all details
- **[CALIBRATION_SYSTEM_SUMMARY.md](CALIBRATION_SYSTEM_SUMMARY.md)** - System overview and design decisions

### Lessons Learned
- **[LESSONS_READ_SOURCE_FIRST.md](LESSONS_READ_SOURCE_FIRST.md)** - Why reading source code first saves time
- **[TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md)** - All issues encountered and fixes applied

---

## 🛠️ Scripts

### Core Scripts
| Script | Purpose | Time | Required |
|--------|---------|------|----------|
| `generate_calibration_data.py` | Generate 1000 images with per-step latents | ~11h | Yes |
| `collect_layer_activations.py` | Collect activation stats from 100 images | ~30m | Yes |
| `analyze_activations.py` | Analyze stats and export config | <1m | Yes |
| `verify_calibration.py` | Check data integrity | <1m | Recommended |

### Helper Scripts
| Script | Purpose |
|--------|---------|
| `rebuild_manifest.py` | Recover corrupted manifest |
| `decode_latents.py` | Decode latents if images failed |

---

## 📋 Quick Commands

### Complete Workflow
```bash
# 1. Generate calibration data (11 hours)
python -m src.generate_calibration_data --num-images 1000 --num-steps 50

# 2. Collect activations (30 minutes)
python -m src.collect_layer_activations --calib-dir calibration_data --num-images 100

# 3. Analyze and export config (<1 minute)
python -m src.analyze_activations \
    --stats calibration_data/activations/layer_statistics.json \
    --export-config quantization_config.json

# 4. Verify everything (<1 minute)
python -m src.verify_calibration --calib-dir calibration_data
```

### Quick Test (10 images, 6 minutes)
```bash
python -m src.generate_calibration_data --num-images 10 --num-steps 50 --calib-dir test
python -m src.collect_layer_activations --calib-dir test --num-images 10
python -m src.analyze_activations --stats test/activations/layer_statistics.json
```

---

## 📊 What You Get

### Output Files
```
calibration_data/
├── samples/              # 51,000 .npz files (per-step latents)
├── latents/              # 1,000 .npy files (final latents)
├── images/               # 1,000 .png files (for FID)
├── activations/
│   ├── layer_statistics.json       # Per-layer stats ⭐
│   └── collection_metadata.json    # Collection info
└── manifest.json                    # All prompts and metadata

quantization_config.json              # Ready-to-use config ⭐
```

### Key Outputs

1. **`layer_statistics.json`** - Per-layer activation statistics:
   ```json
   {
     "mm_block_0": {
       "min": -2.456, "max": 3.123,
       "mean": 0.234, "std": 0.891,
       "percentiles": {"p01": -1.234, "p99": 2.456}
     }
   }
   ```

2. **`quantization_config.json`** - Quantization recommendations:
   ```json
   {
     "layer_configs": {
       "mm_block_0": {
         "weight_bits": 6,
         "activation_bits": 8,
         "clip_percentile": [1, 99]
       }
     }
   }
   ```

---

## ⏱️ Time & Storage

### Time (M4 Max)
- Generate 1000 images: ~11 hours
- Collect activations (100 images): ~30 minutes
- Analyze: <1 minute
- **Total: ~11.5 hours**

### Storage
- 1000 images (no activations): 3.3 GB
- 1000 images + activations (100 subset): 8 GB
- 10 images (test): 200 MB

---

## ✅ Success Criteria

You're ready for quantization when:

- ✓ 1000 images generated
- ✓ All samples/latents/images present
- ✓ Activation statistics collected
- ✓ Quantization config exported
- ✓ Verification passes
- ✓ Total storage ~8 GB

---

## 🔧 Troubleshooting

### Common Issues

**Model state corruption**
- ✓ Already handled by reloading pipeline per image

**Out of memory**
- Reduce images: `--num-images 50`

**Missing samples**
- Resume: `--resume` flag
- Rebuild: `python -m src.rebuild_manifest`

**Wrong activation statistics**
- Regenerate: `--force` flag

See [TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md) for complete list.

---

## 🚀 Next Steps

After collecting calibration data:

1. **Implement TaQ-DiT quantization**
   - Use `quantization_config.json` for per-layer settings
   - Implement timestep-aware bit-widths

2. **Evaluate quantized model**
   - Use `calibration_data/images/` for FID
   - Compare quality vs original

3. **Iterate**
   - Adjust bit-widths based on results
   - Refine quantization strategy

---

## 📖 Further Reading

### Design Decisions
- Why 100 images? → Statistics stabilize with 50-100 samples
- Why 17 timesteps? → Captures all distribution phases efficiently
- Why subsample? → 99% storage reduction vs full collection
- Why reload pipeline? → Avoids model state corruption

### Implementation Details
- Euler sampling with proper `to_d` and `append_dims`
- Timestep conversion: sigmas → timesteps
- Activation collection via monkey-patching
- Storage optimization: prompts instead of conditioning

See [README_CALIBRATION.md](README_CALIBRATION.md) for full explanations.

---

## 🎓 Lessons Learned

Key insight: **Read the source code first!**

Reading DiffusionKit's source before implementing would have saved 5+ hours of debugging.

**What we learned**:
- Proper Euler formula (with `to_d`)
- Broadcasting with `append_dims`
- Timestep conversion requirements
- Modulation caching strategy

See [LESSONS_READ_SOURCE_FIRST.md](LESSONS_READ_SOURCE_FIRST.md) for details.

---

## 📞 Support

If you encounter issues:

1. Check [TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md)
2. Run verification: `python -m src.verify_calibration`
3. Check error messages carefully
4. Review relevant documentation section

---

## 📦 File Structure

```
mlxproject/
├── src/
│   ├── generate_calibration_data.py     # Main generation
│   ├── collect_layer_activations.py     # Activation collection
│   ├── analyze_activations.py           # Analysis
│   ├── verify_calibration.py            # Verification
│   └── rebuild_manifest.py              # Recovery
│
├── calibration_data/                    # Output directory
│   ├── samples/
│   ├── latents/
│   ├── images/
│   ├── activations/
│   └── manifest.json
│
├── CALIBRATION_CHECKLIST.md             # ⭐ Start here
├── QUICKSTART_CALIBRATION.md            # Quick commands
├── README_CALIBRATION.md                # Full documentation
├── CALIBRATION_SYSTEM_SUMMARY.md        # System overview
├── LESSONS_READ_SOURCE_FIRST.md         # Lessons learned
├── TROUBLESHOOTING_GUIDE.md             # Issue fixes
└── INDEX.md                             # This file
```

---

## 🎯 Quick Links

**Getting Started**:
1. Read: [CALIBRATION_CHECKLIST.md](CALIBRATION_CHECKLIST.md)
2. Run: Quick test (10 images)
3. Verify: `python -m src.verify_calibration`
4. Run: Full generation (1000 images)
5. Collect: Activation statistics
6. Analyze: Export quantization config

**Documentation by Role**:
- **First-time user**: [CALIBRATION_CHECKLIST.md](CALIBRATION_CHECKLIST.md)
- **Quick reference**: [QUICKSTART_CALIBRATION.md](QUICKSTART_CALIBRATION.md)
- **Understanding design**: [CALIBRATION_SYSTEM_SUMMARY.md](CALIBRATION_SYSTEM_SUMMARY.md)
- **Implementing quantization**: [README_CALIBRATION.md](README_CALIBRATION.md)
- **Debugging issues**: [TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md)
- **Learning best practices**: [LESSONS_READ_SOURCE_FIRST.md](LESSONS_READ_SOURCE_FIRST.md)

---

## ✨ Summary

Complete, production-ready system for calibration data collection:

- ✅ Generates 1000 images with per-step latents (~11h)
- ✅ Collects activation statistics efficiently (~30m)
- ✅ Analyzes and recommends quantization strategies
- ✅ Exports ready-to-use configuration
- ✅ Includes verification and recovery tools
- ✅ Extensively documented with examples

**Ready to implement TaQ-DiT quantization!** 🚀
