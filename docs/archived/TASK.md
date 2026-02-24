# Tasks

Date: 2026-02-09

- [x] Create project skeleton with docs and module layout
- [x] Add precision schedule policy and integrate into sampler
- [x] Add profiling utilities and baseline comparison path
- [x] Write unit tests and update README

Date: 2026-02-09

- [x] Implement full SD 2.1 MLX T2I pipeline with profiling and HF download

Date: 2026-02-10

- [x] Fix adaLN modulation shape broadcast: extend _fix_adaln_weights_in_place to treat wrong-shape weights as bad; add unit tests (test_diffusionkit_patches)

## Discovered During Work

- [ ] Run preliminary benchmarks (scheduled vs baseline)
- [ ] Add benchmark CLI or script to run fixed prompts and report timings
- [ ] Create notebook for dynamic precision experiments
