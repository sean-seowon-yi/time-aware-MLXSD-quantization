# SD3 Medium MLX Architecture
### `argmaxinc/mlx-stable-diffusion-3-medium`

---

## Inputs

| Tensor | Shape | Description |
|---|---|---|
| `latent_image_embeddings` | `(B, H, W, 16)` | Noisy VAE latents |
| `token_level_text_embeddings` | `(B, 77, 4096)` | T5 token embeddings |
| `pooled_text_embeddings` | `(B, 2048)` | CLIP pooled embeddings |
| `timestep` | `(B,)` | Diffusion timestep |

---

## Global Conditioning

Computed once per forward pass. The output `conditioning` is fed into every `adaLN_modulation` in every block.

```
pooled_text_embeddings (B, 2048)          timestep (B,)
          │                                     │
          ▼                                     ▼
  PooledTextEmbeddingAdapter           TimestepAdapter
  Linear(2048 → 1536)                  sinusoidal_embed(256)
  SiLU                                 Linear(256 → 1536)
  Linear(1536 → 1536)                  SiLU
                                       Linear(1536 → 1536)
          │                                     │
          └──────────────── + ─────────────────┘
                             │
                    conditioning  (B, 1, 1, 1536)
```

---

## Input Embedding

```
latent_image_embeddings                    token_level_text_embeddings
(B, H, W, 16)                              (B, 77, 4096)
          │                                           │
          ▼                                           ▼
  LatentImageAdapter                         context_embedder
  Conv2d(16 → 1536, kernel=2, stride=2)      Linear(4096 → 1536)
  [patchify: each 2×2 patch → 1 token]
          │
          + LatentImagePositionalEmbedding
            (learned embedding, center-cropped to input resolution)
          │
  reshape → (B, S_img, 1, 1536)             (B, S_txt, 1, 1536)

  S_img = (H/2) × (W/2)                     S_txt = 77
        = 1024 for 512×512 input
```

---

## MultiModalTransformerBlock × 24  (blocks `mm_00` – `mm_23`)

Each block processes img and txt **independently** through `pre_sdpa`,
then **jointly** through a single SDPA, then **independently** again through `post_sdpa`.

### Standard Block (blocks 0 – 22, both streams)

```
IMG stream (B, S_img, 1, 1536)                TXT stream (B, S_txt, 1, 1536)
          │                                               │
          │     adaLN_modulation                          │     adaLN_modulation
          │     SiLU → Linear(1536 → 6×1536)              │     SiLU → Linear(1536 → 6×1536)
          │     splits into 6 params:                     │     splits into 6 params:
          │       β₁, γ₁  (pre-attn shift/scale)         │       β₁, γ₁  (pre-attn shift/scale)
          │       α₁       (post-attn gate)               │       α₁       (post-attn gate)
          │       β₂, γ₂  (pre-FFN shift/scale)          │       β₂, γ₂  (pre-FFN shift/scale)
          │       α₂       (post-FFN gate)                │       α₂       (post-FFN gate)
          │                                               │
          ▼                                               ▼
   LayerNorm (norm1)                              LayerNorm (norm1)
   affine_transform(x, shift=β₁, scale=γ₁)       affine_transform(x, shift=β₁, scale=γ₁)
   → modulated_pre_attention                       → modulated_pre_attention
          │                                               │
          ├─► q_proj  Linear(1536 → 1536)           ├─► q_proj  Linear(1536 → 1536)
          ├─► k_proj  Linear(1536 → 1536, no bias)  ├─► k_proj  Linear(1536 → 1536, no bias)
          └─► v_proj  Linear(1536 → 1536)           └─► v_proj  Linear(1536 → 1536)
                │                                               │
                │         QKNorm (RMSNorm per head on Q and K)  │
                │                                               │
                └──────────────────── concat ──────────────────┘
                                          │
                              Q, K, V:  (B, 24 heads, S_img+S_txt, 64)
                                          │
                             PositionalEmbedding (learned, not RoPE)
                             applied at input embedding stage only
                                          │
                                          ▼
                           Joint SDPA  softmax(Q·Kᵀ / √64) · V
                           [Flash attention when seq_len > 1024]
                                          │
                              (B, 24 heads, S_img+S_txt, 64)
                              reshape → (B, S_img+S_txt, 1, 1536)
                                          │
                                       split
                            ┌───────────┴───────────┐
                     img slice                  txt slice
                  (B, S_img, 1, 1536)       (B, S_txt, 1, 1536)
                            │                        │
                            ▼                        ▼
                       o_proj                    o_proj
                  Linear(1536 → 1536)        Linear(1536 → 1536)
                            │                        │
                       × α₁                     × α₁
                            │                        │
                       + residual               + residual          [post_sdpa_res]
                            │                        │
                     LayerNorm (norm2)         LayerNorm (norm2)
                     affine_transform(          affine_transform(
                       shift=β₂, scale=γ₂)      shift=β₂, scale=γ₂)
                            │                        │
                           fc1                      fc1
                     Linear(1536 → 6144)      Linear(1536 → 6144)
                            │                        │
                           GELU                     GELU             [post_gelu]
                            │                        │
                           fc2                      fc2
                     Linear(6144 → 1536)      Linear(6144 → 1536)
                            │                        │
                       × α₂                     × α₂
                            │                        │
                       + residual               + residual
                            │                        │
                            ▼                        ▼
                 IMG out (B, S_img, 1, 1536)   TXT out (B, S_txt, 1, 1536)
```

---

### Block 23 — `mm_23`  (final multimodal block)

The **img stream** is identical to a standard block.
The **txt stream** has `skip_post_sdpa=True`:

```
TXT stream in block 23:
  - adaLN_modulation outputs only 2 params (β₁, γ₁) — no α₁, β₂, γ₂, α₂
  - q_proj, k_proj, v_proj still computed and contributed to joint SDPA
  - o_proj replaced with nn.Identity() (no-op)
  - No norm2, no fc1, no GELU, no fc2
  - TXT output after SDPA is DISCARDED → set to None
  - Only IMG stream continues past block 23
```

---

## After Block 23: IMG stream only

```
IMG stream (B, S_img, 1, 1536)
```

> Note: SD3 Medium has `depth_unified = 0`, so there are **no UnifiedTransformerBlocks**.
> The unified blocks present in Flux-style models do not exist here.

---

## FinalLayer

```
IMG stream (B, S_img, 1, 1536)
          │
          ▼
  adaLN_modulation
  SiLU → Linear(1536 → 2×1536)
  split into (shift, scale)
          │
  LayerNorm (norm_final)
  affine_transform(x, shift, scale)
          │
  Linear(1536 → patch_size² × 16)
       = Linear(1536 → 64)
          │
  unpatchify → (B, H, W, 16)
```

**Output**: predicted noise in VAE latent space, same shape as input `(B, H, W, 16)`.

---

## Architecture Summary

| Property | Value |
|---|---|
| `hidden_size` D | 1536 |
| `num_heads` | 24 |
| `per_head_dim` | 64 |
| `mlp_ratio` (FFN expansion) | 4 × D → fc1: 1536→6144, fc2: 6144→1536 |
| `depth_multimodal` | 24 blocks |
| `depth_unified` | 0 (no unified blocks) |
| `parallel_mlp` | False (FFN is always sequential, after attention) |
| `use_qk_norm` | True — RMSNorm applied per head to Q and K |
| Positional encoding | Learned input embedding (not RoPE) |
| Joint SDPA token order | `[img_tokens \| txt_tokens]` concatenated |
| adaLN params per block | 6 (standard), 2 for `mm_23_txt` |
| `k_proj` bias | False (softmax invariance) |
| Patch size | 2 × 2 |
| VAE latent dim | 16 |

---

## Activation Families Captured by EDA Tracer

| Family | Capture Point |
|---|---|
| `pre_attn` | After adaLN norm1, before Q/K/V — input to attention projections |
| `q_proj` | Output of `q_proj` linear |
| `k_proj` | Output of `k_proj` linear |
| `v_proj` | Output of `v_proj` linear |
| `sdpa_out` | Raw SDPA output before `o_proj` |
| `post_sdpa_res` | After `o_proj` + first residual add, before norm2 + FFN |
| `post_gelu` | After `fc1` + GELU, before `fc2` |

> `mm_23_txt` contributes only `pre_attn`, `q_proj`, `k_proj`, `v_proj` — no `sdpa_out`, `post_sdpa_res`, or `post_gelu`.
