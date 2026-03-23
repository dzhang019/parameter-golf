# Parameter Golf Notes: `daniel-solution-v1`

## Current Observations

- We added a weight-spectrum analysis tool at [analyze_model_weights.py](/scratch/hewittlab/dtz2104/parameter-golf/analyze_model_weights.py) that can read either the raw checkpoint or the compressed artifact and analyze the dequantized weights.
- On the naive baseline (`records/track_10min_16mb/2026-03-17_NaiveBaseline`), the raw `.pt` and compressed `.int8.ptz` give almost identical spectral summaries. This is good: the structure survives quantization/export and is therefore more likely to be architecturally meaningful.
- The most compressible major matrix family in the baseline is `attn.c_q.weight` (`attn_q` in the analysis summary), where `rank_95 / min_dim` is often around `0.46-0.58`.
- `attn.c_k.weight` is somewhat compressible, but materially less so than `Q`.
- `attn.c_v.weight` and the MLP matrices look much fuller-rank.
- There is some depth structure in `Q`: middle-to-late blocks appear more compressible than early blocks in the baseline, but the pattern is not clean and monotone across all modules.
- The current model architecture in [train_gpt.py](/scratch/hewittlab/dtz2104/parameter-golf/train_gpt.py) uses a shared residual width (`model_dim`) across embeddings, skips, attention, MLP, and logits. That means per-layer residual-width schedules are possible, but not a small local patch.

## Current Takeaways

- The evidence supports targeted submodule compression more than global “later layers should just be narrower.”
- Low-rank `Q` is the most plausible first architectural change because:
  - it is supported by the baseline spectrum
  - it preserves residual width and interfaces
  - it can save both params and step time
  - there is prior competition evidence that it works
- We should be careful not to over-interpret weight SVD alone. A trained matrix being compressible does not automatically mean the architecture can be reduced there with no retraining loss.
- Activation-side evidence is the missing piece. If activations also live in a lower-dimensional subspace, that is much stronger support for architectural reduction.

## Results: Low-Rank `Q` Training

We tested training-time low-rank `Q` factorization in `records/track_10min_16mb/2026-03-22_danielsolutionv1/train_gpt.py` by replacing each `Q: 512 -> 512` projection with `512 -> r -> 512`.

### Short screening runs (~35 minutes local)

- Uniform low-rank `Q` did not produce a local step-time win in this implementation.
- `Q_RANK=256` was clearly unhelpful: slower and slightly worse than baseline.
- `Q_RANK=144` and `Q_RANK=160` also looked weak.
- `Q_RANK=128` and `Q_RANK=192` stayed closest to the full-rank baseline and were the only serious candidates worth promoting to longer runs.

### Longer runs (~110 minutes local proxy for the 10-minute challenge wallclock)

Comparison against the long baseline `baseline_sp1024`:

- Baseline full-rank `Q`:
  - steps: `12681`
  - step_avg: `520.48 ms`
  - prequant `val_bpb`: `1.2217`
  - final int8 roundtrip `val_bpb`: `1.2283`
  - artifact: `15.818 MB`
- `Q_RANK=192`:
  - steps: `12611`
  - step_avg: `523.37 ms`
  - prequant `val_bpb`: `1.2219`
  - final int8 roundtrip `val_bpb`: `1.2289`
  - artifact: `15.291 MB`
- `Q_RANK=128`:
  - steps: `12566`
  - step_avg: `525.26 ms`
  - prequant `val_bpb`: `1.2224`
  - final int8 roundtrip `val_bpb`: `1.2298`
  - artifact: `14.749 MB`

### Interpretation

- Full-rank baseline remained best on BPB.
- `Q_RANK=192` was close, but still slightly worse than baseline.
- `Q_RANK=128` was worse than `Q_RANK=192`.
- The factorized `Q` path did not provide a speedup on the local RTX6000 implementation.
- The main value of low-rank `Q` in this codepath is artifact savings, not direct BPB or speed gains.
- This suggests low-rank `Q` is best viewed as a budget-reallocation tool: it may only be worthwhile if the saved bytes are spent on another improvement.

### Spectral breakdown of trained low-rank models

Learned effective `Q = W_up @ W_down` spectra after the long runs:

- Baseline full-rank `Q`:
  - average `r95`: `262.4`
  - average effective rank: `231.3`
- `Q_RANK=192`:
  - average `r95`: `138.9`
  - average effective rank: `115.5`
- `Q_RANK=128`:
  - average `r95`: `101.2`
  - average effective rank: `86.2`

This confirms:

- trained full-rank `Q` is much lower-rank than 512 in practice
- but forcing a low-rank parameterization during training still costs some quality
- `Q_RANK=192` preserves more of the useful subspace than `Q_RANK=128`

## Results: Post-Hoc SVD Truncation

We built `truncation_ablation.py` to apply post-hoc low-rank SVD truncation to trained checkpoints and evaluate the damaged model.

After fixing evaluation mismatches, identity truncation (`rank=512`) matched baseline correctly, so the ablation path is now trustworthy.

### Uniform all-layer `Q` truncation sweep

Prequant `val_bpb` after truncating all 9 `Q` matrices to a uniform rank:

- `512`: `1.221677`
- `480`: `1.221716`
- `448`: `1.221922`
- `416`: `1.222395`
- `384`: `1.223203`
- `352`: `1.224447`
- `320`: `1.226325`
- `288`: `1.229305`
- `256`: `1.233678`
- `224`: `1.240304`
- `192`: `1.250804`
- `160`: `1.267376`
- `128`: `1.296304`
- `96`: `1.356702`

### Interpretation

- Very small damage down to roughly `384`
- Mild but real damage by `352`
- A clear knee around `320-288`
- `256` and below degrades quickly

This is a strong negative result for the naive “train full rank, then hard-SVD it to 192” idea.

The main lesson:

- low spectral concentration does **not** mean post-hoc SVD replacement is safe
- the model tolerates only modest truncation when applied abruptly after training
- training under a rank constraint is qualitatively different from truncating a trained full-rank model

## Results: Equal-Budget Non-Uniform Truncation

We also tested whether rank should be reallocated by depth under a fixed total `Q` parameter budget.

Based on the original spectral analysis, middle / mid-late layers appeared **more compressible**, so we tested schedules with:

- lower rank in the middle
- higher rank on the edges
- equal total `Q` parameter count versus a uniform-rank baseline

Results:

- Uniform `r=160`: `1.267376`
- Equal-budget `160` equivalent (`edges 176`, `middle 128`): `1.266854`
- Uniform `r=192`: `1.250804`
- Equal-budget `192` equivalent (`edges 224`, `middle 128`): `1.251453`
- Uniform `r=224`: `1.240304`
- Equal-budget `224` equivalent (`edges 256`, `middle 160`): `1.240157`

### Interpretation

- Differences are tiny.
- There is no strong evidence that depth-aware rank allocation dramatically beats uniform allocation in the post-hoc truncation setting.
- But the allocation idea is at least plausible: some equal-budget schedules are marginally better than uniform.
- This is weak evidence only. It does **not** rescue post-hoc SVD truncation as a promising compression method by itself.

## Updated Conclusions

- Low-rank `Q` is real spectrally.
- Post-hoc SVD truncation of `Q` is much more destructive than the spectra alone would suggest.
- Training-time low-rank parameterization works much better than post-hoc truncation, but still loses slightly to full-rank baseline in this codepath.
- `Q_RANK=192` is the only low-rank variant that remains mildly interesting, and only as a way to free artifact budget for something else.
- Non-uniform depth allocation may matter a little, but not enough to prioritize on its own.
- The “train larger, then compress into a smaller model” idea is still alive, but the compression step likely needs to be learned or gradual, not a one-shot SVD projection.

## Best Next Experiments

- Run truncation ablations per matrix family and per layer.
- Test adaptive Q rank by depth, not one global `r=192`.
- Consider low-rank Q plus modestly reduced K only where ablations permit.
- Do not touch MLP width based on current evidence unless ablations say otherwise.
- Only attempt variable residual width after we have strong activation-side evidence.
- Train a larger model first, then compress into a smaller architecture within the allowed training window. Possible versions:
  - start with a temporarily over-budget model and replace selected modules with low-rank factorizations during training
  - progressively distill or project high-rank weights into a smaller student architecture before the wallclock ends
  - use the larger model as a fast feature finder, then hand off to the smaller exportable model for the final segment of training

## What PR #215 Is Arguing, Step By Step

Note: PR `#215` is not present in this local checkout, so this section is based on the summary text and what we can infer from the repo architecture and prior runs.

### 1. Start from the standard `Q` projection

In the baseline-style model, query projection is a single matrix:

- `Q = x @ W_q`
- where `W_q` has shape `512 x 512`

That costs:

- `512 * 512 = 262,144` weights per layer

### 2. Observe that `Q` appears spectrally compressible

The claim is that weight analysis showed `Q` behaves like it mostly uses a lower-dimensional subspace.

That means:

- if we compute the singular values of `W_q`
- much of the energy is captured by far fewer than 512 directions

This is the same general signal we saw in the naive baseline:

- `Q` looks more compressible than `K`, `V`, or MLP weights

### 3. Replace one full matrix with two thinner matrices

Instead of:

- `512 -> 512`

PR #215 uses:

- `512 -> 192 -> 512`

So the full matrix is factorized into:

- `c_q_down: 512 -> 192`
- `c_q_up: 192 -> 512`

Parameter count becomes:

- `512 * 192 + 192 * 512 = 196,608`

Savings per layer:

- `262,144 - 196,608 = 65,536`
- about `25%`

### 4. Why this can help even before talking about accuracy

This change can help in two independent ways:

- fewer params in each `Q` projection means smaller artifact footprint
- fewer multiply-adds in the `Q` path means faster training steps

The PR summary claims step time improved from:

- `108 ms` to `77 ms`

That is a very large win in a 600-second budget. More steps often matter as much as or more than small per-step quality improvements.

### 5. Why “effective dimensions 89-114 out of 192” matters

The summary says the trained factorized model used only `89-114` effective dimensions out of `192`.

Interpretation:

- even after giving the model rank `192`
- the learned factorization only meaningfully uses roughly half of those latent directions

This suggests `192` was not a tight bottleneck. It may even have slack.

Important caution:

- “effective dimensions” depends on exactly how they measured it
- it could mean spectral effective rank of the product, activation covariance rank, or something similar
- without the PR code, we should treat the exact definition as uncertain

### 6. Why they say `Q` is special

The summary says:

- `K` had condition numbers around `19-29`
- `V` around `5-8`
- `O` around `531-2620`
- only `Q` was a good low-rank target

The rough intended argument is:

- `Q` seems to already live in a concentrated subspace
- `K`, `V`, and `O` behave more like full-rank operators
- therefore low-rank factorization hurts those more

One caveat here: condition number alone is not enough to establish compressibility. A matrix can have a high condition number and still need many singular directions for good performance. The stronger evidence is cumulative spectral energy, effective rank, and direct truncation ablations.

### 7. How this turns into “2 more layers”

If `Q` is cheaper per layer, the saved parameters can be spent elsewhere.

That means the PR’s gain may come from a bundle of effects:

- low-rank `Q` itself may be neutral or mildly positive
- faster steps give more optimization progress in 600 seconds
- smaller parameter footprint lets them add more layers under the artifact cap

So yes: the “2 more layers” are not free. They are funded by the saved params and maybe by the speed headroom.

This is exactly why we need an ablation table. Otherwise we cannot tell whether the win came from:

- the low-rank inductive bias
- the extra training steps
- the extra depth
- better compression under quantization
- or the combination

## Why PR #215 Matters For Our Direction

It does **not** strongly support “later residual width should shrink.”

It **does** support:

- low-rank factorization of selected submodules
- especially `Q`
- potentially with rank varying by depth

In other words, it is evidence for “compress the right internal operator,” not yet evidence for “change the whole residual stream schedule.”

## Ablations We Should Run For The `Q` Idea

We should separate the following effects.

### A. Pure architecture ablations

Keep everything else fixed and compare:

- baseline full `Q`
- low-rank `Q` with `r=256`
- low-rank `Q` with `r=192`
- low-rank `Q` with `r=160`
- low-rank `Q` with `r=128`

Measure:

- step time
- parameter count
- artifact size after export
- final `val_bpb`

### B. Depth allocation ablations

Keep total artifact budget similar and compare:

- 9L full `Q`
- 9L low-rank `Q`
- 10L low-rank `Q`
- 11L low-rank `Q`

This tells us whether low-rank `Q` is useful by itself or mainly as a way to buy depth.

### C. Equal-step vs equal-wallclock ablations

This is critical.

Compare:

- equal wallclock: normal challenge setting
- equal step count: force both models to train for the same number of optimizer steps

If low-rank `Q` only wins under equal wallclock and not equal steps, then the main benefit is speed.
If it still wins at equal steps, the architecture itself is helping.

### D. Per-layer rank ablations

Try:

- uniform `r=192` at all layers
- higher rank in deep layers, lower in shallow layers
- lower rank in middle layers only
- factorize only selected layers, not all layers

This is high-value because our current baseline spectra suggest `Q` compressibility varies by depth.

### E. Post-hoc truncation ablations before retraining

For an existing trained checkpoint:

- truncate `Q` in one layer at a time to different ranks
- evaluate BPB degradation
- then truncate all `Q` layers together

This is the fastest way to estimate where rank can be removed without immediately spending full training runs.

### F. Quantization interaction ablations

Measure whether low-rank `Q` helps mainly because it compresses/quantizes better:

- compare pre-quant validation
- compare post-quant validation
- compare artifact bytes

If most of the gain appears only after quantization/export, that tells us the benefit is more about artifact efficiency than model quality.

## Questions To Answer Before Building A Full Residual-Width Schedule

- Do activations in later layers also occupy a smaller subspace, or only the weights?
- Is `Q` compressible because it is intrinsically low-rank, or because the current architecture over-allocates query capacity?
- Does `K` have enough slack for a small reduction, or is it much less forgiving than `Q`?
- Does low-rank `Q` mainly buy speed, or does it also improve quality per step?
- Once rank savings are available, is spending them on more layers actually better than spending them on MLP width, hash features, or higher-precision exceptions?

## Immediate Recommendation

The next clean path is:

1. Add checkpoint truncation/eval ablations for `Q` by layer and by rank.
2. Add activation-rank analysis for `Q/K/V` and MLP inputs.
3. If the evidence stays consistent, implement low-rank `Q` with a small rank sweep.
4. Only then consider adaptive-by-depth rank schedules or broader architecture changes.
