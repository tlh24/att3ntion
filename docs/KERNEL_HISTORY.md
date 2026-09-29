# Kernel history (single gather and shared-KV), 2026-09-09 to 2026-09-17

## 1. How to read this

- Every speedup is relative to **that phase's own frozen baseline** (named in the timeline). They are not cumulative; never multiply ratios across phases.
- All numbers are H100 80GB (one 512x32 profile ran on an H200), BF16 inputs, FP32 accumulation, B=1, D=128 unless stated. "Complete" = forward + all five gradients + shared-KV head reductions.
- These are the agents' own measured records, copied from the reports in section 8. Timing scopes differ between phases (callable CUDA events, CUDA-graph replay, isolated kernel); only compare numbers inside one table.
- Terms used below: **CTA** (thread block), **warp** (32 threads), **SM** (one GPU core; H100 has 132), **CTAs/SM** (how many blocks fit on one SM at once, limited by registers and shared memory; also called residency or occupancy), **spill** (registers overflow into slow "local" memory), **MMA** (`mma.sync`, the per-warp tensor-core instruction), **WGMMA** (Hopper's warpgroup tensor-core instruction, 4 warps issue one large matrix multiply), **CuTe** (CUTLASS's layout library that builds WGMMA descriptors), **BK/BJ** (column/row tile width), **LSE** (log-sum-exp saved by the forward for the backward), **R/S pass** (backward kernels that own the R-side and S-side key/value gradients), **dQ pass** (backward kernel for the query gradient).

## 2. Where things stand now

**Release P07** (binary SHA `dcab6619...`) is installed in this worktree. It is the shared-KV path: Hq query heads share one K/V head (Hkv=1), D=128, equal causal windows 16/32/64/128 on both key axes. Automatic dispatch (`fwd_group=0, rs_group=0`) picks the schedule per shape. As of 2026-09-22 the default window in `att3ntion/_single_gather_shared.py` is 128 (was 32).

| Question | Answer (source) |
|---|---|
| Speed at w32 | Complete 1.46-4.88x faster than the strongest measured FBGEMM control over 9 cells (H16/64/128 x N128/256/512); 1.93-2.33x vs the frozen Phase-1 original. H64/N256: 0.600 ms vs FBGEMM 0.963 ms. |
| Forward alone | Still loses 5 of 9 w32 cells to FBGEMM. TLX forward was 1.9-3.0x faster than R26's forward at w32. |
| Other windows | vs the frozen original ported to runtime windows: w16 2.78-3.73x, w64 1.71-1.85x, **w128 only 1.32-1.39x**. No external comparison exists: FBGEMM's released backward supports only w2=32. |
| w128 (new default) is the least tuned window | R/S at w128 uses the retained MMA body (RS30); the WGMMA w128 attempt lost (W128-01). Isolated wide dQ is only 1.04-1.07x. Hopper forward W128 (F19) is used only for H>=64, N>=256. |
| Longer / wider shapes (N2048, H64) | P06 at 128x128: 48.1 ms, vs FBGEMM released 512x32 43.4 ms (10.9% slower, different window) and FBGEMM general 128x128 ~131.7 ms. No numerically valid 128x128 candidate beats FBGEMM 512x32 yet. |
| Numerical weaknesses | 18 pre-existing stress failures remain. In auto mode all 24 scaled-input cases exceed the absolute LSE gate and 8 concentrated-softmax cases fail some gradient gates. Cause: BF16 rounding of products/probabilities before tensor-core GEMMs. Every visible pair is computed, but arithmetic is not exact. |
| Precision fix exists, not installed | A compensated w32 prototype (split each BF16 product into high + residual parts) passes 18 stress cells at 12-17% extra cost over P06 (`overnight_20260915/precision/`). A later compensated package (main checkout `overnight_20260916/`) is 1.2444x the time of P07. A compensated 128x128 candidate at N2048 still fails the concentrated dQ gate in its wide dQ path. |
| Singleton rows | A query with <=1 visible key now gets exactly zero score gradients (fixed inherited bug, O10/CS06). |

Which file holds what (`cuda/`; every shared-KV file below `common.cuh` lives in `cuda/shared_kv/`):

| File | Contents |
|---|---|
| `forward.cu`, `backward.cu` | Legacy 3-gather kernels (`Y_gather_tc`, `Bwd_gather_tc` with `ROLE=BWD_ALL`, bit-exact to main) and the single query-gather variants (`BWD_QUERY_ANCHOR`, `BWD_QUERY_ROWS`), independent heads. |
| `common.cuh` | Window tile bounds `sg_row_bounds` / `sg_col_bounds`, mask packing `sg_pack_mask32`. |
| `single_gather_shared.cu` | Shared-KV portable (non-Hopper) path, entry points, dispatch. |
| `shared_retained_fwd.cuh`, `shared_retained_dq.cuh`, `shared_wide_dq.cuh`, `shared_retained_rs.cuh` | Retained-raw-window MMA schedules (forward, dQ narrow, dQ w64/w128, R/S). |
| `shared_mask_metadata.cuh`, `shared_delta.cuh`, `shared_reduction.cuh` | Per-backward mask/support metadata, delta, head reduction. |
| `shared_hopper.cu`, `shared_hopper_dq.cu`, `shared_hopper_rs.cu` (w32), `shared_hopper_rs64.cu` (w64) | H100-only WGMMA/CuTe kernels, compiled only when `ATT3NTION_CUTLASS_INCLUDE` is set; otherwise the MMA fallback is built. |

Build with CUTLASS (from `overnight_20260915/BUILD_AND_USE.md`; CUTLASS 3.5.1, commit `f7b19de3...`, CUDA 12.6, the include dir must contain `cute/tensor.hpp`):

```bash
export ATT3NTION_CUTLASS_INCLUDE=/path/to/cutlass/include
export TORCH_CUDA_ARCH_LIST=9.0
MAX_JOBS=2 python setup.py build_ext --inplace
# y = single_gather_shared_attention(Q, R, S, Vr, Vs); y.backward(dy)   # window defaults to 128
# Q: [B,H,N,128]; R,S,Vr,Vs: [B,1,N,128]; groups 0/0 = automatic dispatch
```

## 3. Timeline

| Date | Phase | Outcome | Baseline |
|---|---|---|---|
| 09-09 | Exp 1: single query-gather fwd/bwd, D64/128, role-specialized backward | Full causal 8-32x vs paper; loses at H4 N256 w32 (0.55-0.86x) and on the whole shared-KV matrix (FBGEMM 0.01-0.93x) because it walked dense N^2 pairs and replicated KV 64x | Paper ("Fast and Simplex") Triton Listings 1-3, FBGEMM Triton forward, legacy 3-gather |
| 09-10 | Exp 2: window bounds (V1), per-role tiles (V2/V3), D128 one-pass output (V4) | Complete vs V0: core 1.82x, shared-KV 5.35x geomean; beats tuned paper in every independent-head cell (1.39-13.4x). Shared-KV window cells still lose to FBGEMM (0.28-0.69x) | V0 = commit `cf1e258` |
| 09-11 | Shared-KV prototype: native Hkv=1 reads, G heads per CTA | G2 complete only 1.13x at N256 (below the 20% bar). Paper Listing 3 still 1.42x faster. Negative result, left opt-in | Exp-2 final ("prev final") |
| 09-11 to 09-15 | Tight window-relative tiles, Phase-1 cleanup | G2 complete 2.603 -> 1.401 ms (1.86x) at H64 N256 w32 | G2 before patch |
| 09-15 | WGMMA/CuTe bring-up (GEMM probe, packed forward, epilogues) | Standalone only; static-pairing epilogue 2.61x faster than shared atomics | Standalone atomic-epilogue kernel |
| 09-15 | R/S backward rounds R00-R28 | R26: complete 1.243x at N256 (1.341 -> 1.078 ms, graph); windows 16/32/64/128 added | Phase-1 G2 (R00) |
| 09-15 | FBGEMM/TLX baseline integration | R26 1.10-1.35x vs TLX-forward + Triton backward at H64; ties FBGEMM Triton 3.5 at H64/N256 | FBGEMM TLX and Triton |
| 09-15/16 | Overnight O01-O52 on 10 H100s (forward F, dQ Q/WG/W, R/S RS/CS, fusion, precision PR, tape T) -> P01..P07 | P07 as in section 2 | Phase-1 original `c131f6cc` + R26 + FBGEMM envelope |
| 09-16/17 | Compensated w32 package (main checkout, not installed here) | 1.6635x geomean vs FBGEMM; 1.2444x the time of P07 | FBGEMM |
| 09-17 | w128 optimization + window-shape study (N2048) | No valid 128x128 candidate beats FBGEMM 512x32 (43.33 ms); compensated 57.9 ms fails concentrated dQ | FBGEMM released 512x32 |
| 09-17 | Asymmetric 512x32 research kernel (isolated, obelisk/d128) | Checkpoint 2: 53.37 ms vs FBGEMM 43.39 ms (from 91.14 ms); gap is entirely R/S | FBGEMM released 512x32 |
| 09-22 | Owner changed shared-KV default window 32 -> 128 | - | - |

## 4. Kept

| Change | Effect | Where |
|---|---|---|
| Role-specialized backward: compile out the absent softmaxes | 2.3-3.4x vs legacy 3-gather backward; D128 no longer spills | `backward.cu` `Bwd_gather_tc<...,ROLE>` |
| Window metadata -> tile ranges (V1), bitwise identical to dense | D128 N256 w32: 4.7x fwd / 7.8x bwd vs V0 | `common.cuh` |
| Per-regime tile tables (V2/V3): 4 warps beat 8 once nothing spills; 2x16 tiles for windows | Window backward 2.6x (H4 w32), 3.1x (shared w32) | `sg_set_config`, exp2 `candidates/final.json` |
| D128 one-pass 128-channel backward output (V4) | 1.3-1.5x for either backward role | `backward.cu` (`bwd_dh=128`) |
| `__launch_bounds__(threads,2)` / `(threads,3)` register pins | Restored 128/168 registers and parity after drift (trap T2) | `forward.cu` |
| Native shared-KV reads + 2 heads per CTA (G2) | Forward 1.30x; 64x less prepared KV memory | `single_gather_shared.cu` |
| Tight window-relative tile starts, unaligned bool-mask loader | 63% / 81% / 46% fewer cells (fwd / Q-bwd / each R/S); complete 1.86x | `common.cuh` |
| SINGLE_TILE / SINGLE_COL fast paths, dead-merge removal | ~1.1% complete | `forward.cu`, `shared_retained_dq.cuh` |
| CuTe for WGMMA layouts; statically checked register-pairing epilogue | 320-case GEMM probe passes; epilogue 2.61x vs atomics | `shared_hopper*.cu` |
| R26: 2 heads x 2 gradient directions per CTA share raw Q/dY staging | R/S 1.42x; complete 1.243x (`rs_group=4`). R14 kept as bitwise-preserving `rs_group=2` | `shared_retained_rs.cuh` |
| Fusion F01/F02: delta without FP32 temporaries, head reduction split over 4 groups | H64N256 reduction 72.66 -> 18.33 us, delta 25.71 -> 3.96 us | `shared_delta.cuh`, `shared_reduction.cuh` |
| Packed mask + saturated support metadata once per backward (M01, O15) | 2.8-3.5% complete, prep ~2 us | `shared_mask_metadata.cuh` |
| Retained raw window in R/S: async cp.async staging (RS08), sequential head visits (RS18/19), circular windows w64/w128 (RS16, RS30) | RS08 1.768x R/S vs original; RS19 1.897x; RS30 w128 1.392x | `shared_retained_rs.cuh` |
| CTA-uniform singleton dispatch (CS06) | Exact zero score gradients for <=1 visible key with no per-score cost; I06 1.894x original complete | `shared_retained_rs.cuh` |
| dQ: keep raw R (Q01), one-warp w16 geometry (Q08, 1.85-2.16x), sequential heads + deferred prefetch (Q36-39), per-warp support count (Q45), wide dQ W13/W15/W18 | Isolated dQ w32 1.17-1.45x; w64 1.20-1.26x; w128 1.04-1.07x | `shared_retained_dq.cuh`, `shared_wide_dq.cuh` |
| dQ WGMMA: A operands assembled in registers (WG03), static fragment index (WG06, local 128 B -> 0), original pairwise w64 sum order (WG10) | Isolated w16 3.12-4.87x, w32 1.43-1.80x, w64 1.29-1.37x | `shared_hopper_dq.cu` |
| Forward: F04 one-warp w16, F10 register fragments + dead-storage epilogue, F07 grouped row-tile merge (w64/128) | F10 H64N256 121 us vs 134.6; F07 1.21-1.32x | `shared_retained_fwd.cuh` |
| Forward WGMMA: F13 pads raw Vr rows to 136 channels (kills bank conflicts, keeps 4 CTAs/SM); F16/F17 W64; F19 W128 split into first-64-queries + rest | F13 89.8 us vs 134.6 (w32); F19 696 us vs F07 777.5 (w128 H64N256) | `shared_hopper.cu` |
| R/S WGMMA w32: moving 48-column window view + private register P/gA (RS42) | 2.11-2.25x R/S vs original, 168 regs, 3 CTAs/SM | `shared_hopper_rs.cu` |
| BF16 per-head KV partials after FP32 accumulation (RS43/P03) | Halves 4 partial buffers; backward peak about -36% | R/S kernels |
| w64 R/S scratch fold reusing dead Q/dY storage (RS52/P06) | 1.99x R/S vs original at N256 | `shared_hopper_rs64.cu` |
| P07: hoist repeated scaled-Q fragment (forward 6-8%) + phased in-register R/S channel fold (R/S 10-11%, short-scoreboard stalls 23.2% -> 6.8%) | Complete 7-9% over P06, output bits unchanged | `shared_hopper.cu`, `shared_hopper_rs.cu` |
| 512x32 only: put the Hadamard product on the short 32-token axis | Forward 13.8 -> 7.0 ms, dQ 17.6 -> 9.1 ms | `~/att3ntion/research_512x32_20260917/` (not in `cuda/`) |

## 5. Tried and rejected

Most common lesson: a change that pushes registers or shared memory past a CTAs/SM threshold loses, even when it removes work. At 128 threads, 3 CTAs/SM needs about <=168-170 registers and <=75 KB shared memory (the driver also reserves 1024 B per block). Register count alone is a poor predictor (R26 won at 248 registers).

| Idea | Result | Why it lost |
|---|---|---|
| Dense (j,k) traversal under a window mask (exp 1) | w32 cost = full causal; 0.55x vs paper at H4 D128 N256 w32 | Visits masked tiles; fixed by V1 bounds |
| Replicate KV to 64 heads for shared-KV | 0.01-0.93x vs FBGEMM | No on-chip KV reuse; representation alone changed nothing (exp 2 sharing family) |
| 8-warp backward CTAs at D128 | 1 CTA/SM, 12.5% warps active | 4 warps fit 2 CTAs/SM and won |
| 2x16 forward tiles for full causal; 4x32 at D64 small shapes | 0.81-0.84x; 0.86x at H1 D64 N128 | Tile choice must depend on regime |
| G4 grouping, 4 heads per CTA (shared-KV prototype) | 1.03x fwd vs G2 1.30x | 8-warp block fits 1 CTA/SM |
| BK32 (wider column tile) in MMA forward | +22.1% (G1) / +19.1% (G2) latency at N512 | Likely lost prefetch/compute overlap of 2 buffered BK16 tiles (not isolated) |
| BK32 in R/S (R02, R04, R10, R12, R28) | R02 0.96x, R10 0.97x, R28 0.85-0.89x of R00; R04/R12 no better than their BK16 twins | Shared memory cuts residency; 24 B local memory in some |
| Hand-computed WGMMA descriptors | Wrong values | Replaced by CuTe (trap T6) |
| WGMMA epilogue with runtime column matching | 768 B local/thread; 9.52x slower than atomics | Zero ptxas spills did not mean zero local arrays |
| First WGMMA packed forward prototype | Slower despite fewer registers | Swapping instructions alone is not a speedup |
| G1 independent R/S directions (R05, R13); G4/WPH2 (R07) | 0.84-0.97x | Less input reuse |
| Larger simultaneous head groups: G8 (R16, R17), 4 heads x 2 directions (R27), RS12 | R17 1 CTA/SM; R27 1.38x < R26 1.42x; RS12 125 KiB, 1 CTA | Occupancy loss |
| Split the 128 channels into two 64-channel CTAs (R18-R21, R24) | 0.65-0.83x | Repeats score GEMMs and staging; do not retry without removing that |
| Force 3 CTAs/SM with launch bounds (R22, R25) | R22 1.24x < R11 1.36x; R25 256 B local | Shared memory still limits to 2; spills |
| Consume output fragments immediately to cut registers (RS02, RS03) | 0.985x / 1.249x vs original, below RS01 | Extra shuffles outweigh lower registers |
| One head, two directions per CTA (RS04, RS06) | 1.21x / 1.01x | Repeats raw loads; too few active warps |
| Oversized raw capacity (RS07 128 at w32, RS15 w64, RS17 w128, RS28 double buffer) | RS07 1.00x, RS28 0.95x of original; RS15/RS17 slower than R26 | Each hit a 1-CTA/SM cliff |
| Direct warp fold to shared output at w32 (RS09, RS10) | 1.645x / 1.729x < RS08 1.768x | Slower despite fewer barriers |
| 8 sequential head visits (RS20); alternating sweep directions (RS31, RS32) | No gain / slower | Diminishing reuse, longer CTAs |
| Per-score singleton selects (CS03, CS04) | 15-18% slower | Branch per score; CTA-uniform dispatch (CS06) kept |
| Various WGMMA R/S layouts: shared/shared score (RS33b), w32 register score (RS34, 220 regs), prefetch visits (RS35/36), moving-48 alone (RS37), private P/gA alone (RS38), 64 columns (RS44) | All slower than retained MMA or RS42 | Each crossed 3 -> 2 CTAs/SM; only the RS42 combination fit |
| w16 direct-to-global fold (RS39-41) | Slower; 408-608 B stack | Spills |
| w64 moving 80-column WGMMA (RS45); stream w64 values for 3 CTAs (RS46, RS50) | 1.135 / 1.46 / 1.36 ms vs retained 1.069 | Registers (243-250) still limit to 2 CTAs |
| 3-CTA bound on w16 (RS47); BF16x2 value multiply (RS48, RS49) | Near tie or loss (RS48 172 regs, 2 CTAs) | No material gain |
| Scratch fold at w32 (RS51); rolled fold loop (RS53) | 0.466 / 0.724 ms vs 0.401 | Loses 3-CTA residency; serialized reads |
| Direct-global Q/dY instead of shared (DG01) | ~2x slower than phased fold | 234 registers, 2 CTAs |
| W128 WGMMA, 144-column view (W128-01) | R/S 50-82% slower than P06 | 255 regs, 171 KB shared, 208 B local, 1 CTA |
| Forward: probabilities into shared + deferred projection (F03) | 151 us vs 137.6 | Repeated fragment loads beat higher occupancy |
| Forward: full-window raw residency at w64 (F07 variant); sequential head-pair visits (F09); repeated w16 head visits (F12) | No gain or loss | 92 KB -> 2 CTAs; too few active blocks |
| Forward: pad the scaled-Q vector too (F14) | 103 us vs F13 89.8 | 128 -> 146 regs, 4 -> 3 CTAs |
| Forward: fifth producer warp building next A (F15) | 149.9 us vs 134.7 original | 1/4 of the lanes do assembly; halves CTAs |
| Forward W128: stream all rows (F18); quadrant ordering (F20) | F18 superseded; F20 1093.8 us vs F07 778.6 | F18 wastes masked rows on first 64 queries; F20 doubles merges, 256 B local |
| dQ: retain large raw unions (Q04-Q07, Q14-Q17) | 0.37-0.99x | Shared memory -> fewer CTAs; duplicated staging |
| dQ: one warp at w32 (Q08); query x head Cartesian groups (Q28-31); 32 heads at small N | 0.78x; up to 0.90x; loses at H64N128 | Too few CTAs / rectangle waste at w32 |
| dQ: retain whole window at w128; 16-32 head groups (W04, W14, W16, W17) | ~35% loss; slower | Occupancy threshold |
| dQ: reload raw R/Vr per head (W05); stream Vr from global (W08-W11); trim 512 B only (W06) | 0.44-1.10x | Reloads cost more; 1024 B per-block reserve meant the trim did not change residency |
| dQ WGMMA: shared A operands (WG01/02), temporal visits (WG04), two live A fragments (WG08) | 0.78-0.92x; WG04 < WG03; WG08 w16/w32 slower | Staging round trip, register growth |
| Pair-derivative tape (T00-T07, TF00-TF03): store P/gA once, reuse in R/S | 3-7% complete gain | Rejected for research scope (pair-derivative reuse excluded) and 64-512 MiB extra storage |
| Two ICLR-run optimizations O1/O2 (compensated package) | 1.0036x / 1.0068x | Missed the 1.05x promotion rule |
| w128/N2048: fast split R/S | 45.0 ms but ordinary dR/dS/dVr/dVs gates fail | Invalid; enlarging its cache did not fix it |
| w128: deterministic query split | 49.1 ms | Passes ordinary, fails scale-2/concentrated |
| w128: compensate only the R/S direction; residuals on dQ dY-value and projection; delay dQ Jacobian subtraction | 60.9 ms, misses dR/dVr limits; no measurable effect; no change | Error is in the wide dQ tensor-core formulation |
| w128: scalar FP32 dQ | Accurate but 3943.7 ms | Diagnostic only |
| w128: overlap dQ and R/S on separate streams | No latency gain | Kernels did not co-reside |
| w128: unchecked causal shortcut | Faster | Admitted future keys |
| Extend the equal-window kernel to 128x128/256x64/512x32 by a parameter | Not feasible | Head packing undefined above W=64; 168-617 KiB shared per block |
| 512x32: share staged tiles across heads (HV=4) or 4 anchors | No timing change | Not L2-bandwidth bound |
| 512x32: short-axis R/S | 32.0 / 18.9 ms vs 24.8 / 13.7 | 2 CTAs/SM (92-108 KB), latency-bound |
| 512x32: shared-memory A operands; + swizzled Q/dY; two 64-wide chains | 33.4/18.8, 38.8/22.5, 24.9/13.8 ms | More pressure loses a block; same-budget variants unchanged |
| 512x32: force 4 CTAs/SM (K=32 tile + min-blocks 4); register cap at K=48 | 49.8 / 38.9 ms vs 37.5 | K=32 doubles the opposing loop; occupancy is not the lever |
| 512x32: transposed factorization dR = S * (dP^T Q) | Retracted before coding | Pair count is irreducible; 1.47x overhead is window tiling |

## 6. Traps

| # | Trap | Fix |
|---|---|---|
| T1 | A timing run that overlapped a test session on the same pod was contaminated | Discard it; never run tests while a timing suite runs |
| T2 | Making loop bounds runtime values silently raised `Y_gather_tc` registers (128 -> 132 at D64, 168 -> 194 at D128), losing a CTA/SM (up to 1.36x slower) | `__launch_bounds__(threads, 2/3)`; check `logs/resource_usage.py` output reads 128 and 168 before timing |
| T3 | Resetting the running max to -inf corrupted saved LSE on multi-row-tile cases while Y looked fine | Keep the previous running max |
| T4 | Removing a shared-buffer barrier raced between output-channel passes (D128/DHT64 backward) | Barrier stays unconditional |
| T5 | Missing barrier after raw R/Vr staging; reduction combined lanes but missed same-row registers (WGMMA forward) | Fixed; test with sanitizers |
| T6 | Hand-computed WGMMA descriptors compiled but gave wrong values; swizzle depends on each operand's major mode; generic shared stores need an async-proxy fence | Use CuTe (CUTLASS 3.5.1) |
| T7 | Omitting `--expt-relaxed-constexpr` gave warnings and an incorrect binary | Always pass it for CuTe objects |
| T8 | CUTLASS 3.5.1: CuTe types that do not depend on a template parameter get instantiated on the host and fail (F20, 512x32 R/S, `make_gmma_desc`) | Make every shared-memory layout feeding a WGMMA descriptor depend on a template parameter |
| T9 | CUTLASS 3.5.1: one aggregate constexpr ownership assertion over 128 threads was rejected; missing CuTe extended-shape macro (RS37) | Split into individual assertions; define the macro |
| T10 | Loading several .so builds that reuse the same object files coalesced GNU-unique init flags, so a kernel skipped its shared-memory opt-in (O08) | Build with hidden host symbols (O11) |
| T11 | Process-wide "shared-memory opt-in done" bool skipped GPU 1 (invalid argument) (O21) | Device-aware thread-local cache |
| T12 | CUDA-graph capture: ctypes cached the eager stream so launches escaped capture (T01); harness reused device-0 capture stream on device 1 (O41) | Query the current stream inside every call; poison outputs with NaN and require replay to overwrite them |
| T13 | Validation shape selected a different kernel than the timed shape (wide-dQ threshold 16,384 tasks) | Force the same dispatch path in validation |
| T14 | FBGEMM's saved statistic is not a natural-log LSE (`max(log2(e)x)+ln(l)`); its B2 backward used dY's batch stride for delta; shared-KV offsets wrong | Labeled external-only corrections; never copied into our kernels |
| T15 | `ncu` fails with `ERR_NVGPUCTRPERM` on normal pods (g259, obelisk/d128) | Use the `single-admin` pod profile |
| T16 | Per-kernel `torch.profiler` timings vary about +/-20%; profiler sums are not graph latency; first sentinel reading per session is 5-8% cold | Trust only paired, graph-timed benchmarks |
| T17 | Recompilation started before a local edit synced to the pod | Check source SHA256 before and after each build |
| T18 | Unfiltered racecheck reports hazards in PyTorch's FP64 oracle reductions; `regex:` filter syntax rejected | Filter with `kns=<kernel>` to own kernels |
| T19 | A wrapper's runtime index math changed the control's registers (231 -> 235), corrupting a screen | Compile-time specialization; load the immutable control binary |
| T20 | Harness column `event_speedup_ref_vs_provider` means provider/reference, not old/new | Read the convention before quoting |

## 7. Open ideas not yet tried

| Idea | Expected payoff (as stated in source) | Source |
|---|---|---|
| Fuse the four R/S gradients (and maybe dQ) into fewer kernels sharing staged tiles; warp-specialized producer/consumer with deferred reductions | Needed to close 512x32 gap: our 4-gradient R/S (37.5 ms) already exceeds FBGEMM's whole backward (36.3 ms). "Large systems effort, uncertain payoff" | 512x32 `STATUS.md`, `ATTEMPTS.md` 17, 20 |
| Accurate wide dQ tensor-core formulation for w128 | Only remaining accuracy blocker for compensated 128x128 | `w128_optimization/RESULTS.md` |
| Independent (W1,W2) template dims, tiled online softmax, asymmetric shared-gradient schedule | Required for any valid wider/asymmetric claim | `window_shapes/BLOCKERS.md` |
| TMA (Hopper bulk async copy), deeper async overlap, 2-CTA cluster raw-tile sharing (E6) | Not stated; admit only if counters show the cost they remove | `NEXT_EXPERIMENTS.md` |
| Forward: still loses 5/9 w32 cells; graph diagnostic had G2 0.138 ms vs FBGEMM 0.086 ms at N256 | But removing forward entirely gives only ~1.14x complete | `NEXT_EXPERIMENTS.md`, `forward_next_steps.md` |
| Exact cover of the sloped (i,k) band with legal tensor-core rectangles, chosen by a cost model | At w32 masked-cell removal alone bounds R/S at 1.559x (~1.26x total) | `RESEARCH_ASSESSMENT.md` 1; bound from `NEXT_EXPERIMENTS.md` 2 |
| Bounded-storage chunked head ownership for KV gradients | Memory; latency gain not assumed | `RESEARCH_ASSESSMENT.md` 2 |
| Integrate the compensated precision path into the public API | Fixes stress failures at 12-17% cost | `precision/ATTEMPTS.md` |
| D=64 window tuning; 1-warp R/S row blocks for w<=32; wrapper/allocation (V5); CUDA graphs for the independent-head path | Exp 2: D64 w32 loses 0.80x to tuned paper | exp2 `RESULTS.md` |

Note: the flash-style direction-0 R/S rewrite (accumulate G[k,d] over query subtiles) was listed as a remaining 512x32 lever in an earlier summary, but `research_512x32_20260917/ATTEMPTS.md` item 20 (later) retracts the transposed factorization as not helpful. Re-read item 20 before trying it.

## 8. Sources

These detailed reports, the raw benchmark data and the zipped source snapshots of every milestone (P02-P07, R26, Phase 1) were moved out of the repo on 2026-09-22 to `~/att3ntion_archive_20260922/`. `benchmarks/...` paths below are under its `single-gather-night/` folder, and `~/att3ntion/...` paths under its `main-checkout/` folder:

- `benchmarks/results/single_gather/night_20260909/RESULTS.md` (exp 1)
- `benchmarks/results/single_gather/exp2_20260910/RESULTS.md` (exp 2)
- `benchmarks/shared_kv_window_quick/REPORT.md`
- `benchmarks/tight_tiles/README.md`, `progress_summary_20260915.md`, `phase1_h100.md`
- `benchmarks/tight_tiles/rs_iterations_20260915/ATTEMPTS.md`
- `benchmarks/tight_tiles/overnight_20260915/LEDGER.md`, `RESULTS.md`, `BUILD_AND_USE.md`, and `forward/`, `dq/`, `rs/`, `fusion/`, `precision/`, `tape/` `*ATTEMPTS.md`
- `benchmarks/tight_tiles/next_speedup_20260915/NEXT_EXPERIMENTS.md`, `research_audit_20260915/RESEARCH_ASSESSMENT.md`
- `~/att3ntion/overnight_20260917/{w128_optimization,window_shapes,w128_task_compare}/`
- `~/att3ntion/research_512x32_20260917/{README,STATUS,RESULTS,ATTEMPTS}.md`
- `~/att3ntion/overnight_20260916/package/MORNING.md` (compensated package)
