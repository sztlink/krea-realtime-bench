# krea-realtime-bench

Measurements of [Krea Realtime 14B](https://github.com/krea-ai/realtime-video), the open autoregressive video model distilled from Wan 2.1 with Self-Forcing. Vendor numbers are self-reported. These are one lab, one rented H100, one afternoon, and the exact script that produced every number, so the sample size problem has a fix and the fix is you running it.

Method note. The bench does not fork upstream. It instruments the stock code path at runtime (wraps the forward, wraps the cache rebuild, swaps `from_pretrained` for a config-only load that skips a 57 GB download). What gets measured is what they ship.

The model generates video the way an LLM decodes text. Blocks of 3 latent frames, 4 denoising steps per block, and a literal LLM-style KV cache per transformer layer. Post-RoPE keys, contiguous append, sink tokens, rolling eviction. That makes every KV-cache technique from the LLM world applicable to real-time video, and this repo is the measurement floor for that work.

## Numbers (H100 SXM 80GB, 2026-07-24)

Stack. torch 2.8.0, eager (no compile), SageAttention 2.2.1, bf16, 832x480, 9 blocks per run, fixed prompt. Full raw data in `results/`.

| config | fps steady | prefill/block | denoise/forward | peak VRAM |
|---|---|---|---|---|
| kv window 3, 4 steps (the claimed config) | **5.71** | 342 ms | 366 ms | 49.6 GB |
| kv window 3, 5 steps (repo default) | 3.79 | 342 ms | 376 ms | 61.0 GB |
| kv window 6, 4 steps | 3.81 | 714 ms | 398 ms | 55.0 GB |
| kv window 12, 4 steps | 2.55 | 1590 ms | 449 ms | 66.4 GB |
| kv window 21 (global), 4 steps | **OOM** | — | — | >79 GB |

Reading the table. Krea claims 11 fps on a B200 with 4 steps and compile. At roughly 2x the hardware plus compile, that claim is plausible. Multi-seed variance on our runs stays under 1%.

The stability mechanism is the expensive part. The server zeroes and rebuilds the whole KV cache every block with one extra forward over the clean context window. That cost grows linearly with the window and is paid every block, so total cost is quadratic in video length. bf16 KV costs 1.28 GB per latent frame across the 40 layers, and the 21-frame window the offline path uses does not fit in 80GB with the rest of the stack resident.

Two more findings worth your VRAM.

- The official server calls `fuse_projections()` and keeps the unfused q/k/v Linears alive. 17.4B parameters on the GPU for a 14.1B model. That is 6.6 GB doing nothing.
- fp8 weights via the repo's own torchao path halve weight memory (34.9 to 17.4 GB) and cost about 6% fps in eager. Trajectories for both are saved as fidelity references.

## Reproduce it

Requirements. One CUDA GPU with 48GB+ for the small windows (80GB+ for window 12, more than 80GB for window 21), ~60GB disk, python via uv.

```bash
git clone https://github.com/krea-ai/realtime-video
cd realtime-video

# 1. their uv.lock breaks on the clip package with setuptools>=81 build isolation. Fix first:
#    add under [tool.uv.extra-build-dependencies] in pyproject.toml:  clip = ["setuptools<81"]
uv sync
uv pip install libs/sageattention-2.2.1-cp311-cp311-linux_x86_64.whl   # x86_64 only

# 2. models. Note that huggingface-cli is dead in current hub, use `hf`.
hf download Wan-AI/Wan2.1-T2V-1.3B --include "Wan2.1_VAE.pth" --local-dir wan_models/Wan2.1-T2V-1.3B
hf download Wan-AI/Wan2.1-T2V-1.3B --include "google/*" --local-dir wan_models/Wan2.1-T2V-1.3B
hf download Wan-AI/Wan2.1-T2V-1.3B --include "config.json" --local-dir wan_models/Wan2.1-T2V-1.3B
hf download Wan-AI/Wan2.1-T2V-14B --include "config.json" --local-dir wan_models/Wan2.1-T2V-14B
hf download krea/krea-realtime-video krea-realtime-video-14b.safetensors --local-dir checkpoints

# 3. the T5 encoder their code loads exists in no public repo (official Wan ships a .pth,
#    Krea's HF ships no T5 at all). Download the .pth and convert:
curl -L -C - -o wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth \
  https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/models_t5_umt5-xxl-enc-bf16.pth
cp ../krea-realtime-bench/convert_t5.py . && uv run python convert_t5.py

# 4. run
echo "MODEL_FOLDER=wan_models" > .env
cp ../krea-realtime-bench/bench_m0b.py .
uv run python bench_m0b.py          # full sweep, ~10 min of GPU after load
cp ../krea-realtime-bench/bench_fp8.py .
uv run python bench_fp8.py          # fp8 pass, fresh process (a reload in-process OOMs)
```

The bench needs no Wan-14B weights. It instantiates the model from `config.json` and loads the Krea checkpoint on top, which saves you a 57 GB download.

### DGX Spark / GB10 owners, this is for you

A Spark's 128GB unified memory should run the full 21-frame global window that OOMs on an 80GB H100. Nobody has that number yet. Caveats we already know. The uv.lock and the sage wheel are x86_64, so on ARM use NVIDIA's pytorch container, install the deps by hand, and let attention fall back to SDPA. Memory numbers and the recompute curve stay comparable, absolute fps does not, and the receipt is valuable either way.

## The full window curve (H200 141GB, same day)

The H100 hit its wall at the 21-frame window, so we rented the bigger box and finished the curve. Steady-state prefill per block (median of settled blocks, first-touch compile excluded), `bench_kv21.py`, raw data in `results/h200-sxm-2026-07-24/`.

| window (latent frames) | prefill/block | KV cache bf16 | peak alloc | fps steady |
|---|---|---|---|---|
| 3 | 338 ms | 7.7 GB | 49.6 GB | 4.40 |
| 12 | 1592 ms | 19.2 GB | 62.6 GB | 2.38 |
| 15 | 2046 ms | 23.0 GB | 77.9 GB | 2.76 |
| 18 | 2567 ms | 26.8 GB | 85.5 GB | 2.52 |
| 21 | ~2840 ms | 30.7 GB | **93.2 GB** | 2.37 |
| 24 | refused | 34.5 GB allocated | — | — |

- Cross-node sanity holds. Prefill at window 3 and 12 reproduces the H100 numbers within 1% (338 vs 342 ms, 1592 vs 1590 ms). Denoise per forward ran ~14% slower on this H200 node, so treat absolute fps across nodes with care, the curve shape is the finding.
- Linearity confirmed across the whole range, roughly 130 to 143 ms of prefill per context frame, paid every block.
- The 21-frame window peaks at **93.2 GB**. That is why an 80GB H100 OOMs, and the real bar for the full window in bf16.
- **Window 24 is refused by the shipped code.** `WanDiffusionWrapper` hardcodes `seq_len = 32760` (exactly 21 frames at 832x480) and an assert fires in the prefill forward. The "global" window is not just expensive, it is the architectural ceiling of the release. Raising it is a one-line patch, the memory bill afterwards is not.

## The pipeline proxy on a consumer GPU (RTX 4090, same day)

Making the 14B fit a 24GB card takes W4A4 quantization, and the last port of this kind burned about $70 of pod time debugging the pipeline at cloud prices. So the quantization stage gets built and proven on Self-Forcing 1.3B first. Same architecture, same causal loop, same server code, on a local 4090 where every iteration is free. Scripts `bench_1p3b.py` and `make_embedding.py`, raw data in `results/rtx4090-2026-07-24/`.

| config | fps steady | prefill/block | peak VRAM |
|---|---|---|---|
| kv window 3, 4 steps | **7.5** | 161 ms | 11.6 GB |
| kv window 6, 4 steps | 5.4 | 375 ms | 12.6 GB |
| kv window 12, 4 steps | 3.8 | 886 ms | 14.3 GB |
| kv window 21 (global), 4 steps | 2.88 | 2366 ms | **23.6 GB** |

- SDPA eager, bf16, 832x480, 9 blocks, multi-seed variance under 1% on repeat seeds (7.50 and 7.49). Latent trajectories for 3 seeds are saved as fidelity references for the quantized runs to come.
- The recompute curve reproduces the 14B shape. The 1.3B is a faithful laboratory for the full model.
- The global window that OOMs an 80GB H100 at 14B scale fits a 24GB card at 1.3B, barely. Its prefill carries a ~6GB allocation transient and needs `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to survive on 24GB.
- Two release walls nobody documented. The stock T5 loader builds UMT5-XXL in fp32 directly on the GPU, 22.7 GB before any video model loads, so the release cannot even initialize on a 24GB card. `make_embedding.py` computes the prompt embedding on CPU and the T5 never touches VRAM. And the SageAttention wheel shipped in the repo has no sm89 kernels, it imports clean on a 4090 and explodes at runtime, so consumer Ada needs a source build or falls back to SDPA.

### The 6.6 GB fix, measured

The unfused q/k/v finding from the H100 run is now `fuse-dedup.patch`, three lines that drop the original projections after fusion. On the 1.3B it frees 0.44 GB and 212M duplicate parameters, latents stay bit identical under `torch.equal`, fps stays put. Scaled to the 14B geometry that is the 6.6 GB from the finding above. `bench_l1.py` reproduces the measurement. The bench itself keeps instrumenting stock code, the patch is a proposed fix headed upstream.

### A calibration collector that survives the causal loop

W4A4 calibration wants activation statistics from the loop as it really runs. Stock PTQ collectors cache the whole transformer input per step and replay the model later, and a causal server carries gigabytes of mutable KV state inside those very inputs, so replay is not an option here. `collect_wan.py` inverts the flow. Hooks capture the input of every quantization target during real generation, denoise steps and cache recompute both, tagged per call with phase and timestep. One 48 second pass on the 4090 covers the full schedule, 27 calls at each of the 4 denoising timesteps plus 24 recompute passes at t=0, and writes exact channelwise stats plus stratified token reservoirs for every target in every block. The stats already draw the quantization map. Attention and FFN inputs smooth well, the FFN down projection carries post-GELU spikes up to 835x the channel mean, and the cross attention outputs are small enough to just stay bf16. Reports in `results/rtx4090-2026-07-24/` (`collect_summary.json`, `timestep_histogram.json`, `outlier_report.json`), the 2.6 GB of raw token reservoirs travel on request.

### The first W4A4 checkpoint, calibrated from the loop

The collector's reservoirs fed an SVDQuant INT4 pass over all 30 blocks, with DeepCompressor used as a library and its conventions honored end to end. Smooth folded into the weights, a rank 32 low-rank branch, symmetric int4 in groups of 64, and the final rounding done by the stock nunchaku packer, which accepted all 1260 tensors on the first try. The whole calibration took 163 seconds on the 4090. `ptq_wan.py` runs it, `convert_wan.py` emits the two-file nunchaku checkpoint. 0.74 GB of quantized blocks plus 0.28 GB kept in bf16, against 2.6 GB for the same blocks unquantized.

Per-module simulated W4A4 output error against the real-loop reservoirs tells an inverted story. The cross attention output projection, the stream with the wildest outlier channels, came out cleanest at 5.2% median, so it went into the artifact after all. The FFN down projection is the honest worst at 15.7% median, and an ablation on its worst block splits the blame. With fp activations the error drops to 9.4%, with fp weights it only drops to 14.7%, so the activation side dominates, the rank 32 branch buys just 1.4 points there, and the named next lever is the unsigned activation shift the upstream recipe already supports, not a bigger rank and not GPTQ. Raw numbers in `results/rtx4090-2026-07-24/` (`ptq_report.json`, `ablate_ffn2.json`).

### The W4A4 runtime, running

The port is deliberately small. Only the calibrated Linears become `SVDQW4A4Linear`, and the attention math, RoPE, norms, KV cache and server loop stay stock (`nunchaku_causal_wan.py`, loaded from the two-file checkpoint with the key renames the other nunchaku ports use). Before touching the loop, a unit test replays the captured mid-generation fixtures through the real sm89 kernel and compares against the bf16 reference. The kernel lands within a point of the calibration simulation on every slot (`test_w4a4_fixtures.py`), so the whole chain from collector to kernel is numerically coherent.

Then the loop itself, `bench_w4a4.py`, raw data in `results/rtx4090-2026-07-24/results_w4a4.json`.

| config | fps steady | peak VRAM | bf16 reference |
|---|---|---|---|
| kv window 3, 4 steps | **8.8** | 9.4 GB | 7.5 fps, 11.1 GB |
| kv window 21 (global), 4 steps | **4.8** | 21.4 GB | 2.88 fps, 23.6 GB |

- 17% faster on the small window, 67% faster on the global one. The recompute pass is GEMM-heavy, which is exactly what W4A4 accelerates, while the small-window frame is dominated by SDPA. Transformer weights after load are 1.07 GB against 2.84 GB in bf16.
- Latents stay finite across 3 seeds and both windows, and the frames are coherent video (the dancer, the warehouse, the light). Last frames for both windows in `frames/`.
- The W4A4 trajectory diverges from the bf16 one, as 27 autoregressive latent frames of few-percent per-module error must. Divergence is not a quality verdict. The fidelity ruler below is.

### The fidelity ruler, and why trajectory closeness is the wrong ask

Before judging the W4A4 by its distance to the bf16 trajectory, we measured what that distance means. A 0.1% perturbation injected once into a single bf16 forward gets erased by the arithmetic itself, because bf16 resolves about 0.4% per element. A 1% perturbation injected once grows 48.7x over 27 autoregressive frames and heads toward the distance between two different seeds (`ruler_l5_chaos.py`, `chaos_floor.json`). The sampler is chaotic, so no sustained per-step error, including bf16's own rounding, can hold a trajectory close. Any ruler that demands it fails everything.

So the ruler asks what can actually be held. Per-frame divergence against the bf16 reference stays below the distance between two valid videos for the whole clip (0.60x that control early, 0.88x late, 3 seeds). Cross-seed diversity inside the W4A4 model matches the bf16 model at a 0.89 ratio, so the model did not collapse. Latent statistics track bf16 within 10%. A double-length run of 18 chunks holds a settled std slope of 0.0071 per frame against bf16's own 0.0080, meaning the rise is the video gaining motion, not the quantization drifting. And the frame strips stay coherent to the last frame of the double-length run (`frames/strip_bf16_vs_w4a4.png`, rows bf16, W4A4, W4A4 at 18 chunks). All raw numbers in `ruler.json`, scripts `ruler_l5_runs.py` and `ruler_l5_metrics.py`.

Verdict. The W4A4 makes a different video of the same prompt with the same stability, the same diversity and the same statistics as bf16, at 8.8 fps where bf16 gives 7.5. On this model the pipeline holds. The 14B is next, and it needs one A100 pass, not faith.

## The A100 pass, and the proxy paying off (same day)

One rented A100 80GB ran the proven chain against the 14B. Collect on the real loop took 8 minutes, calibration of all 40 blocks took 22.5 minutes, and the stock packer accepted every tensor again. The whole pass cost about $3.60 including a dead community host that billed 35 minutes without ever booting its container. Scripts in `cloud/setup_c.sh` and `cloud/driver_c.sh`, raw numbers in `results/a100-2026-07-24/`.

The receipt that matters is the transfer. Per-stream simulated W4A4 output error, median across blocks, small model against big model.

| stream | 1.3B | 14B |
|---|---|---|
| self qkv | 0.081 | 0.081 |
| self out | 0.122 | 0.128 |
| cross q | 0.120 | 0.079 |
| cross out | 0.052 | 0.048 |
| ffn up | 0.093 | 0.091 |
| ffn down | 0.157 | 0.158 |

The 1.3B predicted the 14B's calibration behavior stream by stream, with the qkv median matching to three decimals. The post-GELU outliers run twice as wild at 14B scale, one channel at 1644x its mean against 835x on the small model, and the smoothing plus the low-rank branch absorb them to the same final error. Debugging on the small model and spending cloud money only on the scaled pass is the whole method, and this table is what it bought.

The converted checkpoint weighs 6.6 GB of quantized blocks plus 4.2 GB kept in bf16, against roughly 25 GB for the same blocks unquantized. Whether it generates on a 24 GB consumer card is the next receipt, and it runs on hardware we own.

## Limitations, stated before you find them

- One prompt (a dancer in a warehouse), one resolution (832x480, the causal path hardcodes its RoPE for it), one GPU class, one day. The prompt is now a flag (`--prompt` or `BENCH_PROMPT`), so widen it.
- Eager only, no `DO_COMPILE`. The compiled number would be higher and less comparable across boxes. The memory numbers and the recompute curve do not depend on it.
- fps here is end-to-end through VAE decode and frame callback, not transformer-only marketing fps.
- n=1 on hardware. That is the whole point of the receipt format below.

## Send back a receipt

Open an issue (or bring it to the Waffle House) with

- `results_m0b/results.json` (the script writes it, includes GPU, torch, per-run timings, peak memory)
- the console log (it records which attention backend loaded)
- anything that broke, with the exact error. Friction reports are receipts too.

## Lineage and what comes next

This is the measurement floor for making the 14B run on consumer GPUs. Next steps live downstream. W4A4 weights via SVDQuant (same pipeline as the [nunchaku Krea 2 port, PR #947](https://github.com/nunchaku-ai/nunchaku/pull/947)), then 4-bit KV cache as the cheap alternative to quadratic recompute, then a trajectory-based fidelity benchmark, all open.

Code here is MIT. The Krea Realtime 14B weights are CC-BY-NC-SA, their license travels with them.
