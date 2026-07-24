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
