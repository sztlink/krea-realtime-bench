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
