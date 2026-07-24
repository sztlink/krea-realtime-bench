"""L4 bench: the 1.3B W4A4 in the real causal loop on the 4090.

Same harness as bench_l1.py, with the L3 checkpoint swapped in via
nunchaku_causal_wan.load_w4a4_blocks. Measures fps/memory, saves latent
trajectories (3 seeds kv3 + the global window) and last frames as the L5 inputs.
Latents are compared against the bf16 L0 references with coarse stats only
(the fidelity verdict is L5's job, by trajectory, not here).
"""
import os, json, time, asyncio, traceback
from pathlib import Path
os.environ.setdefault("DO_COMPILE", "false")
import torch

OUT = Path("results_w4a4"); OUT.mkdir(exist_ok=True)
PROMPT = "A person dancing in an empty warehouse, dramatic lighting, camera static"
CONFIG_PATH = "configs/self_forcing_server.yaml"
QDIR = "ptq_out/self-forcing-1p3b-w4a4"
RES = {"meta": {}, "runs": []}
def flush(): (OUT / "results.json").write_text(json.dumps(RES, indent=1, default=str))
def gb(x): return round(x / 1e9, 3)
def mem():
    free, total = torch.cuda.mem_get_info()
    return {"torch_alloc_gb": gb(torch.cuda.memory_allocated()), "device_used_gb": gb(total - free)}

from release_server import load_merge_config, load_transformer, \
    load_vae, load_pipeline, GenerateParams, GenerationSession, Models
from wan.modules.causal_model import CausalWanModel
from nunchaku_causal_wan import load_w4a4_blocks
def _from_config(path, **kw):
    return CausalWanModel.from_config(CausalWanModel.load_config(str(path)), **kw)
CausalWanModel.from_pretrained = staticmethod(_from_config)

config = load_merge_config(CONFIG_PATH)
RES["meta"] = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
               "checkpoint": QDIR}
t0 = time.time(); transformer = load_transformer(config)
load_w4a4_blocks(transformer.model, QDIR)
torch.cuda.synchronize(); torch.cuda.empty_cache()
RES["meta"]["load"] = {"sec": round(time.time() - t0, 1), **mem()}
RES["meta"]["param_count"] = sum(p.numel() for p in transformer.parameters())

import re as _re
_slug = _re.sub(r"[^a-z0-9]+", "-", PROMPT.lower())[:60].strip("-")
_emb = torch.load(f"embeddings/{_slug}.pt", map_location="cpu", weights_only=True)
_cond = {"prompt_embeds": _emb["prompt_embeds"].to("cuda")}
class _StaticEnc(torch.nn.Module):
    def forward(self, text_prompts): return {k: v.clone() for k, v in _cond.items()}
text_encoder = _StaticEnc()
vae_encoder, vae_decoder = load_vae()
pipeline = load_pipeline(config, torch.cuda.current_device(), transformer, text_encoder, vae_decoder)
models = Models(text_encoder, transformer, pipeline, vae_encoder, vae_decoder)
flush(); print("LOAD OK", json.dumps(RES["meta"], default=str))

def run_cfg(tag, kv_frames=3, steps=4, seed=42, num_blocks=9, save_latents=False):
    print(f"===== RUN {tag} =====")
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    params = GenerateParams(prompt=PROMPT, seed=seed, kv_cache_num_frames=kv_frames,
                            num_blocks=num_blocks, num_denoising_steps=steps)
    n = {"n": 0}; frames = []
    def cb(pixels, frame_ids, event):
        event.synchronize(); n["n"] += pixels.shape[1]
        frames.append(pixels[0, -1].float().cpu())
    rec = {"tag": tag, "kv": kv_frames, "steps": steps, "seed": seed, "ok": False}
    try:
        session = GenerationSession(params, config, frame_callback=cb, models=models)
        block_times = []
        torch.cuda.synchronize(); t_all = time.time()
        for i in range(num_blocks):
            tb = time.time()
            try: session.generate_block(models)
            except asyncio.CancelledError: break
            torch.cuda.synchronize(); block_times.append(round(time.time() - tb, 3))
        wall = time.time() - t_all
        steady_wall = sum(block_times[1:]) if len(block_times) > 1 else wall
        latents = session.all_latents[:, :session.current_start_frame].cpu()
        rec.update({"ok": True, "wall_sec": round(wall, 2), "pixel_frames": n["n"],
            "fps_e2e": round(n["n"] / wall, 2),
            "fps_steady": round(max(0, n["n"] - 6) / steady_wall, 2) if steady_wall else None,
            "peak_alloc_gb": gb(torch.cuda.max_memory_allocated()),
            "latents_finite": bool(torch.isfinite(latents).all()), **mem()})
        if save_latents:
            torch.save(latents, OUT / f"latents_{tag}.pt")
            ref_path = Path("results_1p3b") / f"latents_{tag}.pt"
            if ref_path.exists():
                ref = torch.load(ref_path, map_location="cpu", weights_only=True)
                if ref.shape == latents.shape:
                    d = (latents.float() - ref.float())
                    rec["vs_bf16"] = {
                        "rmse": round(d.pow(2).mean().sqrt().item(), 4),
                        "rel": round((d.pow(2).mean() / ref.float().pow(2).mean()).sqrt().item(), 4),
                        "cos": round(torch.nn.functional.cosine_similarity(
                            latents.flatten().float(), ref.flatten().float(), dim=0).item(), 4)}
        if frames:
            from PIL import Image
            arr = ((frames[-1].clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).numpy()
            Image.fromarray(arr).save(OUT / f"frame_{tag}_last.png")
        session.dispose()
    except Exception:
        rec["error"] = traceback.format_exc(); torch.cuda.empty_cache()
    RES["runs"].append(rec); flush()
    print(json.dumps({k: v for k, v in rec.items()}, default=str)[:600])

run_cfg("kv3_s4_warmup", 3, 4)
run_cfg("kv3_s4", 3, 4, save_latents=True)
run_cfg("kv3_s4_seed43", 3, 4, seed=43, save_latents=True)
run_cfg("kv3_s4_seed44", 3, 4, seed=44, save_latents=True)
run_cfg("kv21_s4", 21, 4, save_latents=True)
print("DONE")
