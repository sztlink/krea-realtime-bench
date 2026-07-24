"""L1 bench: fuse_projections dedup patch on the Self-Forcing 1.3B (4090).

Measures load memory + param count after the patch, runs kv3_s4 (warmup + timed),
and compares latents bit-for-bit against the L0 baseline trajectory
(results_1p3b/latents_kv3_s4.pt) — the compute path is unchanged, so any
difference means the patch broke something. Results in results_l1/results.json.
"""
import os, json, time, asyncio, traceback
from pathlib import Path
os.environ.setdefault("DO_COMPILE", "false")
import torch

OUT = Path("results_l1"); OUT.mkdir(exist_ok=True)
PROMPT = "A person dancing in an empty warehouse, dramatic lighting, camera static"
CONFIG_PATH = "configs/self_forcing_server.yaml"
RES = {"meta": {}, "runs": []}
def flush(): (OUT / "results.json").write_text(json.dumps(RES, indent=1, default=str))
def gb(x): return round(x / 1e9, 3)
def mem():
    free, total = torch.cuda.mem_get_info()
    return {"torch_alloc_gb": gb(torch.cuda.memory_allocated()), "device_used_gb": gb(total - free)}

from release_server import load_merge_config, load_transformer, \
    load_vae, load_pipeline, GenerateParams, GenerationSession, Models
from wan.modules.causal_model import CausalWanModel
def _from_config(path, **kw):
    return CausalWanModel.from_config(CausalWanModel.load_config(str(path)), **kw)
CausalWanModel.from_pretrained = staticmethod(_from_config)

config = load_merge_config(CONFIG_PATH)
RES["meta"] = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
               "patch": "fuse_projections dedup (del q/k/v after fusion)"}
t0 = time.time(); transformer = load_transformer(config); torch.cuda.synchronize()
torch.cuda.empty_cache()
RES["meta"]["load_transformer"] = {"sec": round(time.time() - t0, 1), **mem()}
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
    n = {"n": 0}
    def cb(pixels, frame_ids, event):
        event.synchronize(); n["n"] += pixels.shape[1]
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
        rec.update({"ok": True, "wall_sec": round(wall, 2), "pixel_frames": n["n"],
            "fps_e2e": round(n["n"] / wall, 2),
            "fps_steady": round(max(0, n["n"] - 6) / steady_wall, 2) if steady_wall else None,
            "peak_alloc_gb": gb(torch.cuda.max_memory_allocated()), **mem()})
        latents = session.all_latents[:, :session.current_start_frame].cpu()
        if save_latents:
            torch.save(latents, OUT / f"latents_{tag}.pt")
            ref_path = Path("results_1p3b") / f"latents_{tag}.pt"
            if ref_path.exists():
                ref = torch.load(ref_path, map_location="cpu", weights_only=True)
                rec["latents_shape"] = list(latents.shape)
                rec["ref_shape"] = list(ref.shape)
                if latents.shape == ref.shape:
                    rec["latents_bit_identical"] = bool(torch.equal(latents, ref))
                    rec["latents_max_abs_diff"] = float((latents.float() - ref.float()).abs().max())
        session.dispose()
    except Exception:
        rec["error"] = traceback.format_exc(); torch.cuda.empty_cache()
    RES["runs"].append(rec); flush()
    print(json.dumps(rec, default=str)[:900])

run_cfg("kv3_s4_warmup", 3, 4)
run_cfg("kv3_s4", 3, 4, save_latents=True)
print("DONE")
