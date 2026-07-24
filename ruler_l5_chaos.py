"""L5 chaos-floor control: bf16 with a 0.1% perturbation injected once, first
forward only. Its divergence curve against stock bf16 is the sampler's chaotic
amplification floor. If the W4A4 curve sits near it, trajectory divergence cannot
distinguish quantization from any epsilon perturbation, and the fidelity verdict
must rest on content-level receipts instead."""
import os, json, asyncio
from pathlib import Path
os.environ.setdefault("DO_COMPILE", "false")
import torch

OUT = Path("results_l5"); OUT.mkdir(exist_ok=True)
PROMPT = "A person dancing in an empty warehouse, dramatic lighting, camera static"
CONFIG_PATH = "configs/self_forcing_server.yaml"

from release_server import load_merge_config, load_transformer, \
    load_vae, load_pipeline, GenerateParams, GenerationSession, Models
from wan.modules.causal_model import CausalWanModel
def _from_config(path, **kw):
    return CausalWanModel.from_config(CausalWanModel.load_config(str(path)), **kw)
CausalWanModel.from_pretrained = staticmethod(_from_config)

config = load_merge_config(CONFIG_PATH)
transformer = load_transformer(config)
import re as _re
_slug = _re.sub(r"[^a-z0-9]+", "-", PROMPT.lower())[:60].strip("-")
_emb = torch.load(f"embeddings/{_slug}.pt", map_location="cpu", weights_only=True)
_cond = {"prompt_embeds": _emb["prompt_embeds"].to("cuda")}
class _StaticEnc(torch.nn.Module):
    def forward(self, text_prompts): return {k: v.clone() for k, v in _cond.items()}
vae_encoder, vae_decoder = load_vae()
pipeline = load_pipeline(config, torch.cuda.current_device(), transformer, _StaticEnc(), vae_decoder)
models = Models(_StaticEnc(), transformer, pipeline, vae_encoder, vae_decoder)

CALL = {"n": 0}
orig = models.transformer.forward
def _jitter(t, gen_seed):
    g = torch.Generator(device=t.device).manual_seed(gen_seed)
    noise = torch.randn(t.shape, generator=g, device=t.device, dtype=torch.float32)
    return t + 1e-2 * t.float().std() * noise.to(t.dtype)

def perturbed(*a, **k):
    out = orig(*a, **k)
    if CALL["n"] == 0:
        # the loop unpacks `_, denoised_pred = transformer(...)`: perturb every tensor
        if isinstance(out, tuple):
            out = tuple(_jitter(t, 777 + i) if torch.is_tensor(t) else t
                        for i, t in enumerate(out))
        else:
            out = _jitter(out, 777)
        print("eps injected on first forward")
    CALL["n"] += 1
    return out
models.transformer.forward = perturbed

params = GenerateParams(prompt=PROMPT, seed=42, kv_cache_num_frames=3,
                        num_blocks=9, num_denoising_steps=4)
def cb(pixels, frame_ids, event): event.synchronize()
session = GenerationSession(params, config, frame_callback=cb, models=models)
for _ in range(9):
    try: session.generate_block(models)
    except asyncio.CancelledError: break
torch.cuda.synchronize()
lat = session.all_latents[:, :session.current_start_frame].cpu()
torch.save(lat, OUT / "latents_bf16_eps1e2_s42.pt")

ref = torch.load("results_1p3b/latents_kv3_s4.pt", map_location="cpu", weights_only=True).float()
d = (lat.float() - ref).flatten(2)
rel = (d.pow(2).mean(-1) / ref.flatten(2).pow(2).mean(-1).clamp_min(1e-12)).sqrt()[0]
print("chaos floor rel per frame:", [round(x, 3) for x in rel.tolist()])
(OUT / "chaos_floor.json").write_text(json.dumps({"eps": 1e-2, "rel": rel.tolist()}))
