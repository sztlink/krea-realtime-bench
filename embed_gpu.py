"""Prompt embeddings on the GPU, in seconds instead of minutes.

Why the old path was slow. `make_embedding.py` runs UMT5-XXL on the CPU, because the
stock WanTextEncoder builds it in fp32 on the GPU at 22.7 GB, which did not fit beside
the 14B W4A4. Each prompt cost three to five minutes, and that throttled iteration exactly
where it matters most, which is writing the prompt.

Why it fits now. The weight file is ALREADY bf16 (`...-enc-bf16.safetensors`) and in bf16
the encoder takes about 11.4 GB, which leaves room beside the 1.3B at 2.84 GB. Load once,
encode the whole list, save everything.

Uso:
  .venv/bin/python embed_gpu.py "prompt um" "prompt dois" ...
  python embed_gpu.py --arquivo prompts.txt        # one prompt per line, or name|prompt
"""
import os, re, sys, time, torch
from pathlib import Path
from wan.modules.t5 import umt5_xxl
from wan.modules.tokenizers import HuggingfaceTokenizer
from settings import MODEL_FOLDER
from safetensors.torch import load_file as safe_load_file

args = sys.argv[1:]
prompts = []
if args and args[0] == "--arquivo":
    for linha in Path(args[1]).read_text().splitlines():
        linha = linha.strip()
        if not linha or linha.startswith("#"):
            continue
        prompts.append(linha.split("|", 1)[1] if "|" in linha else linha)
else:
    prompts = args
if not prompts:
    print("uso: embed_gpu.py \"prompt\" [...]  |  embed_gpu.py --arquivo lista.txt")
    sys.exit(1)

out = Path("embeddings"); out.mkdir(exist_ok=True)
faltando = []
for p in prompts:
    slug = re.sub(r"[^a-z0-9]+", "-", p.lower())[:60].strip("-")
    if (out / f"{slug}.pt").exists():
        print(f"ja existe  {slug}")
    else:
        faltando.append((slug, p))
if not faltando:
    print("nada a fazer")
    sys.exit(0)

t0 = time.time()
dev = torch.device("cuda")
# Build on the CPU and only then move. `umt5_xxl` ignores the dtype argument and builds
# in fp32, so building straight on the device would cost the 22.7 GB that drove the
# original CPU path. On the CPU the peak is host RAM, and what reaches the card is bf16.
enc = umt5_xxl(encoder_only=True, return_tokenizer=False, dtype=torch.bfloat16,
               device=torch.device("cpu")).eval().requires_grad_(False)
sd = safe_load_file(os.path.join(MODEL_FOLDER, "Wan2.1-T2V-1.3B",
                                 "models_t5_umt5-xxl-enc-bf16.safetensors"), device="cpu")
enc.load_state_dict(sd)
del sd
enc = enc.to(torch.bfloat16).to(dev)
tok = HuggingfaceTokenizer(name=os.path.join(MODEL_FOLDER, "Wan2.1-T2V-1.3B", "google", "umt5-xxl/"),
                           seq_len=512, clean="whitespace")
print(f"T5 na GPU em {time.time()-t0:.1f}s, {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

with torch.no_grad():
    for slug, p in faltando:
        t = time.time()
        ids, mask = tok([p], return_mask=True, add_special_tokens=True)
        ids, mask = ids.to(dev), mask.to(dev)
        seq_lens = mask.gt(0).sum(dim=1).long()
        emb = enc(ids, mask)
        # ZERO the padding. The tokenizer pads to 512 positions and the encoder produces
        # values at them. The model attends over all 512, so garbage in the padded region
        # dominates conditioning and the output turns to mush. The original
        # make_embedding.py does this, and skipping it costs an afternoon.
        for u, v in zip(emb, seq_lens):
            u[v:] = 0.0
        torch.save({"prompt_embeds": emb.to(torch.bfloat16).cpu(), "prompt": p},
                   out / f"{slug}.pt")
        print(f"EMB-OK {slug}.pt  {time.time()-t:.2f}s", flush=True)
print("DONE", flush=True)
