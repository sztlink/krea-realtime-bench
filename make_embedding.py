"""Build the prompt embedding on CPU, once. The stock WanTextEncoder constructs
UMT5-XXL on GPU in fp32 (22.7 GB), which no 24GB card can hold. This sidesteps
it entirely. Output: embeddings/<slug>.pt with the bf16 prompt_embeds tensor."""
import os, sys, re, torch
from pathlib import Path
from wan.modules.t5 import umt5_xxl
from wan.modules.tokenizers import HuggingfaceTokenizer
from settings import MODEL_FOLDER

PROMPT = os.environ.get("BENCH_PROMPT", "A person dancing in an empty warehouse, dramatic lighting, camera static")
slug = re.sub(r"[^a-z0-9]+", "-", PROMPT.lower())[:60].strip("-")
out = Path("embeddings"); out.mkdir(exist_ok=True)

enc = umt5_xxl(encoder_only=True, return_tokenizer=False, dtype=torch.float32,
               device=torch.device("cpu")).eval().requires_grad_(False)
from safetensors.torch import load_file as safe_load_file
enc.load_state_dict(safe_load_file(os.path.join(MODEL_FOLDER, "Wan2.1-T2V-1.3B",
                    "models_t5_umt5-xxl-enc-bf16.safetensors"), device="cpu"))
tok = HuggingfaceTokenizer(name=os.path.join(MODEL_FOLDER, "Wan2.1-T2V-1.3B", "google", "umt5-xxl/"),
                           seq_len=512, clean="whitespace")
ids, mask = tok([PROMPT], return_mask=True, add_special_tokens=True)
seq_lens = mask.gt(0).sum(dim=1).long()
with torch.no_grad():
    ctx = enc(ids, mask)
for u, v in zip(ctx, seq_lens):
    u[v:] = 0.0
torch.save({"prompt_embeds": ctx.to(torch.bfloat16), "prompt": PROMPT}, out / f"{slug}.pt")
print(f"EMB-OK {slug}.pt shape={tuple(ctx.shape)}")
