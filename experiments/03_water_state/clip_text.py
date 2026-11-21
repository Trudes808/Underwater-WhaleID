"""CLIP ViT-L/14 text embeddings of the prompts in prompts.json (CPU), saved to the cache for the zero-shot scores."""
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
import torch
from transformers import CLIPModel, CLIPTokenizer

HERE = Path(__file__).parent
OUT = Path("/home/sat3737/NOAA_whales/underwater_cache/waterstate/clip_text.npz")
torch.set_num_threads(8)
P = json.loads((HERE / "prompts.json").read_text())
m = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval()
tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
texts = P["air"] + P["underwater"]
with torch.no_grad():
    e = m.get_text_features(**tok(texts, padding=True, return_tensors="pt"))
    if not torch.is_tensor(e):
        e = e.pooler_output
    e = torch.nn.functional.normalize(e, dim=-1).numpy()
np.savez(OUT, emb=e, is_air=np.array([1] * len(P["air"]) + [0] * len(P["underwater"])),
         logit_scale=float(m.logit_scale.exp()), texts=np.array(texts))
print(e.shape, float(m.logit_scale.exp()))
