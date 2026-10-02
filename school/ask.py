"""Let the model write:  python -m school.ask "Once upon a time" [--n 300]
   Use --ckpt runs/school/passed_0_Values.pt to hear the model as it was when it passed that stage."""
import argparse

import torch

from .exams import decode, encode
from .model import GPT, ModelConfig

ap = argparse.ArgumentParser()
ap.add_argument("prompt")
ap.add_argument("--ckpt", default="runs/school/latest.pt")
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--temperature", type=float, default=0.8)
a = ap.parse_args()
ck = torch.load(a.ckpt)
model = GPT(ModelConfig(**ck["cfg"]) if "cfg" in ck else ModelConfig(vocab_size=128))
model.load_state_dict(ck.get("model", ck))
out = model.sample(torch.tensor([encode(a.prompt)]), a.n, a.temperature)
print(decode(out[0].tolist()))
