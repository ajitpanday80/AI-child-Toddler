"""Ask a trained checkpoint a question:  python -m school.ask runs/school/latest.pt "27+45="  """
import sys

import torch

from .curriculum import STOI, encode, decode
from .model import GPT, ModelConfig

d = torch.load(sys.argv[1])
model = GPT(ModelConfig(**d["cfg"]))
model.load_state_dict(d["model"])
model.eval()
for q in sys.argv[2:]:
    out = model.generate(torch.tensor([encode(q)]), STOI["\n"], 12)
    print(q, decode(out[0, len(q):].tolist()).split("\n")[0])
