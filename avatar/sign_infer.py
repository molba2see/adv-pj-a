"""Generate a skeleton-coordinate .npy file from one gloss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from peft import PeftModel
from mGPT.archs.mgpt_vq import VQVae


def main():
    p = argparse.ArgumentParser(); p.add_argument("--adapter", required=True)
    p.add_argument("--vqvae", required=True); p.add_argument("--model-path", required=True)
    p.add_argument("--gloss", required=True); p.add_argument("--out", required=True)
    p.add_argument("--max-new-tokens", type=int, default=40); p.add_argument("--device", default="cuda")
    a = p.parse_args(); d = torch.device(a.device if a.device == "cpu" or torch.cuda.is_available() else "cpu")
    vc = json.loads((Path(a.vqvae) / "config.json").read_text(encoding="utf-8")); stats = np.load(Path(a.vqvae) / "stats.npz")
    vae = VQVae(nfeats=vc["nfeats"], code_num=vc["code_num"], code_dim=512, output_emb_width=512,
                down_t=3, stride_t=2, width=512, depth=3).to(d)
    vae.load_state_dict(torch.load(Path(a.vqvae) / "pytorch_model.bin", map_location=d)); vae.eval()
    tok = AutoTokenizer.from_pretrained(a.adapter, legacy=True)
    base = T5ForConditionalGeneration.from_pretrained(a.model_path); base.resize_token_embeddings(len(tok))
    model = PeftModel.from_pretrained(base, a.adapter).to(d).eval()
    inp = tok([f"Generate sign motion for gloss: {a.gloss}"], return_tensors="pt").to(d)
    ids = model.generate(**inp, max_new_tokens=a.max_new_tokens)[0].tolist()
    start = tok.convert_tokens_to_ids(f"<motion_id_{vc['code_num']}>"); end = tok.convert_tokens_to_ids(f"<motion_id_{vc['code_num'] + 1}>")
    ids = ids[ids.index(start) + 1:] if start in ids else ids
    ids = ids[:ids.index(end)] if end in ids else ids
    motion_ids = {tok.convert_tokens_to_ids(f"<motion_id_{i}>"): i
                  for i in range(vc["code_num"])}
    ids = [motion_ids[x] for x in ids if x in motion_ids]
    if not ids: raise RuntimeError("The model generated no motion tokens")
    with torch.no_grad(): out = vae.decode(torch.tensor(ids, device=d)[None]).squeeze(0).cpu().numpy()
    out = out * stats["std"] + stats["mean"]
    np.save(a.out, out.reshape(out.shape[0], -1, 3).astype(np.float32))
    print(f"saved {a.out}: {out.shape[0]} frames x {out.shape[1] // 3} keypoints")


if __name__ == "__main__": main()
