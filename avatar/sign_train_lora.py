"""LoRA fine-tuning: one gloss -> discrete sign skeleton motion tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration
from peft import LoraConfig, TaskType, get_peft_model

from mGPT.archs.mgpt_vq import VQVae
from mGPT.data.sign_motion import iter_segments


class TokenDataset(Dataset):
    def __init__(self, rows, vae, mean, std, max_frames, device, token_ids):
        self.rows, self.vae, self.mean, self.std = rows, vae, mean, std
        self.max_frames, self.device = max_frames, device
        self.token_ids = token_ids

    def __len__(self): return len(self.rows)

    @torch.no_grad()
    def __getitem__(self, i):
        x = self.rows[i]["motion"][:self.max_frames]
        x = ((x - self.mean) / self.std).astype(np.float32)
        x = torch.from_numpy(x)[None].to(self.device)
        tokens, _ = self.vae.encode(x)
        start, end = self.token_ids[self.vae.quantizer.n_e], self.token_ids[self.vae.quantizer.n_e + 1]
        body = [self.token_ids[x] for x in tokens[0].cpu().tolist()]
        return self.rows[i]["gloss"], [start] + body + [end]


def collate(batch, tokenizer, max_length):
    texts, labels = zip(*batch)
    enc = tokenizer([f"Generate sign motion for gloss: {x}" for x in texts],
                    padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    width = max(len(x) for x in labels)
    y = torch.full((len(labels), width), -100, dtype=torch.long)
    for i, x in enumerate(labels): y[i, :len(x)] = torch.tensor(x)
    return enc, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True); p.add_argument("--vqvae", required=True)
    p.add_argument("--model-path", required=True, help="local T5/Flan-T5 directory")
    p.add_argument("--out", required=True); p.add_argument("--include-face", action="store_true")
    p.add_argument("--landmark-dim", choices=["2d", "3d"], default="3d")
    p.add_argument("--max-frames", type=int, default=128); p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4); p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cuda"); args = p.parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    cfg = json.loads((Path(args.vqvae) / "config.json").read_text(encoding="utf-8"))
    stats = np.load(Path(args.vqvae) / "stats.npz"); mean, std = stats["mean"], stats["std"]
    vae = VQVae(nfeats=cfg["nfeats"], code_num=cfg["code_num"], code_dim=512,
                output_emb_width=512, down_t=3, stride_t=2, width=512, depth=3).to(device)
    vae.load_state_dict(torch.load(Path(args.vqvae) / "pytorch_model.bin", map_location=device)); vae.eval()
    rows = list(iter_segments(args.data_root, args.include_face, args.landmark_dim))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, legacy=True)
    motion_tokens = [f"<motion_id_{i}>" for i in range(cfg["code_num"] + 2)]
    tokenizer.add_tokens(motion_tokens)
    token_ids = {i: tokenizer.convert_tokens_to_ids(f"<motion_id_{i}>")
                 for i in range(cfg["code_num"] + 2)}
    model = T5ForConditionalGeneration.from_pretrained(args.model_path)
    model.resize_token_embeddings(len(tokenizer))
    # q/v are LoRA adapters; shared/lm_head must remain trainable for new motion tokens.
    model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=8,
        lora_alpha=16, lora_dropout=0.05, target_modules=["q", "v"], bias="none",
        modules_to_save=["shared", "lm_head"]))
    model.print_trainable_parameters(); model.to(device)
    ds = TokenDataset(rows, vae, mean, std, args.max_frames, device, token_ids)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        collate_fn=lambda b: collate(b, tokenizer, 128))
    opt = torch.optim.AdamW((x for x in model.parameters() if x.requires_grad), lr=args.lr)
    model.train()
    for epoch in range(args.epochs):
        for enc, labels in loader:
            enc = {k: v.to(device) for k, v in enc.items()}; labels = labels.to(device)
            loss = model(**enc, labels=labels).loss
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        print(f"epoch={epoch + 1}/{args.epochs} loss={loss.item():.5f}")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out); tokenizer.save_pretrained(out)
    (out / "sign_vqvae.json").write_text((Path(args.vqvae) / "config.json").read_text(encoding="utf-8"), encoding="utf-8")


if __name__ == "__main__": main()
