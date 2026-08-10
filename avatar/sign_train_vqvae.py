"""Train a VQ-VAE on Korean sign-language skeleton segments.

Run from the avatar directory, for example:
  python sign_train_vqvae.py --data-root ./재난안전수어영상 --out ./checkpoints/sign_vqvae
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mGPT.archs.mgpt_vq import VQVae
from mGPT.data.sign_motion import compute_stats, iter_segments


class SignDataset(Dataset):
    def __init__(self, rows, mean, std, max_frames):
        self.rows, self.mean, self.std, self.max_frames = rows, mean, std, max_frames

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        x = (self.rows[i]["motion"] - self.mean) / self.std
        x = x[:self.max_frames]
        length = len(x)
        out = np.zeros((self.max_frames, x.shape[-1]), dtype=np.float32)
        out[:length] = x
        return torch.from_numpy(out), length


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--include-face", action="store_true")
    p.add_argument("--landmark-dim", choices=["2d", "3d"], default="3d")
    p.add_argument("--max-frames", type=int, default=128)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--code-num", type=int, default=512)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.max_frames % 8:
        raise ValueError("--max-frames must be divisible by 8 for the default tokenizer")

    rows = list(iter_segments(args.data_root, args.include_face, args.landmark_dim))
    mean, std = compute_stats(args.data_root, args.include_face, args.landmark_dim)
    if not rows:
        raise RuntimeError(f"No {args.landmark_dim} sign segments found")
    ds = SignDataset(rows, mean, std, args.max_frames)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = VQVae(nfeats=mean.size, code_num=args.code_num, code_dim=512,
                  output_emb_width=512, down_t=3, stride_t=2, width=512, depth=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    for epoch in range(args.epochs):
        for motion, lengths in loader:
            motion = motion.to(device)
            pred, commit, _ = model(motion)
            mask = torch.arange(motion.shape[1], device=device)[None, :] < lengths.to(device)[:, None]
            mask = mask.unsqueeze(-1)
            recon = torch.nn.functional.smooth_l1_loss(pred[mask.expand_as(pred)], motion[mask.expand_as(motion)])
            velocity = torch.nn.functional.smooth_l1_loss(
                (pred[:, 1:] - pred[:, :-1])[mask[:, 1:].expand_as(pred[:, 1:])],
                (motion[:, 1:] - motion[:, :-1])[mask[:, 1:].expand_as(motion[:, 1:])])
            loss = recon + 0.5 * velocity + 0.02 * commit
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        print(f"epoch={epoch + 1}/{args.epochs} loss={loss.item():.5f}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "pytorch_model.bin")
    np.savez(out / "stats.npz", mean=mean, std=std)
    (out / "config.json").write_text(json.dumps({"nfeats": int(mean.size), "code_num": args.code_num,
        "include_face": args.include_face, "landmark_dim": args.landmark_dim,
        "coordinate_dim": 3 if args.landmark_dim == "3d" else 2,
        "max_frames": args.max_frames}, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
