# Sign gloss -> skeleton training

This path uses a sign-language-specific VQ-VAE followed by T5 LoRA.  The
default representation is body pose + left hand + right hand (201 values, or
67 keypoints x 3).  Add `--include-face` to include the face tier.

Install the project requirements first, including `peft`:

```powershell
pip install -r requirements.txt
```

Run from the `avatar` directory:

```powershell
python sign_train_vqvae.py `
  --data-root .\재난안전수어영상 `
  --out .\checkpoints\sign_vqvae `
  --epochs 100 --max-frames 128
```

Then train the gloss-to-motion-token adapter. `--model-path` must point to a
local T5/Flan-T5 checkpoint, such as the checkpoint configured in
`configs/lm/default.yaml` after it has been downloaded.

```powershell
python sign_train_lora.py `
  --data-root .\재난안전수어영상 `
  --vqvae .\checkpoints\sign_vqvae `
  --model-path .\deps\flan-t5-base `
  --out .\checkpoints\sign_lora
```

Generate skeleton coordinates for one gloss:

```powershell
python sign_infer.py `
  --adapter .\checkpoints\sign_lora `
  --vqvae .\checkpoints\sign_vqvae `
  --model-path .\deps\flan-t5-base `
  --gloss 춥다1 `
  --out .\outputs\춥다1.npy
```

The output is a float32 NumPy array shaped `(frames, keypoints, 3)`. The
coordinates are de-normalized back to the source keypoint coordinate system.

For the supplied data, most JSON files contain 2D landmarks. Train a separate
2D model with `--landmark-dim 2d` on both tokenizer and LoRA commands. Its
output is shaped `(frames, keypoints, 2)`.
