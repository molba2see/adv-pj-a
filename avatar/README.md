pip install -r requirements.txt

python sign_train_vqvae.py  --data-root .\재난안전수어영상 --out .\checkpoints\sign_vqvae --epochs 100

python sign_train_lora.py --data-root .\재난안전수어영상 --vqvae .\checkpoints\sign_vqvae --model-path .\deps\flan-t5-base --out .\checkpoints\sign_lora

python sign_infer.py --adapter .\checkpoints\sign_lora --vqvae .\checkpoints\sign_vqvae --model-path .\deps\flan-t5-base --gloss 춥다1 --out .\outputs\춥다1.npy

<div align= "center">
    <h1> Official repo for MotionGPT <img src="./assets/images/avatar_bot.jpg" width="35px"></h1>
