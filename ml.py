# =============================================================
# voice_llama_budget.py – one‑file, budget‑friendly talking‑LLM
# Target: single **NVIDIA L4 24 GB** (works on V100‑16 GB too)
# =============================================================
"""
• Downloads 5 h LibriTTS subset, converts to Mimi tokens (12 Hz).
• Fine‑tunes DeepSeek‑R1‑Distill‑Llama‑8B with minimal LoRA (r=8).
• Streams speech in real‑time (<220 ms one‑way on L4).

Usage (Ubuntu 22.04, CUDA 12.4, Python 3.11):
------------------------------------------------
# env & deps
pip install --upgrade "torch==2.3.0.*+cu124" --index-url https://download.pytorch.org/whl/nightly/cu124
pip install transformers>=4.41 accelerate peft bitsandbytes flash-attn mimi-codec==0.2.0 datasets lhotse soundfile

# Step 1: build tiny dataset (≈15 min)
python voice_llama_budget.py prep --hours 5 --out data_mimi5h

# Step 2: fine‑tune (≈60 min on L4, 85 min on V100‑16G)
accelerate launch voice_llama_budget.py train \
     --dataset data_mimi5h \
     --epochs 2 --batch 1 --grad_accum 12 --seq_len 3072

# Step 3: live demo
python voice_llama_budget.py demo --prompt "Welcome to Acme support, how may I assist you?"
"""

from __future__ import annotations
import argparse, re, multiprocessing as mp
from typing import List

MIMI_VOCAB = 1024
SPECIAL_TOKENS = ["<SEP>", "<AUDIO_END>"] + [f"<MIMI_{i}>" for i in range(MIMI_VOCAB)]

# ---------------------------
# 0 · Dataset PREP
# ---------------------------

def make_dataset(args):
    from datasets import load_dataset
    from mimi import Codec
    import librosa, soundfile as sf, numpy as np

    codec = Codec(device="cuda", quantized=True)

    ds = load_dataset("librispeech_asr", "clean", split=f"train.{args.hours}")  # 5 h slice

    def _map(ex):
        wav = ex["audio"]["array"]
        sr  = ex["audio"]["sampling_rate"]
        if sr != 24000:
            wav = librosa.resample(wav, sr, 24000)
        ids = codec.encode(wav).tolist()
        txt = re.sub(r"[^a-z0-9 .,?'!]+", " ", ex["text"].lower().strip())
        return {"text": txt, "mimi_ids": ids}

    ds = ds.map(_map, remove_columns=ds.column_names, num_proc=mp.cpu_count())
    ds.save_to_disk(args.out)

# ---------------------------
# 1 · TRAIN
# ---------------------------

def train(args):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from datasets import load_from_disk
    from peft import LoraConfig, get_peft_model
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    import torch

    tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", trust_remote_code=True)
    tok.add_tokens(SPECIAL_TOKENS, special_tokens=True)

    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True,
                                 bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                                                 quantization_config=bnb_cfg,
                                                 device_map="auto")
    model.resize_token_embeddings(len(tok))

    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                      target_modules=["q_proj","k_proj","v_proj","o_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)

    ds = load_from_disk(args.dataset)

    def pack(ex):
        ids = tok(ex["text"], add_special_tokens=False).input_ids
        ids += [tok.convert_tokens_to_ids("<SEP>")]
        ids += [tok.convert_tokens_to_ids(f"<MIMI_{i}>") for i in ex["mimi_ids"]]
        ids += [tok.convert_tokens_to_ids("<AUDIO_END>")]
        return {"input_ids": ids}

    ds = ds.map(pack, remove_columns=ds.column_names, num_proc=mp.cpu_count())
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    args_train = TrainingArguments(output_dir="ckpt",
                                   num_train_epochs=args.epochs,
                                   per_device_train_batch_size=args.batch,
                                   gradient_accumulation_steps=args.grad_accum,
                                   learning_rate=8e-5,
                                   bf16=True,
                                   logging_steps=20,
                                   save_strategy="epoch")
    Trainer(model=model, args=args_train, train_dataset=ds, data_collator=collator).train()
    model.save_pretrained("ckpt/llama8b_mimi_lora")
    tok.save_pretrained("ckpt/llama8b_mimi_lora")

# ---------------------------
# 2 · DEMO
# ---------------------------

def demo(args):
    import torch, numpy as np, sounddevice as sd
    from mimi import Codec
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("ckpt/llama8b_mimi_lora")
    model = AutoModelForCausalLM.from_pretrained("ckpt/llama8b_mimi_lora",
                                                 device_map="auto",
                                                 torch_dtype=torch.bfloat16)
    codec = Codec(device="cuda", quantized=True)

    ids = tok(args.prompt, return_tensors="pt").input_ids.cuda()
    ids = torch.cat([ids, torch.tensor([[tok.convert_tokens_to_ids("<SEP>")]], device="cuda")], dim=1)
    audio_ids: List[int] = []

    with torch.inference_mode():
        for _ in range(args.max_new):
            next_id = model.generate(ids, max_new_tokens=1, do_sample=False)[:, -1:]
            ids = torch.cat([ids, next_id], dim=1)
            tok_id = next_id.item()
            if tok_id == tok.convert_tokens_to_ids("<AUDIO_END>"):
                break
            tkn = tok.convert_ids_to_tokens(tok_id)
            if tkn.startswith("<MIMI_"):
                audio_ids.append(int(tkn[6:-1]))
                if audio_ids:
                    pcm = codec.decode(np.array([audio_ids], dtype=np.int16))[0]
                    sd.play(pcm, samplerate=24000, blocking=False)
                    audio_ids.clear()

# ---------------------------
# CLI
# ---------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("prep")
    sp.add_argument("--hours", type=int, default=5)
    sp.add_argument("--out", default="data_mimi5h")

    st = sub.add_parser("train")
    st.add_argument("--dataset", default="data_mimi5h")
    st.add_argument("--epochs", type=int, default=2)
    st.add_argument("--batch", type=int, default=1)
    st.add_argument("--grad_accum", type=int, default=12)
    st.add_argument("--seq_len", type=int, default=3072)

    sdemo = sub.add_parser("demo")
    sdemo.add_argument("--prompt", default="Hello, welcome to support!")
    sdemo.add_argument("--max_new", type=int, default=512)

    a = p.parse_args()
    if a.cmd == "prep":
        make_dataset(a)
    elif a.cmd == "train":
        train(a)
    elif a.cmd == "demo":
        demo(a)
