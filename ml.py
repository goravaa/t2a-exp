# =============================================
# llama_voice_pipeline.py (monorepo style)
# =============================================
"""
End‑to‑end script bundle to turn DeepSeek‑R1‑Distill‑Llama‑8B into a Mimi‑spoken model.
Run sections as stand‑alone modules or call with CLI flags.
Requirements (install once):
    pip install --upgrade transformers datasets acceleratE peft bitsandbytes flash-attn mimi-codec==0.2.0
    # H100 driver: CUDA 12.4, torch 2.3 nightly wheels recommended.
"""

import argparse, os, json, re, time, math, multiprocessing as mp
from dataclasses import dataclass
from typing import List, Dict, Any

# ------------------------------
# 0 · Global constants & helpers
# ------------------------------
MIMI_VOCAB_SIZE = 1024
SPECIAL_TOKENS = ["<SEP>", "<AUDIO_END>"] + [f"<MIMI_{i}>" for i in range(MIMI_VOCAB_SIZE)]

# ------------------------------
# 1 · Dataset prep
# ------------------------------
"""
We use **LibriTTS clean‑100 + VCTK** (≈ 125 h) for rapid training. Both are CC‑BY 4.0.
You can swap any HF speech‑dataset name.
"""

def make_dataset(args):
    from datasets import load_dataset, Dataset, concatenate_datasets, DatasetDict, Features, Sequence, Value
    import numpy as np, soundfile as sf
    from mimi import Codec

    codec = Codec(device="cuda", quantized=True)

    def _process(example):
        # → audio to Mimi token ids (list[int])
        wav = example["audio"]["array"]
        if example["audio"]["sampling_rate"] != 24000:
            # quick resample for demo; replace with torchaudio.kaldi_resample for prod
            import librosa
            wav = librosa.resample(wav, example["audio"]["sampling_rate"], 24000)
        token_ids = codec.encode(wav).tolist()
        text = example["text"].lower().strip()
        # simple ascii clean
        text = re.sub(r"[^a-z0-9 .,?'!]+", " ", text)
        return {
            "text": text,
            "mimi_ids": token_ids
        }

    print("\n▶ Downloading LibriTTS & VCTK …")
    ds1 = load_dataset("librispeech_asr", "clean", split="train.100")
    ds2 = load_dataset("vctk", split="train")
    ds = concatenate_datasets([ds1, ds2]).shuffle(seed=42)

    print("▶ Converting to Mimi tokens … (GPU)")
    ds = ds.map(_process, remove_columns=ds1.column_names, num_proc=mp.cpu_count())

    print("▶ Saving Arrow file:", args.out)
    ds.save_to_disk(args.out)

# ------------------------------
# 2 · Fine‑tuning LoRA head
# ------------------------------

def train(args):
    import torch, bitsandbytes as bnb
    from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
                              DataCollatorForLanguageModeling)
    from peft import LoraConfig, get_peft_model
    from datasets import load_from_disk

    # tokenizer patch
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    added_n = tok.add_tokens(SPECIAL_TOKENS, special_tokens=True)
    print(f"Added {added_n} Mimi tokens to tokenizer; new vocab = {len(tok)}")

    # model load w/ 4‑bit weights for speed
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.resize_token_embeddings(len(tok))

    # LoRA config (Q, K, V, O)
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora)

    # dataset load & packaging
    ds = load_from_disk(args.dataset)

    def _pack(example):
        txt_ids = tok(example["text"], add_special_tokens=False).input_ids
        mimi_tokens = [tok.convert_tokens_to_ids("<SEP>")]
        mimi_tokens += [tok.convert_tokens_to_ids(f"<MIMI_{i}>") for i in example["mimi_ids"]]
        mimi_tokens += [tok.convert_tokens_to_ids("<AUDIO_END>")]
        return {"input_ids": txt_ids + mimi_tokens}

    ds = ds.map(_pack, remove_columns=ds.column_names, num_proc=mp.cpu_count())

    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    targs = TrainingArguments(
        output_dir="checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=200,
        logging_steps=20,
        bf16=True,
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=collator
    )
    print("▶ Starting fine‑tune…")
    trainer.train()
    trainer.save_model("checkpoints/llama8b-mimi-lora")

# ------------------------------
# 3 · Streaming demo
# ------------------------------

def demo(args):
    import torch, numpy as np, sounddevice as sd
    from mimi import Codec
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint,
                                                 torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    codec = Codec(device="cuda", quantized=True)

    prompt = args.prompt
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    ids = torch.cat([ids, torch.tensor([[tok.convert_tokens_to_ids("<SEP>")]], device="cuda")], dim=1)

    audio_buf = []
    with torch.inference_mode():
        for _ in range(args.max_new):
            next_id = model.generate(ids, max_new_tokens=1, do_sample=False, pad_token_id=tok.eos_token_id)[:, -1:]
            ids = torch.cat([ids, next_id], dim=1)
            token = next_id.item()
            if token == tok.convert_tokens_to_ids("<AUDIO_END>"):
                break
            tok_str = tok.convert_ids_to_tokens(token)
            if tok_str.startswith("<MIMI_"):
                audio_buf.append(int(tok_str[6:-1]))
                if len(audio_buf) % 1 == 0:
                    pcm = codec.decode(np.array([audio_buf], dtype=np.int16))[0]
                    sd.play(pcm, samplerate=24000, blocking=False)
                    audio_buf.clear()
    sd.wait()

# ------------------------------
# CLI entry
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeek‑8B Mimi voice pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p0 = sub.add_parser("prep")
    p0.add_argument("--out", default="dataset_mimi")

    p1 = sub.add_parser("train")
    p1.add_argument("--dataset", default="dataset_mimi")
    p1.add_argument("--base_model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    p1.add_argument("--epochs", type=int, default=3)
    p1.add_argument("--batch", type=int, default=1)
    p1.add_argument("--grad_accum", type=int, default=8)
    p1.add_argument("--lr", type=float, default=5e-5)

    p2 = sub.add_parser("demo")
    p2.add_argument("--checkpoint", default="checkpoints/llama8b-mimi-lora")
    p2.add_argument("--tokenizer", default="tokenizer_mimi")
    p2.add_argument("--prompt", default="Hello, how may I assist you today?")
    p2.add_argument("--max_new", type=int, default=1024)

    args = parser.parse_args()

    if args.cmd == "prep":
        make_dataset(args)
    elif args.cmd == "train":
        train(args)
    elif args.cmd == "demo":
        demo(args)
