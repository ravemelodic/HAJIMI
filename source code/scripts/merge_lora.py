"""
把 LoRA adapter merge 进 base model，输出普通 HF 模型。
Mac 上不需要 PEFT，直接加载 merged model 就行。

用法:
    python scripts/merge_lora.py
    python scripts/merge_lora.py --base Qwen/Qwen3-8B --lora models/monologue_lora --out models/merged_qwen3_8b
"""

import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge(base_model_id, lora_path, output_path):
    print(f"Loading base model: {base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="cpu",   # merge 在 CPU 上做，避免显存不够
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

    print(f"Loading LoRA adapter: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)

    print("Merging LoRA weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen3-8B")
    parser.add_argument("--lora", default="models/monologue_lora")
    parser.add_argument("--out",  default="models/merged_qwen3_8b")
    args = parser.parse_args()

    merge(args.base, args.lora, args.out)
