#Module D
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BNB_AVAILABLE = False
try:
    from transformers import BitsAndBytesConfig
    import bitsandbytes
    BNB_AVAILABLE = True
except ImportError:
    pass

from src.prompts import PERSONA_SYSTEM_PROMPTS, PERSONA_SYSTEM_PROMPTS_ZH, format_monologue_prompt


class MonologueGenerator:

    def __init__(self, model_id="Qwen/Qwen3-8B", device=None,
                 lora_path=None, use_4bit=True,
                 temperature=0.85, max_new_tokens=2048,
                 enable_thinking=False):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking

        quantization_config = None
        if use_4bit and device == "cuda" and BNB_AVAILABLE:
            print("加载 Qwen3-8B...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif use_4bit and not BNB_AVAILABLE:
            print("bitsandbytes 不可用")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        device_map = "auto" if self.device == "cuda" else {"": self.device}
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        if lora_path is not None:
            from peft import PeftModel
            print(f"加载 LoRA: {lora_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_path)

        self.model.eval()

    def generate(self, emotion, emotion_scores, body_language, persona="general",
                 lang="en", system_prompt_override=None, temperature=None):
        if system_prompt_override:
            system_prompt = system_prompt_override
        else:
            prompt_map = PERSONA_SYSTEM_PROMPTS_ZH if lang == "zh" else PERSONA_SYSTEM_PROMPTS
            if persona not in prompt_map:
                print(f"未知人格 '{persona}'，改用 general")
                persona = "general"
            system_prompt = prompt_map[persona]

        user_prompt = format_monologue_prompt(emotion, emotion_scores, body_language, lang=lang)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Qwen3 默认开 thinking 模式会产出大量 <think>...</think>，
        # 把 max_new_tokens 全消耗在推理上导致独白为空，所以关掉
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature if temperature is not None else self.temperature,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        monologue = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

        return self._clean_output(monologue)

    def _clean_output(self, text):
        # 去掉 <think>...</think> 块（思考模式残留）
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        text = text.strip()

        prefixes = [
            "Inner monologue:", "Cat's inner monologue:", "Monologue:",
            "*inner monologue*", "Here's the monologue:",
            "内心独白:", "猫咪内心独白:", "独白:",
        ]
        for prefix in prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        if text.startswith("'") and text.endswith("'"):
            text = text[1:-1]

        return text.strip("*").strip()
