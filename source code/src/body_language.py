#Module B
import json
import re
import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from src.prompts import BODY_LANGUAGE_SYSTEM_PROMPT, BODY_LANGUAGE_USER_PROMPT


# 解析失败时用这个兜底
DEFAULT_BODY_LANGUAGE = {
    "ears": "unknown",
    "eyes": "unknown",
    "tail": "unknown",
    "body_posture": "unknown",
    "mouth": "unknown",
    "whiskers": "unknown",
    "overall_tension": "unknown",
    "additional_observations": "parse failed",
}


class BodyLanguageAnalyzer:

    def __init__(self, model_id="Qwen/Qwen3-VL-2B-Instruct", device=None, max_new_tokens=512):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.max_new_tokens = max_new_tokens

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        # device_map="auto" 在 MPS 上行为不稳定
        device_map = "auto" if self.device == "cuda" else {"": self.device}
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()

    def analyze(self, image, max_retries=2):
        for attempt in range(max_retries + 1):
            messages = [
                {"role": "system", "content": BODY_LANGUAGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": BODY_LANGUAGE_USER_PROMPT},
                    ],
                },
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text], images=[image],
                padding=True, return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,  # 贪婪解码，JSON 输出更稳定
                ) #简单来说就是 这个部分我们必须要输出结构性json，如果说你sampling的话会乱。模型每一步都得选概率最高的token。

            # 只取新生成的部分
            trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]
            output_text = self.processor.batch_decode(
                trimmed, skip_special_tokens=True,
            )[0].strip()

            result = self._parse_json(output_text)
            if result is not None:
                return result

            print(f"第 {attempt + 1} 次 JSON 解析失败: {output_text[:200]}")

        print("体态分析解析全部失败，返回默认值")
        return DEFAULT_BODY_LANGUAGE.copy()

    def _parse_json(self, text):
        """试几种方式解析模型输出的 JSON"""
        # 先直接解析
        try:
            return self._validate(json.loads(text))
        except json.JSONDecodeError:
            pass

        # 找 {...} 块
        m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if m:
            try:
                return self._validate(json.loads(m.group()))
            except json.JSONDecodeError:
                pass

        # 最后用正则逐字段提取
        fields = ["ears", "eyes", "tail", "body_posture", "mouth",
                  "whiskers", "overall_tension", "additional_observations"]
        result = {}
        for field in fields:
            m = re.search(rf'"{field}"\s*:\s*"([^"]*)"', text)
            result[field] = m.group(1) if m else "unknown"

        known = sum(1 for v in result.values() if v != "unknown")
        return result if known >= len(fields) // 2 else None

    def _validate(self, data):
        for field in ["ears", "eyes", "tail", "body_posture", "mouth", "whiskers", "overall_tension"]:
            if field not in data:
                data[field] = "unknown"
        if "additional_observations" not in data:
            data["additional_observations"] = ""
        return data
