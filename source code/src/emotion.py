#Module C
import clip
import torch
from PIL import Image

from src.prompts import CLIP_STRATEGIES, build_body_anchored_prompt


class EmotionClassifier:

    EMOTIONS = ["relaxed", "curious", "fearful", "aggressive", "playful", "content"]

    def __init__(self, model_id="ViT-L/14", device=None):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        self.model, self.preprocess = clip.load(model_id, device=self.device)
        self.model.eval()

        # 提前算好静态策略的文本特征，后面每次推理不用重复算
        self._text_features_cache = {}
        self._precompute_text_features()

    def _precompute_text_features(self):
        for strategy_name, prompts in CLIP_STRATEGIES.items():
            if prompts is None:  # body_anchored 每次都不一样，跳过
                continue
            texts = [prompts[emotion] for emotion in self.EMOTIONS]
            text_tokens = clip.tokenize(texts).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_text(text_tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            self._text_features_cache[strategy_name] = feats

    def classify(self, image, strategy="descriptive", body_language=None, temperature=100.0):
        """返回 {emotion: probability} 字典"""
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        if strategy == "body_anchored":
            if body_language is None:
                print("body_anchored 需要 body_language，改用 descriptive")
                strategy = "descriptive"
                text_features = self._text_features_cache["descriptive"]
            else:
                texts = [build_body_anchored_prompt(e, body_language) for e in self.EMOTIONS]
                text_tokens = clip.tokenize(texts).to(self.device)
                with torch.no_grad():
                    text_features = self.model.encode_text(text_tokens)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        else:
            if strategy not in self._text_features_cache:
                raise ValueError(f"未知策略: {strategy}")
            text_features = self._text_features_cache[strategy]

        similarity = (image_features @ text_features.T) * temperature
        probs = similarity.softmax(dim=-1).cpu().numpy()[0]

        return {e: float(p) for e, p in zip(self.EMOTIONS, probs)}

    def classify_all_strategies(self, image, body_language=None):
        """跑所有策略，消融实验用"""
        results = {}
        for name in CLIP_STRATEGIES:
            results[name] = self.classify(image, strategy=name, body_language=body_language)
        return results
