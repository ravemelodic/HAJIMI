import torch
import gc
from PIL import Image

from src.detector import CatDetector
from src.body_language import BodyLanguageAnalyzer
from src.emotion import EmotionClassifier
from src.monologue import MonologueGenerator
from src.utils import load_config


class HajimiPipeline:

    def __init__(self, config=None, low_memory=None):
        if config is None:
            config = load_config()

        self.config = config
        self.low_memory = low_memory if low_memory is not None \
            else config.get("low_memory_mode", False)

        if self.low_memory:
            print("按需加载模型")
            self.detector = None
            self.body_analyzer = None
            self.emotion_classifier = None
            self.monologue_gen = None
        else:
            print("加载所有模型...")
            self._load_all_models()

    def _load_all_models(self): #加载四个模型：yolo qwen3vl clip 和 qwen8b
        yolo_cfg = self.config["models"]["yolo"]
        bl_cfg = self.config["models"]["body_language"]
        clip_cfg = self.config["models"]["clip"]
        mono_cfg = self.config["models"]["monologue"]
        device = self.config.get("device", None)

        print("  YOLO...")
        self.detector = CatDetector(
            model_path=yolo_cfg["model_id"],
            conf_threshold=yolo_cfg["confidence_threshold"],
            bbox_padding=yolo_cfg["bbox_padding"],
        )

        print("  Qwen3-VL...")
        self.body_analyzer = BodyLanguageAnalyzer(
            model_id=bl_cfg["model_id"],
            device=device,
            max_new_tokens=bl_cfg.get("max_new_tokens", 512),
        )

        print("  CLIP...")
        self.emotion_classifier = EmotionClassifier(
            model_id=clip_cfg["model_id"],
            device=device,
        )

        print("  Qwen3-8B...")
        lora_path = mono_cfg.get("lora_path") if mono_cfg.get("use_lora", False) else None
        self.monologue_gen = MonologueGenerator(
            model_id=mono_cfg["model_id"],
            device=device,
            lora_path=lora_path,
            use_4bit=(mono_cfg.get("precision", "4bit") == "4bit"),
            temperature=mono_cfg.get("temperature", 0.85),
            max_new_tokens=mono_cfg.get("max_new_tokens", 300),
            enable_thinking=mono_cfg.get("enable_thinking", False),
        )
        print("全部加载完成!")

    def _load_single_model(self, name): #按需加载模型
        cfg = self.config
        device = cfg.get("device", None)

        if name == "detector":
            yolo_cfg = cfg["models"]["yolo"]
            self.detector = CatDetector(
                model_path=yolo_cfg["model_id"],
                conf_threshold=yolo_cfg["confidence_threshold"],
                bbox_padding=yolo_cfg["bbox_padding"],
            )
        elif name == "body_analyzer":
            bl_cfg = cfg["models"]["body_language"]
            self.body_analyzer = BodyLanguageAnalyzer(
                model_id=bl_cfg["model_id"], device=device,
                max_new_tokens=bl_cfg.get("max_new_tokens", 512),
            )
        elif name == "emotion_classifier":
            clip_cfg = cfg["models"]["clip"]
            self.emotion_classifier = EmotionClassifier(
                model_id=clip_cfg["model_id"], device=device,
            )
        elif name == "monologue_gen":
            mono_cfg = cfg["models"]["monologue"]
            lora_path = mono_cfg.get("lora_path") if mono_cfg.get("use_lora", False) else None
            self.monologue_gen = MonologueGenerator(
                model_id=mono_cfg["model_id"], device=device,
                lora_path=lora_path,
                use_4bit=(mono_cfg.get("precision", "4bit") == "4bit"),
                temperature=mono_cfg.get("temperature", 0.85),
                max_new_tokens=mono_cfg.get("max_new_tokens", 300),
                enable_thinking=mono_cfg.get("enable_thinking", False),
            )

    def _unload_model(self, attr_name): #卸载模型
        obj = getattr(self, attr_name, None)
        if obj is not None:
            del obj
            setattr(self, attr_name, None)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def run(self, image, persona="general", clip_strategy="descriptive", lang="en"):#main function
        # Step 1: 检测猫
        if self.low_memory:
            self._load_single_model("detector")
        detections = self.detector.detect(image)
        if self.low_memory:
            self._unload_model("detector")

        if not detections:
            return {"num_cats": 0, "detections": [], "error": "未检测到猫咪"}

        # Step 2: 体态分析
        if self.low_memory:
            self._load_single_model("body_analyzer")
        for det in detections: #对于每一个超过置信度的猫，都会分析
            det["body_language"] = self.body_analyzer.analyze(det["crop"])
        if self.low_memory:
            self._unload_model("body_analyzer")

        # Step 3: 情绪分类
        if self.low_memory:
            self._load_single_model("emotion_classifier")
        for det in detections:
            body_lang = det["body_language"] if clip_strategy == "body_anchored" else None
            det["emotions"] = self.emotion_classifier.classify(
                det["crop"], strategy=clip_strategy, body_language=body_lang,
            )
            det["top_emotion"] = max(det["emotions"], key=det["emotions"].get)
        if self.low_memory:
            self._unload_model("emotion_classifier")

        # Step 4: 独白生成
        if self.low_memory:
            self._load_single_model("monologue_gen")
        for det in detections:
            det["monologue"] = self.monologue_gen.generate(
                emotion=det["top_emotion"],
                emotion_scores=det["emotions"],
                body_language=det["body_language"],
                persona=persona,
                lang=lang,
            )
        if self.low_memory:
            self._unload_model("monologue_gen")

        results = []
        for det in detections:
            results.append({
                "bbox": det["bbox"],
                "confidence": det["confidence"],
                "body_language": det["body_language"],
                "emotions": det["emotions"],
                "top_emotion": det["top_emotion"],
                "monologue": det["monologue"],
            })

        return {"num_cats": len(results), "detections": results}
    #返回字典
