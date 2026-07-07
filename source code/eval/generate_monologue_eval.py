"""
生成 LLM 人工评估数据（两阶段 + LoRA 消融对比）
Phase 1: YOLO + Qwen3-VL + CLIP → 收集 body_lang / emotions（只跑一次）
Phase 2a: Qwen3-8B base → 生成 base 独白
Phase 2b: 加载 LoRA adapter → 生成 lora 独白
输出: eval/monologue_eval_data.json，每条带 condition="base"|"lora"
"""

import gc
import json
import random
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.detector import CatDetector
from src.body_language import BodyLanguageAnalyzer
from src.emotion import EmotionClassifier
from src.monologue import MonologueGenerator
from src.utils import load_config

PROJECT_ROOT = Path(__file__).parent.parent
EVAL_DIR = PROJECT_ROOT / "data" / "eval_set"
OUTPUT_PATH = PROJECT_ROOT / "eval" / "monologue_eval_data.json"

PERSONAS = ["catgirl", "hachimi", "maodie", "general"]
N_IMAGES = 5


def get_lora_path():
    try:
        cfg = load_config()
        if cfg["monologue"].get("use_lora"):
            p = PROJECT_ROOT / cfg["monologue"]["lora_path"]
            if p.exists():
                return str(p)
    except Exception:
        pass
    fallback = PROJECT_ROOT / "models" / "monologue_lora"
    return str(fallback) if fallback.exists() else None


def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        print(f"  显存已释放，当前占用: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


def main():
    # ── 选图 ─────────────────────────────────────────────────────────
    all_images = sorted(EVAL_DIR.glob("eval_0[0-4]*.jpg"))
    selected = random.sample(all_images, min(N_IMAGES, len(all_images)))
    backup = [p for p in all_images if p not in selected]
    random.shuffle(backup)

    # ── Phase 1: YOLO + Qwen3-VL + CLIP ──────────────────────────────
    print("Phase 1: 加载 YOLO / Qwen3-VL / CLIP ...")
    detector = CatDetector()
    body_analyzer = BodyLanguageAnalyzer()
    clip_classifier = EmotionClassifier()

    phase1 = []
    for img_path in list(selected) + backup:
        if len(phase1) >= N_IMAGES:
            break
        print(f"  检测: {img_path.name}")
        img = Image.open(img_path)
        detections = detector.detect(img)
        if not detections:
            print(f"    未检测到猫，换下一张")
            continue
        crop = detections[0]["crop"]
        body_lang = body_analyzer.analyze(crop)
        emotions = clip_classifier.classify(crop, strategy="descriptive")
        top_emotion = max(emotions, key=emotions.get)
        print(f"    情绪: {top_emotion} ({emotions[top_emotion]:.1%})")
        phase1.append({"img_path": img_path, "body_lang": body_lang,
                       "emotion": top_emotion, "emotion_scores": emotions})

    print(f"\nPhase 1 完成，共 {len(phase1)} 张图")

    del body_analyzer, clip_classifier, detector
    free_gpu()

    # ── Phase 2: Base + LoRA 两轮生成 ────────────────────────────────
    lora_path = get_lora_path()
    print(f"\nLoRA path: {lora_path}")

    eval_items = []

    for condition, use_lora in [("base", False), ("lora", True)]:
        actual_lora = lora_path if use_lora else None
        print(f"\n{'='*50}")
        print(f"Phase 2 [{condition}]: 加载 Qwen3-8B (lora={actual_lora is not None}) ...")

        mono_gen = MonologueGenerator(lora_path=actual_lora)

        for entry in phase1:
            img_path = entry["img_path"]
            print(f"\n  生成独白 [{condition}]: {img_path.name}")
            for persona in PERSONAS:
                monologue = mono_gen.generate(
                    emotion=entry["emotion"],
                    emotion_scores=entry["emotion_scores"],
                    body_language=entry["body_lang"],
                    persona=persona,
                )
                print(f"    [{persona}] {monologue[:60]}...")
                eval_items.append({
                    "id": f"{img_path.stem}_{persona}_{condition}",
                    "image_path": str(img_path),
                    "emotion": entry["emotion"],
                    "emotion_scores": entry["emotion_scores"],
                    "body_language": entry["body_lang"],
                    "persona": persona,
                    "condition": condition,
                    "monologue": monologue,
                })

        del mono_gen
        free_gpu()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_items, f, indent=2, ensure_ascii=False)

    print(f"\n生成完成! {len(eval_items)} 条独白 (base={N_IMAGES*len(PERSONAS)}, lora={N_IMAGES*len(PERSONAS)}) -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
