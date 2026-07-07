# HAJIMI 单模块测试脚本
# 用法: python test_modules.py [模块名]
# 模块名: a, b, c, d, all

import sys
import os
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_test_image():
    """获取测试图片"""
    from PIL import Image
    import requests
    from io import BytesIO

    test_path = PROJECT_ROOT / "data" / "test_cat.jpg"
    if test_path.exists():
        return Image.open(test_path).convert("RGB")

    print("下载测试图片...")
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    response = requests.get(url, timeout=15, headers=headers)
    if not response.headers.get("content-type", "").startswith("image"):
        url = "https://cdn.pixabay.com/photo/2024/02/28/07/42/european-shorthair-8601492_640.jpg"
        response = requests.get(url, timeout=15, headers=headers)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    test_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(test_path)
    return image


def test_a():
    """测试模块A: YOLO 猫咪检测"""
    print("=" * 50)
    print("测试模块A: YOLO 猫咪检测")
    print("=" * 50)

    from src.detector import CatDetector
    detector = CatDetector()
    print("✓ CatDetector 加载成功")

    image = get_test_image()
    detections = detector.detect(image)

    print(f"✓ 检测到 {len(detections)} 只猫咪")
    for i, det in enumerate(detections):
        print(f"  #{i+1}: conf={det['confidence']:.1%}, bbox={det['bbox']}")

    if detections:
        # 保存裁剪图
        crop_path = PROJECT_ROOT / "data" / "test_cat_crop.jpg"
        detections[0]["crop"].save(crop_path)
        print(f"✓ 裁剪图已保存: {crop_path}")

    return detections[0]["crop"] if detections else image


def test_b(image=None):
    """测试模块B: Qwen3-VL 体态分析"""
    print("=" * 50)
    print("测试模块B: Qwen3-VL 体态分析")
    print("=" * 50)

    if image is None:
        crop_path = PROJECT_ROOT / "data" / "test_cat_crop.jpg"
        if crop_path.exists():
            from PIL import Image
            image = Image.open(crop_path).convert("RGB")
        else:
            image = get_test_image()

    from src.body_language import BodyLanguageAnalyzer
    print("加载 Qwen3-VL-2B 模型...")
    analyzer = BodyLanguageAnalyzer()
    print("✓ 加载成功")

    result = analyzer.analyze(image)
    print("✓ 体态分析结果:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    return result


def test_c(image=None, body_language=None):
    """测试模块C: CLIP 情绪分类"""
    print("=" * 50)
    print("测试模块C: CLIP 情绪分类")
    print("=" * 50)

    if image is None:
        crop_path = PROJECT_ROOT / "data" / "test_cat_crop.jpg"
        if crop_path.exists():
            from PIL import Image
            image = Image.open(crop_path).convert("RGB")
        else:
            image = get_test_image()

    from src.emotion import EmotionClassifier
    print("加载 CLIP ViT-L/14...")
    classifier = EmotionClassifier()
    print("✓ 加载成功")

    strategies = ["simple", "descriptive", "expert"]
    if body_language:
        strategies.append("body_anchored")

    for strat in strategies:
        bl = body_language if strat == "body_anchored" else None
        result = classifier.classify(image, strategy=strat, body_language=bl)
        top = max(result, key=result.get)
        print(f"✓ {strat}: {top} ({result[top]:.1%})")

    return classifier.classify(image, strategy="descriptive")


def test_d(emotion=None, emotion_scores=None, body_language=None):
    """测试模块D: Qwen3-8B 内心独白"""
    print("=" * 50)
    print("测试模块D: Qwen3-8B 内心独白生成")
    print("=" * 50)

    if emotion is None:
        emotion = "curious"
    if emotion_scores is None:
        emotion_scores = {"curious": 0.5, "relaxed": 0.3, "playful": 0.2}
    if body_language is None:
        body_language = {
            "ears": "forward", "eyes": "wide open", "tail": "up high",
            "body_posture": "sitting upright", "mouth": "closed relaxed",
            "whiskers": "forward", "overall_tension": "low"
        }

    from src.monologue import MonologueGenerator
    print("加载 Qwen3-8B (4-bit)...")
    generator = MonologueGenerator()
    print("✓ 加载成功")

    for persona in ["catgirl", "hachimi", "maodie", "general"]:
        mono = generator.generate(emotion, emotion_scores, body_language, persona)
        print(f"\n✓ {persona}:")
        print(f"  {mono[:150]}..." if len(mono) > 150 else f"  {mono}")


def main():
    parser = argparse.ArgumentParser(description="HAJIMI 模块测试")
    parser.add_argument("module", nargs="?", default="all",
                        choices=["a", "b", "c", "d", "all"],
                        help="要测试的模块 (a/b/c/d/all)")
    args = parser.parse_args()

    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    if args.module == "a" or args.module == "all":
        crop = test_a()
        print()

    if args.module == "b" or args.module == "all":
        body_language = test_b()
        print()

    if args.module == "c" or args.module == "all":
        emotions = test_c(body_language=body_language if args.module == "all" else None)
        print()

    if args.module == "d" or args.module == "all":
        if args.module == "all":
            top_emotion = max(emotions, key=emotions.get)
            test_d(top_emotion, emotions, body_language)
        else:
            test_d()


if __name__ == "__main__":
    main()
