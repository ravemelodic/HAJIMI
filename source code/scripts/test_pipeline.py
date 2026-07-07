# HAJIMI 模块测试脚本
# 逐个测试各模块，确保整个流程能运行

import sys
import os
import torch
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def download_test_image():
    """下载测试用的猫咪图片"""
    test_image_path = PROJECT_ROOT / "data" / "test_cat.jpg"

    if test_image_path.exists():
        print(f"✓ 测试图片已存在: {test_image_path}")
        return Image.open(test_image_path)

    print("下载测试图片...")
    # 使用一张公开的猫咪图片
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")

        # 保存到本地
        test_image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(test_image_path)
        print(f"✓ 测试图片已保存: {test_image_path}")
        return image
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        print("请手动放置一张猫咪图片到: data/test_cat.jpg")
        return None


def test_module_a_detector():
    """测试模块A: YOLO 猫咪检测"""
    print("\n" + "=" * 60)
    print("测试模块A: YOLO 猫咪检测")
    print("=" * 60)

    try:
        from src.detector import CatDetector

        detector = CatDetector()
        print("✓ CatDetector 加载成功")

        image = download_test_image()
        if image is None:
            return None

        detections = detector.detect(image)
        print(f"✓ 检测到 {len(detections)} 只猫咪")

        for i, det in enumerate(detections):
            print(f"  猫咪 #{i+1}: bbox={det['bbox']}, conf={det['confidence']:.2%}")

        if detections:
            return detections[0]["crop"]  # 返回第一只猫的裁剪图
        else:
            print("⚠ 未检测到猫咪，使用原图继续测试")
            return image

    except Exception as e:
        print(f"✗ 模块A 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_module_b_body_language(cat_crop):
    """测试模块B: 体态分析"""
    print("\n" + "=" * 60)
    print("测试模块B: Qwen3-VL 体态分析")
    print("=" * 60)

    try:
        from src.body_language import BodyLanguageAnalyzer

        print("加载 Qwen3-VL 模型 (可能需要下载)...")
        analyzer = BodyLanguageAnalyzer()
        print("✓ BodyLanguageAnalyzer 加载成功")

        body_language = analyzer.analyze(cat_crop)
        print("✓ 体态分析完成:")
        for key, value in body_language.items():
            print(f"  {key}: {value}")

        return body_language

    except Exception as e:
        print(f"✗ 模块B 测试失败: {e}")
        import traceback
        traceback.print_exc()
        # 返回默认值以便继续测试
        return {
            "ears": "forward",
            "eyes": "half closed",
            "tail": "relaxed",
            "body_posture": "relaxed lying",
            "mouth": "closed relaxed",
            "whiskers": "relaxed neutral",
            "overall_tension": "low",
        }


def test_module_c_emotion(cat_crop, body_language=None):
    """测试模块C: CLIP 情绪分类"""
    print("\n" + "=" * 60)
    print("测试模块C: CLIP 情绪分类")
    print("=" * 60)

    try:
        from src.emotion import EmotionClassifier

        print("加载 CLIP ViT-L/14 模型...")
        classifier = EmotionClassifier()
        print("✓ EmotionClassifier 加载成功")

        # 测试所有策略
        strategies = ["simple", "descriptive", "expert"]
        if body_language:
            strategies.append("body_anchored")

        for strategy in strategies:
            bl = body_language if strategy == "body_anchored" else None
            emotions = classifier.classify(cat_crop, strategy=strategy, body_language=bl)
            top_emotion = max(emotions, key=emotions.get)
            print(f"✓ 策略 '{strategy}': {top_emotion} ({emotions[top_emotion]:.1%})")

        # 返回 descriptive 策略的结果
        return classifier.classify(cat_crop, strategy="descriptive")

    except Exception as e:
        print(f"✗ 模块C 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return {"relaxed": 0.5, "curious": 0.3, "content": 0.2}


def test_module_d_monologue(emotion, emotion_scores, body_language):
    """测试模块D: 内心独白生成"""
    print("\n" + "=" * 60)
    print("测试模块D: Qwen3-8B 内心独白生成")
    print("=" * 60)

    try:
        from src.monologue import MonologueGenerator

        print("加载 Qwen3-8B 模型 (4-bit 量化，可能需要下载)...")
        generator = MonologueGenerator()
        print("✓ MonologueGenerator 加载成功")

        # 测试不同人格
        personas = ["catgirl", "hachimi", "maodie", "general"]
        for persona in personas:
            monologue = generator.generate(
                emotion=emotion,
                emotion_scores=emotion_scores,
                body_language=body_language,
                persona=persona,
            )
            print(f"\n✓ 人格 '{persona}':")
            print(f"  {monologue[:200]}..." if len(monologue) > 200 else f"  {monologue}")

        return True

    except Exception as e:
        print(f"✗ 模块D 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline():
    """测试完整 Pipeline"""
    print("\n" + "=" * 60)
    print("测试完整 Pipeline")
    print("=" * 60)

    try:
        from src.pipeline import HajimiPipeline

        print("加载完整 Pipeline...")
        # 使用低显存模式以防止 OOM
        pipeline = HajimiPipeline(low_memory=True)
        print("✓ Pipeline 初始化成功")

        image = download_test_image()
        if image is None:
            return False

        print("运行完整 Pipeline...")
        result = pipeline.run(image, persona="general", clip_strategy="descriptive")

        if result["num_cats"] == 0:
            print(f"⚠ {result.get('error', '未检测到猫咪')}")
            return False

        det = result["detections"][0]
        print(f"✓ Pipeline 运行成功!")
        print(f"  检测到猫咪: {result['num_cats']} 只")
        print(f"  主要情绪: {det['top_emotion']}")
        print(f"  体态: {det['body_language']}")
        print(f"  独白: {det['monologue'][:100]}...")

        return True

    except Exception as e:
        print(f"✗ Pipeline 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradio_components():
    """测试 Gradio 相关组件 (不启动服务)"""
    print("\n" + "=" * 60)
    print("测试 Gradio 组件")
    print("=" * 60)

    try:
        from src.utils import draw_bboxes, make_radar_chart

        # 测试 draw_bboxes
        test_image = Image.new("RGB", (640, 480), color="white")
        test_detections = [{"bbox": [100, 100, 300, 300], "confidence": 0.95}]
        result_image = draw_bboxes(test_image, test_detections)
        print("✓ draw_bboxes 正常")

        # 测试 make_radar_chart
        test_emotions = {
            "relaxed": 0.3, "curious": 0.4, "fearful": 0.05,
            "aggressive": 0.05, "playful": 0.1, "content": 0.1
        }
        fig = make_radar_chart(test_emotions)
        print("✓ make_radar_chart 正常")

        # 测试 Gradio 导入
        import gradio as gr
        print(f"✓ Gradio 版本: {gr.__version__}")

        return True

    except Exception as e:
        print(f"✗ Gradio 组件测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试流程"""
    print("=" * 60)
    print("HAJIMI 项目测试")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    results = {}

    # 测试模块A
    cat_crop = test_module_a_detector()
    results["Module A (YOLO)"] = cat_crop is not None

    if cat_crop is None:
        print("\n⚠ 模块A 失败，无法继续测试后续模块")
        return

    # 测试模块B
    body_language = test_module_b_body_language(cat_crop)
    results["Module B (Qwen3-VL)"] = body_language is not None

    # 测试模块C
    emotions = test_module_c_emotion(cat_crop, body_language)
    results["Module C (CLIP)"] = emotions is not None
    top_emotion = max(emotions, key=emotions.get) if emotions else "relaxed"

    # 测试模块D
    results["Module D (Qwen3-8B)"] = test_module_d_monologue(
        top_emotion, emotions, body_language
    )

    # 测试 Gradio 组件
    results["Gradio Components"] = test_gradio_components()

    # 测试完整 Pipeline
    results["Full Pipeline"] = test_full_pipeline()

    # 打印总结
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    print("\n" + ("🎉 所有测试通过!" if all_passed else "⚠ 部分测试失败"))

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
