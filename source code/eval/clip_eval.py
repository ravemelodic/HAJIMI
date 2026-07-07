"""
CLIP 情绪分类评估脚本
命令行运行版本，等同于 03_clip_ablation.ipynb 的评估逻辑

用法:
    python eval/clip_eval.py
    python eval/clip_eval.py --strategy descriptive
"""

import json
import argparse
from pathlib import Path

from PIL import Image
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.detector import CatDetector
from src.emotion import EmotionClassifier
from src.body_language import BodyLanguageAnalyzer

EMOTIONS = ["relaxed", "curious", "fearful", "aggressive", "playful", "content"]


def load_ground_truth(annotations_dir: Path) -> dict:
    """加载并合并两位标注者的标签"""
    ann1_path = annotations_dir / "annotator_1.json"
    ann2_path = annotations_dir / "annotator_2.json"

    with open(ann1_path, "r") as f:
        ann1 = json.load(f)
    with open(ann2_path, "r") as f:
        ann2 = json.load(f)

    # 合并标签 (一致取该标签，不一致取标注者1)
    ground_truth = {}
    for img_name in ann1:
        e1 = ann1[img_name].get("primary_emotion", "")
        e2 = ann2.get(img_name, {}).get("primary_emotion", "")
        if e1:
            ground_truth[img_name] = e1 if e1 == e2 else e1

    return ground_truth


def evaluate(strategy: str = None):
    """运行评估"""
    project_root = Path(__file__).parent.parent
    eval_dir = project_root / "data" / "eval_set"
    annotations_dir = project_root / "eval" / "annotations"

    # 加载 ground truth
    gt = load_ground_truth(annotations_dir)
    print(f"Ground truth: {len(gt)} 张图片")

    # 加载模型
    print("加载模型...")
    detector = CatDetector()
    classifier = EmotionClassifier()
    body_analyzer = BodyLanguageAnalyzer() if strategy in (None, "body_anchored") else None

    # 确定要评估的策略
    strategies = [strategy] if strategy else ["simple", "descriptive", "body_anchored", "expert"]

    for strat in strategies:
        print(f"\n{'=' * 50}")
        print(f"评估策略: {strat}")
        print(f"{'=' * 50}")

        gt_labels, predictions = [], []

        for img_path in sorted(eval_dir.glob("*.jpg")):
            if img_path.name not in gt:
                continue

            img = Image.open(img_path)
            detections = detector.detect(img)

            if not detections:
                continue

            crop = detections[0]["crop"]
            body = body_analyzer.analyze(crop) if strat == "body_anchored" and body_analyzer else None
            emotions = classifier.classify(crop, strategy=strat, body_language=body)
            pred = max(emotions, key=emotions.get)

            gt_labels.append(gt[img_path.name])
            predictions.append(pred)

        # 输出结果
        acc = accuracy_score(gt_labels, predictions)
        print(f"\n准确率: {acc:.1%}")
        print(f"\n分类报告:")
        print(classification_report(gt_labels, predictions, labels=EMOTIONS, zero_division=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP 情绪分类评估")
    parser.add_argument("--strategy", type=str, default=None,
                        choices=["simple", "descriptive", "body_anchored", "expert"],
                        help="指定单个策略评估，不指定则评估所有策略")
    args = parser.parse_args()
    evaluate(args.strategy)
