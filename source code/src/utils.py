import yaml
from pathlib import Path
from PIL import Image, ImageDraw


def load_config(config_path=None):
    if config_path is None:
        project_root = Path(__file__).parent.parent
        config_path = project_root / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def draw_bboxes(image, detections):
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
        label = f"Cat #{i+1} ({conf:.1%})"
        text_bbox = draw.textbbox((x1, y1 - 20), label)
        draw.rectangle(text_bbox, fill="lime")
        draw.text((x1, y1 - 20), label, fill="black")
    return img_draw

# 向外扩展15%裁切
def crop_with_padding(image, bbox, padding=0.15):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    pad_w, pad_h = int(w * padding), int(h * padding)
    img_w, img_h = image.size
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(img_w, x2 + pad_w)
    y2 = min(img_h, y2 + pad_h)
    return image.crop((x1, y1, x2, y2))


def make_radar_chart(emotions):
    import plotly.graph_objects as go

    emotion_labels = {
        "relaxed":    "Relaxed",
        "curious":    "Curious",
        "fearful":    "Fearful",
        "aggressive": "Aggressive",
        "playful":    "Playful",
        "content":    "Content",
    }

    categories = [emotion_labels.get(e, e) for e in emotions]
    values = list(emotions.values())

    # 闭合多边形 （雷达图）
    categories.append(categories[0])
    values.append(values[0])

    fig = go.Figure(data=go.Scatterpolar(
        r=values, theta=categories,
        fill="toself",
        fillcolor="rgba(99, 110, 250, 0.3)",
        line=dict(color="rgb(99, 110, 250)", width=2),
        marker=dict(size=6),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1],
                                   tickvals=[0.2, 0.4, 0.6, 0.8, 1.0])),
        showlegend=False,
        title="Emotion Distribution",
        width=450, height=400,
        margin=dict(l=60, r=60, t=60, b=60),
    )
    return fig
