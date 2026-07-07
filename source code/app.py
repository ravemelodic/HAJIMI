import time
import traceback

import gradio as gr
from PIL import Image

from src.pipeline import HajimiPipeline
from src.utils import draw_bboxes, make_radar_chart, load_config
from src.prompts import PERSONA_SYSTEM_PROMPTS, PERSONA_SYSTEM_PROMPTS_ZH

DOG_CLASS_ID = 16

pipeline = None


def initialize_pipeline():
    global pipeline
    if pipeline is None:
        print("Loading models for the first time...")
        pipeline = HajimiPipeline()
        print("All models loaded.")


def _has_dog(image):
    results = pipeline.detector.model(image, verbose=False)
    for r in results:
        for cls_id in r.boxes.cls:
            if int(cls_id.item()) == DOG_CLASS_ID:
                return True
    return False


def _get_cat_idx(cat_choice):
    try:
        return int(cat_choice.split()[1]) - 1 if cat_choice else 0
    except Exception:
        return 0


# Phase 1: detect + body language + CLIP
def detect_and_analyze(image):
    initialize_pipeline()

    if image is None:
        return (image,
                gr.update(visible=False),
                gr.update(value="Please upload an image first.", visible=True),
                None, {}, [])

    detections = pipeline.detector.detect(image)

    if not detections:
        msg = ("That looks like a dog! HAJIMI only understands cats."
               if _has_dog(image)
               else "No cat detected. Please upload a photo containing a cat.")
        return (image,
                gr.update(visible=False),
                gr.update(value=msg, visible=True),
                None, {}, [])

    for det in detections:
        det["body_language"] = pipeline.body_analyzer.analyze(det["crop"])
        det["emotions"] = pipeline.emotion_classifier.classify(det["crop"], strategy="descriptive")
        det["top_emotion"] = max(det["emotions"], key=det["emotions"].get)

    annotated = draw_bboxes(image, detections)
    n = len(detections)

    cat_choices = (["Cat 1"] if n == 1
                   else [f"Cat {i+1} ({d['confidence']:.0%})" for i, d in enumerate(detections)])
    cat_label = f"{n} cat{'s' if n > 1 else ''} detected" + (" — select one" if n > 1 else "")

    state = [{
        "bbox": d["bbox"],
        "confidence": d["confidence"],
        "body_language": d["body_language"],
        "emotions": d["emotions"],
        "top_emotion": d["top_emotion"],
    } for d in detections]

    first = state[0]
    return (
        annotated,
        gr.update(choices=cat_choices, value=cat_choices[0], label=cat_label, visible=(n > 1)),
        gr.update(value="", visible=False),
        make_radar_chart(first["emotions"]),
        first["body_language"],
        state,
    )


def update_emotion_display(state, cat_choice):
    if not state:
        return None, {}
    det = state[min(_get_cat_idx(cat_choice), len(state) - 1)]
    return make_radar_chart(det["emotions"]), det["body_language"]


# Phase 2: generate monologue
def get_system_prompt(persona, lang_choice):
    lang = "zh" if lang_choice == "Chinese" else "en"
    prompt_map = PERSONA_SYSTEM_PROMPTS_ZH if lang == "zh" else PERSONA_SYSTEM_PROMPTS
    return prompt_map.get(persona, "")


def generate_monologue(state, cat_choice, persona, lang_choice, custom_prompt, temperature):
    if not state:
        return "Please click 'Detect & Analyze' first."

    det = state[min(_get_cat_idx(cat_choice), len(state) - 1)]
    lang = "zh" if lang_choice == "Chinese" else "en"

    print(f"[generate] persona={persona} lang={lang} emotion={det['top_emotion']}")
    t0 = time.time()
    try:
        monologue = pipeline.monologue_gen.generate(
            emotion=det["top_emotion"],
            emotion_scores=det["emotions"],
            body_language=det["body_language"],
            persona=persona,
            lang=lang,
            system_prompt_override=custom_prompt.strip() if custom_prompt.strip() else None,
            temperature=temperature,
        )
    except Exception as e:
        traceback.print_exc()
        return f"Generation failed:\n{type(e).__name__}: {e}"

    print(f"[generate] done in {time.time() - t0:.1f}s, {len(monologue)} chars")

    if not monologue.strip():
        monologue = ("(Empty output — try increasing max_new_tokens in config.yaml "
                     "or make sure enable_thinking=false)")

    emotion = det["top_emotion"]
    conf = det["emotions"][emotion]
    persona_names = {
        "catgirl": "Catgirl",
        "hachimi":  "Hachimi",
        "maodie":  "Maodie",
        "general": "General",
    }
    header = (
        f"Emotion: {emotion} ({conf:.0%})\n"
        f"Persona: {persona_names.get(persona, persona)}\n"
        f"{'─' * 44}\n\n"
    )
    return header + monologue


def create_demo():
    persona_choices = [
        ("Catgirl",  "catgirl"),
        ("Hachimi",    "hachimi"),
        ("Maodie",   "maodie"),
        ("General",   "general"),
    ]

    with gr.Blocks(title="HAJIMI") as demo:
        gr.Markdown("""
        # Holistic Animal Judgment via Intelligent Multimodal Inference
        """)

        state = gr.State([])

        # Row 1: input + detection result
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="Upload Cat Image", height=320)
                analyze_btn = gr.Button("Detect & Analyze", variant="primary", size="lg")

            with gr.Column(scale=1):
                detection_image = gr.Image(label="Detection Result", height=320)
                error_msg = gr.Textbox(label="", visible=False, interactive=False, lines=2)

        cat_selector = gr.Radio(choices=[], label="Select Cat", visible=False)

        # Row 2: emotion chart + body language (equal height via min_height)
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                emotion_plot = gr.Plot(label="Emotion Distribution")
            with gr.Column(scale=1):
                body_lang_json = gr.JSON(label="Body Language Analysis", min_height=400)

        # Row 3: controls + monologue
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                lang_selector = gr.Radio(
                    ["English", "Chinese"], value="English",
                    label="Language",
                )
                persona_selector = gr.Radio(
                    choices=persona_choices, value="general",
                    label="Cat Persona",
                )
                temperature_slider = gr.Slider(
                    minimum=0.5, maximum=1.3, value=0.85, step=0.05,
                    label="Temperature",
                )
                generate_btn = gr.Button("Generate Monologue", variant="secondary", size="lg")

            with gr.Column(scale=2):
                monologue_out = gr.Textbox(
                    label="Cat's Inner Monologue",
                    lines=10, interactive=False,
                    placeholder="Click 'Detect & Analyze' first, then 'Generate Monologue'.",
                )

        # Advanced: view/edit system prompt
        with gr.Accordion("Advanced: View / Edit System Prompt", open=False):
            gr.Markdown("The system prompt below controls the cat's persona. You can edit it before generating.")
            prompt_display = gr.Textbox(
                label="System Prompt (editable)",
                value=PERSONA_SYSTEM_PROMPTS["general"],
                lines=8,
                interactive=True,
            )
            reset_btn = gr.Button("Reset to Default", size="sm")

        gr.Markdown("---\n**Stack**: YOLOv8 · CLIP ViT-L/14 · Qwen3-VL-2B · Qwen3-8B+LoRA  |  COMP7065 Mini-Project")

        # Event bindings
        analyze_btn.click(
            fn=detect_and_analyze,
            inputs=[input_image],
            outputs=[detection_image, cat_selector, error_msg, emotion_plot, body_lang_json, state],
        )

        cat_selector.change(
            fn=update_emotion_display,
            inputs=[state, cat_selector],
            outputs=[emotion_plot, body_lang_json],
        )

        persona_selector.change(
            fn=get_system_prompt,
            inputs=[persona_selector, lang_selector],
            outputs=[prompt_display],
        )
        lang_selector.change(
            fn=get_system_prompt,
            inputs=[persona_selector, lang_selector],
            outputs=[prompt_display],
        )
        reset_btn.click(
            fn=get_system_prompt,
            inputs=[persona_selector, lang_selector],
            outputs=[prompt_display],
        )

        generate_btn.click(
            fn=generate_monologue,
            inputs=[state, cat_selector, persona_selector, lang_selector, prompt_display, temperature_slider],
            outputs=[monologue_out],
        )

    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="path to config yaml (default: configs/config.yaml)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="create public gradio link")
    args = parser.parse_args()

    # 把 config 路径传给 pipeline（通过覆盖 load_config 默认路径）
    if args.config:
        from src import utils
        _orig_load = utils.load_config
        utils.load_config = lambda path=None: _orig_load(args.config)

    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
