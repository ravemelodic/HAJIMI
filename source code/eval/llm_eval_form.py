import json
import csv
import random
from pathlib import Path
from datetime import datetime

import gradio as gr
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_FILE = PROJECT_ROOT / "eval" / "human_eval_results.csv"

LABELS = {
    "中文": {
        "title": "# HAJIMI LLM 内心独白人工评估",
        "desc": "请根据猫咪图片和生成的内心独白，在 4 个维度上打分（1-5 分）。",
        "evaluator": "评估者姓名",
        "eval_ph": "输入你的名字",
        "monologue": "内心独白",
        "relevance": "相关性（独白匹配情绪?）",
        "consistency": "人格一致性",
        "creativity": "创意",
        "humor": "幽默",
        "notes": "备注（可选）",
        "notes_ph": "任何额外评价...",
        "submit": "提交评分",
        "status": "状态",
    },
    "English": {
        "title": "# HAJIMI LLM Monologue Human Evaluation",
        "desc": "Rate each cat monologue on 4 dimensions (1–5).",
        "evaluator": "Evaluator Name",
        "eval_ph": "Enter your name",
        "monologue": "Inner Monologue",
        "relevance": "Relevance (monologue matches emotion?)",
        "consistency": "Persona Consistency",
        "creativity": "Creativity",
        "humor": "Humor",
        "notes": "Notes (optional)",
        "notes_ph": "Any additional comments...",
        "submit": "Submit Rating",
        "status": "Status",
    },
}


def load_eval_items():
    path = PROJECT_ROOT / "eval" / "monologue_eval_data.json"
    if not path.exists():
        print(f"警告: 未找到评估数据，请先运行 generate_monologue_eval.py")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_rating(evaluator_name, image_id, relevance, consistency, creativity, humor, notes):
    write_header = not RESULTS_FILE.exists()
    with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "evaluator", "image_id",
                             "relevance", "consistency", "creativity", "humor", "notes"])
        writer.writerow([datetime.now().isoformat(), evaluator_name, image_id,
                         relevance, consistency, creativity, humor, notes])
    return f"✓ Saved / 已保存  ({image_id})"


def create_eval_form():
    eval_items = load_eval_items()
    random.shuffle(eval_items)
    items = eval_items[:20]
    n = len(items)

    # component refs for language update
    all_mono, all_rel, all_con, all_cre, all_hum = [], [], [], [], []
    all_notes_boxes, all_submit_btns, all_status_boxes = [], [], []

    with gr.Blocks(title="HAJIMI LLM Eval") as demo:
        with gr.Row():
            lang = gr.Radio(["中文", "English"], value="中文",
                            label="Language / 语言", scale=0)

        title_md = gr.Markdown(LABELS["中文"]["title"])
        desc_md = gr.Markdown(LABELS["中文"]["desc"])
        evaluator = gr.Textbox(label=LABELS["中文"]["evaluator"],
                               placeholder=LABELS["中文"]["eval_ph"])

        if not items:
            gr.Markdown("**请先运行 generate_monologue_eval.py 生成评估数据**")
        else:
            for i, item in enumerate(items):
                with gr.Group():
                    condition = item.get("condition", "")
                    cond_badge = f"  `{condition}`" if condition else ""
                    gr.Markdown(f"### #{i+1}{cond_badge}  —  emotion: **{item.get('emotion','?')}**  |  persona: **{item.get('persona','?')}**")

                    item_id = gr.State(value=item.get("id", f"item_{i}"))

                    with gr.Row():
                        img_path = item.get("image_path", "")
                        if Path(img_path).exists():
                            gr.Image(value=img_path, label="", height=260, show_label=False)

                        with gr.Column():
                            mono = gr.Textbox(
                                value=item.get("monologue", ""),
                                label=LABELS["中文"]["monologue"],
                                lines=5, interactive=False,
                            )
                            all_mono.append(mono)

                    with gr.Row():
                        rel = gr.Slider(1, 5, step=1, value=3, label=LABELS["中文"]["relevance"])
                        con = gr.Slider(1, 5, step=1, value=3, label=LABELS["中文"]["consistency"])
                        cre = gr.Slider(1, 5, step=1, value=3, label=LABELS["中文"]["creativity"])
                        hum = gr.Slider(1, 5, step=1, value=3, label=LABELS["中文"]["humor"])
                        all_rel.append(rel); all_con.append(con)
                        all_cre.append(cre); all_hum.append(hum)

                    notes_box = gr.Textbox(label=LABELS["中文"]["notes"],
                                           placeholder=LABELS["中文"]["notes_ph"])
                    submit_btn = gr.Button(LABELS["中文"]["submit"])
                    status_box = gr.Textbox(label=LABELS["中文"]["status"], interactive=False)
                    all_notes_boxes.append(notes_box)
                    all_submit_btns.append(submit_btn)
                    all_status_boxes.append(status_box)

                    submit_btn.click(
                        fn=save_rating,
                        inputs=[evaluator, item_id, rel, con, cre, hum, notes_box],
                        outputs=status_box,
                    )

        # --- language toggle ---
        def switch_lang(choice):
            L = LABELS[choice]
            updates = [
                gr.update(value=L["title"]),
                gr.update(value=L["desc"]),
                gr.update(label=L["evaluator"], placeholder=L["eval_ph"]),
            ]
            for _ in range(n):
                updates += [
                    gr.update(label=L["monologue"]),
                    gr.update(label=L["relevance"]),
                    gr.update(label=L["consistency"]),
                    gr.update(label=L["creativity"]),
                    gr.update(label=L["humor"]),
                    gr.update(label=L["notes"], placeholder=L["notes_ph"]),
                    gr.update(value=L["submit"]),
                    gr.update(label=L["status"]),
                ]
            return updates

        all_outputs = [title_md, desc_md, evaluator]
        for mono, rel, con, cre, hum, nb, sb, stb in zip(
            all_mono, all_rel, all_con, all_cre, all_hum,
            all_notes_boxes, all_submit_btns, all_status_boxes
        ):
            all_outputs += [mono, rel, con, cre, hum, nb, sb, stb]

        lang.change(fn=switch_lang, inputs=lang, outputs=all_outputs)

    return demo


if __name__ == "__main__":
    demo = create_eval_form()
    demo.launch(server_port=7861)
