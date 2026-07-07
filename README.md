# HAJIMI - 猫咪情绪理解系统

**H**olistic **A**nimal **J**udgment via **I**ntelligent **M**ultimodal **I**nference

## 项目简介

HAJIMI 是一个基于多模态 AI 的猫咪情绪理解系统。用户上传一张猫咪照片，系统会自动完成：

1. **猫咪检测** — 使用 YOLOv8 定位图片中的猫咪
2. **体态分析** — 使用 Qwen3-VL-2B 分析猫咪的耳朵、尾巴、姿势等体态特征
3. **情绪分类** — 使用 CLIP 零样本分类，判断猫咪的情绪状态
4. **内心独白** — 使用 Qwen3-8B (LoRA 微调) 生成拟人化的猫咪内心独白 (4 种人格: 猫娘/哈基米/耄耋/通用)

## 系统架构

```
图片输入 -> [YOLO 检测] -> 裁剪猫咪区域
                              |
                              v
                     [Qwen3-VL 体态分析] -> 结构化体态 JSON
                              |                    |
                              v                    v
                     [CLIP 情绪分类] -----> 情绪概率分布
                                                   |
                                                   v
                     [Qwen3-8B + LoRA] -----> 内心独白文本
```

## 模型配置

| 模块 | 模型 | 精度 | 显存占用 |
|------|------|------|---------|
| 猫咪检测 | YOLOv8n | FP16 | ~0.5GB |
| 体态分析 | Qwen3-VL-2B-Instruct | FP16 | ~4GB |
| 情绪分类 | CLIP ViT-L/14 | FP16 | ~1.5GB |
| 内心独白 | Qwen3-8B + LoRA | 4-bit | ~5GB |
| **合计** | | | **~11GB** |

> 推荐 GPU: NVIDIA RTX 4080/4090 (16GB+) 或同等显存的显卡

## 快速开始

### 1. 环境设置

```bash
cd hajimi

# 方式一: 使用设置脚本 (推荐)
python setup.py --install

# 方式二: 手动安装
pip install -r requirements.txt
```

### 2. 验证环境

```bash
# 检查依赖是否安装成功
python setup.py --verify

# 逐个测试各模块
python test_modules.py a      # 测试 YOLO 检测
python test_modules.py b      # 测试 Qwen3-VL 体态分析
python test_modules.py c      # 测试 CLIP 情绪分类
python test_modules.py d      # 测试 Qwen3-8B 内心独白
python test_modules.py all    # 测试全部模块
```

### 3. 数据准备 (可选)

运行 Jupyter Notebook 下载数据集和预缓存模型：

```bash
jupyter notebook notebooks/01_data_preparation.ipynb
```

### 4. LoRA 微调（可选）

```bash
jupyter notebook notebooks/02_lora_training.ipynb
```

### 5. 启动 Demo

```bash
python app.py
```

浏览器访问 `http://localhost:7860` 即可使用。

## 项目结构

```
hajimi/
├── notebooks/                  # Jupyter Notebooks
│   ├── 01_data_preparation     # 数据下载 + 模型缓存
│   ├── 02_lora_training        # LoRA 微调
│   └── 03_clip_ablation        # CLIP 消融实验
├── src/                        # 核心代码
│   ├── detector.py             # 模块A: YOLO 猫咪检测
│   ├── body_language.py        # 模块B: 体态分析
│   ├── emotion.py              # 模块C: 情绪分类
│   ├── monologue.py            # 模块D: 内心独白
│   ├── pipeline.py             # Pipeline 编排器
│   └── prompts.py              # Prompt 模板
├── app.py                      # Gradio Demo
├── eval/                       # 评估脚本
├── data/                       # 数据目录
├── docs/                       # 文档
├── models/                     # 模型权重
└── configs/                    # 配置文件
```

## 评估方案

- **CLIP 消融实验**: 4 种 prompt 策略在 50 张标注图片上的准确率对比
- **LLM 人工评估**: 5 分 Likert 量表评估独白的相关性、人格一致性、创意和幽默感
- **知识蒸馏评估**: Claude Opus 4.6 生成合成数据 -> LoRA 微调 Qwen3-8B，对比微调前后人格表现

## 课程信息

COMP7065 Mini-Project | 团队成员: Huang Jimin, WU Xizhe
