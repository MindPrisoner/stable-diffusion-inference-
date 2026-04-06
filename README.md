# Stable Diffusion Inference

这是一个本地 Stable Diffusion 1.5 推理项目，用于在已有模型权重的基础上生成图片，并对比不同推理参数对结果的影响。

## 项目目标

- 使用本地模型进行文本生成图像
- 测试 `negative prompt` 的效果
- 对比不同 `guidance scale` 的输出差异
- 对比不同 `num inference steps` 对生成质量的影响

## 目录结构

```text
stable_diffusion_inference/
├── generate.py              # 推理入口
├── requirements.txt         # Python 依赖
├── outputs/                 # 生成结果保存目录
└── README.md
```

## 依赖安装

建议先创建虚拟环境，再安装依赖：

```bash
pip install -r requirements.txt
```

项目主要依赖：

- `torch`
- `diffusers`
- `transformers`
- `safetensors`
- `accelerate`

## 模型准备

`generate.py` 中默认使用本地模型路径：

```python
/mnt/d/AIModels/sd15
```

这表示模型权重需要提前下载到本地，并且目录中应包含 Stable Diffusion 1.5 所需文件。

如果你的模型存放位置不同，只需要修改 `generate.py` 里的 `model_path` 即可。

## 运行方式

直接执行：

```bash
python generate.py
```

脚本会自动：

1. 创建 `outputs/` 目录
2. 加载本地 Stable Diffusion 模型
3. 根据 prompt 生成图片
4. 保存输出到 `outputs/`

## 推理参数

`generate_image()` 支持以下参数：

- `prompt`：正向提示词
- `negative_prompt`：负向提示词
- `steps`：推理步数
- `guidance`：Classifier-Free Guidance 强度
- `height` / `width`：生成分辨率
- `output_path`：图片保存路径

默认示例配置为：

- `steps = 20`
- `guidance = 7.5`
- `height = 384`
- `width = 384`

## 结果示例

项目中已经保存了多张结果图，用于对比不同设置：

- `outputs/no_negative.png`
- `outputs/have_negative.png`
- `outputs/guidance_5.png`
- `outputs/guidance_7.5.png`
- `outputs/guidance_10.png`
- `outputs/step_10.png`
- `outputs/step_20.png`
- `outputs/step_30.png`
- `outputs/generated_sd15.png`

## 实验观察

这个项目主要适合观察下面几类现象：

- 加入 `negative prompt` 后，画面中的噪点和畸变通常会减少
- `guidance` 较小时，模型更自由，但可能更发散
- `guidance` 较大时，结果更贴近提示词，但可能牺牲一点自然度
- `steps` 较少时，细节不足
- `steps` 增加后，图像一般更稳定，但推理时间也会更长

## 注意事项

- 该脚本使用 `local_files_only=True`，不会在线下载模型
- 如果有 GPU，脚本会优先使用 CUDA
- 如果没有 GPU，会回退到 CPU，但推理会明显更慢
- `variant="fp16"` 适合半精度模型文件，若你的本地模型结构不同，可能需要同步调整加载参数

