# LingBot-VA 推理调试指南

按以下步骤可在单机上把 **Image-to-Video-Action (i2va)** 推理跑通。

## 1. 环境

- Python 3.10、PyTorch 2.9、CUDA 12.6（与 README 一致）
- 安装依赖后，**推理** 时 `transformer` 的 `attn_mode` 必须为 `"torch"` 或 `"flashattn"`，不能为 `"flex"`

## 2. 下载模型

从 [HuggingFace](https://huggingface.co/robbyant/lingbot-va-base) 或 [ModelScope](https://modelscope.cn/models/Robbyant/lingbot-va-base) 下载 **lingbot-va-base**，得到本地目录，例如：

```text
/path/to/lingbot-va-base/
├── vae/
├── tokenizer/
├── text_encoder/
└── transformer/
```

## 3. 设置推理用 attn_mode

编辑 **`<模型目录>/transformer/config.json`**，将 `"attn_mode"` 改为 `"torch"` 或 `"flashattn"`：

```json
"attn_mode": "torch"
```

（训练时为 `"flex"`，推理必须改掉，否则会报错。）

## 4. 准备首帧图像（i2va）

使用 **robotwin_i2av** 配置时，需要 3 张首帧 PNG，放在 `example/robotwin/` 下，文件名为：

- `observation.images.cam_high.png`
- `observation.images.cam_left_wrist.png`
- `observation.images.cam_right_wrist.png`

**快速生成占位图**（仅用于跑通流程）：

```bash
cd /path/to/lingbot-va
python example/robotwin/create_dummy_images.py
```

会在这 3 个文件名下生成 256x320 的占位图。

## 5. 单 GPU 跑 i2va

在仓库根目录下执行：

```bash
export LINGBOT_VA_MODEL_PATH=/path/to/lingbot-va-base
bash script/run_i2va_single_gpu.sh
```

未设置 `LINGBOT_VA_MODEL_PATH` 或路径不对时，脚本会报错并提示。

- 结果会写到 `save_root` 下的 `real/<prompt>_<时间>/`（默认 `./train_out` 可改）。
- 其中会保存 `latents_*.pt`、`actions_*.pt`，以及用首帧 + 预测 latent 解码得到的 `demo.mp4`（在 generate 流程里）。

## 6. 可选：换配置 / 减少步数

- 使用 **demo** 配置（2 视角、不同 action 维度）时，可改用 `demo_i2av`，并设置 `example/demo/` 下对应名称的 PNG（见 `va_demo_cfg.obs_cam_keys`）。
- 在对应 config 里可调：
  - `num_inference_steps` / `action_num_inference_steps`：减小可加快推理（质量会下降）；
  - `num_chunks_to_infer`：i2va 生成的总 chunk 数；
  - `frame_chunk_size`：每个 chunk 的帧数。

## 7. 常见错误

| 现象 | 处理 |
|------|------|
| `attn_mode` 相关报错 | 确认 `transformer/config.json` 里为 `"torch"` 或 `"flashattn"` |
| 找不到 `observation.images.*.png` | 在 `example/robotwin/` 下运行 `create_dummy_images.py` 或自行放置同名 PNG |
| CUDA OOM | 使用 README 中的 offload（VAE、text_encoder 放到 CPU），或减小 `frame_chunk_size` / 推理步数 |
| 找不到 `wan_va` 模块 | 在 **仓库根目录**（含 `wan_va` 的上一级）执行 `bash script/run_i2va_single_gpu.sh` |

## 8. Server 模式（与仿真器联调）

若要与 RoboTwin 等仿真器联调，使用 **server** 模式：

- 启动推理服务：`bash evaluation/robotwin/launch_server.sh`（需先设好 `wan22_pretrained_model_name_or_path` 或 `LINGBOT_VA_MODEL_PATH`）
- 再在另一终端启动 client / 仿真器，见 README 的 RoboTwin 部署说明。

上述步骤可保证 i2va 推理从环境、模型、首帧到单 GPU 脚本整条链路打通；若某一步报错，把报错信息与对应步骤号贴出来即可继续排查。

---

## 9. 下载 Post-Training 数据集（用于微调）

**该数据集仅在 HuggingFace 提供，ModelScope 无此数据集。**

- 数据集：`robbyant/robotwin-clean-and-aug-lerobot`（LeRobot 格式，用于 post-training 微调）

**一键下载到仓库下默认目录：**

```bash
# 需先安装: pip install huggingface_hub
python script/download_dataset.py
```

**下载到指定目录：**

```bash
python script/download_dataset.py /path/to/save
```

下载完成后，训练时设置环境变量即可：

```bash
export LINGBOT_VA_DATASET_PATH=/path/to/robotwin-clean-and-aug-lerobot
NGPU=8 bash script/run_va_posttrain.sh
```

配置里已支持从 `LINGBOT_VA_DATASET_PATH` 读取数据集路径（见 `wan_va/configs/va_robotwin_train_cfg.py`）。
