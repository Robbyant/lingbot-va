# LingBot-VA 在 RoboTwin-2.0 上 Eval 说明

## 前置条件

- **RoboTwin-2.0** 已下载，且 **assets** 已就绪（你当前路径：`/mnt/users/wangyuxuan-20250915/EAI/RoboTwin-2.0`）
- **LingBot-VA** 模型目录：`lingbot-va-base`（或通过 `LINGBOT_VA_MODEL_PATH` 指定）
- 推理时 `lingbot-va-base/transformer/config.json` 中 `attn_mode` 为 `"torch"` 或 `"flashattn"`（你当前已是 `flashattn`，无需改）

## 环境依赖（重要）

Eval 时 **client 会启动 RoboTwin 仿真**，当前 Python 环境必须能 `import sapien`，否则会报错：

```text
ModuleNotFoundError: No module named 'sapien'
```

请在本机 **用于跑 eval 的 conda 环境**（如 `lingbot-va`）中安装 RoboTwin 仿真依赖，例如：

```bash
# 0. setuptools 需 <82，否则 sapien 的 pkg_resources 会报错
pip install 'setuptools<82'

# 1. 系统（若未装）
sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools

# 2. 在 lingbot-va 环境中安装（不覆盖已有 torch）
pip install sapien==3.0.0   # 或 3.0.0b1（若你从源码/其它源安装）
pip install open3d scipy mplib gymnasium trimesh imageio pydantic zarr h5py
pip install -r /mnt/users/wangyuxuan-20250915/EAI/RoboTwin-2.0/script/requirements.txt
```

**Curobo（运动规划）**：RoboTwin 的 env 依赖 [NVlabs/curobo](https://github.com/NVlabs/curobo)，需在 RoboTwin 仓库下按官方步骤安装：

```bash
cd /mnt/users/wangyuxuan-20250915/EAI/RoboTwin-2.0/envs
git clone https://github.com/NVlabs/curobo.git
cd curobo && pip install -e . --no-build-isolation && cd ../..
```

若与当前 PyTorch/CUDA 版本不兼容，可参考 [RoboTwin 安装文档](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) 使用与 LingBot-VA 兼容的 torch 版本后再装 curobo。  
并按需执行 RoboTwin 的 `script/_install.sh`（pytorch3d、mplib/sapien 补丁等）。

## 一键 Eval（同机 Server + Client）

在 **lingbot-va 仓库根目录** 执行：

```bash
export ROBOTWIN_ROOT=/mnt/users/wangyuxuan-20250915/EAI/RoboTwin-2.0
export LINGBOT_VA_MODEL_PATH=/mnt/users/wangyuxuan-20250915/EAI/lingbot-va/lingbot-va-base  # 可选，默认即仓库下 lingbot-va-base

# 默认：任务 adjust_bottle，test_num=100，结果到 ./results
bash script/run_eval_robotwin.sh

# 指定结果目录、任务、测试次数（快速试跑建议 test_num=2）
bash script/run_eval_robotwin.sh ./results adjust_bottle 2
```

脚本会先启动 LingBot-VA server（WebSocket），再启动 RoboTwin eval client，结果在 `save_root`（默认 `./results`）下。

## 仅跑 Client（Server 已另起）

若已在其他终端启动 LingBot-VA server（例如 `bash evaluation/robotwin/launch_server.sh`），可只跑 client：

```bash
export ROBOTWIN_ROOT=/mnt/users/wangyuxuan-20250915/EAI/RoboTwin-2.0
bash script/run_eval_robotwin_client_only.sh ./results adjust_bottle 2
```

默认连接 `PORT=29056`，可通过 `export PORT=29056` 修改。

## 可选：快速试跑脚本

```bash
bash script/run_eval_robotwin_quick.sh
```

等价于：`run_eval_robotwin.sh ./results adjust_bottle 2`，用于快速验证流程。

## 结果与指标

- 视频与可视化：`save_root/stseed-<seed>/visualization/<task_name>/`
- 成功率等：`save_root/stseed-<seed>/metrics/<task_name>/res.json`（`succ_num`、`total_num`、`succ_rate`）

## 常见问题

| 现象 | 处理 |
|------|------|
| `ModuleNotFoundError: No module named 'sapien'` | 在当前环境中安装 sapien 及 RoboTwin 依赖（见上方「环境依赖」） |
| `attn_mode` 相关报错 | 确认 `transformer/config.json` 中为 `"torch"` 或 `"flashattn"` |
| 未找到 `ROBOTWIN_ROOT/assets` | 先下载 RoboTwin assets，或设置 `ROBOTWIN_ROOT` 到正确路径 |
| 未找到 `curobo` / `CuroboPlanner` | 在 RoboTwin 下安装 curobo：`cd $ROBOTWIN_ROOT/envs && git clone https://github.com/NVlabs/curobo.git && cd curobo && pip install -e . --no-build-isolation` |
| Server 启动失败 | 检查 GPU、CUDA、以及模型路径 `LINGBOT_VA_MODEL_PATH` |
