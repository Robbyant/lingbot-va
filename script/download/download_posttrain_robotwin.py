#!/usr/bin/env python3
"""
从 HuggingFace 或 ModelScope 下载 LingBot-VA RoboTwin 后训权重 (lingbot-va-posttrain-robotwin)。
用于 RoboTwin 2.0 评测时达到论文报告成功率，需先下载再设置 LINGBOT_VA_MODEL_PATH。

用法:
  python script/download/download_posttrain_robotwin.py
  python script/download/download_posttrain_robotwin.py /path/to/save
  HF_ENDPOINT=https://hf-mirror.com python script/download/download_posttrain_robotwin.py  # 国内镜像

下载完成后评测前设置:
  export LINGBOT_VA_MODEL_PATH=/path/to/lingbot-va-posttrain-robotwin
  bash script/run_eval_robotwin.sh
"""
import os
import sys

REPO_ID_HF = "robbyant/lingbot-va-posttrain-robotwin"
REPO_ID_MS = "Robbyant/lingbot-va-posttrain-robotwin"


def main():
    # 仓库根目录 = script/download 的上级的上级
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_dir = os.path.join(repo_root, "lingbot-va-posttrain-robotwin")
    local_dir = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else default_dir

    use_modelscope = os.environ.get("USE_MODELSCOPE", "").lower() in ("1", "true", "yes")

    if use_modelscope:
        try:
            from modelscope import snapshot_download
        except ImportError:
            print("请先安装: pip install modelscope")
            sys.exit(1)
        print(f"正在从 ModelScope 下载: {REPO_ID_MS}")
        print(f"保存到: {local_dir}")
        os.makedirs(local_dir, exist_ok=True)
        try:
            path = snapshot_download(REPO_ID_MS, local_dir=local_dir)
        except Exception as e:
            print(f"下载失败: {e}")
            sys.exit(1)
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("请先安装: pip install huggingface_hub")
            sys.exit(1)
        print(f"正在从 HuggingFace 下载: {REPO_ID_HF}")
        print(f"保存到: {local_dir}")
        os.makedirs(local_dir, exist_ok=True)
        try:
            path = snapshot_download(
                repo_id=REPO_ID_HF,
                local_dir=local_dir,
            )
        except Exception as e:
            print(f"下载失败: {e}")
            print("  国内用户可试: HF_ENDPOINT=https://hf-mirror.com python script/download/download_posttrain_robotwin.py")
            sys.exit(1)

    print(f"\n下载完成: {path}")
    print("RoboTwin 评测前设置:")
    print(f"  export LINGBOT_VA_MODEL_PATH={path}")
    print("  或在运行脚本前: LINGBOT_VA_MODEL_PATH=%s bash script/run_eval_robotwin.sh" % path)


if __name__ == "__main__":
    main()
