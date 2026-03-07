#!/usr/bin/env python3
"""
从 HuggingFace 下载 LingBot-VA post-training 数据集 (robotwin-clean-and-aug-lerobot)。
README 中该数据集仅在 HuggingFace 提供，ModelScope 无此数据集。

用法:
  python script/download_dataset.py
  python script/download_dataset.py /path/to/save
"""
import os
import sys

def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_dir = os.path.join(repo_root, "robotwin-clean-and-aug-lerobot")
    local_dir = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else default_dir

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("请先安装: pip install huggingface_hub")
        sys.exit(1)

    repo_id = "robbyant/robotwin-clean-and-aug-lerobot"
    print(f"正在从 HuggingFace 下载数据集: {repo_id}")
    print(f"保存到: {local_dir}")
    os.makedirs(local_dir, exist_ok=True)

    try:
        path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
        )
    except Exception as e:
        print(f"下载失败: {e}")
        sys.exit(1)

    print(f"\n下载完成: {path}")
    print("训练时设置数据集路径:")
    print(f"  export LINGBOT_VA_DATASET_PATH={path}")

if __name__ == "__main__":
    main()
