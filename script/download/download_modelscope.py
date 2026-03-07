#!/usr/bin/env python3
"""
从 ModelScope 下载 LingBot-VA 模型 (lingbot-va-base)。
用法:
  python script/download_modelscope.py
  python script/download_modelscope.py /path/to/save
"""
import os
import sys

def main():
    # 默认保存到仓库根目录下的 lingbot-va-base
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_dir = os.path.join(repo_root, "lingbot-va-base")
    local_dir = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else default_dir

    try:
        from modelscope import snapshot_download
    except ImportError:
        print("请先安装 modelscope: pip install modelscope")
        sys.exit(1)

    model_id = "Robbyant/lingbot-va-base"
    print(f"正在从 ModelScope 下载: {model_id}")
    print(f"保存到: {local_dir}")
    os.makedirs(local_dir, exist_ok=True)

    try:
        path = snapshot_download(model_id, local_dir=local_dir)
    except Exception as e:
        print(f"下载失败: {e}")
        sys.exit(1)

    print(f"\n下载完成: {path}")
    print("推理前请设置并修改 transformer 的 attn_mode:")
    print(f"  export LINGBOT_VA_MODEL_PATH={path}")
    print("  # 编辑 {}/transformer/config.json 将 attn_mode 改为 \"torch\" 或 \"flashattn\"".format(path))

if __name__ == "__main__":
    main()
