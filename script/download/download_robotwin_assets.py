#!/usr/bin/env python3
"""
RoboTwin 2.0 测评 assets 下载（带重试 + 可选国内镜像）。
HuggingFace 连接中断时可用此脚本重试；国内建议先设 HF_ENDPOINT 镜像。

用法:
  python script/download_robotwin_assets.py
  python script/download_robotwin_assets.py /mnt/users/wangyuxuan-20250915/EAI/RoboTwin
  HF_ENDPOINT=https://hf-mirror.com python script/download_robotwin_assets.py /path/to/RoboTwin
"""
import os
import sys
import time

def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bench_root = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.environ.get("BENCH_ROOT", os.path.join(repo_root, "..", "RoboTwin"))
    bench_root = os.path.normpath(bench_root)
    assets_dir = os.path.join(bench_root, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    # 国内可设 HF_ENDPOINT=https://hf-mirror.com 加速/避免连接中断
    hf_endpoint = os.environ.get("HF_ENDPOINT", "")
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
        print(f"Using HF_ENDPOINT: {hf_endpoint}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("请先安装: pip install huggingface_hub")
        sys.exit(1)

    repo_id = "TianxingChen/RoboTwin2.0"
    patterns = ["background_texture.zip", "embodiments.zip", "objects.zip"]
    max_retries = 3

    print(f"下载 RoboTwin2.0 assets 到: {assets_dir}")
    for attempt in range(1, max_retries + 1):
        try:
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=patterns,
                local_dir=assets_dir,
                repo_type="dataset",
            )
            print("下载完成.")
            break
        except Exception as e:
            print(f"第 {attempt}/{max_retries} 次尝试失败: {e}")
            if attempt < max_retries:
                wait = 10 * attempt
                print(f"{wait}s 后重试...")
                time.sleep(wait)
            else:
                print("多次重试仍失败。建议：")
                print("  1) 国内用户设置镜像后重试: HF_ENDPOINT=https://hf-mirror.com python script/download_robotwin_assets.py", bench_root)
                print("  2) 或浏览器打开 https://huggingface.co/datasets/TianxingChen/RoboTwin2.0 手动下载上述 zip 放到", assets_dir, "后执行:")
                print("     cd " + bench_root + " && bash script/_download_assets.sh  # 仅解压与配置")
                sys.exit(1)

    # 解压与配置（与 RoboTwin 原脚本一致）
    import subprocess
    orig_cwd = os.getcwd()
    try:
        os.chdir(assets_dir)
        for name in ["background_texture.zip", "embodiments.zip", "objects.zip"]:
            if os.path.isfile(name):
                subprocess.check_call(["unzip", "-o", name], shell=False)
                os.remove(name)
        os.chdir(bench_root)
        if os.path.isfile(os.path.join(bench_root, "script", "update_embodiment_config_path.py")):
            subprocess.check_call([sys.executable, "script/update_embodiment_config_path.py"], cwd=bench_root)
        print("解压与路径配置完成.")
    finally:
        os.chdir(orig_cwd)

if __name__ == "__main__":
    main()
