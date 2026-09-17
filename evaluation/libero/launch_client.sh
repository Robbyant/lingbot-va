START=0
END=10

# Added MUJOCO_GL=egl: headless simulation
unset http_proxy https_proxy socks_proxy all_proxy HTTP_PROXY HTTPS_PROXY SOCKS_PROXY ALL_PROXY

MUJOCO_GL=egl PYTHONPATH="$(cd "$(dirname "$0")/../.." && pwd)" python evaluation/libero/client.py \
    --libero-benchmark libero_10 \
    --port 29056 \
    --test-num 50 \
    --task-range $START $END \
    --out-dir outputs/libero
