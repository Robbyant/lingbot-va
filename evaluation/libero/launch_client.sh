START=0
END=10

# 若 EGL 报错，可先: sudo apt install -y libosmesa6 ，并取消下行注释或 export MUJOCO_GL=osmesa
# MUJOCO_GL=osmesa

python3 -m evaluation.libero.client \
    --libero-benchmark libero_10 \
    --port 29056 \
    --test-num 50 \
    --task-range $START $END \
    --out-dir outputs/libero