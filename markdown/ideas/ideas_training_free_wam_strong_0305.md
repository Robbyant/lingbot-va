# WAM（World Action Model）推理加速：补充创新点（training-free）

这份文件专门补充 **training-free 的 accelerate WAM** 点子（不写 MPC/规划/奖励那类“提高成功率为主”的内容）。主清单在 `ideas/ideas_0305.md`，这里给你一些 **更工程化但依然“WAM 特有”** 的加速补充，重点围绕：

- **KV cache 的冷热分层/压缩/量化**
- **world token 的 ROI 更新（只算“动的/因果相关”的部分）**
- **跨周期 warm-start 与 early-exit**
- **CPU↔GPU 搬运与算子并行重叠**

---

## 创新点 A：KV Cache 冷热分层 + 冷层量化（Hot/Cold KV Tiering + Quantized Cold Cache）

- **WAM 特性**：闭环 episode 里 cache 会增长；action 侧会反复 attend 历史 world KV，但真正“有效”的往往是最近窗口 + 少量长期记忆。
- **算法（training-free）**：
  - 维护 **Hot KV**（最近 \(W\) 帧，高精度 bf16/fp16）与 **Cold KV**（更旧部分，量化存储 int8/fp8 或 cpu-pinned）。
  - action 阶段默认只看 Hot + 少量 Cold summary（或按需解量化一小段）。
  - 当触发事件（接触/遮挡/目标切换）才临时扩大可见 Cold 范围。
- **落点**：`modules/model.py` 的 cache 存取/valid selection（写入时打 hot/cold tag；读取时拼接/解量化）。
- **加速来源**：缩短 action 注意力的有效 K/V；减少显存占用与 cache 管理成本；offload 场景收益更大。
- **风险/验证**：量化误差影响精细操作；用分阶段策略（精细阶段禁用 cold 量化或增大 Hot 窗口）。

---

## 创新点 B：Cache 的 “Delta Write” （无更新即不写，避免无效 cache 维护）

- **WAM 特性**：很多周期里 world 其实未更新（或只更新很小一块），但 naive 实现仍会写入整段 KV，带来 cache 膨胀与带宽浪费。
- **算法**：
  - 当触发器判定本周期 world 不需要更新（或只更新 ROI），则 **禁止写入 pred cache**（或只写 ROI token 的 KV）。
  - action 分支使用上次的 KV（或上次 + ROI 增量）。
- **落点**：`wan_va_server.py` 调用 transformer 时的 `update_cache` 策略 + `modules/model.py` 中 cache 写入逻辑。
- **加速来源**：减少 cache 写带宽、减少 cache 长度增长、减少后续注意力计算。
- **风险/验证**：需要严格保证因果一致性；用日志记录每次“跳写”的比例与失败案例。

---

## 创新点 C：World 的 ROI Token Update（基于“运动/变化”更新 world token）

- **WAM 特性**：控制里大量背景是静态的；真正重要的是 “手+物体+接触区域”。
- **算法（training-free）**：
  - 用廉价的变化检测（多视角 downsample diff / 光流近似 / mask 变化）得到 ROI。
  - 将 ROI 映射到 latent patch/token，只有 ROI token 进入 world 去噪更新；其余 token 直接复用上一周期 world latent（以及 KV）。
  - ROI token 的写入以 block 为粒度（利于 cache 管理）。
- **落点**：`wan_va_server.py::_encode_obs_with_cache()` 产出 ROI mask；world 循环里按 mask 构建子序列（需要在数据打包/patch 逻辑处支持子集）。
- **加速来源**：world token 数显著减少 → 注意力边数下降 → NFE 总成本下降。
- **风险/验证**：ROI 映射误差；先用粗 block（如 8×8 patch）降低错杀风险。

---

## 创新点 D：Action Diffusion Warm-start（跨周期动作 latent 热启动）

- **WAM 特性**：相邻控制周期最优动作通常连续；action 去噪从纯噪声开始浪费。
- **算法**：
  - 用上周期 action latent 的末态 \(a_{t}^{*}\) 作为本周期初值，加小噪声后只做少量去噪（或直接从中间 sigma 开始）。
  - 结合你的 “动作维度冻结/早停”（`ideas_0305.md` 的创新点 7）进一步减少步数。
- **落点**：`wan_va_server.py` action loop：维护 `prev_action_latent`，改变初始化与 timesteps 起点。
- **加速来源**：减少 action NFE（尤其 action_steps 很高时）。
- **风险/验证**：遇到突变场景会陷入局部；用 “变化检测/事件触发” 强制重启（random init）。

---

## 创新点 E：Scheduler Early-Exit（用收敛判据提前结束扩散）

- **WAM 特性**：很多周期里去噪很快收敛，后续 steps 的增益小。
- **算法（training-free）**：
  - **监测信号（不额外 forward）**：在每次 `scheduler.step()` 你本来就有 `model_pred_t`（如 \(v_t\)/\(\epsilon_t\)）以及更新后的 latent \(x_{t+1}\)。用它们构造“相对变化量”：
    - \(\Delta_v(t)=\dfrac{\|v_t-v_{t-1}\|_2}{\|v_t\|_2+\varepsilon}\)（或把 \(v\) 换成 \(\epsilon\)/\(\hat{x}_0\)）
    - \(\Delta_x(t)=\dfrac{\|x_{t+1}-x_t\|_2}{\|x_t\|_2+\varepsilon}\)
    - world 分支可只在 ROI/block 上算 \(\Delta_x\)（与“ROI token update”配合更稳）；action 分支在 action latent 维度上算即可。
  - **抗抖/滞回**：对 \(\Delta_v,\Delta_x\) 做一个轻量 EMA（或滑窗均值）得到 \(\widehat{\Delta}(t)\)，并设置“连续命中”计数器 `hit`：
    - 若 \(\widehat{\Delta}(t) < \tau(t)\)，则 `hit += 1`，否则 `hit = 0`。
    - 当 `hit >= m` 且已完成最少步数 `t >= t_min`，触发 early-exit。
  - **阈值怎么设（关键是归一化 + 随噪声阶段调度）**：
    - 固定阈值起步：\(\tau(t)=\tau_0\)，例如 \(\tau_0\in[10^{-3},10^{-2}]\)（依赖归一化后量纲）。
    - 更稳的做法：随噪声强度收紧/放宽，比如 \(\tau(t)=c\cdot \sigma_t\) 或 \(\tau(t)=c\cdot \sigma_t^2\)（早期 \(\sigma\) 大允许更大变化，后期更严格）。
    - 分阶段：接触/精细阶段使用更小阈值、更大的 \(m\)（更保守），粗阶段相反。
  - **防误退 guardrails**：
    - 设 `t_min`（至少跑完前 \(p\%\) steps 才允许早停），避免一开始就误判“变化小”。
    - 若触发“事件/突变”（例如观测 diff/接触检测/目标切换），本周期禁用 early-exit 或直接重启（参考 D 的“变化触发强制重启”）。
    - 可加一个“最终质量兜底”检查：例如 early-exit 前再看一次 \(\Delta_x\) 是否也低于阈值（双条件）以降低误判。
- **落点**：`wan_va_server.py` 的 video/action loop；或 `FlowMatchScheduler.step()` 外围做。
- **加速来源**：直接减少 NFE。
- **风险/验证**：阈值敏感；用离线 sweep 找 Pareto。

---

## 创新点 F：CPU↔GPU 搬运与算子重叠（Encode/Preprocess Overlap with CUDA Streams）

- **WAM 特性**：offload + 多视角 preprocess 常让 CPU↔GPU 变成瓶颈，尤其 server 长跑时会出现尾延迟。
- **算法**：
  - 使用 pinned memory + 异步 H2D；VAE encode 放到独立 CUDA stream；与 transformer forward 重叠。
  - 图像 resize/stack 在 CPU 侧也可并行（线程池）并提前准备下一周期数据。
- **落点**：`wan_va_server.py::_encode_obs_with_cache()` 与 `_compute_kv_cache()`：把 preprocess/transfer/encode 拆分成可重叠阶段。
- **加速来源**：减少 pipeline 泡沫、降低 P90/P99 时延。
- **风险/验证**：实现复杂；用 Nsight/torch profiler 验证 overlap 是否生效。

---

## 创新点 G：自适应 Offload 策略（基于命中率/负载的 VAE 常驻调度）

- **WAM 特性**：offload 带来搬运开销，但在 encode reuse 命中率高时 VAE 常驻 GPU 可能更划算。
- **算法**：
  - 在线统计：encode reuse 命中率、VAE encode 的占比、GPU 余量。
  - 动态决定：VAE 是否常驻 GPU（或只常驻高频视角的编码路径）。
- **落点**：`VA_Server.__init__` 与 `_encode_obs_with_cache()`：根据统计切换 `self.vae`/`self.streaming_vae` 的 device。
- **加速来源**：避免“搬运主导”的坏工况，尤其在高帧率/多视角时。
- **风险/验证**：频繁切换 device 反而更慢；需加入 hysteresis（滞回）。


