# WAM（World Action Model）推理加速：创新点清单（training-free 为主）

下面的点尽量围绕 **WAM 的独特性**（动作会改变世界、需要闭环控制、world/action 联合生成且可复用缓存），避免照搬纯视频生成或常见 VLA（BC/RT/纯 policy）套路。每条都给出：机制、落点、加速来源、风险与验证方式。

---

## 背景：当前推理耗时主要来自哪里

以当前实现（`wan_va/wan_va_server.py`）为例，推理开销通常由以下部分主导：

- **NFE（transformer 前向次数）**：video 分支 `num_inference_steps` + action 分支 `action_num_inference_steps`。
- **Token 数 / 注意力有效边数**：world latent token 远多于 action token；action 阶段还会 attend 到 world 的 KV cache。
- **Cache 维护成本**：KV cache 增长、slot 管理、以及（若设计不当）无约束的可见范围带来的注意力开销。
- **观测编码与数据搬运**：多视角图像 resize/stack + VAE encode；offload 时还有 CPU↔GPU 传输。

因此，推理加速的“杠杆”主要是：**减少 NFE、减少有效 token/边、复用/压缩 cache、减少不必要的 world 去噪、减少数据搬运**。

---

## 设计原则（WAM 专属）

- **动作优先**：若最终指标是控制成功率，世界生成只需提供“对动作有用”的信息（不是高保真视频）。
- **干预因果**：动作是干预变量（intervention），world 必须对 action 敏感；world 的哪些 token 对 action 重要应可识别并被优先计算。
- **闭环与低延迟**：每个控制周期必须准时输出动作；允许“粗动作 + 快速修正”的策略。
- **缓存可迁移**：同一 episode 内、甚至相似场景间，许多计算（KV、prompt embedding、静态背景 token）可复用。

---

## 创新点 1：Action-first 世界“懒惰化”（Lazy World Denoising, LWD）

- **WAM 特性**：world 只是 action 的上下文，不一定需要每个周期都高质量去噪。
- **算法**：
  - 用一个廉价的 **world-need 指标** 决定是否跑 world 去噪：例如观测变化幅度、接触/抓取状态变化、动作不确定性（action 方差/熵）、或“动作对 world token 的注意力集中度”。
  - 若指标低：本周期 **跳过/极少** world steps（例如 `video_exec_step=0~2`），直接用已有 world cache 生成 action。
  - 若指标高：再启用正常 world 去噪。
- **落点**：`wan_va_server.py::_infer()` 中 video loop 的 step budget 动态化（per-chunk/per-cycle）。
- **加速来源**：直接砍掉大量 world NFE。
- **风险/验证**：
  - 风险：遇到“需要精确世界状态”的时刻误判。
  - 验证：按任务阶段分桶（抓取前/抓取中/放置前）统计跳过 world 的成功率变化；并用“误判率”分析指标设计。

---

## 创新点 2：多速率联合生成（Multi-rate WAM Sampling）

- **WAM 特性**：动作需要高频更新，世界状态（尤其背景/静态部分）可以低频更新。
- **算法**：
  - world 每 \(K\) 个控制周期才更新一次（或每个 chunk 只在首周期更新）。
  - action 每个周期都更新；action 的 attention 只看“最近一次更新的 world KV + 最新观测条件 token（很短）”。
  - 可加一个轻量的 **观测增量 token**：只编码/写入最新帧的少量 summary token，而不是全量世界 latent。
- **落点**：server 模式下 `compute_kv_cache` 与 `infer` 的调用节奏；以及 cache 可见范围控制。
- **加速来源**：world 侧 NFE 与 token 更新频次下降；action 侧每步 K/V 更短。
- **风险/验证**：
  - 风险：world 低频导致“状态滞后”。
  - 验证：对比不同 \(K\) 下的成功率/动作抖动/碰撞率，并量化延迟降低。

---

## 创新点 3：WAM 专属“因果 cache 视野裁剪”（Causal Cache Cropping）

- **WAM 特性**：动作只需要与当前动作因果相关的 world 部分（例如手、被操作物体、容器口等）。
- **算法**：
  - 用 action→world 的注意力或梯度（近似）在推理时在线得到 **重要 world token 子集**。
  - 将 KV cache 按重要性裁剪：action 阶段只 attend 到 Top-\(M\) 的 world KV（或 ROI KV）。
  - 重要性可以是：
    - 最近几步 action token 对 world token 的平均 attention；
    - 或 “动作预测对 world token 的敏感度” 的低成本近似（例如 last-layer attention 聚合）。
- **落点**：`modules/model.py` 的 cache 读出（`valid` selection）处引入“重要 token mask”；或在写入时就打标分类。
- **加速来源**：减少 action 阶段的 K/V 长度，降低注意力计算。
- **风险/验证**：
  - 风险：裁剪错误导致关键几何信息缺失。
  - 验证：记录被裁剪 token 的类别/位置；对失败案例可视化 attention 以调参。

---

## 创新点 4：KV cache 在线压缩成“世界摘要 token”（Online World Summarization）

- **WAM 特性**：对控制而言，世界只需少量摘要（物体相对位姿、可达性、接触）即可指导动作。
- **算法（training-free）**：
  - 周期性将较旧的 world KV 压缩成少数 \(K\) 个 **summary KV**（例如用 attention pooling / mean pooling / PCA 低秩投影）。
  - 将被压缩的原 KV slots 释放，保持 cache 上限。
  - action 阶段优先看 summary + 最近窗口内的高分辨 KV。
- **落点**：`modules/model.py` cache 管理（slot 释放策略）+ 一个压缩函数（可放在 `utils/`）。
- **加速来源**：控制 cache 长度，避免长 episode 变慢；action 注意力更短。
- **风险/验证**：
  - 风险：摘要丢失细粒度操作信息。
  - 验证：按“接触/精细操作阶段”关闭压缩或提高 \(K\)；做阶段自适应。

---

## 创新点 5：用“速度场历史”做多步外推（AB2/AB3 for Flow, training-free）

- **WAM 特性**：你们的 `FlowMatchScheduler.step()` 是显式 Euler（速度场积分），天然可做多步法。
- **算法**：
  - 保存前一步（或前两步）的速度 \(v_{t-1}, v_{t-2}\)。
  - 用 Adams–Bashforth（2/3 阶）更新：
    - AB2：\(\Delta x \approx \Delta\sigma \cdot (3/2\,v_t - 1/2\,v_{t-1})\)
    - AB3：\(\Delta x \approx \Delta\sigma \cdot (23/12\,v_t - 16/12\,v_{t-1} + 5/12\,v_{t-2})\)
  - 在更少 steps 下维持质量（或相同 steps 下更好质量）。
- **落点**：`wan_va/utils/scheduler.py` 新增 `step_ab2/ab3`；`wan_va_server.py` 循环里维护速度历史。
- **加速来源**：允许把 steps 降得更激进而不崩（减少 NFE）。
- **风险/验证**：
  - 风险：外推不稳定，尤其在 guidance 强或噪声大区域。
  - 验证：对比 Euler/AB2 在 10/15/20 steps 下的成功率与动作平滑度。

---

## 创新点 6：WAM 版 speculative control：快动作提案 + 单步世界验真 + 局部修正

- **WAM 特性**：控制可以容忍“先给一个可执行动作”，再快速修正（闭环）。
- **算法**：
  1) 用极少 steps（甚至仅 action 分支）产生 action 提案 \(a_t\)。
  2) 用 1 次（或极少）world 更新预测 \(\hat{o}_{t+1}\) 的关键摘要（不需要整段视频）。
  3) 若验真失败（违反约束/目标偏差大），再追加少量 steps 对 action 做 refinement（或回退到完整采样）。
- **落点**：server 模式最自然；需要一个廉价的“验真器”（可用现有 world head 的低成本统计，或基于任务约束的几何判据）。
- **加速来源**：大多数时刻只走快路径；仅少数困难时刻走慢路径。
- **风险/验证**：
  - 风险：验真器设计不良导致误放行。
  - 验证：统计快路径命中率、回退率、以及回退带来的尾延迟。

---

## 创新点 7：动作维度的“冻结/早停”（Action Token Freezing）

- **WAM 特性**：很多动作维度在大多数阶段接近常量（例如部分关节、或某些夹爪维度）。
- **算法（training-free）**：
  - 在 action 去噪过程中，监测每个 action 维度（或 token）的变化量 \(|\Delta a|\)。
  - 若连续 \(m\) 步 \(|\Delta a| < \epsilon\)，则将该维度标记为 frozen：后续 steps 不再更新（或仅做一次低频更新）。
  - 对 frozen 维度可以在 transformer 输出后做 mask，或在输入噪声中置零相应更新。
- **落点**：`wan_va_server.py` action loop + `actions_mask` 扩展；或在 `postprocess_action` 前做冻结逻辑。
- **加速来源**：减少有效 action token 更新与注意力计算（尤其在更细粒度 token 化时效果更明显）。
- **风险/验证**：冻结阈值/窗口需要按任务阶段自适应。

---

## 创新点 8：观测编码复用（Obs/VAE Encode Reuse with Change Detection）

- **WAM 特性**：相邻控制周期多视角图像变化可能很小；重复 VAE encode 浪费。
- **算法**：
  - 对输入图像做低成本 hash/差分（例如 downsample 后 L2、或 SSIM 近似）。
  - 若变化小于阈值：复用上次的 encoded latent（或仅对变化最大的相机视角重编码）。
  - 对 wrist camera 可按运动幅度自适应刷新频率。
- **落点**：`wan_va_server.py::_encode_obs()` 外围加缓存与变化检测。
- **加速来源**：减少 VAE encode 与 CPU↔GPU 传输（offload 时收益更大）。
- **风险/验证**：快速运动/遮挡变化需要强制刷新。

---

## 创新点 9：WAM 目标导向的 step 自适应（Goal-Progress Adaptive Step Budget）

- **WAM 特性**：控制目标通常有可计算的进度指标（距离、门角度、抓取状态、容器对齐）。
- **算法（training-free）**：
  - 在每次输出 action 前，用一个廉价的 proxy 估计“本周期动作对目标推进是否足够”（例如从 action 大小、方向一致性、或 world 摘要预测中估计）。
  - 若推进明显：减少后续采样 steps；若推进停滞/反向：增加 steps 或触发回退策略。
- **落点**：server loop 中的 step controller；与上面的 speculative control 可组合。
- **加速来源**：多数“简单阶段”少算，困难阶段多算。
- **风险/验证**：需要设计任务无关的通用 proxy（或按任务族提供不同 proxy）。

---

## 创新点 10：控制专用的“低保真 world”通道（World-for-Control Channel）

- **WAM 特性**：控制不需要高保真像素细节，而需要几何/接触/可达性。
- **算法**：
  - 不改变训练（或极少改动），在推理时从现有 latent 中提取一个极低维的控制特征（例如每帧若干统计：均值/方差、或固定池化得到的 K 个 token）。
  - action 阶段只 attend 到这些控制 token + 最近窗口内的少量高分辨 token（必要时再补全）。
- **落点**：在 world cache 写入后追加一段 pooled token；action 阶段只读 pooled token（类似“在线瓶颈”）。
- **加速来源**：大幅缩短 action 侧的 K/V。
- **风险/验证**：对精细插入/对齐任务，可能需要阶段性打开高分辨 KV。

---

## 创新点 11：候选动作并行 + 单次世界评估（Batch Candidates, Shared World KV）

- **WAM 特性**：world KV 可共享；动作候选可以并行评估，比串行多次采样更划算。
- **算法**：
  - 以 batch 方式并行生成 \(K\) 个 action 候选（少步数/不同噪声 seed）。
  - 复用同一份 world KV（以及 prompt embedding），用廉价 world 评估器/约束检查选最优。
  - 只对选中的候选做后续 refinement（可选）。
- **落点**：`wan_va_server.py` 将 action 采样 batch 化（维持 world cache 相同），以及选择器实现。
- **加速来源**：用并行吞吐换更少回退与更少长采样；在 GPU 上常更划算。
- **风险/验证**：batch 增大显存；需要找到“选优 proxy”。

---

## 创新点 12：训练-推理对齐的“交错式联合步”但保持 cache（Interleaved Joint-Step with Cache）

- **WAM 特性**：训练里 world/action 联合序列 + 因果 mask；推理里分阶段。对齐可减少步数敏感性。
- **算法（尽量 training-free）**：
  - 每个大步只做 **一次** transformer 调用，但在 cache 中交错写入：
    1) 读 world cache，更新一小步 world（写入 pred KV）
    2) 立刻在同一步用更新后的 KV 更新 action（或用同一 forward 的共享 trunk，输出两个 head）
  - 目标是：在不显著增加 token 的前提下，把“2 次 forward/步”变成“1 次 forward/步”（或减少总步数）。
- **落点**：需要对 `modules/model.py` 的前向做轻量封装（共享 block 计算、双 head 输出），或复用 `forward_train` 的拼接思路但用 cache 控制稀疏可见范围。
- **加速来源**：减少总前向次数（NFE）。
- **风险/验证**：实现复杂；需要确保 mask/缓存可见性不会引入未来泄露。

---

## 创新点 13：KV cache 冷热分层 + 冷层量化（Hot/Cold KV Tiering + Quantized Cold Cache）

- **WAM 特性**：episode 变长时 cache 增长拖慢 action 注意力；但真正有用的通常是最近窗口 + 少量长期记忆。
- **算法（training-free）**：
  - 将 KV 分为 **Hot**（最近 \(W\) 帧 bf16/fp16）与 **Cold**（更旧部分 int8/fp8 或 cpu-pinned）。
  - action 阶段默认只 attend Hot + 少量 Cold summary；事件触发时临时扩大可见 Cold。
- **落点**：`modules/model.py` cache 存取与 `valid` selection（读时拼接/解量化）。
- **加速来源**：缩短有效 K/V、降显存、避免长 episode 变慢（offload 场景更明显）。
- **风险/验证**：量化误差；用分阶段策略（精细操作阶段禁用或提高精度）。

---

## 创新点 14：Cache 的 Delta Write（无更新就不写，避免无效 cache 维护）

- **WAM 特性**：很多周期 world 不更新或只更新 ROI，但 naive 仍写整段 KV，导致 cache 膨胀与带宽浪费。
- **算法**：当本周期 world 不更新（或只更新 ROI），则 **禁止写入 pred cache**（或只写 ROI token 的 KV）。
- **落点**：`wan_va_server.py` 的 `update_cache` 策略 + `modules/model.py` cache 写入逻辑。
- **加速来源**：减少 cache 写带宽、控制 cache 长度增长、降低后续注意力开销。
- **风险/验证**：需保证因果一致性；记录“跳写比例 vs 失败案例”。

---

## 创新点 15：World 的 ROI Token Update（按“动的/因果相关的”token 更新 world）

- **WAM 特性**：背景静态，关键是“手+物体+接触区域”；全量 world token 去噪浪费。
- **算法（training-free）**：
  - 用廉价变化检测（多视角 downsample diff / 光流近似）得到 ROI。
  - 仅对 ROI token 做 world 更新，其余 token 直接复用上一周期 world latent/KV。
- **落点**：ROI 从 `_encode_obs` 外围产出；world 数据打包/patch 逻辑支持子集 token（block 粒度先做）。
- **加速来源**：world token 数显著下降 → 注意力边数下降 → NFE 成本下降。
- **风险/验证**：ROI 映射误差；先用粗 block 降错杀风险。

---

## 创新点 16：Action Diffusion Warm-start（跨周期动作 latent 热启动）

- **WAM 特性**：相邻控制周期动作连续；每次从纯噪声采样浪费。
- **算法**：用上周期 action latent 末态做本周期初值（加小噪声/从中间 sigma 开始），并结合早停/冻结降低 steps。
- **落点**：`wan_va_server.py` action loop 维护 `prev_action_latent`，改变初始化与 timesteps 起点。
- **加速来源**：减少 action NFE（action_steps 高时收益更明显）。
- **风险/验证**：突变场景会卡住；用事件触发强制重启（random init）。

---

## 创新点 17：Scheduler Early-Exit（用收敛判据提前结束扩散）

- **WAM 特性**：很多周期去噪很快收敛，后续 steps 增益小。
- **算法（training-free）**：监测 \(\|v_t-v_{t-1}\|\) / \(\|\Delta x\|\) 等，连续低于阈值则提前结束该分支（world 或 action）。
- **落点**：`wan_va_server.py` 的 video/action loop 或 `FlowMatchScheduler.step()` 外围。
- **加速来源**：直接减少 NFE。
- **风险/验证**：阈值敏感；做离线 sweep 找 Pareto。

---

## 创新点 18：CPU↔GPU 搬运与算子重叠（Encode/Transfer Overlap with CUDA Streams）

- **WAM 特性**：offload + 多视角 preprocess 常让 CPU↔GPU 搬运成为瓶颈，拉高尾延迟。
- **算法**：pinned memory + 异步 H2D；VAE encode 独立 CUDA stream；与 transformer forward 重叠；CPU preprocess 线程化预取下一周期。
- **落点**：`wan_va_server.py::_encode_obs*()` 与 `_compute_kv_cache()` 的 pipeline 拆分与重叠。
- **加速来源**：减少 pipeline 泡沫，显著改善 P90/P99。
- **风险/验证**：工程复杂；用 profiler 验证 overlap 生效。

---

## 建议的落地顺序（按“最可能立刻提速/风险最低”排序）

1. **LWD（跳过/少跑 world）** + **自适应 step budget**（最直接砍 NFE）
2. **AB2/AB3 多步法**（不改模型，仅改 scheduler/循环）
3. **Causal Cache Cropping / Summary token**（减少 action 阶段 attention）
4. **Obs/VAE encode 复用**（工程上收益稳定）
5. **Speculative control（快路径 + 验真 + 回退）**（闭环友好，尾延迟需评估）

---

## 评估指标（建议同时记录）

- **吞吐/时延**：ms/step、ms/chunk、P50/P90/P99 时延、GPU 利用率、显存峰值。
- **控制质量**：成功率、碰撞率、动作抖动（\|\Delta a\|）、阶段性失败分布（抓取/放置/开合）。
- **计算分解**：world NFE vs action NFE、cache 长度随时间变化、VAE encode/CPU↔GPU 传输占比。

