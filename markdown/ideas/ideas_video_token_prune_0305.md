# Video / World Token Prune 创新点（training-free）

目标：对 **生成的 video/world token 长度** 做裁剪或压缩，降低 attention 计算与 cache 成本，且不改变训练。最稳的做法是先做在 **KV/注意力侧**（等价于 token prune），再考虑真正缩短前向序列。

---

## 设计原则

- **优先做“谁 attend 谁”的裁剪**：world token 仍存在，但在后续 attention 里只让一部分 token 参与（缩短有效 K/V），不改输入 token 形状与 mesh_id，工程风险小。
- **再考虑真正缩短序列**：merge/evict 等需动 cache 写入与打包逻辑，适合在 KV prune 验证后再做。

---

## 创新点 1：Action-conditioned KV Token Prune（最推荐、可行性最高）

**目标**：action 分支 attend 的 world KV 从“全量视频 token”变为“Top-M 因果相关 token”。

**重要性打分（training-free）**：

- 用 **action token → world token 的 attention 权重**（最后 1~2 层即可）：
  - \(\mathrm{imp}(w_i) = \sum_{a \in \text{action tokens}} \mathrm{Attn}(a \to w_i)\)
- 可在多 step/多层做 EMA 平滑，减少抖动。

**Prune 规则**：

- 保留 **Top-M** world tokens（或按 block 保留 Top-B blocks）。
- **强制保留**：wrist/手附近 ROI、最近帧 token、变化显著区域（见创新点 3）。

**落点**：

- 在 transformer 的 attention **读 cache 时**做 K/V 的 gather（只给注意力一个缩短的 K/V），不改 latent / mesh_id 生成。

**加速来源**：显著降低 **action 阶段** attention 的 K/V 长度；长 episode 时延与 cache 增长更可控。

**创新性**：WAM 专属“因果裁剪”——world 为 action 服务，只保留对动作预测重要的 token。

**风险/验证**：M 过小可能丢关键几何；可先做 M 的 sweep，并记录被裁掉的 token 空间/时间分布。

---

## 创新点 2：Temporal Token Eviction + Summary Tokens

**目标**：把“很久以前的 video tokens”从全量保留改为少量“世界摘要 token”，使有效 token 长度稳定在上限。

**做法**：

- **Hot**：最近 \(W\) 帧 world tokens 全保留。
- **Cold**：更早的帧压缩成 \(K\) 个 summary tokens（mean pooling / attention pooling / PCA 低秩等，均 training-free）。
- action 分支默认只 attend：**summary + 最近 \(W\) 帧**。

**落点**：cache 的写入/淘汰策略；summary 作为额外 KV 写入，不改变主前向的 token 形状定义。

**加速来源**：控制 token 长度不随 episode 增长爆炸；降低长序列时延与显存。

**创新性**：把“世界记忆”做成可控的长期摘要，而非无限增长的 KV。

**风险/验证**：摘要丢失细粒度操作信息；可在接触/精细阶段关闭压缩或增大 \(K\)。

---

## 创新点 3：Change-driven Spatial ROI Prune

**目标**：只保留“动的/相关的”空间区域对应 token，其余视为静态背景。

**ROI 信号（training-free）**：

- 多视角下采样 diff、近似光流、分割 mask 变化。
- 末端速度大时扩大 ROI；静止时缩小 ROI。

**工程要点**（当前 world latent 为多相机在宽度维拼接）：

- 每个相机 ROI 映射到其对应的 **latent 水平区间**。
- 先按**粗 block**（如 8×8 latent patch）做，鲁棒且易实现。

**落点**：

- 建议先做“ROI 决定 KV 可见性”（即 KV prune），不立刻做“减少前向 token 数”。

**加速来源**：action attention 与长序列 cache 均受益；ROI 也可作为动态 step budget 的触发器。

**创新性**：利用 WAM 多视角结构，把视觉变化直接映射到 token 级计算裁剪。

---

## 创新点 4：ToMe / Token Merging on World Tokens（真缩短序列，研究向）

**目标**：在 world 分支去噪过程中，将相似/低重要性 token **合并**，真实减少序列长度。

**做法（training-free）**：

- 每隔若干步，对 world tokens 做相似度（cosine / L2），将最相似的若干对 **merge**（如加权平均）。
- 保留映射表，必要时可“反投影”回原网格（通常不完全可逆）。

**难点**：

- 需同步处理 **mesh_id**、token 打包、cache slot 对应关系。
- 对扩散去噪稳定性有影响（尤其 early steps）。

**建议**：作为论文亮点合适；工程上建议在 **KV prune（创新点 1/2/3）** 验证后再上。

---

## 落地顺序建议

| 顺序 | 方案 | 说明 |
|------|------|------|
| 1 | Action-conditioned KV prune（创新点 1） | 改动集中、不破坏形状，对“video token 长度带来的注意力成本”最直接 |
| 2 | Temporal eviction + summary（创新点 2） | 控制长 episode 的 token 上限，与 1 可组合 |
| 3 | Change-driven ROI（创新点 3） | 先用于 KV prune / 触发器，再考虑真减 token |
| 4 | ToMe / token merging（创新点 4） | 真剪序列长度，研究型，放在最后 |

---

## 落点与依赖小结

- **创新点 1**：transformer 内 attention 读 cache 时的 K/V 索引/gather；需暴露 1~2 层 action→world attention 权重（或等价重要性）。
- **创新点 2**：cache 管理（写入、淘汰、summary 生成与拼接）。
- **创新点 3**：`_encode_obs` 外围的 ROI 映射 + 同上 cache/attention 的可见性。
- **创新点 4**：world 数据打包、patch、mesh_id 与 cache slot 一致性。

---

## 评估指标建议

- 有效 world token 数 / K-V 长度分布；attention 计算量（FLOPs 或等价）。
- 控制质量：成功率、碰撞率、阶段失败分布；与“不 prune”的 A/B。
- 延迟与显存：P50/P90 时延；cache 峰值；长 episode 的时延增长曲线。
