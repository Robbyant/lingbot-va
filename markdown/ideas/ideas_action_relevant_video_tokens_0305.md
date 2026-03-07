# 识别“对下一个动作强相关”的 video/world tokens（training-free 创新点）

目标：在 WAM（world/action 联合扩散）推理中，在线识别当前生成的 **video/world tokens** 里哪些对 **下一步动作** 最关键，并 **优先保留/更新** 这些 tokens（其余 token 可被裁剪为不可见、压缩为摘要、或低频刷新）。

> 关键建议：优先做 **KV 可见性裁剪**（token 仍存在，但后续注意力只看关键 token），工程风险远低于“真正删 token 让前向变短”。后者可作为第二阶段。

---

## 0) 定义：什么叫“对下一个动作强相关”

给定 world token \(w_i\)、下一步动作输出 \(a\)（或 action 去噪过程中的噪声预测 \(\\epsilon_a\) / 速度场 \(v_a\)），我们希望估计某种 “重要性”：

- **Sensitivity/Influence**：\( \\partial a / \\partial w_i \) 大
- **Information**：移除/扰动 \(w_i\) 会明显改变动作分布（均值/方差）
- **Causal utility**：\(w_i\) 能解释 action 的决策（而不是仅相关）

training-free 的核心：用 **模型本身的 attention/输出变化** 做 proxy，而不是再训练一个 importance head。

---

## 1) Action→World 注意力归因（最可行、WAM 专属）

### 1.1 单层/多层 attention 聚合

**思路**：在 action 分支 forward 时，拿到 action tokens 对 world tokens 的 attention 权重，把它当作 “action 正在读取的信息”。

- **importance**：
  \[
  \\mathrm{imp}(w_i) = \\sum_{a \\in \\mathcal{A}} \\mathrm{Attn}(a \\rightarrow w_i)
  \]
  可对最后 1–2 层、或全部层加权求和（越靠后越贴近输出）。

### 1.2 跨 step 的 EMA 稳定化（减少抖动）

- 维护 \( \\hat{\\mathrm{imp}}_t = \\alpha \\hat{\\mathrm{imp}}_{t-1} + (1-\\alpha)\\mathrm{imp}_t \)
- 用 \(\\hat{\\mathrm{imp}}_t\) 做 Top‑K/Top‑Blocks 选择。

### 1.3 强制保留项（防止“注意力偏见”漏掉关键几何）

即便 attention 给低分，也强制保留：

- **最近帧** tokens（短期因果）
- **变化显著 ROI** tokens（见第 3 节）
- **wrist/手附近** 的 token 块（可用 state/相机先验映射）

**优势**：不改训练、信号天然存在；最贴合 “world 为 action 服务” 的 WAM 叙事。  
**风险**：attention ≠ 因果；需要用扰动验证（第 2 节）做 sanity check。

---

## 2) 反事实扰动评分（更因果，但更耗）

### 2.1 Token dropout / mask 的 action 变化量

**思路**：对候选 token 集合做小扰动，看 action 输出改变多少。

- 扰动方式：
  - **drop**：把 token 的 K/V 置零或替换为 mean token
  - **noise**：加小噪声扰动 token 表示
- 评分：
  - \(\\Delta_a = \\| a - a^{(\\text{drop } i)} \\|\)
  - 或动作分布差异（若你有多样采样）：KL/方差变化

### 2.2 “二阶段”快速筛选（降低开销）

先用 attention 得到 Top‑M 候选，再对这 M 个做反事实验证，最终保留 Top‑K（K≪M）。

**优势**：更接近因果影响，解释性强。  
**代价**：需要额外 forward（可只在低频或关键阶段运行）。

---

## 3) Change/Flow 驱动的 token 重要性（与多视角观测强耦合）

### 3.1 观测变化 → latent block 变化 → token 重要性

**思路**：动作决策往往依赖 “正在变化的部分”（手、物体、接触）。用低成本图像变化检测做 ROI，再映射到 token 网格。

- 变化信号：
  - 多视角 downsample diff（mean abs diff）
  - 近似光流（低分辨率即可）
  - 目标/手/物体 mask 变化（若你有分割器）
- 映射方式：
  - 先按 **粗 block**（例如 latent 8×8 patch）做 ROI，降低错杀风险
  - 对多相机拼接的 latent：每个相机对应固定的宽度区间，ROI 映射到对应 slice

### 3.2 与 attention 融合（最稳的组合）

最终分数：

\[
\\mathrm{score}(w_i)=\\lambda\\,\\hat{\\mathrm{imp}}_{\\text{attn}}(w_i) + (1-\\lambda)\\,\\hat{\\mathrm{imp}}_{\\text{change}}(w_i)
\]

**优势**：training-free、成本低、能覆盖“注意力没看但其实关键在动”的情况。  
**风险**：变化不等于因果（例如光照变化）；靠强制保留/阈值鲁棒化。

---

## 4) Action-uncertainty 驱动的重要性（不确定性越大越需要更多 token）

**思路**：当 action 分支对世界不确定时，通常需要更多上下文 token。反过来：能显著降低 action 不确定性的 token 更重要。

- 做法：
  - 对同一 obs，采样多次 action（不同 seed/少量步）得到方差 \(\\mathrm{Var}(a)\)
  - 逐步引入 token 子集（从 Top‑K 开始）观察方差下降速度
  - 能最快降低方差的 tokens 视为关键

**优势**：直接服务“下一步动作稳定/确定”。  
**代价**：需要少量多样采样（可低频做）。

---

## 5) 梯度/一阶敏感度近似（更直接，但需要可导获取）

**思路**：用一阶近似度量 token 对动作的影响（类似 saliency）。不更新参数，但允许取梯度。

- 定义标量目标：例如 action 预测的 L2 范数、或某些维度的 logit/均值
- 计算：
  - \(\\mathrm{imp}(w_i)=\\|\\nabla_{w_i} \\mathcal{L}\\|\) 或 \(\\|w_i\\odot \\nabla_{w_i}\\mathcal{L}\\|\)
- 实用技巧：
  - 只对最后 1–2 层 token 取梯度（降开销）
  - 只在关键阶段触发（接触/遮挡）

**优势**：比 attention 更接近“影响”。  
**风险**：实现侵入+开销；且梯度噪声大，需要平滑/分块。

---

## 6) 如何“保留这些 tokens”：三种落地形态（从易到难）

### 6.1 KV 可见性裁剪（最推荐）

- world token 仍写 cache，但 action 阶段只读取 Top‑K 的 world K/V（其余视为被 prune）
- 好处：不改 `mesh_id` / patch 打包；不破坏张量形状；容易 A/B。

### 6.2 Cache 生命周期管理（Hot/Cold + Summary）

- 保留关键 tokens 为 Hot，低重要 tokens 变 Cold（压缩/量化/低频更新）
- 适合长 episode，避免 token 长度增长导致越跑越慢。

### 6.3 真正缩短序列（token merging / ToMe）

- 对低重要 tokens 做 merge，减少 forward token 数
- 工程侵入最大，建议在 6.1/6.2 验证收益后再做。

---

## 7) 实验与验证（必须做的 sanity checks）

- **Token recall**：关键物体/手附近 tokens 是否在 Top‑K 覆盖率高（可视化到像素/patch）。
- **Ablation**：
  - 只用 attention / 只用 change / 融合
  - K 从小到大扫（Pareto：时延 vs 成功率）
- **反事实验证**：随机丢 token vs 丢低分 token vs 丢高分 token，比较 action 变化量与任务成功率。
- **长序列曲线**：episode 越长，是否还能保持时延不增长（cache hygiene 成功与否）。

