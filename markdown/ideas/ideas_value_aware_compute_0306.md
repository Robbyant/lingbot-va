# WAM 推理加速：Value-Aware Compute 创新点（0306）

单纯把 `num_inference_steps` 调小（如 10→3）难以作为创新点——审稿人会认为 diffusion/flow matching 减步数、蒸馏、一致性采样已有大量工作。

真正能站住脚的角度：在 WAM 里，**world 和 action 不是两个独立生成任务**，而是一个闭环控制系统里**两种不同时间尺度、不同价值密度**的计算。创新应写成 **「如何把有限的 denoising budget 在 world/video 与 action 之间按控制价值动态分配」**，而不是「把 diffusion steps 调小」。

当前实现（`wan_va_server.py`）已给出切入点：先 video loop 再 action loop，两个 budget 分开配置；可在此基础上做**预算分配策略、world/action 耦合、停止准则、cache 复用**。

---

## 1. Action-Conditioned Dynamic Budget

**核心**：不是固定 `video=3, action=10`，而是每个 control cycle 动态决定  
- 这一步要不要更新 world  
- world 需要几步  
- action 需要几步  

预算由 **action relevance** 决定，而非仅看图像变化。

**信号示例**：
- 当前 action 的不确定性 / 多样性
- action 对 world token 的注意力集中度
- 预测动作与上一步动作的差异
- 接触 / 抓取 / 放置等关键事件
- world prediction 对 action 的边际增益

**Novelty 表述**：  
*"We allocate denoising budget according to the control value of world updates, rather than uniformly reducing diffusion steps."*

与传统 diffusion 的区别：传统是样本质量导向；此处是**控制收益导向**；预算分配对象是**两个耦合分支**，不是单一生成器。

---

## 2. Multi-Rate World-Action Sampling

**思想**：
- action **高频**更新
- world **低频**更新
- 大部分周期复用旧 world cache
- 仅在关键时刻刷新 video/world branch

机器人控制里动作环通常比世界建模环更高频，设定自然。

**Novelty 表述**：  
*"Asynchronous denoising for embodied world-action models."*

与一般 video diffusion 的区别：world 不是最终输出目标，而是 action 的上下文；可允许 world 低频、action 高频；本质是控制系统里的**多速率采样**。

---

## 3. Action-Guided ROI World Denoising

**思想**：不要让 world branch 每次对整张 latent grid 同等强度更新，而是  
- 重点更新与当前动作相关的区域（gripper 附近、被操作物体、容器口等）  
- 背景区域复用 cache 或低频更新  

**ROI 来源**：
- wrist camera / end-effector pose
- 上一步 action rollout 落点
- action→world attention map
- optical flow / obs difference

**Novelty 表述**：  
*"We reduce world denoising by exploiting the action-conditioned spatial sparsity of embodied interaction."*

比通用 sparse diffusion 更合理：ROI 不是纯视觉，而是 **causal to action**。

---

## 4. Speculative Control with World Verification

**流程**：
1. 用极少 steps 快速生成 action proposal
2. 用 1 小步 world update 预测该 action 的后果
3. 若 world verification 通过 → 直接执行
4. 若不通过 → 追加 refinement steps

类比 LLM speculative decoding，但此处：proposal 是 action，verifier 是 world model，目标是 control success。

**Novelty 表述**：  
*"Fast action proposal, cheap world verification, selective refinement."*

WAM-specific：利用同时有 world 与 action 两个头。

---

## 5. 论文主线 Framing

**主标题建议**（避免「fewer diffusion steps」）：  
- **Value-Aware Compute Allocation for World-Action Models**  
- 或 **Action-Centric Adaptive Sampling for World-Action Models**

**主张**：
- WAM 的 world/video 与 action 分支对控制的价值密度不同
- 统一固定步数低效
- 应根据动作相关性、阶段、置信度，把 compute budget 动态分配到最有用的 branch / token / timestep

---

## 6. 推荐组合方案

落地时建议做成一个完整系统：

- **动态 world/action 步数分配**
- **world 低频、action 高频**
- **action-guided ROI world update**
- **world verification 触发 fallback refinement**

**快路径**：不更新或少更新 world，小步数出 action，用局部 ROI 或 summary world 做验证。  
**慢路径**：关键周期 full world refresh，增加 action refinement，重建可靠 world context。

---

## 7. 避免被说「只是工程 trick」

需证明：

1. **不是所有 step reduction 都一样**：相同 NFE 下，你的分配策略优于固定减步。
2. **world/action 联合分配优于只压 video 或只压 action**：做 branch-level ablation。
3. **改善来自 WAM 特有结构**：例如  
   - action uncertainty 能预测 world refresh 必要性  
   - ROI world tokens 与 action success 强相关  
   - speculative verification 显著降低尾延迟而不掉成功率  

**评测建议**：success rate、average NFE、P50/P90 latency、failure mode breakdown、不同任务阶段的 budget 分布。

---

## 8. 关于 env_step 占比

若 `env_step_update` 占端到端时间很高（如 80%+），则：

- 减少 inference steps 对**模型推理时间**有效
- 对**端到端 wall-clock** 改善会被环境仿真掩盖

论文中建议拆开报告：**model-only inference latency** 与 **end-to-end control latency**。

---

## 9. 一句话总结

最有新意的方向不是「把 diffusion steps 调小」，而是：  
**让 world 和 action 在闭环控制中按价值、按阶段、按空间区域、按验证结果动态消耗计算。**  
这才是从「生成模型加速」走向「world-action model 专属推理机制」的差异点。
