# 动态 Step Budget：Lazy World Denoising / Multi-rate 实现思路

目标：**每控制周期动态决定 world/action 的步数**，在不动模型结构的前提下砍 world NFE（通常是最大头）。对应主清单 `ideas_0305.md` 创新点 1/2/9/11。

---

## 1. 核心直觉

- **World 分支**：token 多、attention 重，是推理大头；但很多周期里“世界没怎么变”或“action 不需要新世界信息”。
- **做法**：大多数周期 world **少算/不算**（用旧 world cache/旧 world latent）；少数关键周期 world **多算**。
- **实现**：每个 control cycle 动态决定 `world_steps`、`action_steps`（以及是否强制刷新观测/缓存）。

---

## 2. Lazy World Denoising（LWD）

**含义**：world 按需算——当“世界没怎么变 / action 不需要新世界”时，world 去噪步数从 full 降到 0~2。

**代码落点（最小改动）**  
`wan_va_server.py::_infer()` 当前逻辑：

- `self.scheduler.set_timesteps(video_inference_step)`
- `timesteps = timesteps[:video_step]`（当 `video_step != -1`）

把 **`video_step` 从固定 config 改为每周期计算的 `world_steps`**：

- **world_steps = 0**：跳过 video loop。需注意：若 `frame_st_id == 0`，仍需把 `latents[:, :, 0:1]` 设为 `init_latent`（第一帧条件注入），否则偏离有条件生成。
- **world_steps = 1~3**：跑极少步，维持 KV/latent 的“新鲜度”。
- **world_steps = full**：关键周期跑满。

**world_need 指标（training-free、低成本）**  
结合已有观测变化检测（如 `_obs_change_score` / obs ref）：

- **obs_change**：多视角 downsample 的 mean-abs-diff。
- **state_delta**：`np.max(np.abs(state - prev_state))`。
- **action_delta**：上一周期输出动作幅度/jerk，如 `||a_t - a_{t-1}||`。
- **外部强事件**：碰撞、抓取成功、目标切换等 → 强制 `world_steps = full`。

**简单分段规则示例**：

```python
if force_refresh or obs_change > T_hi or state_delta > S_hi:
    world_steps = FULL
elif obs_change < T_lo and state_delta < S_lo and action_delta < A_lo:
    world_steps = 0
else:
    world_steps = 2  # 小步刷新
```

---

## 3. Multi-rate（多速率 / 快慢双环）

**含义**：world 低频更新、action 高频更新；不必每个控制周期都更新 world。

**实现要点**：

- 维护计数器 `world_update_countdown`（或等价逻辑）。
- 每次 `infer()`：
  - 若 countdown == 0 或事件触发：跑一次 world 更新（少步或 full），然后 countdown = K。
  - 否则：本周期 `world_steps = 0`，只跑 action。
- action 分支继续使用**上一轮** world 的 cache/latent；需保证在“更新周期”里 world cache 确实被更新（如 `_compute_kv_cache()` 或 video loop 最后一次 forward 的 `update_cache=1`）。

**落点**：server 内 `infer()` 的调用节奏 + `_infer()` 的 step 覆盖逻辑。

---

## 4. Goal-Progress Adaptive Budget（创新点 9 的加速版）

**含义**：用任务进度 proxy 调预算——越接近目标/越简单 → 步数越少；停滞/反向/恢复 → 步数增加。

**实现**：step controller 多一个输入 `progress_proxy`（末端到目标距离、抓取状态、门角度等），并入 LWD 的分段规则，例如：

- 进度明显推进 → 减少 world/action steps。
- 进度停滞或反向 → 增加 steps 或触发回退策略。

---

## 5. Batch Candidates（创新点 11，用并行换更少大步数）

**含义**：用**小步数**并行生成 K 个 action 候选，用廉价选择器（规则/约束/碰撞几何）选一个，再对选中者做少量 refinement（可选）。目标是降低“单路径长 steps 兜底”的概率，从而降低**平均** NFE 与尾延迟。

**实现要点**：

- action latent 的 batch 维扩成 K，**共享同一份 world cache**（不复制 world token）。
- 跑 `action_steps_small`，选优后再单独跑 `action_steps_refine`（可选）。

---

## 6. 最小可落地改法（建议起步）

1. 在 `VA_Server` 增加 **step controller** 状态：`prev_state`、`prev_action`、`prev_obs_change`（或等价）。
2. `_infer()` 支持 **override**：`world_steps_override`、`action_steps_override`，用它们截断 `timesteps` 长度。
3. **world_steps = 0 的边界**：确保 `frame_st_id == 0` 时 `latent_cond` 仍注入（否则有条件生成会偏）。
4. 先只用 **obs_change + state_delta** 做 ablation：`world_steps ∈ {0, 2, FULL}`，扫阈值看延迟/成功率 Pareto。

---

## 7. 评估指标建议

- 延迟：每周期 P50/P90/P99；world NFE 分布。
- 控制质量：成功率、碰撞率、动作抖动；按阶段（抓取前/中/放置）分桶。
- 统计：world_steps=0 / 2 / full 的占比；误判（关键周期被判为 0）与漏判比例。
