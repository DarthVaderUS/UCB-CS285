# CS285 HW1 全流程系统化学习指南（Push-T Imitation Learning）

## 0. 文档定位与学习目标

这份文档的目标不是“告诉你跑一条命令”，而是让你从三个层次彻底掌握 HW1：

1. 理论层：你在学什么机器学习问题，为什么这样设计。
2. 工程层：每个工具和每段代码在整个实验链路中的作用。
3. 实验层：如何稳定复现、如何验证达标、如何写出高质量作业报告。

读完并实践后，你应当能做到：

- 独立解释 Behavior Cloning、Action Chunking、Flow Matching 的核心思想。
- 独立重写本作业关键模块（数据、模型、训练、评估）。
- 独立排查常见问题（数据下载损坏、日志缺失、训练不达标等）。
- 独立完成结果分析与报告撰写。

---

## 1. 作业任务全景

### 1.1 作业核心任务

你要在 Push-T 环境做 imitation learning（模仿学习），实现两种策略：

- MSE Policy（监督回归版行为克隆）
- Flow Matching Policy（基于速度场学习的生成式策略）

并完成完整训练与评估。

### 1.2 指标要求（来自作业说明）

- MSE 最终 reward 至少 0.5
- Flow Matching 最终 reward 至少 0.7

### 1.3 你本次已经完成的实测结果

- MSE: 0.55809（达标）
- Flow: 0.843（达标）

说明：当前实验设置已经满足作业精度要求。

---

## 2. 项目结构与职责划分

```text
hw1/
├─ pyproject.toml
├─ README.md
├─ src/
│  └─ hw1_imitation/
│     ├─ data.py
│     ├─ model.py
│     ├─ evaluation.py
│     ├─ train.py
│     └─ modal_train.py
└─ exp/
   └─ seed_xxx_.../
      ├─ log.csv
      └─ wandb/
```

### 2.1 各文件作用

- data.py
    - 下载与读取 Push-T 数据集（zarr 格式）
    - 数据标准化 Normalizer
    - 构造 chunk 数据集（state, action_chunk）

- model.py
    - BasePolicy 接口
    - MSEPolicy 实现
    - FlowMatchingPolicy 实现
    - build_policy 工厂函数

- train.py
    - 参数配置（TrainConfig）
    - 初始化数据、模型、优化器
    - 主训练循环
    - 周期评估与最终评估
    - 日志落盘

- evaluation.py
    - 在真实环境 rollout 评估策略
    - 记录 mean reward 与视频
    - 保存 checkpoint artifact

- modal_train.py
    - 可选：把训练搬到 Modal 远程环境

---

## 3. 理论知识点（考试式理解）

## 3.1 Imitation Learning 与 Behavior Cloning

你有专家数据集 D = {(s_t, a_t)}，目标是学习策略 π(a|s)，让策略在环境中表现接近专家。

最基础做法是 Behavior Cloning：把它看成监督学习。

- 输入：状态 s_t
- 输出：动作 a_t（或动作序列）
- 损失：通常是 MSE

优点：简单稳定，易训练。
缺点：分布偏移（训练看到的是专家状态，测试看到的是自己策略访问到的状态）。

## 3.2 Action Chunking

本作业不是单步动作预测，而是一次预测未来 K 步动作 chunk。

- 记 chunk 大小为 K
- 模型输出形状是 (batch, K, action_dim)

好处：

- 减少每步重新推理的频率
- 提升动作序列平滑性
- 更接近“短时规划”

## 3.3 标准化（Normalization）

为什么要做？

- 状态和动作量纲不同，数值范围差异大
- 不标准化会导致训练难、学习率敏感、收敛慢

做法：

- state_norm = (state - mean_state) / std_state
- action_norm = (action - mean_action) / std_action

推理时要反标准化动作再送进环境。

## 3.4 MSE Policy 的数学形式

模型：

- 输入状态 s
- 输出动作块 A_hat

损失：

$$
\mathcal{L}_{mse} = \lVert A_{hat} - A \rVert_2^2
$$

本质：直接学习从状态到动作序列的回归映射。

## 3.5 Flow Matching Policy 的数学形式

Flow Matching 的直觉：

- 从噪声动作 A0 出发
- 定义一个时间参数 tau 从 0 走到 1
- 学一个速度场 v_theta(s, A_tau, tau)
- 让动作沿速度场演化，最终到达专家动作 A

训练时：

1. 采样噪声 A0
2. 采样 tau ~ Uniform(0, 1)
3. 线性插值

$$
A_{tau} = tau * A + (1 - tau) * A0
$$

4. 目标速度

$$
v^* = A - A0
$$

5. 最小化

$$
\mathcal{L}_{flow} = \lVert v_{theta}(s, A_{tau}, tau) - (A - A0) \rVert_2^2
$$

推理时（Euler 积分）：

$$
A^{k+1} = A^k + \Delta t * v_{theta}(s, A^k, t_k)
$$

从高斯噪声初始化，重复 num_steps 次得到动作块。

## 3.6 评估指标

在 Push-T 环境中 rollout 多个 episode（本作业是 100 个），记录每个 episode 的 max reward，再求平均：

- eval/mean_reward = mean(max_reward over episodes)

这是你是否达标的核心依据。

---

## 4. 工具栈全量解析（你需要会用什么）

## 4.1 Python 包管理与环境

### 4.1.1 uv（课程推荐）

README 推荐使用 uv：

- uv sync 安装依赖
- uv run 执行脚本

优点：速度快，环境管理一致。

### 4.1.2 conda + pip（你这次实际也用了）

你当前实际流程是：

- conda 管理基础环境
- pip install -e . 安装项目
- python 直接跑训练脚本

这条路径也可行。

## 4.2 PyTorch

你需要掌握：

- nn.Module 定义模型
- forward 逻辑（这里封装成 sample_actions / \_predict_velocity）
- loss.backward
- optimizer.step
- DataLoader 批加载
- to(device) 设备迁移

## 4.3 numpy

用于：

- 数据统计均值标准差
- 数组转换
- 环境输出处理

## 4.4 zarr

Push-T 数据存储格式，读取快，适合大规模数组。

你在 data.py 里通过 zarr.open(..., mode="r") 读取 state/action/episode_ends。

## 4.5 gymnasium + gym-pusht

用于在线评估策略：

- gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")
- env.step(action)
- env.reset(seed)

## 4.6 wandb

用于实验追踪、视频、模型 artifact。

离线模式：

- 设置 WANDB_MODE=offline
- 不依赖网络，也能生成本地日志

在线模式：

- wandb login 后可在网页查看曲线和视频

## 4.7 tyro

把 dataclass 直接转命令行参数，降低参数管理复杂度。

你可以直接写：

- --policy-type flow
- --num-epochs 400
- --flow-num-steps 10

## 4.8 Modal（可选）

远程训练平台。HW1 通常本地 CPU 就够。

---

## 5. 关键代码设计思路（按模块拆解）

## 5.1 data.py 设计思路

### 5.1.1 Normalizer

设计点：

- 以数据统计构建 mean/std
- std 做最小值截断，防止除零
- 提供 normalize 与 denormalize 对称接口

意义：

- 训练和推理阶段输入输出规范一致

### 5.1.2 Chunk 数据集构建

核心逻辑：

- 按 episode 边界滑窗
- 每个样本取起点 t
- 输入 state[t]
- 标签 action[t:t+K]

关键细节：

- 不能跨 episode 拼接 chunk
- 通过 episode_ends 限定有效起点范围

## 5.2 model.py 设计思路

### 5.2.1 BasePolicy 抽象接口

统一两种策略 API：

- compute_loss
- sample_actions

好处：

- train/eval 代码对具体策略类型解耦
- build_policy 工厂函数可替换模型实现

### 5.2.2 MSEPolicy

设计点：

- MLP 输入 state_dim
- MLP 输出 chunk_size \* action_dim
- reshape 成 (B, K, A)
- compute_loss 内部调用 sample_actions，逻辑一致

### 5.2.3 FlowMatchingPolicy

设计点：

- 额外输入 tau
- 网络输入是 concat(state, flatten(action_chunk), tau)
- 输出 velocity，与动作块同形状

训练路径：

- 随机 noise
- 随机 tau
- 构造插值 action
- 回归目标速度

采样路径：

- 从高斯噪声 action_chunk 初始化
- 每步计算 velocity
- Euler 更新

## 5.3 train.py 设计思路

### 5.3.1 参数配置集中化

TrainConfig 管理所有超参数，便于：

- CLI 覆盖
- 复现实验
- wandb 记录

### 5.3.2 训练循环结构

标准模板：

1. 取 batch
2. 前向算 loss
3. zero_grad
4. backward
5. step
6. log
7. 周期性 evaluate

最终再做一次 evaluate（防止最后一步没命中 eval_interval）。

### 5.3.3 日志与实验目录

目录命名包含：

- seed
- 时间戳
- exp_name

好处：

- 多实验不会互相覆盖
- 可追踪性强

## 5.4 evaluation.py 设计思路

### 5.4.1 chunk 执行逻辑

每次用策略生成 K 步动作，环境逐步消费这些动作。

这样与训练目标（动作 chunk）保持一致。

### 5.4.2 视频记录

仅前若干 episode 记录视频，避免日志过重。

### 5.4.3 checkpoint artifact

每次评估可存一版模型，方便回溯最佳策略。

---

## 6. 实验复现详细步骤（你可以照着一条条做）

## 6.1 环境准备

### 6.1.1 进入项目目录

PowerShell:

```powershell
cd "D:\大二\其他\self learning\UCB CS285 Deep Reinforcement Learning\hw\hw1"
```

### 6.1.2 安装依赖（你已经执行成功）

```powershell
pip install -e .
```

## 6.2 快速连通性检查（建议每次改代码后先做）

### 6.2.1 冒烟训练（很短）

```powershell
$env:WANDB_MODE="offline"
$env:PYTHONPATH="src"
python src/hw1_imitation/train.py --policy-type mse --num-epochs 2 --batch-size 128 --eval-interval 1000000 --num-video-episodes 0 --exp-name mse_smoke
```

目标：

- 能下载数据
- 能训练
- 能评估
- 能写出 exp 目录

## 6.3 完整训练（达标主实验）

### 6.3.1 MSE 完整训练

```powershell
$env:WANDB_MODE="offline"
$env:PYTHONPATH="src"
python src/hw1_imitation/train.py --policy-type mse --num-epochs 400 --eval-interval 1000000000 --exp-name mse_full
```

### 6.3.2 Flow 完整训练

```powershell
$env:WANDB_MODE="offline"
$env:PYTHONPATH="src"
python src/hw1_imitation/train.py --policy-type flow --num-epochs 400 --eval-interval 1000000000 --exp-name flow_full
```

说明：

- 这里 eval_interval 设置很大，目的是只在最后评估一次（最终 run_training 会补一次最终 evaluate）。

## 6.4 结果目录检查

你当前生成了：

- exp/seed_42_20260427_195930_mse_full
- exp/seed_42_20260427_200605_flow_full

每个目录至少应包含：

- log.csv
- wandb/

---

## 7. 结果解读与对比结论

## 7.1 达标情况

- MSE = 0.55809（超过 0.5）
- Flow = 0.843（超过 0.7）

## 7.2 为什么 Flow 往往更高

直觉上，Flow 模型不是“直接猜最终动作”，而是在动作空间学习一个连续演化过程，具备更强的表达能力，尤其适合多峰或复杂动作分布。

## 7.3 loss 与 reward 的关系

- loss 下降通常是好信号
- 但最终以环境 reward 为准
- 训练目标和真实控制效果不是完全等价

---

## 8. 常见问题与排障手册

## 8.1 数据下载损坏（你已遇到）

症状：

- urllib.error.ContentTooShortError

处理：

```powershell
Remove-Item -Path "data/pusht.zip" -ErrorAction SilentlyContinue
```

然后重新跑训练，触发重下载。

## 8.2 日志里看不到 eval/mean_reward

你当前 Logger 的 CSV 表头是在第一次 log 时固定的。若第一次是 train/loss，则后续 eval 字段不在 CSV header 中。

建议：

- 优先从 wandb 历史查看评估指标
- 或改 Logger 逻辑支持动态扩展 header

## 8.3 训练慢

可尝试：

- 减小 num_video_episodes
- 减少评估频率
- 用 GPU
- 合理调 batch_size

## 8.4 不达标

优先调参顺序建议：

1. num_epochs（先增加）
2. hidden_dims（容量）
3. lr（常见范围 1e-4 到 5e-4）
4. flow_num_steps（Flow 推理质量）

---

## 9. 报告撰写建议（直接可用模板）

## 9.1 必写内容

1. 两种策略实现说明
2. 模型结构（层数、宽度、激活）
3. 训练设置（epoch、batch、lr、chunk_size）
4. 最终 reward 与阈值对比
5. rollout 现象分析（视频观察）

## 9.2 推荐图表

- train/loss vs step（MSE 与 Flow 各一条）
- eval/mean_reward 对比条形图

## 9.3 结论写法建议

- 先给数字结论（是否达标）
- 再给机制解释（为什么 Flow 更优）
- 最后给不足与改进方向（比如日志结构、评估频率、调参空间）

---

## 10. 系统化掌握路线（建议 3 轮学习法）

## 第 1 轮：复现导向（你已经完成）

目标：跑通并达标。

你已完成。

## 第 2 轮：原理导向

你应亲自推导并手写：

- Flow loss 从何而来
- Euler 采样为什么有效
- chunk_size 对控制频率和性能的影响

## 第 3 轮：工程导向

建议你亲手做 3 个小改动：

1. 给 Logger 加动态 header，保证 CSV 包含 eval 指标
2. 增加保存 best reward checkpoint
3. 新增配置开关：是否每 N step 存视频

做到这一步，你就不仅“会做作业”，而是掌握了一个完整的深度学习实验工程闭环。

---

## 11. 一页速查（临考/提交前）

- 数据：state + action + episode_ends（zarr）
- 任务：state -> action_chunk
- 模型：MSE 与 Flow 两种策略
- 训练：Adam + mini-batch + 周期评估
- 评估：100 episodes mean reward
- 阈值：MSE >= 0.5, Flow >= 0.7
- 你实测：MSE 0.55809, Flow 0.843
- 交付：代码 + exp 日志目录 + 报告分析

---

## 12. 附录：关键命令清单

### 安装

```powershell
pip install -e .
```

### MSE 冒烟

```powershell
$env:WANDB_MODE="offline"
$env:PYTHONPATH="src"
python src/hw1_imitation/train.py --policy-type mse --num-epochs 2 --batch-size 128 --eval-interval 1000000 --num-video-episodes 0 --exp-name mse_smoke
```

### MSE 完整

```powershell
$env:WANDB_MODE="offline"
$env:PYTHONPATH="src"
python src/hw1_imitation/train.py --policy-type mse --num-epochs 400 --eval-interval 1000000000 --exp-name mse_full
```

### Flow 完整

```powershell
$env:WANDB_MODE="offline"
$env:PYTHONPATH="src"
python src/hw1_imitation/train.py --policy-type flow --num-epochs 400 --eval-interval 1000000000 --exp-name flow_full
```

### 检查实验目录

```powershell
Get-ChildItem exp
```

---

如果你愿意，我下一步可以继续给你补两份配套资料：

1. 实验报告模板（含图表占位、可直接填）
2. 一份“手把手代码精讲版”，逐行解释 model.py 与 train.py 每个设计选择
