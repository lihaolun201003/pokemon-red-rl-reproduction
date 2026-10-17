# 论文与上游配置差异

来源：[论文 v2](https://arxiv.org/html/2502.19920v2)，上游固定 SHA 见 `UPSTREAM.json`。`configs/poke_red_v2.yaml` 保留原文件，`configs/paper_baseline.yaml` 是独立的论文参数配置。后者**不能消除论文文字与上游环境代码本身的差异**，尚不代表严格数值复现。

| 项目 | 官方 YAML | 论文 | paper_baseline |
|---|---|---|---|
| workers × rollout | 8 × 64 | 32 × 2048 | 32 × 2048 |
| epochs / minibatches | 3 / 8 | 3 / 8 | 3 / 8 |
| gamma / lambda | .997 / .95 | .997 / .95 | .997 / .95 |
| PPO clip / value coef | .2 / .5 | .2 / .5 | .2 / .5 |
| LR / optimizer | .0003 / AdamW | .0003 / AdamW | 相同 |
| max gradient norm | .5 | .5 | .5 |
| entropy beta | .01 → 0 | 0 | 0 |
| 初始 episode budget | YAML 范围 40960..81920；wrapper 实际取最大值 | 10240 | 10240 |
| event reward | .5 × 4 = 2 | 2 | 2 |
| exploration / coordinate | .5 × 1 × .1 = .05 | .005 | .5 × .1 × .1 = .005 |
| level weight | 0 | .5 | .5 × 1 |
| healing coefficient | 0 | 2.5 | .5 × 5 |

Smoke 配置保留论文参数和完整官方网络，仅将 workers=2、worker_steps=16、epochs=1、minibatches=2、checkpoint_interval=1；上限为 256 个累计环境步。默认验证目标是 32 步，再恢复到累计 64 步。

观测为灰度 72×80×3 和 48×48×1 探索图、被上游 mask 选中的事件标志、6 个 HP 比例和 6 个等级 / 100。送入模型前图像变为 CHW，所有模态为 float32；图像、地图、事件范围 [0,1]，game_state 声明 [-1,1]。具体事件维度从上游 `create_event_flag_mask` 计算，不手工改动。动作下标**必须保持官方顺序**：0 DOWN、1 LEFT、2 RIGHT、3 UP、4 A、5 B、6 START；每次按 8 帧，松开 16 帧。

代码与论文的已知差异（保留官方实现，不擅改算法）：

- 上游等级奖励先对每只宝可梦减 2，并额外扣除初始宝可梦的 4 级，再以历史最高值计算 22 级后的缩放；论文公式直接使用等级总和。
- 上游治疗使用全队 HP 总和 / 最大 HP 总和的差值，排除升级时的回血，并过滤队伍大小变化和濒死情况；论文写作每只宝可梦的 HP 分数之和，并提及升级回血。
- 新事件使 budget 增加 2048，但单次检测到多个新事件时也只增加一次；原代码统计的 reward events 与观测 mask 不完全相同。
- 只以步数耗尽结束（Gymnasium truncation），不把黑屏、全队倒下或到达城市当作通关 terminal。neroRL 将 done/truncation 合并后交给原版 GAE。
- seed 控制 Python/NumPy/PyTorch 和动作采样，不改写初始存档中的 Game Boy RNG。恢复时重开 episode，不恢复模拟器中途 RAM。
- 仓库未找到附带的兼容预训练权重；不下载第三方不同框架权重充当论文模型。

后续若进行严格论文复现，应先与作者确认这些差异，再另立实验配置。本阶段只验证可运行性，不比较论文分数。
