# 评估结果汇总（eval-protocol-v1，含 extended check v1-ext）

- 生成时间（UTC）：2026-10-17T17:20:54.726277+00:00
- 输入 runs：11 个 results.json（modes: checkpoint, random_actions, untrained；variants: v1, v1-ext）
- 设备：cuda；base seed：[100000]
- v1：steps/episode=[4096]，episodes/episode 组=[3]；v1-ext：steps/episode=[8192]，episodes/episode 组=[5]
- 数据全部来自 `scripts/eval_checkpoint.py` 的 `results.json`，未经人工修饰。

## 1. Stochastic 策略评估（v1；每个 run 3 个 episode；mean ± std 来自该组 episode）

| Run | 训练步数 | n | episode reward | length | coord_count | maps visited | badges | levels_sum | party_size | deaths | max_map_progress | mt_moon | cerulean | 动作熵(bit) | 最大动作占比 | eval steps | eval 秒数 | eval steps/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eval_10k | 10,000 | 3 | 0.3867 ± 0.1925 | 4,096.0 ± 0.0 | 77.33 ± 38.51 | 1.67 ± 0.47 | 0.00 ± 0.00 | 6.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.67 ± 0.47 | 0.000 ± 0.000 | 0.000 ± 0.000 | 2.800 | 0.170 | 16,384 | 93.4 | 175.4 |
| eval_100k | 100,000 | 3 | 2.6650 ± 2.9133 | 4,096.0 ± 0.0 | 133.00 ± 16.99 | 2.33 ± 0.47 | 0.00 ± 0.00 | 6.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 0.000 ± 0.000 | 0.000 ± 0.000 | 2.681 | 0.202 | 16,384 | 82.4 | 198.8 |
| eval_300k | 300,000 | 3 | 4.1783 ± 2.4736 | 4,096.0 ± 0.0 | 169.00 ± 34.67 | 3.67 ± 0.47 | 0.00 ± 0.00 | 6.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 1.33 ± 0.47 | 0.000 ± 0.000 | 0.000 ± 0.000 | 2.592 | 0.229 | 16,384 | 87.1 | 188.1 |
| eval_500k | 500,000 | 3 | 0.5800 ± 0.5813 | 4,096.0 ± 0.0 | 82.67 ± 69.31 | 1.67 ± 0.94 | 0.00 ± 0.00 | 7.00 ± 1.41 | 1.33 ± 0.47 | 0.33 ± 0.47 | 0.67 ± 0.94 | 0.000 ± 0.000 | 0.000 ± 0.000 | 2.520 | 0.242 | 16,384 | 87.9 | 186.5 |
| eval_1m | 1,000,000 | 3 | 2.7933 ± 2.7165 | 4,096.0 ± 0.0 | 158.67 ± 36.11 | 3.00 ± 0.00 | 0.00 ± 0.00 | 6.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 0.000 ± 0.000 | 0.000 ± 0.000 | 2.675 | 0.198 | 16,384 | 81.1 | 202.1 |
| baseline_untrained | 0 (untrained) | 3 | 1.1100 ± 1.3154 | 4,096.0 ± 0.0 | 88.67 ± 74.59 | 1.67 ± 0.94 | 0.00 ± 0.00 | 6.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.67 ± 0.94 | 0.000 ± 0.000 | 0.000 ± 0.000 | 2.803 | 0.164 | 16,384 | 88.5 | 185.2 |

## 2. Deterministic 策略评估（v1；每 run 1 个 argmax episode；轨迹唯一，方差按定义为 0）

| Run | 训练步数 | terminated_by | length | episode reward | coord_count | maps visited | badges | levels_sum | party_size | deaths | max_map_progress | mt_moon | cerulean | 动作熵(bit) | 最大动作占比 | eval steps | eval 秒数 | eval steps/s | 视频 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eval_10k | 10,000 | step_cap | 4,096.0 | 0.0050 | 1.00 | 1.00 | 0.00 | 6.00 | 1.00 | 0.00 | 0.00 | 0.000 | 0.000 | 0.007 | 1.000 | 16,384 | 93.4 | 175.4 | artifacts/training/seed_0/eval/videos/eval_10k.mp4 |
| eval_100k | 100,000 | step_cap | 4,096.0 | 0.0450 | 9.00 | 1.00 | 0.00 | 6.00 | 1.00 | 0.00 | 0.00 | 0.000 | 0.000 | 0.054 | 0.995 | 16,384 | 82.4 | 198.8 | artifacts/training/seed_0/eval/videos/eval_100k.mp4 |
| eval_300k | 300,000 | step_cap | 4,096.0 | 0.0050 | 1.00 | 1.00 | 0.00 | 6.00 | 1.00 | 0.00 | 0.00 | 0.000 | 0.000 | 0.009 | 0.999 | 16,384 | 87.1 | 188.1 | artifacts/training/seed_0/eval/videos/eval_300k.mp4 |
| eval_500k | 500,000 | step_cap | 4,096.0 | 0.0050 | 1.00 | 1.00 | 0.00 | 6.00 | 1.00 | 0.00 | 0.00 | 0.000 | 0.000 | 0.019 | 0.999 | 16,384 | 87.9 | 186.5 | artifacts/training/seed_0/eval/videos/eval_500k.mp4 |
| eval_1m | 1,000,000 | step_cap | 4,096.0 | 0.0050 | 1.00 | 1.00 | 0.00 | 6.00 | 1.00 | 0.00 | 0.00 | 0.000 | 0.000 | 0.547 | 0.875 | 16,384 | 81.1 | 202.1 | artifacts/training/seed_0/eval/videos/eval_1m.mp4 |
| baseline_untrained | 0 (untrained) | step_cap | 4,096.0 | 0.0050 | 1.00 | 1.00 | 0.00 | 6.00 | 1.00 | 0.00 | 0.00 | 0.000 | 0.000 | 0.000 | 1.000 | 16,384 | 88.5 | 185.2 | artifacts/training/seed_0/eval/videos/baseline_untrained.mp4 |

## 3. 随机动作基线（v1；均匀随机，无模型）

| Run | episodes | episode reward | length | coord_count | maps visited | badges | levels_sum | party_size | deaths | max_map_progress | mt_moon | cerulean | 动作熵(bit) | 最大动作占比 | eval steps | eval 秒数 | eval steps/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_random_actions | 3 episodes | 0.3650 ± 0.0414 | 4,096.0 ± 0.0 | 73.00 ± 8.29 | 2.00 ± 0.00 | 0.00 ± 0.00 | 6.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 0.000 ± 0.000 | 0.000 ± 0.000 | 2.807 | 0.147 | 12,288 | 29.8 | 412.2 |

## 4. Extended check（v1-ext：每 run 5 stochastic + 1 deterministic × 8192 步，单列统计）

- v1-ext 与 v1 的 episode 预算不同（8192 vs 4096 步、5 vs 3 个 stochastic episode），reward/coords 等绝对值**不可直接跨协议比较**；该小节用于 n=5 降低方差，并在同一协议内比较 untrained / 300k / 500k / 1M。
- v1-ext 的 deterministic episode 同样只作轨迹记录（argmax 退化问题见 v1 表），不用于模型选择。

**Stochastic episodes（v1-ext）**

| Run | 训练步数 | n | episode reward | length | coord_count | maps visited | badges | levels_sum | party_size | deaths | max_map_progress | mt_moon | cerulean | 动作熵(bit) | 最大动作占比 | eval steps | eval 秒数 | eval steps/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ext_300k | 300,000 | 5 | 3.0880 ± 2.3740 | 8,192.0 ± 0.0 | 197.60 ± 46.27 | 4.00 ± 0.63 | 0.00 ± 0.00 | 6.60 ± 1.20 | 1.20 ± 0.40 | 0.20 ± 0.40 | 1.40 ± 0.49 | 0.000 ± 0.000 | 0.000 ± 0.000 | 2.577 | 0.244 | 49,152 | 229.4 | 214.2 |
| ext_500k | 500,000 | 5 | 0.8282 ± 0.8728 | 8,192.0 ± 0.0 | 103.00 ± 55.92 | 2.20 ± 1.17 | 0.00 ± 0.00 | 8.20 ± 4.40 | 1.80 ± 1.60 | 0.40 ± 0.80 | 1.00 ± 0.89 | 0.000 ± 0.000 | 0.000 ± 0.000 | 2.548 | 0.222 | 49,152 | 277.5 | 177.1 |
| ext_1m | 1,000,000 | 5 | 3.8270 ± 2.7737 | 8,192.0 ± 0.0 | 205.40 ± 25.61 | 4.00 ± 0.89 | 0.00 ± 0.00 | 6.00 ± 0.00 | 1.00 ± 0.00 | 0.20 ± 0.40 | 1.20 ± 0.40 | 0.000 ± 0.000 | 0.000 ± 0.000 | 2.679 | 0.190 | 49,152 | 227.7 | 215.9 |
| ext_untrained | 0 (untrained) | 5 | 1.4720 ± 1.5032 | 8,192.0 ± 0.0 | 134.40 ± 104.85 | 2.20 ± 1.60 | 0.00 ± 0.00 | 6.00 ± 0.00 | 1.00 ± 0.00 | 0.20 ± 0.40 | 0.60 ± 0.80 | 0.000 ± 0.000 | 0.000 ± 0.000 | 2.803 | 0.167 | 49,152 | 281.0 | 174.9 |

**Deterministic episode（v1-ext）**

| Run | 训练步数 | terminated_by | length | episode reward | coord_count | maps visited | badges | levels_sum | party_size | deaths | max_map_progress | mt_moon | cerulean | 动作熵(bit) | 最大动作占比 | eval steps | eval 秒数 | eval steps/s | 视频 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ext_300k | 300,000 | step_cap | 8,192.0 | 0.0050 | 1.00 | 1.00 | 0.00 | 6.00 | 1.00 | 0.00 | 0.00 | 0.000 | 0.000 | 0.005 | 1.000 | 49,152 | 229.4 | 214.2 | - |
| ext_500k | 500,000 | step_cap | 8,192.0 | 0.0050 | 1.00 | 1.00 | 0.00 | 6.00 | 1.00 | 0.00 | 0.00 | 0.000 | 0.000 | 0.010 | 0.999 | 49,152 | 277.5 | 177.1 | - |
| ext_1m | 1,000,000 | step_cap | 8,192.0 | 0.0050 | 1.00 | 1.00 | 0.00 | 6.00 | 1.00 | 0.00 | 0.00 | 0.000 | 0.000 | 0.545 | 0.875 | 49,152 | 227.7 | 215.9 | - |
| ext_untrained | 0 (untrained) | step_cap | 8,192.0 | 0.0050 | 1.00 | 1.00 | 0.00 | 6.00 | 1.00 | 0.00 | 0.00 | 0.000 | 0.000 | 0.000 | 1.000 | 49,152 | 281.0 | 174.9 | - |

## 5. 数据来源与运行元数据

| Run | variant | results.json | results sha256 | checkpoint sha256 | checkpoint env steps | device | steps/ep | episodes | base seed | eval steps | eval 秒数 | created_utc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eval_10k | v1 | artifacts/training/seed_0/eval/eval_10k/results.json | c03779bded11ba37 | f37d57fe69ddedc9 | 10,000 | cuda | 4,096 | 3 | 100,000 | 16,384 | 93.4 | 2026-10-17T16:53:13.088066+00:00 |
| eval_100k | v1 | artifacts/training/seed_0/eval/eval_100k/results.json | ebe9a6c571acfde8 | 3565588cecde490c | 100,000 | cuda | 4,096 | 3 | 100,000 | 16,384 | 82.4 | 2026-10-17T16:54:41.482265+00:00 |
| eval_300k | v1 | artifacts/training/seed_0/eval/eval_300k/results.json | dddb141344342c50 | 82e9cbfb53803a93 | 300,000 | cuda | 4,096 | 3 | 100,000 | 16,384 | 87.1 | 2026-10-17T16:56:14.537841+00:00 |
| eval_500k | v1 | artifacts/training/seed_0/eval/eval_500k/results.json | fbe85333e373b6ed | c48e058081a62ec1 | 500,000 | cuda | 4,096 | 3 | 100,000 | 16,384 | 87.9 | 2026-10-17T16:57:48.591616+00:00 |
| eval_1m | v1 | artifacts/training/seed_0/eval/eval_1m/results.json | 273dd623b343a720 | 7870310d7432e520 | 1,000,000 | cuda | 4,096 | 3 | 100,000 | 16,384 | 81.1 | 2026-10-17T16:59:16.349280+00:00 |
| baseline_untrained | v1 | artifacts/training/seed_0/eval/baseline_untrained/results.json | f7e2bb10bc51df2e | - | - | cuda | 4,096 | 3 | 100,000 | 16,384 | 88.5 | 2026-10-17T17:00:50.648063+00:00 |
| baseline_random_actions | v1 | artifacts/training/seed_0/eval/baseline_random_actions/results.json | c45498d2bcc95ac6 | - | - | cuda | 4,096 | 3 | 100,000 | 12,288 | 29.8 | 2026-10-17T17:01:25.820491+00:00 |
| ext_300k | v1-ext | artifacts/training/seed_0/eval/ext_300k/results.json | bfc3d9554aa6e8e1 | 82e9cbfb53803a93 | 300,000 | cuda | 8,192 | 5 | 100,000 | 49,152 | 229.4 | 2026-10-17T17:11:12.222878+00:00 |
| ext_500k | v1-ext | artifacts/training/seed_0/eval/ext_500k/results.json | 822a884c27a64a04 | c48e058081a62ec1 | 500,000 | cuda | 8,192 | 5 | 100,000 | 49,152 | 277.5 | 2026-10-17T17:15:55.763654+00:00 |
| ext_1m | v1-ext | artifacts/training/seed_0/eval/ext_1m/results.json | 2f2e7096e49a7b2d | 7870310d7432e520 | 1,000,000 | cuda | 8,192 | 5 | 100,000 | 49,152 | 227.7 | 2026-10-17T17:19:49.653059+00:00 |
| ext_untrained | v1-ext | artifacts/training/seed_0/eval/ext_untrained/results.json | 4d93affa4feed635 | - | - | cuda | 8,192 | 5 | 100,000 | 49,152 | 281.0 | 2026-10-17T17:07:16.835218+00:00 |

## 6. 说明与限制

- 确定性评估方差为 0：每个 run 只有 1 个 deterministic episode（策略 argmax）；初始存档固定 Game Boy RNG，且每个 episode 都新建 PyBoy 环境实例，因此同一 checkpoint 的确定性轨迹唯一。deterministic 表中的数值不是统计估计，方差按定义为 0，不参与显著性比较，也不应与其他 run 的 deterministic 值做方差检验。
- stochastic 评估方差来自策略采样：v1 每个 run 3 个 stochastic episode（seed = base seed + 0..2），v1-ext 每个 run 5 个（seed = base seed + 0..4），base seed=100000；表中 mean±std 是该组 episode 的样本统计（n=3 或 n=5），样本很小，std 只用于观察波动，不应据此做显著性结论。
- 评估步数不属于训练预算：eval_steps = episodes × steps_per_episode，只为评估推理消耗；它不推进训练计数器、不进入 PPO 更新、也不写入训练 run 目录。表中“训练步数”列来自 checkpoint.environment_steps（未训练/随机基线的训练步数为 0）。
- eval 秒数与吞吐包含环境构建、reset 与视频编码，且不同设备（CPU/GPU）与机器负载下不可直接横向比较；吞吐列仅作本机运行记录。
- 动作熵：基于该组 episode 聚合动作计数计算的 Shannon 熵（bit），7 个动作均匀分布时最大为 log2(7)≈2.807；最大动作占比 = 出现最多的单一动作的比例。
- 官方环境只记录前两个徽章（badges = badge_1 + badge_2）；mt_moon / cerulean 是 0/1 标记的组内均值（到达过该地点的 episode 比例）。
- 所有数值直接来自各 results.json，按原样聚合，未做筛选、平滑或人工修饰。
- v1-ext 是独立扩展检查（5 stochastic + 1 deterministic × 8192 步），其评估步数单列统计、不计入训练预算，也不并入 v1 对比表；v1 与 v1-ext 的 episode 长度不同，reward/coords 的绝对值不可直接相加或比较。
- 每个 run 的详细逐 episode / 每步指标见各自目录下的 `results.json`；本文件只做聚合。

## 7. 异常与警告

- 无
