# 后续 Agent 接管说明

1. 项目 `C:\Users\lihao\Desktop\pokemon-red-rl-reproduction`，解释器 `.venv\Scripts\python.exe`。基础真实验收已经通过，不必重装。当前 torch 2.5.1+cu121，CUDA True，RTX 4060 8 GB，系统 RAM 约 16 GB。
2. 正确 ROM 是 `local/PokemonRed.gb`，1 MiB，SHA1 `ea9bcae617fdf159b045185467ae58b2e4a48b9a`。用户确认合法使用权后，从 pret/pokered 固定 SHA `af519899719f0754965776faac0e836a3b906e6d` + RGBDS 1.0.4 构建，未修改字节。旧中文 `PokemonRed.gbc` 不兼容，不要再适配或换名使用。
3. 官方初始状态已在 `local/has_pokedex_nballs.state`，SHA256 `0ff708b3dcdfa6c1f5fc07687b1641247017b1eb8a9a8b7c6eb0dd988431bd7e`。初始坐标 5/3/map40，杰尼龟 Level6、HP22/22；英文原版继续运行正常。
4. 真实 CUDA 训练已完成一次 32 步更新，再恢复到累计 64 步。最新真实 Checkpoint：`artifacts/smoke-gpu-resumed/checkpoint_000000064.pt`；原始 32 步权重：`artifacts/smoke-gpu/checkpoint_000000032.pt`。`artifacts/synthetic-*` 仅合成测试，不要混淆。
5. 恢复用原输出的 `config.yaml`，严格校验配置。恢复模型、AdamW、主进程 RNG、update 和总步数，调度按 update 继续；不恢复中途模拟器或 worker RNG，会重开 episode。输出必须全新目录，`--total-steps` 是累计整批目标。
6. 观看命令见 README，可用 CPU/CUDA。Smoke 权重仍明确标记 Untrained Agent。画面、150 帧录像解码和步数上限正常结束均通过；Q/Esc 交互退出未完成独立验证，物理 Esc 曾停止 Computer Use，不能把它算作游戏按键验收。
7. `configs/smoke.yaml` 为 2×16、1 epoch、2 minibatches，累计硬上限 256。用户没有授权在本阶段长训；不要自动追加训练。论文 `paper_baseline.yaml` 规模不能直接套用到本机 16 GB RAM。
8. PPO、GAE、网络和奖励复用官方 neroRL `poke_red`，SHA 见 `UPSTREAM.json`。不要改算法来解决输入问题。已知论文/源码差异见 `docs/baseline_comparison.md`。
9. 构建源码和工具在忽略的 `work/`：w64devkit 2.10.1（Make4.4.1/GCC16.2）、RGBDS1.0.4、pokered。`scripts/build_red.ps1` 可在已有工具上重复构建并校验。Cygwin 安装曾被自动审批拒绝，便携工具方案已成功，不必安装 WSL/Cygwin。
10. 当前报告 `docs/reports/phase1_validation.md`，历史修改版失败报告 `rom_cuda_validation_previous.md`；Junit、训练 JSONL、TensorBoard 和最终读回证据在 `artifacts/`。自动测试 11 passed。不要重新训练来生成相同证据。
11. ROM、状态、源码素材、模型、视频、工具和大日志均不上传。原始上游 Git 归档 `local/upstream.git` 也含状态历史，禁止推送。本地公开代码保留 MIT/NOTICE，尚未创建 GitHub 仓库。
