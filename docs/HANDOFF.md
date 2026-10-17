# 下一位 Agent 的接管说明

1. 项目 `C:\Users\lihao\Desktop\pokemon-red-rl-reproduction`；Python `.venv\Scripts\python.exe`。先读 `docs/reports/progress.md` 和 `phase1_validation.md`。不必重装环境。
2. 用户已提供 `local/PokemonRed.gbc`（2 MiB，中文修改版，SHA1 `944ff0b461fb570d2c8aa99d819e8c8df0bdc446`）。可启动和按 Enter，但与官方初始状态执行不兼容：两帧后 party=57、Level=0、HP=57/14592、PC=0x0038，首步奖励 1258.005。不要重复对这份输入训练，不要更改奖励/内存地址/状态来强行通过。诊断命令 `scripts/probe_rom.py --rom local/PokemonRed.gbc` 退出码 2。
3. 仍需合法且匹配的英文原版 ROM（预期 SHA1 `ea9bcae617fdf159b045185467ae58b2e4a48b9a`），放 `local/PokemonRed.gb`。官方状态已按用户授权从固定上游复制到 `local/has_pokedex_nballs.state`，SHA256 `0ff708b3dcdfa6c1f5fc07687b1641247017b1eb8a9a8b7c6eb0dd988431bd7e`，不会上传。
4. 先 doctor、pytest、check_game 128 步，再按 README 训练到 32、恢复到 64、最后 watch 验证画面和退出。完成即停，不在 Codex 中长训。
5. 官方分支/固定 SHA 见 `UPSTREAM.json`。局部修复 dtype、close、headless speed、worker 参数/清理、有限数值检查和打包依赖。PPO 损失、模型、GAE、AdamW 复用官方。
6. `smoke.yaml` 联调；`paper_baseline.yaml` 保留论文参数。代码与论文的奖励/episode 差异见 `docs/baseline_comparison.md`，不静默改变算法。
7. `artifacts/synthetic-cpu/checkpoint.pt` 仅合成测试。真实预期 `artifacts/smoke/checkpoint_000000032.pt` 尚不存在。恢复使用原输出的 `config.yaml`，恢复模型/AdamW/RNG/调度 update，不恢复运行中的游戏。
8. RTX 4060 8 GB + 16 GB RAM；当前已安装 `torch==2.5.1+cu121`，CUDA True，CPU/CUDA Tensor 和官方网络前后向、CUDA 合成 PPO 更新/恢复均实测通过。不要重新安装。官方 wheel 已校验 SHA256，缓存 `work/torch-2.5.1+cu121-cp310-cp310-win_amd64.whl`；完整依赖 `requirements-lock-cuda.txt`。
9. 输出有配置、seed、Git SHA、输入 hash、JSONL、TensorBoard、checkpoint；目录全新，total-steps 是累计整批目标；Smoke 上限 256。
10. ROM、状态、模型、大日志均忽略。原始上游 Git 保存在 `local/upstream.git`，含状态历史，禁止推送。上游 `.state` 只作本地参考，不作为公开素材。
11. 核心验收未完成，不创建 Public 仓库。待真实验收通过、报告更新且认证正常，用户已条件性授权发布 `pokemon-red-rl-reproduction`。保留 MIT/NOTICE，先审查文件与历史。
