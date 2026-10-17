# 下一位 Agent 的接管说明

1. 项目 `C:\Users\lihao\Desktop\pokemon-red-rl-reproduction`；Python `.venv\Scripts\python.exe`。先读 `docs/reports/progress.md` 和 `phase1_validation.md`。不必重装环境。
2. 用户明确没有 ROM，真实游戏验收阻塞。不要下载 ROM，不要拿合成 checkpoint 宣称真实训练成功。
3. 合法英文 ROM 放 `local/PokemonRed.gb`，兼容状态放 `local/has_pokedex_nballs.state`。核对 ROM SHA1，记录状态 SHA256。可用 `scripts/capture_state.py` 手动产生状态，真实入口仍需验证。
4. 先 doctor、pytest、check_game 128 步，再按 README 训练到 32、恢复到 64、最后 watch 验证画面和退出。完成即停，不在 Codex 中长训。
5. 官方分支/固定 SHA 见 `UPSTREAM.json`。局部修复 dtype、close、headless speed、worker 参数/清理、有限数值检查和打包依赖。PPO 损失、模型、GAE、AdamW 复用官方。
6. `smoke.yaml` 联调；`paper_baseline.yaml` 保留论文参数。代码与论文的奖励/episode 差异见 `docs/baseline_comparison.md`，不静默改变算法。
7. `artifacts/synthetic-cpu/checkpoint.pt` 仅合成测试。真实预期 `artifacts/smoke/checkpoint_000000032.pt` 尚不存在。恢复使用原输出的 `config.yaml`，恢复模型/AdamW/RNG/调度 update，不恢复运行中的游戏。
8. RTX 4060 8 GB + 16 GB RAM；当前 CPU torch。必要时独立终端装 `torch==2.5.1+cu121` 并复测。不能因 nvidia-smi 正常就报告 CUDA 运算通过。
9. 输出有配置、seed、Git SHA、输入 hash、JSONL、TensorBoard、checkpoint；目录全新，total-steps 是累计整批目标；Smoke 上限 256。
10. ROM、状态、模型、大日志均忽略。原始上游 Git 保存在 `local/upstream.git`，含状态历史，禁止推送。上游 `.state` 只作本地参考，不作为公开素材。
11. 核心验收未完成，不创建 Public 仓库。待真实验收通过、报告更新且认证正常，用户已条件性授权发布 `pokemon-red-rl-reproduction`。保留 MIT/NOTICE，先审查文件与历史。
