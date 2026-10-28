# 进度记录

## 2026-10-28：Qwen3-4B 预训练规划

- 新增 docs/pretraining-plan.md：目标、模块接口、数据、预算、训练配方、阶段验收及集群执行流程。
- 核实官方架构与预训练规模，区分同架构研究训练和官方能力复现；以累计 100B token 为待资源核实的建议基线。
- 当前目录不是 Git 仓库，没有 remote；本轮未初始化 Git、提交、联网服务器或运行训练。实施下一步为项目定位与资源盘点。

## 2026-10-28：Guqq2 实际资源与单卡计划

- 按用户确认 remote clone 至 /home/xmz/pretraining，使用短时 Slurm nvidia-smi 验证 RTX 5090 32607 MiB；只做资源盘点，未训练或安装环境。
- 新增 docs/guqq2-plan.md，更新总计划：CPU optimizer offload、1024 序列、microbatch 1、32K token/update、100M 试训再评估 1B；100B/多卡方案降为扩容参考。
- 根盘仅余 42G，完整恢复点约 50–80GB，必须先落实持久存储；未操作未知未挂载磁盘。MemAvailable 约 234GiB，内存容量可试 offload，CPU load 影响留待 profiling。
- 记录代理恢复、用户授权精确终止旧会话、Slurm 与存储信息，以及 Windows SSH/CRLF 和 MemAvailable 经验。
