# Guqq2 实际配置下的 Qwen3-4B 计划

2026-10-28 核实。此文是当前执行基线；[总计划](pretraining-plan.md) 中的大规模路线保留作扩容参考。

## 实测资源

| 项目 | 实际结果 |
| --- | --- |
| 项目仓库 | /home/xmz/pretraining，origin=https://github.com/Mingzhe-Xuan/pretraining.git |
| Slurm | compute 分区，node42，一个节点、一张 GPU |
| GPU | NVIDIA GeForce RTX 5090，32607 MiB 显存，查询时空闲 31922 MiB |
| 驱动 / 功率上限 | 580.178.04 / 575W |
| 配置 CPU / RAM | 48 CPU / 257787 MiB（约 252GiB） |
| 实时负载快照 | CPU load 约 40，Slurm 显示 IDLE；free -h 显示约 234GiB available、201GiB buff/cache |
| 已挂载持久存储 | 根/home 共用 1.8T 文件系统，使用率 98%，仅余 42G |
| 其他块设备 | sda 7.3T、sdb1 3.6T NTFS 未挂载；归属和用途未核实 |
| 调度限制 | 分区 MaxTime=UNLIMITED；无 Slurm 用户作业，长期配额未确认 |
| 资源诊断 | 已通过 Slurm 运行 nvidia-smi；没有训练吞吐实测 |

Slurm accounting 未启用支持 sacctmgr 的插件，因此不能由查询结果断言无限卡时。CPU load 约 40，不能以 Slurm idle 断言独占整机；load 包含可运行和不可中断任务，不等于 CPU 利用率。补充 free 检查显示约 234GiB 可用，最初的 34GiB free 低值主要来自缓存，不能据此认定主机内存不足。

## 结论与前置工作

可以把这台机器作为从零训练完整 Qwen3-4B 的工程验证平台，但现状不适合直接启动 100B token 主跑。主要限制是单卡吞吐、32GB 显存、持久存储不足；CPU offload 与当前较高 load 的相互影响需 profiling 确认。

完整混合精度 AdamW 状态约 48–64GB，尚未包含激活；单张 32GB 卡无法按普通 AdamW 全驻留 GPU。完整恢复点约 50–80GB，当前 42G 甚至无法稳妥保存一个，更无法保留旧点并原子写入新点。

先落实至少约 200GB 可用持久空间供短跑的环境、数据、两个恢复点及临时文件；建议争取 300GB 以上余量。不得清理未知任务数据，也不把 /dev/shm 当持久空间。未挂载大盘只是资源线索，不是可直接使用的授权目录；需确认归属、已有内容和挂载方案。

CPU/RAM 方面，以 64–96GiB 主机内存和 8–16 CPU 作为 offload 试验预算，实际以完整 step/RSS 测量修正；约 234GiB available 支持从内存容量角度试验。仍需遵守调度、检查其他负载影响；不能仅提高 Slurm 请求就推断拥有独占 CPU。

## 当前建议配置

| 项目 | 起始选择 |
| --- | --- |
| 模型 | 完整官方 Qwen3-4B 架构、随机初始化；复用官方 tokenizer |
| 优化器 | AdamW，先评估 DeepSpeed ZeRO-2 + CPU optimizer offload / CPUAdam |
| 精度 | BF16；验证实际 GPU kernel 支持后启用 |
| 初始长度 | 1024；稳定后测 2048，再评估 4096 |
| microbatch | 1 |
| 梯度累积 | 32（1024 长度时为 32768 token/update） |
| 2048 长度对比 | 累积改为 16，保持相同名义 token/update |
| 显存控制 | 每层 activation checkpointing，关闭 KV cache，分块/融合交叉熵 |
| Attention | 先以兼容的 PyTorch SDPA 建立参照，再按数值和 kernel 测试选择高效实现 |
| LR 起点 | 1e-4，betas=(0.9,0.95)，weight decay=0.1，clip=1.0 |
| 100M 日程 | 约 2M token warmup、88M stable、10M decay；保留 decay 前恢复点 |
| 评估 | 每约 5M token 固定小验证集，各语言/领域分别报告 |
| checkpoint | 初始每 30–60 分钟，完整保存后决定周期，最近 2 个+少量里程碑 |
| 初次 Slurm profiling | 1 GPU、8 CPU、96G RAM、30 分钟；仅在资源落实后提交 |

CPU offload 的机制和 CPUAdam 参考 [DeepSpeed 官方教程](https://github.com/deepspeedai/DeepSpeed/blob/master/docs/_tutorials/zero-offload.md)。单卡 ZeRO 不提供跨卡分片收益，主要收益来自 offload。若仍 OOM，先定位激活/logits/状态来源，再评估 ZeRO-3 参数 offload；额外传输可能明显降低吞吐，不默认使用 NVMe offload。

环境选用支持 Blackwell sm_120 的 PyTorch/CUDA 组合，先从 CUDA 12.8+ 的官方二进制候选验证，不把旧 CUDA wheel 套用到 5090。参考 [PyTorch 官方维护者说明](https://discuss.pytorch.org/t/support-for-nvidia-rtx-5090-cuda-sm-120/223683/2)。固定通过实际测试的版本；BF16、SDPA、CPUAdam、可选融合 loss/attention 分别验证前向、反向、优化器 step 与恢复。所需 CPUAdam/其他扩展编译必须通过 Slurm，不能在登录节点做。

32K token/update 是单卡工程试跑起点，低于长期大规模配方；先验证吞吐与收敛，不宣称最优。100M token 约 3052 个更新，实际按有效 token 和末批处理计数。累计 1B 也仅属低预算研究模型，不代表充分预训练的通用能力。

## 时间预算：先测量，再定规模

应测量的是**完整 4B、含 CPU offload、数据读取和 checkpoint/验证开销的端到端有效 token/s**，不能用 RTX 5090 标称峰值或推理速度推算训练时间。

下面是假设吞吐的敏感性分析，均非本机实测；时间不包含排队和中断：

| 端到端有效 token/s | 100M token | 1B token | 100B token |
| --- | --- | --- | --- |
| 100 | 11.57 天 | 115.74 天 | 31.71 年 |
| 500 | 2.31 天 | 23.15 天 | 6.34 年 |
| 1000 | 1.16 天 | 11.57 天 | 3.17 年 |

公式：天数 = token / token每秒 / 86400；年按 365 天计算。先测 200–500 个完整训练 step，剔除启动/预热并单独报告其成本；再做一次完整保存、重启、恢复和验证，得出包含固定开销的估计。实际 CPU 竞争可能造成大幅波动。

若希望 30 天内完成 100B，需要持续约 38580 token/s 的端到端吞吐；当前没有证据支持本配置达到该目标，不能承诺。若项目目标必须为充分预训练的 4B，后续应扩容多卡和持久存储，并重新做预算。

## 数据与阶段安排

沿用总计划的 tokenizer、数据去重/去污染、manifest 和可恢复采样规范。100M 试跑的数据只需约 0.4GB uint32 token 主体，1B 约 4GB，原文/索引/缓存另计；当前磁盘瓶颈主要是环境和完整优化器 checkpoint。先准备数百万 token 的工程小样，再按预算制作约 100M 的去重训练集，绝不先下载整库。

初步混合仍为英文 45%、中文 35%、代码 10%、数学/STEM 10%，来源不足时显式重定比例并冻结版本；小预算不适合同时加入大量数据配方消融。验证集先用约 1M–5M token 的固定集合，训练中抽取固定小子集，完整评估在里程碑做。验证 token 不计训练预算。

1. **资源落实**：确认任务数据根目录、至少约 200GB 持久空间、CPU/RAM 可用性与长期卡时。当前停在此处，不启动训练。
2. **环境与工程闭环**：建立专属 venv，验证 5090 kernels；小架构验证 shift/mask、过拟合、保存与恢复；随后对完整 4B 做短跑。
3. **完整 4B profiling**：比较 1024/2048，在相同 token/update 下记录显存峰值、RSS、CPU offload 开销、token/s、保存大小/耗时；至少完整恢复一次。
4. **100M token 试训**：随机初始化训练主候选，分域验证 loss、固定续写和恢复验收。小样工程数据不得泄漏到固定验证集。
5. **是否累计至 1B**：仅在吞吐/资源预算可接受且验证趋势正常时延续；扩展日程从 decay 前点分支或显式 rewarm，不悄然重置 scheduler。100B 与 32K 留作独立扩容决策。

截至本次盘点，只 clone 了源码并做轻量资源诊断，没有创建环境、下载模型/语料、修改服务器源码或启动训练。
