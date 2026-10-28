# Qwen3-4B 从零预训练计划

制定日期：2026-10-28。状态：规划完成，尚未开始实现和训练。

## 1. 目标与推荐路线

在本项目随机初始化并训练一个采用 Qwen3-4B 架构的中英双语 Base 模型。复用官方 tokenizer 和结构配置，不加载官方模型权重。默认交付 4K 上下文的自回归语言模型，之后按预算扩展到 32K；指令微调、思维链训练和 RL 属于后续独立任务。

建议采用“工程闭环 → 1B token 试训 → 累计 100B token 主训练 → 评估后扩展到 300B–1T”的路线。100B 是首个研究规模目标，不是能力达标承诺；最终 token 预算由实际可用卡时、吞吐和数据质量决定。

官方披露 Qwen3 系列使用约 36T token，且包含分阶段预训练和后训练。因此本计划实现的是相同架构的自主预训练，不承诺复现官方能力。参考：[Qwen3 官方说明](https://qwenlm.github.io/blog/qwen3/)。

## 2. 已知条件与实施前置条件

- 当前本地只有 AGENTS.md 和空文档目录；没有可复用训练实现，当前目录不是 Git 仓库，也没有已确认 remote。
- 可使用 Guqq2，但 GPU 型号、卡数、节点互联、Slurm 分区、单作业时限、长期配额、存储和已有数据均未核实。
- 本轮仅制定计划，没有远程连接、创建环境、下载数据或提交作业。
- 实施第一步需确定 Git remote、服务器项目路径，以及任务专属环境/数据/输出根目录；不猜测或覆盖已有项目。
- 同时确认总卡时/费用上限、中文优先程度、目标用途，以及是否需要 32K。默认以中英通用研究 Base 模型、100B token 为方案基线。

## 3. 架构与初始化

以 [Qwen3-4B-Base 官方 config](https://huggingface.co/Qwen/Qwen3-4B-Base/blob/main/config.json) 为依据，实施时固定模型仓库 revision 和配置哈希：

| 项目 | 值 |
| --- | --- |
| 层数 / hidden size / FFN intermediate | 36 / 2560 / 9728 |
| Q heads / KV heads / head dimension | 32 / 8 / 128 |
| 词表大小 | 151936 |
| embedding 与 LM head | 共享权重 |
| RoPE theta / 架构最大位置 | 1000000 / 32768 |
| RMSNorm epsilon / 初始化标准差 | 1e-6 / 0.02 |
| 激活 / attention dropout / attention bias | SiLU / 0 / false |

采用 Transformers 原生 Qwen3 实现作为正确性基线，保留 QK-Norm、GQA 和完整架构语义。注意 Q 投影维度为 32×128=4096，不能按 hidden size / heads 推断 head dimension。只获取 config/tokenizer 文件，通过 `AutoModelForCausalLM.from_config` 等等价路径初始化；训练入口禁止传入预训练权重作为初始化源。记录种子、实际参数量、初始化配置和 Git commit。

初期输入长度 4096，不修改其架构最大位置定义。训练时关闭 KV cache，使用 BF16（硬件支持时）、activation checkpointing 和经过验证的高效 attention。大词表采用分块/融合交叉熵降低 logits 显存；先验证其数值与梯度等价性，再启用。

## 4. 工程方案与模块边界

首选 PyTorch + Transformers Qwen3 + FSDP2 + Distributed Checkpoint。理由是模型实现与导出路径直接，适合先在单节点验证 4B 训练。FSDP2 的状态分片和保存接口参考 [PyTorch 官方文档](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)。选定兼容版本后锁定依赖，不以动态 latest 作为可复现配置。

若多节点通信成为实测瓶颈，再评估 Megatron 类张量/流水线并行后端，并先完成权重转换与数值对齐；首版不同时维护两套训练后端。小模型可用 DDP 建立参照。

以下目录为待实施结构，目前不创建空实现。每个模块落地时包含 README.md，说明职责、输入输出与运行方式：

```text
configs/                  # 模型、数据混合、训练阶段、Slurm 资源配置
src/pretraining/
  model/                  # 配置校验、随机初始化、HF 导出及对齐
  data/                   # 清洗、去重、tokenize、分片、packing、可恢复采样
  train/                  # 分布式训练、优化器、调度器、有效 token 计数
  checkpoint/             # 分布式保存、恢复、原子完成标记及保留策略
  evaluation/             # 验证集 loss、固定续写、下游基准与报告
scripts/
  slurm/                  # CPU 预处理、GPU 测试/训练/评估作业入口
  tools/                  # manifest 校验、存储估算、导出工具
tests/                    # 单元测试、分布式恢复与端到端验收
docs/agents/              # 状态、测试、环境、远程连接、进度与经验
```

核心接口：数据管线输出版本化 manifest、token shard 和文档边界；loader 输出 input_ids、labels 与有效 loss mask。训练器消费这些 batch，按非 padding 的有效预测 token 计数；checkpoint 同时保存模型/优化器、scheduler、RNG、数据版本、采样与 packing 游标、全局步数和 token 计数。评估消费不可变 checkpoint 和固定评估配置，输出机器可读指标及摘要。

生成物使用任务专属目录并加入 .gitignore；数据、权重、日志、checkpoint 和批量结果不进入 Git。普通 Git 单文件必须小于 10 MiB。导出模型由数据通道保存/传输。

## 5. 数据方案

首轮按实际 tokenizer 产生的有效 token 配比：英文通用/教育 45%，中文通用/教育 35%，代码 10%，数学/STEM 10%。这是试验起点，不是官方配方；类别需互斥标记，避免教育数据与 STEM 重复计算。

- 英文候选：[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)。
- 中文候选：[OpenCSG Chinese FineWeb-Edu](https://huggingface.co/datasets/opencsg/chinese-fineweb-edu)。固定具体版本；不同同名系列可能面向 SFT，不能自动替换为预训练语料。
- 代码与数学：实施时选择具有明确来源、版本与可用许可的公开语料，记录仓库级/文档级来源。尚未选定，不把候选数据量算作可用供给。
- 数据登记包括来源、revision、许可/来源限制、语言、质量规则、文档数、去重前后 token 数、下载与处理哈希。供应商 token 数不能替代本 tokenizer 的实测值。

管线顺序：来源登记 → 文本抽取与规范化 → 语言/质量/敏感信息过滤 → 精确和近似去重 → 与评测集去污染 → 按文档组划分 train/validation → tokenize → packing/分片 → 校验 manifest。跨来源去重也需覆盖；同源近重复文档不得分到不同 split。

固定每种语言/领域的验证集，初始总量约 10M–50M token，规模依供给调整。训练混合与验证集版本分别冻结。每个来源记录唯一 token 量、实际暴露量与重复轮数；数据不足时优先减预算或调整比例，不默认无限重复。

使用 uint32 token 存储：词表超过 65535，uint16 会溢出。每个文档显式 EOS；初版采用连续文本流 packing，允许跨文档因果上下文，边界仍保留供统计与后续实验；不悄然引入与其不兼容的 block attention mask。无聊天模板包装。label shift 只执行一次，并校验 padding、EOS 和 loss mask。

数据处理走 Slurm CPU 作业，输出可重入 shard；训练前分片落盘，默认离线运行。先处理小样本测吞吐与压缩率，再扩大到完整训练预算，避免无预算地下载整库。

## 6. 计算与存储预算

采用粗略训练计算量 F≈6ND，N 暂按 4×10^9；未计入长上下文 attention、重计算和评估等开销。下表仅展示“8 卡集群总有效吞吐相当于每卡持续 100 TFLOP/s”的假设，不代表 Guqq2 性能：

| 累计训练 token | 粗略 FLOPs | 假设下纯训练时间 | 用途 |
| --- | --- | --- | --- |
| 1B | 2.4×10^19 | 0.35 天 | 试训与工程验证 |
| 100B | 2.4×10^21 | 34.7 天 | 推荐首个主训练目标 |
| 300B | 7.2×10^21 | 104.2 天 | 数据/算力允许后扩展 |
| 1T | 2.4×10^22 | 347.2 天 | 更充分预训练，需较大资源投入 |

实际以端到端有效 token/s 为准：天数 = 目标有效 token / 集群有效 token/s / 86400。卡时 = 天数×24×卡数；费用 = 卡时×单价。报告同时包含训练吞吐、含保存/评估的作业吞吐和排队时间；不使用硬件峰值直接承诺工期。

8×80GB GPU 可作为首轮 profiling 的参考拓扑，不是已确认资源或无条件显存保证。少卡可尝试 FSDP 与梯度累积，但须先测 OOM、吞吐和总卡时。4B 的混合精度 AdamW 模型状态可粗估 12–16 bytes/parameter，即全局约 48–64GB，另加激活、通信 buffer、临时 logits 和碎片；按具体实现实测。

100B 唯一 token 的 uint32 裸 token 文件约 400GB；300B 约 1.2TB，1T 约 4TB，另加索引和原文。重复训练暴露不必重复存 token。100B 方案初步预留约 2–5TB 总空间，但必须用小样本压缩率和实际原文规模复核。每个完整恢复 checkpoint 预算约 50–80GB，保留最近 2–3 个及少量里程碑，包含暂存/导出空间；测 I/O 后再确定保存周期。

## 7. 训练初始配方

以下为待试验参数，不冒充官方超参数：

- AdamW，betas=(0.9, 0.95)，eps=1e-8，weight decay=0.1；bias/norm 等参数分组不衰减，验证共享 embedding 仅加入一次。
- 4K 上下文；先以约 0.5M token/update 建立基线，若吞吐、梯度稳定，再尝试约 1M。全局 batch = microbatch×数据并行度×梯度累积×序列长度；变长/mask 时另外计实际有效 token。
- 峰值 LR 在 1e-4、2e-4、3e-4 间做同数据顺序、同 token 预算的短跑；默认候选 2e-4。先以约 200M token warmup 为起点，低预算调试需单独缩短并标注，避免整个试验仍在 warmup。
- 梯度裁剪 1.0，监控梯度范数、非有限值、loss spike、各域验证 loss、有效 token/s、显存峰值、通信和数据等待。
- 主跑采用按 token 定义的 warmup–stable–decay，100B 预算示例为 0.2B warmup、89.8B stable、10B decay 至峰值的 10%。保留进入 decay 前 checkpoint，追加预算时从该分支制定新日程；若从衰减末端继续则显式设计 rewarm，不能无记录地重置 LR。
- 单步优化以有效预测 token 归一化；分布式/梯度累积不允许按不等长 microbatch 的 loss 直接等权平均。
- profiling 后以约 30–60 分钟为 checkpoint 周期起点，同时服从 Slurm 时限；只有完整保存并写入完成标记的版本才可恢复。

## 8. 阶段、产物与验收门槛

| 阶段 | 工作及产物 | 进入下一阶段的条件 |
| --- | --- | --- |
| P0：项目与资源基线（约 0.5–1 工作日） | 初始化 Git/确认 remote；建立模块 README；登记 Slurm/GPU/配额/网络/磁盘；确认总预算 | 资源表、数据源登记、任务目录和预算可执行 |
| P1：环境与数据小样（约 2–4 工作日） | 创建专属 venv 并锁依赖；固定 tokenizer/config；完成小样本数据管线及 manifest | 文档/来源去重、split 隔离、token 范围和 packing 测试通过 |
| P2：正确性闭环（约 2–4 工作日） | 小架构沿用 Qwen3 实现测试；4B 短跑；保存/恢复/导出 | 小数据过拟合成功；无 NaN；中断恢复与连续跑在约定数值容差内一致；HF 导出 logits 对齐 |
| P3：4B profiling 与试训（约 2–4 工作日 + GPU 时间） | 比较 microbatch/重计算/attention；短 LR 对比；累计约 1B token 主候选试训 | 得到显存和端到端吞吐；固定验证 loss 改善；数据与恢复检查通过，100B 预算估算可接受 |
| P4：主训练（按实测吞吐） | 延续已冻结的训练分支，累计到 100B；定期保存和评估 | 无未解释的数值异常；数据/预算符合版本记录；完成 Base checkpoint、学习曲线和消融/配置报告 |
| P5：可选扩展 | 300B–1T；或独立进行 4K→8K→32K 长上下文阶段 | 有额外预算；长上下文独立 profiling，短上下文回归与长文评估通过 |
| P6：交付（约 1–2 工作日 + 评估时间） | HF 模型/tokenizer、完整恢复点、模型卡、数据与训练配置、指标报告 | 可在干净环境加载；产物哈希与训练 provenance 完整 |

时间是工程工作量估计，不含队列等待和大规模下载/预处理。P2 的缩小模型仅用于工程验证，不能替代完整 4B 的性能验证。P3 的超参数对比分支单独计卡时；只有可继续且数据/日程兼容的候选分支计入主跑累计 token。

P4 每 100M–500M token 做轻量分域验证，每 5B–10B 做完整固定评估，按开销调节。发现非有限梯度、持续异常 loss、数据损坏或错误恢复时自动停止并保留诊断；不能靠跳过任意坏 batch 掩盖问题。每个阶段开始前在 test.md 写测试计划，每次提交前记录实际检查结果。

## 9. 测试与评估设计

- 数据单测：确定性 tokenize、uint32 往返、EOS/label shift、pack 边界、有效 token 统计、源间去重与 split 无泄漏、manifest 损坏检测。
- 模型单测：官方配置字段/参数量、QK-Norm/GQA 结构、共享 embedding、随机初始化 provenance、因果 mask；高效 attention 与参照实现小输入前向/梯度对齐。
- 分布式集成：单卡与多卡在相同有效 batch 下更新对齐；连续跑与保存后重启比较后续 batch、LR、token 计数、loss 与参数。固定软件/拓扑先达到可恢复；改变 world size 作为独立测试，不默认承诺精确续训。
- 端到端：Slurm 小数据过拟合、完整 4B 短跑、故障中断/恢复、导出后标准 Transformers 加载。小样本过拟合阈值和数值容差在实施测试计划中预先固定。
- 预训练主指标：各域 held-out cross-entropy/perplexity，按有效 token 加权，监测训练/验证差距；同 tokenizer 才直接比较 PPL。
- 下游固定基准：英文 HellaSwag/ARC、知识 MMLU、中英 C-Eval/CMMLU；数学 GSM8K、代码 HumanEval 等作为阶段性诊断，不作为初期硬门槛。固定版本、prompt、few-shot、解码和评测样本；检查污染。执行生成代码时使用隔离、受限环境。
- 随机初始化、自己的里程碑 checkpoint 与官方 Base 可作参考对照；官方对照推理成本单列。Base 模型使用续写/标准 few-shot 协议，聊天表现不作为预训练完成标准。

## 10. 服务器与可复现性流程

严格遵守 AGENTS.md：每次远程连接前在 docs/agents/gpu.md 记录用途与权限；源码先在本地修改、检查、提交并 push，再在服务器 pull。当前无 remote，故需先完成 P0 的项目定位。

需要联网时先建立并维持唯一专用代理 SSH 会话，普通工作连接使用 `ClearAllForwardings=yes`；通过显式 SOCKS 代理完成端到端健康检查，再进行联网 pull/下载/安装。已有仓库优先 pull，不存在则从已确认 remote clone。代理故障仅按规范允许的诊断例外处理。计算节点网络独立验证，正式训练默认只读已准备好的本地数据。收尾关闭任务代理并确认端口释放。

训练、评估、批处理、编译与明显负载测试全部通过 Slurm；登录节点仅做允许的轻量管理。环境用 Python 创建专属 venv，记录 Python/PyTorch/Transformers/CUDA/NCCL、驱动/GPU 信息、安装命令、锁文件和框架 commit；依赖编译转入 Slurm。

每个运行保存配置哈希、Git commit、环境快照、Slurm job ID、数据 manifest、随机种子、拓扑与指标。state.md 维护快照并追加阶段历史，update.md 按提交粒度登记，test.md 记录计划/结果；连续失败三次先查 lessons.md 并补充经验。禁止覆盖其他任务数据、环境和实验结果。

## 11. 最先执行的工作

1. 确认/建立项目 Git remote，明确 Guqq2 上的任务路径和持久化存储。
2. 按连接规范盘点卡型、可分配卡数、互联、配额、时限和磁盘，形成资源表。
3. 依据预算选定首阶段 token 上限；固定架构、tokenizer 和候选数据版本。
4. 实现可测试的小数据闭环，并用 Slurm 跑完整 4B 性能测试。
5. 用测得的 token/s 和数据供给重算 100B 主跑预算，再冻结主训练方案。
