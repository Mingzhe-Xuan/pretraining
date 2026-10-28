# 经验记录

## 2026-10-28：预训练工程规划

- Qwen3-4B 的 head_dim 由配置显式给出，不能从 hidden_size / num_attention_heads 推断；共享 embedding 与 QK-Norm 也必须保留。
- 151936 词表超过 uint16 范围，token shard 使用 uint32 等足够宽的类型。
- 断点续训需要保存数据采样与 packing 状态；仅恢复模型和优化器不能保证数据序列连续。
- 空目录不等于已初始化的 Git 仓库。服务器 pull/clone 前必须先确定 remote 和项目路径。
- Windows 沙箱内的默认用户可能不读取真实用户的 SSH 配置；访问获授权服务器时需要明确配置路径及允许读取该配置的执行上下文。
- PowerShell 将多行文本通过原生 stdin 发送给 Linux 时可能带 CRLF；应在远端执行前去除行尾 CR，否则 curl URL 等参数可能含隐藏字符。Windows 原生命令参数传递还可能吞掉内层双引号，复杂远端脚本优先通过 stdin 传递并处理换行。
- Slurm IDLE 不等于无系统负载；FreeMem 不等于 MemAvailable。判断 offload 内存容量应使用后者并区分文件缓存；load average 也不能直接当 CPU 利用率。
