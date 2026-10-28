# 经验记录

## 2026-10-28：预训练工程规划

- Qwen3-4B 的 head_dim 由配置显式给出，不能从 hidden_size / num_attention_heads 推断；共享 embedding 与 QK-Norm 也必须保留。
- 151936 词表超过 uint16 范围，token shard 使用 uint32 等足够宽的类型。
- 断点续训需要保存数据采样与 packing 状态；仅恢复模型和优化器不能保证数据序列连续。
- 空目录不等于已初始化的 Git 仓库。服务器 pull/clone 前必须先确定 remote 和项目路径。
