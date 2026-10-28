# Guqq2 系统盘占用排查

日期：2026-10-28，约 16:58–17:01；Slurm 只读作业 452、453。以下 GB/TB 为十进制，du 为文件实际分配空间，大文件列表为逻辑大小。未删除、迁移或清理任何数据。

## 全盘与可见性

- 根盘 /dev/nvme0n1p2：df 已用 1,822,912,028,672 bytes（约 1.82TB），可用 44,395,700,224 bytes（约 44.4GB），98%。
- 当前 xmz 账号无免密 sudo，`du -x` 仅统计到约 142.4GB，出现 322 条权限错误。
- 因此约 1.68TB 的使用量尚未归因。不可见区域包括其他用户 home、/root、/var/lib/docker 与 /var/lib/containerd；不能断言它们各自占用多少，也不能将可读目录合计视为整盘总量。
- 当前用户 `lsof +L1` 只见小型 memfd（内存文件），未见可以解释根盘差额的大型已删除文件；此结果不覆盖其他用户或 root 的进程。

## 可读目录

| 路径 | 分配空间 | 说明 |
| --- | --- | --- |
| /home/xmz | 55.5GB | 当前账号全部数据 |
| /opt | 40.1GB | 主要是 ABACUS 和 Intel 工具 |
| /usr | 22.0GB | 系统库/软件 |
| /var | 11.8GB | 含约 3.1GB 日志 |
| /package | 5.8GB | 软件目录 |
| /tmp | 4.7GB | 主要可见 pip 解包目录及其他任务临时文件 |

以上是同层目录，可用于可见区域比较；以下细目包含于上表中，不可重复相加。

| 路径 | 分配空间 |
| --- | --- |
| /home/xmz/generate_md | 43.3GB |
| /home/xmz/generate_md/results | 19.3GB |
| /home/xmz/generate_md/.venv-smoacs-blackwell | 8.2GB |
| /home/xmz/generate_md/.venv-smoacs-postprocess | 7.6GB |
| /home/xmz/generate_md/.venv-smoacs-paper | 7.1GB |
| /home/xmz/.cache/pip | 12.0GB |
| /opt/abacus | 21.8GB |
| /opt/intel | 12.9GB |
| /var/log/journal | 3.1GB |

新 pretraining 仓库仅约 0.45MB，没有训练数据、模型或 checkpoint。

在 xmz 目录内，超过 1GiB 的普通文件有两份约 1.25GB 的 libtorch_cuda.so，分别位于 postprocess/blackwell 环境；另有 generate_md/results/smoacs/assets-199/reproducibility/section_2_4/templates_masks/initial_mp_nsites2_30__mask_dict_super_common_level5.pkl，约 1.11GB。它们已计入目录总量；不能只凭文件名判定可删除。

## 如何继续定位

若需查清剩余约 1.68TB，需由有权限的管理员通过 Slurm 对根盘执行只读 `du -x`，按同层目录展开其他用户 home、/root 与容器存储；若完整 du 与 df 仍有显著差异，再检查所有用户的已删除仍打开文件、挂载覆盖数据和文件系统元数据。当前无权限直接完成这些统计。

本次不将其他项目环境、实验结果、缓存或系统软件视作可清理内容。清理属于独立任务，应先明确归属、保留要求及可重新生成性。
