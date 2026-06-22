# 1 vime MoE R3对照实验

本文记录用vime框架复现MoE模型R3（Rollout Router Replay）对照实验的步骤：镜像拉取、容器启动、数据与权重准备、训练脚本说明，以及官方CI测试环境对照。文中路径以 `/data/nfs_87/kaiyuan/` 为示例前缀，请读者按自己的NFS或本地挂载地址替换。

| 步骤 | 内容 |
|:-----|:-----|
| 1.2 | Docker镜像拉取与容器启动 |
| 1.3 | 数据集、HF权重、torch_dist转换 |
| 1.4 | 训练脚本与关键参数（GSPO + R3开关） |
| 1.5 | 日志监控与结果指标 |
| 1.6 | 官方测试 `test_qwen3_30B_A3B_r3.py` 对照 |
| 附录 | vime代码更新（可选） |
| 附录 | 执行脚本样例 |

Author: kaiyuan

Email: kaiyuanxie@yeah.net

## 1.1 实验概览

**目标**：在相同硬件与超参下，对比开启与关闭 `--use-rollout-routing-replay` 时，Qwen3-30B-A3B MoE GSPO训练的稳定性、训推一致性与reward/AIME走势。

**硬件**：单机8×GPU（Hopper或同等级算力平台）。

**软件栈**：vime（Megatron训练 + vLLM推理 + Ray调度），Colocate模式（8卡同时承担训练与rollout）。

**算法**：GSPO（`--advantage-estimator gspo`）+ TIS；R3组额外开启 `--use-rollout-routing-replay`。

**数据**：
- 训练：DAPO-Math-17k（DeepScaler reward）
- 评测：AIME-2024（独立测试集，训前 + 每20 step，pass@16）

**默认步数**：每组200 step，不保存checkpoint；顺序脚本先跑R3再跑no-R3。

vime环境与用法亦可参考官方快速上手：[vime Quick Start（中文）](https://github.com/vllm-project/vime/blob/main/docs/zh/get_started/quick_start.md)。

## 1.2 运行环境与镜像

**镜像拉取**

```
docker pull inferactinc/public:vime-vllm-latest
```

镜像已预装Megatron-LM、vLLM、Ray及vime，一般无需额外编译依赖。

**容器启动**

将宿主机存储挂载进容器，代码、数据、日志建议放在同一挂载根目录下。以下 `-v` 左侧为宿主机路径，右侧为容器内路径，按实际环境修改：

```
docker run -itd --name vime \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /your/host/nfs:/data/nfs_87 \
  inferactinc/public:vime-vllm-latest \
  bash
```

进入容器：

```
docker exec -it vime bash
```

下文示例根路径记为 `${WORK_ROOT}=/data/nfs_87/kaiyuan`，对应目录约定：

| 用途 | 示例路径（请替换前缀） |
|:-----|:----------------------|
| vime代码 | `${WORK_ROOT}/RL/vime` |
| 训练脚本 | `${WORK_ROOT}/RL/run_scripts` |
| 数据集 | `${WORK_ROOT}/datasets` |
| torch_dist权重 | `${WORK_ROOT}/models` |
| 共享HF权重 | `/data/nfs_87/models`（或自行指定） |
| 训练日志 | `${WORK_ROOT}/RL/logs` |

## 1.3 数据与模型准备

**数据集下载**

训练集 `zhuzilin/dapo-math-17k`、测试集 `zhuzilin/aime-2024` 可用 `run_scripts/download_datasets.sh` 一键下载（需容器内可访问HuggingFace，按需配置代理）。

期望生成：
- `${WORK_ROOT}/datasets/dapo-math-17k/dapo-math-17k.jsonl`
- `${WORK_ROOT}/datasets/aime-2024/aime-2024.jsonl`

**代理注意**：非交互式 `docker exec bash -c` 往往不会自动加载shell代理；下载脚本内应显式 `export http_proxy=...`；数据集建议走官方Hub，`unset HF_ENDPOINT`（部分镜像站会导致dataset API失败）。

**HF权重与torch_dist转换**

HF checkpoint（rollout用），示例：

```
/data/nfs_87/models/Qwen3-30B-A3B
```

Megatron `torch_dist`（训练 `--ref-load` 用，需8卡空闲转换），示例：

```
${WORK_ROOT}/models/Qwen3-30B-A3B_torch_dist
```

转换前确认GPU空闲，必要时停止占用显存的vLLM或残留训练进程。转换脚本见 `${WORK_ROOT}/RL/run_scripts/convert_qwen3_30b.sh`，日志默认 `${WORK_ROOT}/RL/logs/convert.log`。

容器内手动转换参考：

```
cd ${WORK_ROOT}/RL/vime
source scripts/models/qwen3-30B-A3B.sh
PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
  tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint /data/nfs_87/models/Qwen3-30B-A3B \
  --save ${WORK_ROOT}/models/Qwen3-30B-A3B_torch_dist
```

## 1.4 训练脚本与关键参数

对照实验涉及两个脚本（放在 `${WORK_ROOT}/RL/run_scripts/`）：

| 脚本 | 作用 |
|:-----|:-----|
| `run-kaiyuan-bench-r3-qwen3-30B-A3B.sh` | 单组训练；环境变量 `MODE=r3` 或 `MODE=no-r3` |
| `run-r3-compare-sequential.sh` | 顺序启动：R3 200 step → no-R3 200 step |

**R3开关**（唯一实验变量）：

```
# R3组
--use-rollout-routing-replay

# no-R3组：不加上述flag
```

**核心训练参数摘要**：

| 类别 | 参数 |
|:-----|:-----|
| 算法 | `--advantage-estimator gspo --use-tis --eps-clip 4e-4` |
| Reward | `--rm-type deepscaler` |
| Rollout | batch 8，每prompt 4样本，GBS 32，max response 8192 |
| 并行 | TP=4，CP=2，EP=8，SP，colocate 8 GPU |
| vLLM | DeepEP + triton MoE backend，`--vllm-gpu-memory-utilization 0.8` |
| 学习率 | 1e-6 constant，optimizer CPU offload |
| 评测 | `--eval-interval 20`，AIME pass@16（每题16样本，eval max len 16384） |
| 步数 | `NUM_ROLLOUT=200`，不保存checkpoint |

具体启动命令见文末「执行脚本样例」。

**日志位置**

单组日志命名示例：

```
${WORK_ROOT}/RL/logs/bench_r3_{r3|no-r3}_Qwen3-30B-A3B_{timestamp}.log
```

顺序对照完成后，宿主机侧可能生成 `r3_compare_COMPLETE_*.txt`，内含两组日志路径。

## 1.5 监控与结果解析

**训练是否正常**

```
grep train_metric_utils ${WORK_ROOT}/RL/logs/bench_r3_r3_*.log | tail -3
nvidia-smi
```

正常Colocate下8卡利用率接近满载，显存约70GB+/卡。

**关键指标**

| 指标 | 日志关键字 | 含义 |
|:-----|:-----------|:-----|
| 训推一致性 | `train/train_rollout_logprob_abs_diff` | 训练端与rollout端logprob绝对差，R3应显著更低 |
| 训练reward | `rollout/raw_reward` | DAPO-Math rollout批次平均reward |
| 测试集AIME | `eval/aime` | AIME-2024 pass@16（独立测试集） |
| 单步耗时 | `perf/step_time` | 含rollout+train，eval step会尖峰 |

**AIME评测节奏**

- 训前：`eval 0`（480条生成，约15–20分钟）
- 训练中：step 20/40/…/200 各触发一次
- 指标字段：`eval/aime`（准确率），`eval/aime/truncated_ratio`（截断比例）

## 1.6 测试环境参考

vime仓库内官方E2E测试与本次bench参数对齐，可作为对照基准。

**文件**：`vime/tests/test_qwen3_30B_A3B_r3.py`

| 项 | CI测试默认 | 本次bench |
|:---|:-----------|:----------|
| 模型 | Qwen3-30B-A3B | 同左 |
| GPU | 8 colocate | 同左 |
| 算法 | GSPO + TIS + R3 | 同左；bench增加no-R3对照 |
| num-rollout | 3（CI） | 200 |
| eval | 可选，`n-samples-per-eval-prompt 1` | 每20 step，pass@16 |
| DeepEP | `VIME_TEST_USE_DEEPEP=1` | 开启 |
| max-tokens-per-gpu | 2048（tight memory） | 2048 |
| 路径 | `/root/models`、`/root/datasets` | 挂载目录下的示例路径 |

与 `scripts/run-qwen3-30B-A3B.sh` 相比，bench脚本主要差异：自定义挂载路径、无checkpoint save、`MODE`控制R3开关、完整200 step + 完整AIME eval。

## 1.7 常见故障

| 现象 | 原因 | 处理 |
|:-----|:-----|:-----|
| `RM for deepscale not implemented` | 脚本CRLF截断flag | `sed -i 's/\r$//'` 后检查 `--rm-type deepscaler` |
| `optimizer cpu offload needs --use-precision-aware-optimizer` | flag被截断 | 检查 `--use-precision-aware-optimizer` 完整 |
| 脚本启动后立刻退出 | `pkill`误杀自身 | bench脚本已收窄匹配，避免杀掉正在执行的bench脚本 |
| GPU占满但0%利用率 | 僵尸vLLM/Ray | 停止残留进程后重新启动训练 |
| scp长flag截断 | 上传脚本后flag被截断 | 上传后 `grep` 验证 `deepscaler`、`precision-aware-optimizer` 等关键flag |

**相关文档**：
- [vime Quick Start（中文）](https://github.com/vllm-project/vime/blob/main/docs/zh/get_started/quick_start.md)
- 实验结果分析：`博客文章/R3原理与测试.md`

**附录A vime代码更新（可选）**

`inferactinc/public:vime-vllm-latest` 镜像内已自带vime，直接跑官方 `scripts/run-qwen3-30B-A3B.sh` 或本文bench脚本即可。仅在需要试用仓库最新commit、或自行修改vime源码时，才需要将代码同步到挂载目录并重新安装：

```
# 将本地vime与run_scripts拷入挂载目录后，去除Windows CRLF（路径请自行替换）
sed -i 's/\r$//' /data/nfs_87/kaiyuan/RL/run_scripts/*.sh

# 容器内以可编辑方式安装
cd /data/nfs_87/kaiyuan/RL/vime
pip install -e . --no-deps --no-build-isolation
```

**附录B 执行脚本样例**

以下命令假设容器名为 `vime`，示例工作根目录为 `/data/nfs_87/kaiyuan`（请替换为自己的挂载路径）。

**下载数据集**

```
docker exec vime bash /data/nfs_87/kaiyuan/RL/run_scripts/download_datasets.sh
```

**HF → torch_dist 转换（后台）**

```
docker exec -d vime bash /data/nfs_87/kaiyuan/RL/run_scripts/convert_qwen3_30b.sh
docker exec vime tail -f /data/nfs_87/kaiyuan/RL/logs/convert.log
```

**单组训练：R3（容器内）**

```
docker exec vime bash -lc \
  'cd /data/nfs_87/kaiyuan/RL/run_scripts && \
   sed -i "s/\r$//" run-kaiyuan-bench-r3-qwen3-30B-A3B.sh && \
   NUM_ROLLOUT=200 MODE=r3 bash run-kaiyuan-bench-r3-qwen3-30B-A3B.sh'
```

**单组训练：no-R3**

将上式 `MODE=r3` 改为 `MODE=no-r3`。

**顺序对照（宿主机后台：R3 200 step → no-R3 200 step）**

```
nohup bash /data/nfs_87/kaiyuan/RL/run_scripts/run-r3-compare-sequential.sh \
  > /data/nfs_87/kaiyuan/RL/logs/r3_compare_launch.log 2>&1 &
```

**查看训练进度**

```
grep train_metric_utils /data/nfs_87/kaiyuan/RL/logs/bench_r3_r3_*.log | tail -5
grep train_metric_utils /data/nfs_87/kaiyuan/RL/logs/bench_r3_no-r3_*.log | tail -5
```

**官方CI测试（容器内，需先prepare数据与权重）**

```
cd /data/nfs_87/kaiyuan/RL/vime
python tests/test_qwen3_30B_A3B_r3.py
```
