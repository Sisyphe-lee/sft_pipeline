# Training

训练入口是 [`train_sft.py`](train_sft.py)。它读取 Data Selection 生成的 manifest，
按 recipe 加载和处理 SFT decision，然后训练显式指定的模型 checkpoint。

```text
SFT JSONL + 图片
      │
      ├─ selection policy ─→ manifest
      │
      └────────────────────→ train_sft.py + recipe ─→ checkpoint
```

## Data Selection 的输入与输出

### 输入：SFT JSONL

原始训练输入是 PhysicalAgent 已经导出的 `sft_turns.jsonl` 或等价的聚合 JSONL。
它不是 simulator 的原始 rollout，也不是 Evaluation Review。每行是一条 assistant
decision，主要包含：

- 当前 decision 的文本和图片上下文；
- 本轮 assistant reasoning、文本和 tool request；
- `task.id`、`task.seed`、`trajectory_id`、turn 等元数据；
- outcome、训练资格和图片等素材的相对路径。

图片和证据文件不写进 manifest。训练时仍按 JSONL 中的相对路径读取，所以移动数据时
要保持 JSONL 与素材的目录关系。

常用输入有两类：

| 输入 | 路径 | 用途 |
| --- | --- | --- |
| PhysicalAgent episode 导出 | `/robby/share/3D/lcy/PhysicalAgent/data/**/sft_turns.jsonl` | `kimi12_32x40_20260910` 等按 episode 选择的数据 |
| RT3 聚合训练集 | `<RT3 数据根>/sft/lcy_full_20260911/train_le196608.jsonl` | `20260911_20646rows_h40e76994` |

RT3 数据根通常是
`/robby/share/3D/lingcong.hyb/rt3_sft_20260911_20646rows_h40e76994/`，在不同容器中
可能挂载到其他前缀。

### Policy：决定保留哪些行

Policy 位于 [`configs/data_policies/`](../configs/data_policies/)。最常用的是
[`successful.yaml`](../configs/data_policies/successful.yaml)，默认保留官方成功且允许训练的
trajectory，并对 decision 去重。Policy 还可以限制 task、teacher、failure code，或按
`(task.id, task.seed)` 切 validation。

去重按 `(task.id, trajectory_id, turn)` 判断，相同身份的内容冲突会报错。
Failure-code 过滤只检查当前 decision 的执行结果，历史失败不会因此排除恢复决策。

Policy 文件里每个键是一个独立的筛选策略，文件本身就是要用的策略组合；不写或取默认值的键不生效。

| 策略键 | 作用范围 | 效果 |
| --- | --- | --- |
| `official_success_only` | decision | 只留官方成功的 trajectory |
| `require_formal_training_eligible` | decision | 只留允许正式训练的行 |
| `task_ids`、`teacher_models`、`unit_allowlist` | decision | 限定 task、teacher 或 `<task>/<seed>` |
| `exclude_failure_codes` | trajectory 内的单条 decision | 去掉自身工具调用返回所列 failure code 的 decision；它仍留在后续 decision 的历史上下文里，只是不再监督 |
| `max_failed_decisions: {limit, codes}` | 整条 trajectory | 带所列 failure code 的 decision 超过 `limit` 条时整条丢弃 |
| `max_task_length_quantile: q` | 整条 trajectory | 同 task 内至少有 `q` 比例的 trajectory 比它短时整条丢弃；长度按 decision 数计，等长的同留同丢。`0.8` 即丢掉每个 task 最长的约五分之一 |
| `exclude_labeled_trajectories: [field=value, ...]` | 整条 trajectory | 标签文件里该 trajectory 有任一所列 `field=value` 时整条丢弃 |
| `exclude_labeled_steps: [field, ...]` | trajectory 内的单条 decision | 去掉标签记录在所列字段下点名的 step 所在的 decision |

`ignore_success_terminal_failure: true` 让官方成功 trajectory 的最后一条 decision 不计为失败：
evaluator 在任务谓词成立时立即结束 episode，这条命令会被截断并带上 failure code。
[`successful_no_hard_failures.yaml`](../configs/data_policies/successful_no_hard_failures.yaml)
是"成功轨迹 + 去掉调用被拒或执行失败的 decision"这一组合，轨迹本身不丢。

新增策略只需在 `select_sft_rows.py` 里加一个函数并登记到策略表：只看单行的放
`ROW_STRATEGIES`，需要整条轨迹的放 `TRAJECTORY_STRATEGIES`。各策略独立判断，互不依赖顺序，
`selection.excluded_reasons` 分别记录每个策略排除的行数。

```bash
python data_pipeline/select_sft_rows.py \
  --dataset /data/run-a/sft_turns.jsonl /data/run-b/sft_turns.jsonl \
  --policy configs/data_policies/successful.yaml \
  --output /data/selections/train.manifest.json
```

`--dataset` 可以接多份 JSONL，按给定顺序扫描。`--data-root` 不指定时取所有源文件目录的
共同父目录；源文件不在同一合理数据根下时应显式指定。

### 标签文件：把看过轨迹的结论接进筛选

官方成功不等于任务真的做完：checker 只检查少数几何条件，并在条件成立的瞬间结束 episode。
`exclude_labeled_*` 两个策略读取 `--labels` 给出的 JSONL 标签文件，每行一条记录，
用 `trajectory_id` 或 `task` 加整数 `seed` 绑定到 trajectory，其余字段自由定义；人工审核
结论和模型判定用同一种格式，可以同时传多个文件。没有任何标签记录的 trajectory 不受影响，
其数量记在 `selection.unlabelled_trajectories`，标签文件的 SHA256 记在 `selection.labels`。

[`audit_success_episodes.py`](../data_pipeline/audit_success_episodes.py) 用一个
OpenAI-compatible 的视觉模型逐条判定 manifest 里的 trajectory，输入是任务指令、动作日志、
全程与结尾的视频抽帧和最终三路相机画面，输出 `final_state`、`process`、`incident`
三项结论、各自的依据，以及导致事故的 `incident_steps`。每个 `--base-url` 同时发
`--concurrency-per-url` 个请求（默认 1），已有记录的 trajectory 重跑时跳过：

```bash
source /personal/kimi.env
python data_pipeline/audit_success_episodes.py \
  --manifest /data/selections/train.manifest.json \
  --output /data/labels/train.kimi_audit.jsonl \
  --base-url "$KIMI_BASE_URL_1" "$KIMI_BASE_URL_2" \
  --model /robby/share/3D/checkpoints/Kimi-K3
```

### 输出：selection manifest

Data Selection 只产生一份 v2 JSON manifest：

- `base_root` 和 `sources` 指向源 JSONL；
- `train`、`validation` 保存被选中行的 source、行号、byte offset/length 和分组信息；
- `selection` 保存 policy、输入行数、保留行数和排除原因。

`base_root` 相对 manifest 所在目录，`sources` 相对数据根；图片则相对各自的源 JSONL。
训练根据 offset 直接读取源 JSONL，不复制 SFT 正文，不复制图片，也不生成另一份审计文件。
重新筛选可以覆盖 manifest，但不会覆盖源数据。训练期间必须保持源 JSONL 和 manifest 不变。
重新筛选失败时已有清单保持原样。旧 v1 manifest 需重新筛选生成 v2；训练不对源文件做 hash/stat 检查。

目前已有两份常用 selection 产物：

| 名称 | Manifest | Data root | 内容 |
| --- | --- | --- | --- |
| `kimi12_32x40_20260910` | [`configs/datasets/kimi12_32x40_20260910/train.manifest.json`](../configs/datasets/kimi12_32x40_20260910/train.manifest.json) | `/robby/share/3D/lcy/PhysicalAgent/data` | 1,276 个源文件中选择 17,929 / 39,889 条 decision，无 validation |
| `20260918_25175rows_50task1201unit_h40e76994` | [`configs/datasets/20260918_25175rows_50task1201unit_h40e76994/train.manifest.json`](../configs/datasets/20260918_25175rows_50task1201unit_h40e76994/train.manifest.json) | `/robby/share/3D`；batch pod 使用 `/robby/share/s7/3D` | 1,989 个新 harness authoritative source 中选择 25,175 / 78,029 条 decision，覆盖 1,201 个成功 unit、50 task，无 validation；使用 [`successful.yaml`](../configs/data_policies/successful.yaml) |
| `20260911_20646rows_h40e76994` | `<RT3 数据根>/manifests/20260911_20646rows_h40e76994.manifest.json` | manifest 内的相对 `base_root` | 20,646 条 decision，无 validation |

第一份 manifest 已归档到 Git，但仍需要外部 JSONL 和图片。第二份 manifest、JSONL 和图片
都保存在 RT3 共享数据根中；详细目录结构见
[`docs/RT3_SFT.md`](../docs/RT3_SFT.md#一数据)。

如果数据搬到了新挂载点，不需要修改 manifest，可以在训练时传：

```bash
--data-root /new/path/to/the/same/data-root
```

这里的目录必须与 manifest 的 `sources` 以及 JSONL 内的素材相对路径匹配。

### 2026-09-24 Kimi 5000-unit 数据的成功子集

Selection manifest 位于共享存储：
`/robby/share/3D/lcy/PhysicalAgent/manifests/20260924_kimi_5000_success.manifest.json`。
数据根是 `/robby/share/3D/lcy/PhysicalAgent/data/formal50x100_expert100_native_steps_kimi4_16w_60call_sft_6300_20260924/`；
batch pod 挂载前缀为 `/robby/share/s7/3D`，启动时用 `--data-root` 指向同一数据根。

按原 Run 的 `artifacts.tsv` 定位 4,996 个已有 SFT source，使用
[`successful.yaml`](../configs/data_policies/successful.yaml) 从 164,437 行中选择
46,020 行，覆盖 50 tasks、2,740 个成功 unit / trajectory，无 validation；
118,417 行因非官方成功排除。1181-unit 独立超时补跑不在本 manifest 中。
manifest 使用相对 `base_root` 和 source 路径，不复制正文或图片。
manifest SHA256：`ef2d853b359d3dd795e97a6502186625f8f78ac10373d6bec549a935b5058b93`。

筛选代码为 `d7d0ea7f4030d9546b92747f712a604b0471a5e7`，运行退出码 0。
已核对 unit 覆盖、路径和行身份唯一性，并抽查 102 行 offset、监督元数据及图片引用。
9B 正式训练复用 [`qwen35_9b_sft.yaml`](../configs/recipes/qwen35_9b_sft.yaml)；
数据/训练 CPU 回归通过（57 tests），Job dry-run、shell、模型分片、依赖版本和新输出目录检查通过。
按用户要求不单独提交 smoke；目标 processor 的精确长度预检由正式 Job 在训练前执行。
正式代码固定为 `3ffbd0b5cc2134ef9073189c0d8ab3eccc0df7bb`。
2026-10-09 提交 9B × 32 B300（4 node × 8）、5 epochs、global batch 32、梯度累积 1 的
Job [345007892](https://aistudio.alipay.com/project/job/detail/345007892)，提交后状态 queued，尚未开始训练。
先前的 16 卡 Job 345007463 在排队时已停止，改用上述 32 卡 Job。
运行名 `sft_20260924_kimi_5000_success_9b_mn4_ep5`；输出位于
`/robby/share/s7/3D/lcy/PhysicalAgent/training/checkpoints/<run>`，日志位于
`/robby/share/s7/3D/lcy/PhysicalAgent/training/logs/train_<run 去掉 sft_ 前缀>`。
导出的 `estimated_tokens` 是请求序列化字节数的粗估，不能当作真实模型 token 长度；
最终保留行数以训练入口的精确长度预检为准。

同一批 4,996 个 source 另有一份去掉失败 decision 的清单：
`/robby/share/3D/lcy/PhysicalAgent/manifests/20260924_kimi_5000_success_no_hard_failures.manifest.json`，
使用 [`successful_no_hard_failures.yaml`](../configs/data_policies/successful_no_hard_failures.yaml)。
它是上面 46,020 行的子集，保留 36,581 行，2,740 个 trajectory 和 50 个 task 都在；
去掉的 9,439 行是自身工具调用被拒或执行失败的 decision，成功 trajectory 的最后一条不计。
manifest SHA256：`d430122e7dae76a62a2571733c843767a372d5d82ffedc6b39e0a893c03c8e5e`，
筛选代码为 `92689001d569f40e252ee2bda85008b92258e3c2`，运行退出码 0；
数据/训练 CPU 回归通过（74 tests），已抽查 120 行 offset、行身份及图片引用。
这份清单尚未用于训练，也没有做目标 processor 的精确长度预检。

在它之上再按标签文件筛选的清单：
`/robby/share/3D/lcy/PhysicalAgent/manifests/20260924_kimi_5000_success_no_hard_failures_audited.manifest.json`，
使用 [`successful_no_hard_failures_audited.yaml`](../configs/data_policies/successful_no_hard_failures_audited.yaml)
和同一共享存储 `labels/` 目录下的两份标签文件：

| 标签文件 | 记录 | SHA256 |
| --- | --- | --- |
| `20260924_kimi_5000_success.kimi_audit.jsonl` | 2,740 条 trajectory 各一条 | `8cb655c71cbde22e5a3bd2fb53a1e0e14bb87e8ee0f3c7215a0a7218ea36df15` |
| `20260924_kimi_5000_human_review.jsonl` | 人工审核的 45 个 unit | `70f944b4aecc0061463cabeff9d96ac100681df34a47bdd2f551b756151eabc4` |

Kimi 标签由 `audit_success_episodes.py`（代码 `28b46b6`，中途改为每端点 2 并发的 `40c43e8`，
rubric 未变，SHA256 `e1e657f9…`）在 Kimi-K3 上对每条 trajectory 判一次得到，无失败。结论分布：
`final_state` 未达成 361、无法判断 70；`process` 走捷径 293；`incident` 掉落或碰飞 207、
碰撞 93、动作过猛 24。其中 112 个 unit 在标定时用早一版 rubric 判过一次，
两次“是否整条排除”的结论有 106 个一致。
人工审核覆盖成功数最少的 11 个 task：35 个 unit 判为有问题，`dump_bin_bigbin` 的 10 个无法判断。

清单保留 30,979 行，是上面 36,581 行的子集，覆盖 2,272 个 trajectory、48 个 task。
去掉的 5,602 行里，5,383 行来自整条排除的 468 个 trajectory（最终状态未达成、走捷径、
人工判为有问题或无法判断），219 行是保留 trajectory 中导致事故的那条 decision；
事故之后的补救步骤仍然监督。`dump_bin_bigbin` 和 `rotate_qrcode` 没有剩余 trajectory，
`shake_bottle` 剩 4 个、`shake_bottle_horizontally` 剩 7 个，其余大多被判为走捷径。
manifest SHA256：`5f4bd334a44866c173441c00e2f790eddc33735341d7199aca3f8c52e2c008fb`，
筛选代码为 `3e90545cc0a13a4c67ad3e55ac6c48d812343200`，运行退出码 0；
数据/训练 CPU 回归通过（77 tests）。已核对：保留的 trajectory 都没有排除类标签，
排除的 trajectory 都有；保留行都不覆盖事故 step，额外去掉的行都覆盖；
并抽查 120 行 offset、行身份及图片引用。
2026-10-10 用这份清单提交 9B × 32 B300（4 node × 8）、5 epochs、global batch 32、梯度累积 1 的
Job [345136898](https://aistudio.alipay.com/project/job/detail/345136898)，提交后状态 queued，尚未开始训练。
recipe（[`qwen35_9b_sft.yaml`](../configs/recipes/qwen35_9b_sft.yaml)）、卡数和梯度累积与对照
Job 345007892 相同，换了 manifest；训练代码固定为 `168c364b639bf6428703e853c8059eb946bb6a01`，
比对照 Job 的 `3ffbd0b` 多出 W&B 的分段、task 和吞吐指标，loss 与优化路径不变。
此前两个 16 卡 Job（345133187 固定在 `3ffbd0b`，345134319 固定在 `1fb8499`）都在排队时停止，没有产生输出。
运行名 `sft_20260924_kimi_5000_success_audited_9b_mn4_ep5`，输出和日志目录规则同上。
Job dry-run、manifest SHA、首图可见性、训练加载器读取和新输出目录检查通过；
未单独提交 smoke，目标 processor 的精确长度预检由正式 Job 在训练前执行。

## 各类配置文件的作用

| 文件 | 作用 |
| --- | --- |
| `configs/data_policies/*.yaml` | Data Selection 规则：哪些 decision 进入 manifest |
| `configs/datasets/**/train.manifest.json` | 已定稿的数据选择结果：从哪些 JSONL 读取哪些行 |
| `configs/recipes/*.yaml` | 训练参数：学习率、epoch、长度、采样、保存频率等 |
| `configs/training/*.json` | DeepSpeed 配置，由 recipe 引用 |
| `training/train_sft.py` | 单机或多机训练入口 |
| `<output>/resolved_config.json` | 实际运行参数、manifest/recipe hash 和数据处理统计 |

常用正式 recipe：

| Recipe | 适用范围 |
| --- | --- |
| [`qwen35_9b_sft.yaml`](../configs/recipes/qwen35_9b_sft.yaml) | Qwen3.5-9B 通用正式配置；沿用 main 的正式参数，5 epochs、global batch 32、196,608 token、`length_grouped`，每个 epoch 保存一次；梯度累积按设备数自动计算（16 卡为 2） |
| [`full_sft.yaml`](../configs/recipes/full_sft.yaml) | 通用起点；3 epochs、global batch 32、73,728 token、历史裁剪、`turn_uniform`、冻结视觉、ZeRO-3 |
| [`sft_20260918_25175rows_50task1201unit_h40e76994_mn4.yaml`](../configs/recipes/sft_20260918_25175rows_50task1201unit_h40e76994_mn4.yaml) | 新 harness 50-task 数据的 27B、4 node × 8 GPU 配方；10 epochs、196,608 token、`length_grouped`，每个 epoch 保存一次 |
| [`sft_20260911_20646rows_h40e76994_27b_mn4.yaml`](../configs/recipes/sft_20260911_20646rows_h40e76994_27b_mn4.yaml) | 20,646 行数据的 27B、4 node × 8 GPU 配方；196,608 token、`length_grouped` |

`parity_9b_single_gpu*.yaml` 只用于短回归测试，不是正式训练 recipe。模型路径、manifest、
data root 和输出目录属于每次运行，必须通过 CLI 指定，不应写入可复用 recipe。

## 启动 SFT

4B、9B、27B 都通过 `--model` 显式选择，没有默认模型尺寸。下面是单机 8 卡示例：

```bash
ENABLE_MY_FLASH_ATTN_HOOK=0 python -m torch.distributed.run \
  --standalone --nproc_per_node=8 training/train_sft.py \
  --model /models/Qwen3.8-27B \
  --data-manifest configs/datasets/kimi12_32x40_20260910/train.manifest.json \
  --data-root /robby/share/3D/lcy/PhysicalAgent/data \
  --recipe configs/recipes/full_sft.yaml \
  --output /outputs/sft-run-001 \
  --run-name sft-run-001
```

每设备 batch 固定为 1。未显式设置梯度累积时：

```text
gradient_accumulation_steps = global_batch_size / WORLD_SIZE
```

因此默认 global batch 32 必须能被进程数整除。多机 RT3 配方的完整启动模板在
[`jobs/train_20260911_20646rows_h40e76994_27b_mn4_robbys7.yml`](../jobs/train_20260911_20646rows_h40e76994_27b_mn4_robbys7.yml)。

`jobs/` 保存训练、smoke 模板及辅助脚本：`tools/preflight_train.sh` 检查 Job 环境和数据，
`tools/rendezvous.sh` 协调多节点地址；模板中的共享盘挂载、代码和输出路径需按实际环境设置。
RT3 平台提交方式与环境要求见 [RT3_SFT.md](../docs/RT3_SFT.md) 和 [SETUP.md](../docs/SETUP.md)。

Recipe 的 key 与 CLI 参数同名，只是使用下划线。CLI 参数会覆盖 recipe；例如临时改变
保存频率可以追加 `--save-steps 200`。新训练必须使用新的空输出目录。

## 训练行为

- 只监督当前 assistant completion，包括 reasoning、文本、tool request 和结束标记；
  system、历史、observation、tool result 和图片只作为上下文。
- 默认 loss 是 `chunked_nll` 的 decision 均值，即每条 decision 等权。
- 默认 `turn_uniform`，每个 epoch 随机遍历保留的 decision。
- 默认开启历史裁剪。只有超长行才从最旧历史开始整轮删除；当前输入和完整 completion
  不会被截断，仍然超限的行会在训练前过滤。
- Validation 默认关闭。需要时先在 policy 中设置 `validation_fraction`，再在 recipe 中
  设置 `validation: true`。

采样策略、loss 或长度规则的详细语义以 recipe 和 `train_sft.py --help` 为准。环境评测与
训练独立，训练 loss 不能代替成功率；checkpoint 使用 [`evaluation/run.sh`](../README.md#evaluation)
评测。

## W&B 指标

训练命令加 `--report-to wandb`(或 recipe 设 `report_to: wandb`)即可记录,`--report-to none` 关闭。默认
`WANDB_MODE=online`,本地事件流落在 `<output>/wandb/`(`WANDB_DIR` 可覆盖),环境变量优先;run 名的优先级是
`--run-name`、`WANDB_NAME`、输出目录名,密钥只走 `WANDB_API_KEY`,不写进 YAML。

```bash
export WANDB_API_KEY=<你的 api key>
export WANDB_BASE_URL=http://33.180.4.104   # 容器内网 IP;本机用 http://robby-wandb.robbyant.cn
export WANDB_ENTITY=lingcong-hyb            # 现役 run 都在这个 entity
export WANDB_PROJECT=physicalagent-sft
wandb login --host="$WANDB_BASE_URL"        # --host 要和 WANDB_BASE_URL 一致
# 在原训练命令后加:--report-to wandb --run-name <实验名>
```

离线跑完用 `wandb sync <output>/wandb/offline-run-...` 补传。只有全局 rank 0 通过 Trainer 原生回调记
`train/loss`、`train/learning_rate`、`train/grad_norm`、`train/epoch`,横轴是 `train/global_step`,频率由
`logging_steps` 控制。在这之上另有三张表,分段曲线经 Trainer 的 `_metrics` 一起出,`mfu/` 与 `task/` 用
`commit=False` 搭在 Trainer 那次 log 上:

* `train/` 里除 Trainer 原生指标外,多四条分段曲线 `reasoning_token_accuracy`、`tool_call_token_accuracy`、
  `reasoning_loss`、`tool_call_loss`(分界是 `</think>`,见 `training/span_metrics.py`)。
* `mfu/`:`mfu`、`model_tflops_per_gpu`、`tokens_per_sec`、`step_time_s`、`tokens_per_step`、
  `peak_memory_gb` (`training/mfu.py`,峰值按卡型与精度查表,查不到就只记吞吐)。
* `task/<task>/`:每个 task 六条,`token_accuracy`、`reasoning_token_accuracy`、`tool_call_token_accuracy`、
  `loss`、`reasoning_loss`、`tool_call_loss`(`training/task_metrics.py`)。每一步只出这一步的原始数,不做滑动
  平均,这一步没出现的 task 不更新。task 列表从 manifest 的 `task_id` 来,不写死。

**统计口径跟着 `loss_normalization` 走。** `train/` 的分段曲线和 `task/` 那六条用同一套公式
(`span_metrics.py` 的 `metrics_from`),整个 optimizer step 的账(world × accum 个 micro-batch)合起来算一次:

| `loss_normalization` | loss | accuracy |
| --- | --- | --- |
| `turn_mean`(现役配方) | 每条 decision 先除自己的有效 token 数,再等权平均 | 每条 decision 先算自己的 correct/count,再等权平均 |
| `token_mean` | Σnll / Σtoken | Σcorrect / Σtoken |

所以单 task 的 run 里 `train/loss`、`train/mean_token_accuracy`、`train/reasoning_loss` 等六条应当与
`task/<task>/*` 逐位相等,`training/test_task_metrics.py` 钉了这条。`train/mean_token_accuracy` 是**名字沿用
TRL 的、值换成我们自己算的**(TRL 原本每个 micro-batch 先算比值、再对 accum 个比值等权平均,和上表对不上);
`train/loss` 仍是 HF Trainer 自己那条(turn_mean 下同口径、同一个数),不覆盖。

run 的 config 来自 `resolved_config.json`;训练、保存及最终验证完成后显式 `finish()`,保证在
`--hard-exit-after-train` 之前收尾,异常退出标记为失败。续训默认新建一个 W&B run,从恢复后的 global step
继续记,不回放历史指标。

指标实现同步自 main `92d5897`。更新代码只对之后启动或恢复的进程生效；运行中的 Job
不会热更新，也不能补出此前未采集的分段及 task 指标。

## 输出与恢复

训练输出目录主要包含：

| 文件 | 作用 |
| --- | --- |
| `resolved_config.json` | 展开的运行参数、数据与 recipe 身份、长度处理统计 |
| `length_policy_plan.json` | 精确 token 长度检查和历史裁剪计划 |
| `checkpoint-N/` | 可恢复的 Trainer/DeepSpeed checkpoint |
| `train_results.json` | 完成训练后的汇总指标 |
| `eval_results.json` | 启用 validation 时的 teacher-forced loss |

恢复时必须使用原来的模型、manifest、data root、recipe、参数和输出目录：

```bash
ENABLE_MY_FLASH_ATTN_HOOK=0 python -m torch.distributed.run \
  --standalone --nproc_per_node=8 training/train_sft.py \
  --model /models/Qwen3.8-27B \
  --data-manifest configs/datasets/kimi12_32x40_20260910/train.manifest.json \
  --data-root /robby/share/3D/lcy/PhysicalAgent/data \
  --recipe configs/recipes/full_sft.yaml \
  --output /outputs/sft-run-001 \
  --run-name sft-run-001 \
  --resume-from-checkpoint /outputs/sft-run-001/checkpoint-800
```

入口会用 `resolved_config.json` 检查恢复参数，防止用不同数据或 recipe 接着训练。

## 可选功能与环境

- W&B 默认关闭。追加 `--report-to wandb --run-name <name>` 后启用；默认写 online run，
  是否在线上传由 `WANDB_MODE` 和运行环境决定。
- OSS checkpoint 上传默认关闭，只有设置 `--oss-bucket` 后才启用；OSS 副本不包含完整
  optimizer/RNG 状态，不能代替本地续训 checkpoint。
- 依赖列表见
  [`configs/training/qwen3_5_sft_requirements.txt`](../configs/training/qwen3_5_sft_requirements.txt)。
  FlashAttention、FLA、CUDA/NVRTC 和 DeepSpeed 必须与实际 GPU 环境匹配。
- 部分 AIStudio 镜像不能使用 `python -m torch.distributed.run`，此时用同一环境的 Python
  直接执行 `"$(command -v torchrun)"`。

修改数据加载、processor、labels、长度、采样或 loss 后，CPU 回归入口是：

```bash
python -m pytest -q data_pipeline training
```

CPU 测试不能替代目标模型上的多卡 smoke。

### FlashNext 适配分支

`feat/flashnext-sft` 在现有 Transformers / TRL / DeepSpeed 入口上适配
Qwen3.8-Flash-Next，尚未通过完整模型多卡训练验收。
9B recipe、依赖环境及默认加载顺序保持原样。

新增显式 `--zero3-init`（recipe 中为 `zero3_init: true`），要求 DeepSpeed
stage 3；它在加载权重前初始化分片配置，避免每个进程先加载完整模型。
省略此选项时沿用已有加载顺序及运行记录字段，兼容已有 9B checkpoint 的恢复检查。
参数统计读取 ZeRO 的全局参数大小，避免把未 gather 的参数错误统计为零。

`--flashnext-qsa` 仅适用于 Transformers 5.16.0 的 `qwen4_exp`，显式要求
`--zero3-init`、SDPA 和全参语言训练。它保留 HF 的参数、RoPE、gate 和多模态处理，
将 QSA 的稠密 mask 换成分块计算的 token 索引，调用 Miles 的稀疏前向/反向 kernel。
仅支持单条、不 padding、不 packing、无 KV cache 的 decision。PLE 保持可训练；
视觉塔沿用既有冻结设置，GDN 使用环境中的 FLA。未迁移 Miles 的 RL 入口。

Kernel 原样取自 Miles `05bd9cb2afc9f6a1e94544e8fb4e9918771e763b` 的
`miles/kernels/attention/qsa/sparse_attention.py`，保留 Apache-2.0 许可证。
实验 recipe 为 [`qwen38_flashnext_sft_experimental.yaml`](../configs/recipes/qwen38_flashnext_sft_experimental.yaml)：
LR 1e-5、warmup 5%、5 epochs、global batch 32、decision 均值 loss。
原子加的 sparse backward 不承诺逐 bit 确定性；不会把稠密模型的 FLOPs 公式
套到 MoE/PLE 上，仍记录实际吞吐、step 时间、显存与分段/task loss。

升级包单独安装至 overlay，见
[`flashnext_hf_overlay_requirements.txt`](../configs/training/flashnext_hf_overlay_requirements.txt)，
不改写 9B 环境。Job 由 [`make_flashnext_job.py`](../jobs/tools/make_flashnext_job.py)
生成（`--help` 查看必须显式提供的路径），启动入口为
[`run_flashnext_sft.sh`](../jobs/tools/run_flashnext_sft.sh)。正式 Job 内先做 GPU kernel
数值/梯度校验，通过后直接执行现有训练入口，不另提交 smoke Job。

2026-10-11：原 9B 环境回归 85 passed、2 skipped（缺少 FlashNext 架构时跳过）；
FlashNext Job 依赖栈的后端及指标检查 17 passed、5 skipped（GPU 检查待 Job 内执行）。
CPU 验证覆盖 QSA token 集合/输出/梯度、真实微型 VLM 的 PLE/MoE 反向、TRL loss、
保存重载；使用官方 FlashNext processor 抽查真实 manifest 的首/中/末三行，
精确 token 前缀与 reasoning/tool-call mask 均通过。这些结果不代表完整模型多卡验收。

32 B300 Job [345328886](https://aistudio.alipay.com/project/job/detail/345328886)
代码固定 `fa6c84d07a90fd7492ce19e195ded3a6a0e5437c`；在环境版本断言处失败，
未进入 kernel 校验或模型加载。原因是 Job 的 overlay 路径漏写 `cpfs/`，Python
忽略不存在的 PYTHONPATH 项并回退至镜像里的旧 Transformers。
复用 30,979 decisions 的 audited manifest，SHA256
`5f4bd334a44866c173441c00e2f790eddc33735341d7199aca3f8c52e2c008fb`。
基座为官方 BF16 `Qwen/Qwen3.8-Flash-Next`，revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540`；全部 131 分片已逐文件校验。
共享盘输出为 `<PhysicalAgent>/training/checkpoints/sft_kimi5000_audited_flashnext_bf16_mn4_ep5_20261011`，
日志在相邻 `training/logs/` 的同名目录，每节点保留退出码。

修复版 Job [345330057](https://aistudio.alipay.com/project/job/detail/345330057)
固定代码 `1e25d67242de1241879631ec0047415779480aee`，输出/日志使用上述 run 名加
`_retry1` 的新目录。运行前检查依赖文件存在性、版本及实际导入路径，并保留 startup 日志。
60002 上启动脚本的环境检查已通过。这里的挂载关系已用 `findmnt` 核实：
`/robby/share/cpfs/lingcong.hyb/...` 在 Job 中是
`/robby/share/s7/cpfs/lingcong.hyb/...`；`3D` 数据仍映射到 `/robby/share/s7/3D/...`。
不可套用其他旧训练目录的前缀转换。

### 2026-09-18：50-task 新 Harness 数据

成功数据清单：`configs/datasets/20260918_25175rows_50task1201unit_h40e76994/train.manifest.json`，
25,175 decisions / 1,201 成功 units。正式 mn4 recipe 与 main 的 32-task 实验参数逐项一致。
数据/训练回归已通过（57 tests）；Job YAML 和 shell 检查、平台 dry-run 已通过。
排队的 smoke Job 339053177 已取消，改在 WebIDE 60003 上用 4B × 8 卡运行。
数据预检通过，16 行均未过滤；前两步 loss 为 1.504、1.330，`checkpoint-2`、
Trainer 状态和 resolved config 已落到 CPFS。第三步等待 checkpoint 同步上传 OSS，
因此这里记录为训练与本地保存链路通过，完整三步退出仍待上传结束。
正式训练代码固定为 `badc0772cec2952bf1cfe24a975797aac183957f`。
27B × 32 卡正式 Job [339057655](https://aistudio.alipay.com/project/job/detail/339057655)
已提交，当前状态为 queued；尚未开始正式训练。
