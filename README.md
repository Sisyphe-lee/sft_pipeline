# sft-pipeline

“Learn from Experience” 项目的数据清洗和训练。输入是 EdgeBench 上源模型（Kimi-K3 + Codex）rollout 的抓包，
输出是训练清单：哪些步算 loss。rollout 的采集、桥接抓包和 checkpoint 的评测在 EdgeBench 仓库。

## 两步

| | 做什么 | 入口 |
|---|---|---|
| 清洗 | 抓包 → 训练行 → 标注 → 训练清单 | `data_pipeline/clean.py` |
| 训练 | 按清单和 recipe 训练 | `training/train_sft.py`（见“现状”） |

## 清洗

一条命令做完，做什么由 policy 文件决定：

    python3 data_pipeline/clean.py \
        --policy configs/data_policies/pattern_all.yaml \
        --capture <抓包目录> [<抓包目录> ...] \
        --runs <harness 的 runs 目录> \
        --tasks <EdgeBench 任务定义目录> \
        --data <工作目录> \
        --output <清单>.json

| 参数 | 含义 |
|---|---|
| `--policy` | 清洗策略，`configs/data_policies/*.yaml` |
| `--capture` | 桥接的抓包目录（里面是 `resp_<id>/`），每个桥接一个，可以给多个 |
| `--runs` | harness 的运行目录，结构为 `<runs>/<rollout 编号>/<任务名>/`，用来取评分记录 |
| `--tasks` | 任务定义（`<任务名>.json`），用来判断分数越高越好还是越低越好；不给则按越高越好 |
| `--data` | 工作目录，中间结果都放这里，可重复使用 |
| `--output` | 输出的训练清单 |
| `--only` | 只处理编号里含这段文字的 rollout（任务名或运行编号），调试用 |

Kimi 地址从环境变量 `KIMI_URLS`（空格分隔，以 `/v1` 结尾）读取，没有则读 `/personal/kimi.env`。
集群上放在 tmux 里跑。中断后重跑同一条命令，已完成的部分会跳过。

### 策略文件

| 文件 | 含义 |
|---|---|
| `pattern_all.yaml` | 实验组：12 个 pattern 任何一个标到的步算 loss |
| `control_random.yaml` | 对照组：同一批 rollout，每条随机抽同样多的步 |
| `all_turns.yaml` | 普通 SFT：所有步都算 loss，不调用 Kimi |

可改的项：

| 项 | 含义 |
|---|---|
| `arm` | `labelled` 标到的步 / `random_matched` 随机抽同样多 / `all` 全部 |
| `patterns` | 只用其中几个 pattern，如 `[C1, D3]`；空为全部 |
| `seed` | 随机抽样的种子 |
| `labelling.sheet` | 标注说明。改了内容就会重新标注，旧结果保留 |
| `labelling.readings` | Kimi 通读几遍、各用哪份说明 |
| `labelling.per_url` | 每个 Kimi 地址的并发数 |

同一个 `--data` 下换策略时，导出和标注直接复用：先跑实验组，再跑对照组只需几秒。

### 内部做了什么

`clean.py` 依次做下面几件事，每件的结果都在 `--data` 里：

1. **导出**（`rows/<rollout>.jsonl`）：模型的每次回复一行，含它看到的对话、它的回复（思考、文字、工具调用）和工具定义。所有回复都保留。
2. **渲染**（`work/<rollout>/transcript.txt`）：把这些行排成带步号的日志，第 N 步就是第 N 行；同时用规则找出提交、出结果、结尾这些位置。
3. **通读**（`work/<rollout>/<名字>.json`）：Kimi 读整条日志，报出规则找不到的位置，比如意外结果、同一方向第二次失败、被搁置的想法。
4. **标注**（`work/<rollout>/judged-<说明指纹>/`）：每个位置一个请求，Kimi 按 `patterns/label-sheet.md` 回答是非题并填步号。程序检查答案和步号，全部成立才打标签。
5. **汇总**（`labels-<说明指纹>.jsonl`）：每个标签一条记录。
6. **选择**（`--output`）：按策略写清单。清单按文件和字节位置指向训练行，不复制数据。没选中的步不算 loss，但仍在后面步的历史里。

标签只有标和不标。12 个 pattern 的定义就是 `patterns/label-sheet.md` 里各自的是非题；改定义时要同时改
`data_pipeline/label.py` 里的 `EVIDENCE`，`python -m pytest -q data_pipeline/test_label.py` 会检查两边是否一致。

## 现状（2026-10-11）

- 清洗：`all_turns.yaml` 在一条 Kimi rollout 上跑通（212 行）。需要 Kimi 的两种策略，其标注部分是从 EdgeBench 移植的
  （在那边用 7 条 rollout 对照 Opus 测过），在本仓库的入口下还在调试。
- 训练：`training/`、`jobs/`、`configs/recipes`、`configs/training` 是从 physicalagent-sft-pipeline
  （分支 `feat/flashnext-sft`，提交 `dc7854c`）原样复制的，还没去掉图像和机器人相关的部分，目前不能运行。
- 2026 年 9 月的 Kimi rollout 只能用来调流程，不能用来训练：那批数据里模型几乎不思考（见 EdgeBench `docs/kimi-serving.md`）。
- 还没做：按最终成绩筛选 rollout、验证集划分、对标注结果的抽查。
