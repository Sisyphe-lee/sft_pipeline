# sft-pipeline

“Learn from Experience” 项目的数据清洗和训练。输入是 EdgeBench 上源模型（Kimi-K3 + Codex）rollout 的抓包，
输出是训练清单：哪些步算 loss。rollout 的采集、桥接抓包和 checkpoint 的评测在 EdgeBench 仓库。

## 三步

| | 做什么 | 入口 | 配置 |
|---|---|---|---|
| 标注 | 抓包 → 训练行 → Kimi 标注 → 标签文件。离线做一次 | `data_pipeline/annotate.py` | `configs/labelling/*.yaml` |
| 选数据 | 标签 + 策略 → 训练清单。不调用模型，几秒完成，可反复换策略 | `data_pipeline/select_rows.py` | `configs/data_policies/*.yaml` |
| 训练 | 按清单和 recipe 训练 | `training/train_sft.py`（见“现状”） | `configs/recipes/*.yaml` |

## 标注

    python3 data_pipeline/annotate.py \
        --config configs/labelling/kimi_v1.yaml \
        --capture <抓包目录> [<抓包目录> ...] \
        --runs <harness 的 runs 目录> \
        --tasks <EdgeBench 任务定义目录> \
        --data <数据目录>

| 参数 | 含义 |
|---|---|
| `--config` | 标注设置：标哪些 pattern、Kimi 通读几遍、每个地址的并发数 |
| `--capture` | 桥接的抓包目录（里面是 `resp_<id>/`），每个桥接一个，可以给多个 |
| `--runs` | harness 的运行目录，结构为 `<runs>/<rollout 编号>/<任务名>/`，用来取评分记录 |
| `--tasks` | 任务定义（`<任务名>.json`），用来判断分数越高越好还是越低越好；不给则按越高越好 |
| `--data` | 数据目录。写出 `rows/`（训练行）、`work/`（中间结果）、`labels.jsonl`（标签） |
| `--only` | 只处理编号里含这段文字的 rollout（任务名或运行编号），调试用 |

Kimi 地址从环境变量 `KIMI_URLS`（空格分隔，以 `/v1` 结尾）读取，没有则读 `/personal/kimi.env`。集群上放在 tmux 里跑。

标注结果可以反复使用，重跑同一条命令时只问还没问过的：
- 中断后重跑，已完成的跳过。
- 新增 rollout，只标新增的。
- 改了某个 pattern，只重标这一个，其他 pattern 的标签原样保留。
- 每次重跑前的 `labels.jsonl` 会另存为 `labels.before-<时间>.jsonl`。

### pattern 的定义

一个 pattern 一个文件：`patterns/<名字>.yaml`。Kimi 每次只被问“一个位置 + 一个 pattern”，所以 pattern 之间互不影响。

| 字段 | 含义 |
|---|---|
| `places` | 在哪类位置上找这个 pattern（`start`、`submit`、`result`、`end` 由规则定位，其余由 Kimi 通读报出） |
| `questions` | 是非题，`expect` 是“算这个 pattern”所需的答案 |
| `steps` | 要 Kimi 指出的关键步，按先后顺序；`run` 表示这一步必须执行了命令，`submit` 表示必须是提交，`inside` 表示必须落在标出的范围内 |
| `notes` | 给 Kimi 的补充说明 |

所有答案都符合 `expect`、关键步齐全且顺序正确，才打标签；这由程序判断，不由 Kimi 判断。
改 pattern 就是改这个文件；新增 pattern 就是加一个文件。`patterns/common.md` 是所有 pattern 共用的说明，改它会让全部重标。

## 选数据

    python3 data_pipeline/select_rows.py \
        --policy configs/data_policies/pattern_all.yaml \
        --rows <数据目录>/rows \
        --labels <数据目录>/labels-<指纹>.jsonl [更多标签文件 ...] \
        --output <清单>.json

| 策略文件 | 含义 |
|---|---|
| `pattern_all.yaml` | 实验组：12 个 pattern 任何一个标到的步算 loss |
| `control_random.yaml` | 对照组：同一批 rollout，每条随机抽同样多的步 |
| `all_turns.yaml` | 普通 SFT：所有步都算 loss，不需要标签 |

| 策略项 | 含义 |
|---|---|
| `arm` | `labelled` 标到的步 / `random_matched` 随机抽同样多 / `all` 全部 |
| `patterns` | 只用其中几个 pattern，如 `[C1, D3]`；空为全部 |
| `seed` | 随机抽样的种子 |

`--exclude` 用来叠加修正：文件里每行 `{"trajectory_id": ..., "steps": [...]}`，这些步即使被标到也不选。

清单按文件和字节位置指向训练行，不复制数据。没选中的步不算 loss，但仍在后面步的历史里。

## 标注内部做了什么

`annotate.py` 依次做下面几件事，结果都在 `--data` 里：

1. **导出**（`rows/<rollout>.jsonl`）：模型的每次回复一行，含它看到的对话、它的回复（思考、文字、工具调用）和工具定义。所有回复都保留。
2. **渲染**（`work/<rollout>/transcript.txt`）：把这些行排成带步号的日志，第 N 步就是第 N 行；同时用规则找出提交、出结果、结尾这些位置。
3. **通读**（`work/<rollout>/<名字>.json`）：Kimi 读整条日志，报出规则找不到的位置，比如意外结果、同一方向第二次失败、被搁置的想法。
4. **标注**（`work/<rollout>/labels/<pattern>-<指纹>/<位置>.json`）：每个“位置 + pattern”一个请求，Kimi 回答是非题并填关键步，程序决定是否打标签。
5. **汇总**（`labels.jsonl`）：按各 pattern 当前的定义，每个标签一条记录。

## 现状（2026-10-11）

- 导出、渲染和选数据在一条 Kimi rollout 上跑通（212 行）。Kimi 标注部分是从 EdgeBench 移植的
  （在那边用 7 条 rollout 对照 Opus 测过），在本仓库的入口下还在调试。
- 训练：`training/`、`jobs/`、`configs/recipes`、`configs/training` 是从 physicalagent-sft-pipeline
  （分支 `feat/flashnext-sft`，提交 `dc7854c`）原样复制的，还没去掉图像和机器人相关的部分，目前不能运行。
- 2026 年 9 月的 Kimi rollout 只能用来调流程，不能用来训练：那批数据里模型几乎不思考（见 EdgeBench `docs/kimi-serving.md`）。
- 还没做：按最终成绩筛选 rollout、验证集划分、对标注结果的抽查。
