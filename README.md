# sft-pipeline

“Learn from Experience” 项目的数据清洗和训练。输入是 EdgeBench 上源模型（Kimi-K3 + Codex）rollout 的抓包，
输出是训练清单：哪些步算 loss。rollout 的采集、桥接抓包和 checkpoint 的评测在 EdgeBench 仓库。

## 三步

| | 做什么 | 入口 | 配置 |
|---|---|---|---|
| 标注 | 抓包 → 训练行 → 量长度 → Kimi 标注 → 标签文件。离线做一次 | `data_pipeline/annotate.py` | `configs/labelling/*.yaml` |
| 选数据 | 标签 + 策略 → 训练清单。不调用模型，几秒完成，可反复换策略 | `data_pipeline/select_rows.py` | `configs/data_policies/*.yaml` |
| 训练 | 按清单和 recipe 训练 | `training/train_sft.py`（见“现状”） | `configs/recipes/*.yaml` |

## 标注

    python3 data_pipeline/annotate.py \
        --config configs/labelling/kimi_v1.yaml \
        --capture <抓包目录> [<抓包目录> ...] \
        --runs <harness 的 runs 目录> \
        --tasks <EdgeBench 任务定义目录> \
        --student <学生模型目录> \
        --data <数据目录>

| 参数 | 含义 |
|---|---|
| `--config` | 标注设置：标哪些 pattern、Kimi 通读几遍、每个地址的并发数 |
| `--capture` | 桥接的抓包目录（里面是 `resp_<id>/`），每个桥接一个，可以给多个 |
| `--runs` | harness 的运行目录，结构为 `<runs>/<rollout 编号>/<任务名>/`，用来取评分记录 |
| `--tasks` | 任务定义（`<任务名>.json`），用来判断分数越高越好还是越低越好；不给则按越高越好。集群上是 `/personal/edgebench_runtime/dataset/tasks` |
| `--student` | 学生模型目录（只读 tokenizer 和 `chat_template.jinja`）。给了就量出每行在学生模型下的长度，并找出训练器不收的行 |
| `--data` | 数据目录。写出 `rows/`（训练行）、`tokens/`（长度）、`work/`（中间结果）、`labels.jsonl`（标签） |
| `--only` | 只处理编号里含这段文字的 rollout（任务名或运行编号），调试用 |

Kimi 地址从环境变量 `KIMI_URLS`（空格分隔，以 `/v1` 结尾）读取，没有则读 `/personal/kimi.env`。集群上放在 tmux 里跑。
多条 rollout 放在一条命令里一起跑，才能用满并发。

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
| `steps` | 要 Kimi 指出的关键步，按先后顺序。`run`：这一步必须执行了命令；`submit`：必须是提交；`within: N` / `apart: N`：离前一个关键步最多 / 至少 N 步 |
| `span` | 标签落在哪些步上。`from`/`to`/`max`：两个关键步之间的所有步，超过 `max` 步就不打标签；或 `steps`：只标列出的关键步 |
| `notes` | 给 Kimi 的补充说明 |

所有答案都符合 `expect`、关键步齐全且符合上面的规则，才打标签；这由程序判断，不由 Kimi 判断。

改 pattern 就是改这个文件，新增 pattern 就是加一个文件：
- 改 Kimi 读到的内容（`title`、`questions`、`steps` 的名字、文字、顺序和 `run`、`notes`）：只重标这一个 pattern。
- 改程序规则（`span`、`within`、`apart`、`submit`、`places`）：已有回答不重问，重跑时按新规则重新算标签。
- 改 `patterns/common.md`（所有 pattern 共用的说明）：全部重标，尽量不动。

## 选数据

    python3 data_pipeline/select_rows.py \
        --policy configs/data_policies/pattern_all.yaml \
        --rows <数据目录>/rows \
        --labels <数据目录>/labels.jsonl [更多标签文件 ...] \
        --output <清单>.json

| 策略文件 | 含义 |
|---|---|
| `pattern_all.yaml` | 实验组：12 个 pattern 任何一个标到的步算 loss |
| `control_random.yaml` | 对照组：同一批 rollout，每条随机抽同样多的步 |
| `all_turns.yaml` | 普通 SFT：所有可用的步都算 loss，不需要标签 |

| 策略项 | 含义 |
|---|---|
| `arm` | `labelled` 标到的步 / `random_matched` 随机抽同样多 / `all` 全部 |
| `patterns` | 只用其中几个 pattern，如 `[C1, D3]`；空为全部 |
| `seed` | 随机抽样的种子 |
| `require_reasoning` | 回复里没有思考文字的步不选（默认 true；训练器也收不了这种步） |
| `max_tokens` | 超过这个长度的步不选，要和训练 recipe 的 `max_input_tokens` 一致（默认 196608） |

三种策略都只在“可用的步”里选：是普通的一步（不是上下文压缩的请求）、回复带思考、训练器能收、长度不超限。
后两条用的是标注时量出的长度（`<数据目录>/tokens`，自动读取；`--tokens` 可另指定），没量过则不检查并在输出里注明。
清单的 `selection` 里记着每条 rollout 有多少步可用、多少步因为什么不可用。

`--exclude` 用来叠加修正：文件里每行 `{"trajectory_id": ..., "steps": [...]}`，这些步即使被标到也不选。

清单按文件和字节位置指向训练行，不复制数据。没选中的步不算 loss，但仍在后面步的历史里。

## 训练前检查

    python3 training/tools/check_collate.py --model <学生模型目录> --manifest <清单>.json --rows 20

把清单里的行送进训练器自己的代码（读清单、按位置取行、渲染和切出算 loss 的部分），打印每行的长度和
算 loss 的两段（思考、回复）的开头。不加载权重，不需要 GPU。换数据或改导出格式后先跑这个。

## 标注内部做了什么

`annotate.py` 依次做下面几件事，结果都在 `--data` 里：

1. **导出**（`rows/<rollout>.jsonl`）：模型的每次回复一行，含它看到的对话、它的回复（思考、文字、工具调用）和工具定义。
   所有回复都保留。消息按学生模型的对话模板能接受的形式存：开头的两条 system 消息并成一条，工具调用的参数存成对象。
2. **渲染**（`work/<rollout>/transcript.txt`）：把这些行排成带步号的日志，第 N 步就是第 N 行；同时用规则找出提交、出结果、结尾这些位置。
3. **量长度**（`tokens/<rollout>.jsonl`，需要 `--student`）：用学生模型的模板和 tokenizer 按训练器的方式渲染每一行，记下 token 数和训练器会拒收的行。
4. **通读**（`work/<rollout>/<名字>.json`）：Kimi 读整条日志，报出规则找不到的位置，比如意外结果、同一方向第二次失败、被搁置的想法。
5. **标注**（`work/<rollout>/labels/<pattern>-<指纹>/<位置>.json`）：每个“位置 + pattern”一个请求，Kimi 回答是非题并填关键步，程序决定是否打标签。
6. **汇总**（`labels.jsonl`）：按各 pattern 当前的定义，数据目录里所有已标 rollout 的标签，每个标签一条记录。

## 采集 rollout 时要满足的条件（在 EdgeBench 里设置）

- 桥接用 2026-10-11 之后的版本（保留历史里的思考文字）。之前的抓包里模型几乎不思考，不能用来训练。
- 运行时设 `SFORGE_RAW_CAPTURE_ROOT`：抓包才会带 rollout 编号，否则导出认不出是哪个任务。
- 实验文件里用 `extra_env: {CODEX_AUTO_COMPACT_TOKEN_LIMIT: "<N>"}` 限制 Kimi 的上下文：同样的内容学生模型的
  token 数是 Kimi 的 1.03–1.4 倍，不限制的话长 rollout 后段的步会超过训练上限而用不上。

## 现状

训练部分（`training/`、`jobs/`、`configs/recipes`、`configs/training`）是从
physicalagent-sft-pipeline（分支 `feat/flashnext-sft`，提交 `dc7854c`）复制的，只把读数据和 collate 这一段接通并验证了；
模型加载和训练循环还没在本仓库跑过。还没做：按最终成绩筛选 rollout、验证集划分。
