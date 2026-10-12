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
| `--config` | 标注设置：标哪些 pattern、Kimi 通读几遍、每个地址的并发数、一次请求问多少、思考档位。`kimi_v1.yaml` 标全部 12 个；`kimi_v1_four.yaml` 只标 A1、A4、D4、L1，时间约为三分之一 |
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
- 换一份配置（多标几个 pattern、多读一遍），只补缺的通读和没问过的问题。已经读过的那几遍不会因为改了 `read_request` 重读，
  要重读就删掉 `work/<rollout>/<名字>.json`。
- 每次重跑前的 `labels.jsonl` 会另存为 `labels.before-<时间>.jsonl`。

### pattern 的定义

一个 pattern 一个文件：`patterns/<名字>.yaml`。一次请求问相邻的几个位置和它们各自的 pattern（同一次提交的“提交前”和“出分后”总在一起），
答案按“位置 + pattern”分开存，所以 pattern 之间互不影响。

| 字段 | 含义 |
|---|---|
| `places` | 在哪类位置上找这个 pattern（`start`、`submit`、`result`、`end` 由规则定位，其余由 Kimi 通读报出）。通读在某次出分处报了一个位置时，这次出分也算那一类位置 |
| `skip` | 位置的事实符合时不问，如 B1 的 `mark: [new_best]`：刚刷新最高分，谈不上“回到最佳版本” |
| `effort` | 这个 pattern 单独用哪个思考档位问（`low`/`high`/`max`）；不写则用配置里的。B1 要核对文件到底恢复成了什么，用 `high` |
| `questions` | 是非题，`expect` 是“算这个 pattern”所需的答案 |
| `steps` | 要 Kimi 指出的关键步，按先后顺序。`run`：这一步必须执行了命令；`submit`：必须是提交；`within: N` / `apart: N`：离前一个关键步最多 / 至少 N 步 |
| `span` | 标签落在哪些步上。`from`/`to`/`max`：两个关键步之间的所有步，超过 `max` 步就不打标签；或 `steps`：只标列出的关键步 |
| `notes` | 给 Kimi 的补充说明 |

所有答案都符合 `expect`、关键步齐全且符合上面的规则，才打标签；这由程序判断，不由 Kimi 判断。

改 pattern 就是改这个文件，新增 pattern 就是加一个文件：
- 改 Kimi 读到的内容（`title`、`questions`、`steps` 的名字、文字、顺序和 `run`、`notes`）：只重标这一个 pattern。
- 改程序规则（`span`、`within`、`apart`、`submit`、`places`、`skip`、`effort`）：已有回答不重问，重跑时按新规则重新算标签。
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
4. **通读**（`work/<rollout>/<名字>.json`）：Kimi 读整条日志，报出规则找不到的位置。几遍通读同时进行：`timeline` 写出整条 rollout 的经过
   （判定时作为背景），并报同一方向第二次失败、开销大的动作、探测性提交、恢复旧版本；`ideas` 专找被搁置的想法和实验记录；`surprises` 专找意外结果。
5. **标注**（`work/<rollout>/labels/<pattern>-<指纹>/<位置>.json`）：相邻的位置合成一个请求，Kimi 对每个“位置 + pattern”回答是非题并填关键步，
   程序决定是否打标签。每个请求的思考过程、耗时和 token 数在 `work/<rollout>/requests/`。
6. **汇总**（`labels.jsonl`）：按各 pattern 当前的定义，数据目录里所有已标 rollout 的标签，每个标签一条记录。

## 采集 rollout 时要满足的条件（在 EdgeBench 里设置）

- 桥接用 2026-10-11 之后的版本（保留历史里的思考文字）。之前的抓包里模型几乎不思考，不能用来训练。
- 运行时设 `SFORGE_RAW_CAPTURE_ROOT`：抓包才会带 rollout 编号，否则导出认不出是哪个任务。
- 实验文件里用 `extra_env: {CODEX_AUTO_COMPACT_TOKEN_LIMIT: "140000"}` 限制 Kimi 的上下文：同样的内容学生模型的
  token 数是 Kimi 的 1.03–1.4 倍，140000 × 1.4 正好在训练上限 196608 以内；不限制的话 Codex 到约 245000 才压缩，
  长 rollout 后段的步会超过训练上限而用不上。

## 标注耗时

时间几乎全花在 Kimi 的输出（主要是思考）上，所以按“并发位分钟”算：一个并发位被占用的分钟数；4 个地址 × 2 并发时除以 8 就是实际时间。
下表是 4 条 4 小时的 rollout（212–542 步）平均每条，2026-10-12 实测：

| 配置 | 通读 | 判定 | 合计 | 8 并发成批标 |
|---|---|---|---|---|
| `kimi_v1.yaml`：12 个 pattern，三遍通读 | 9 | 16–21 | 25–30 | 3–4 分钟一条 |
| `kimi_v1_four.yaml`：A1、A4、D4、L1，只读 `timeline` 一遍 | 6 | 4 | 10 | 约 1.2 分钟一条 |

先用 `kimi_v1_four.yaml` 标，以后对同一个数据目录换 `kimi_v1.yaml` 重跑，只补另外两遍通读和没问过的问题。
单独标一条 40 分钟的 rollout（147 步）：前者 3.3 分钟，再补后者 1.7 分钟；时间主要是 `timeline` 那一遍通读（3–9 分钟）。
地址和别的任务共用，对方占用时会更慢。

最初是每个“位置 + pattern”单独一个请求、三遍通读都用 `high`，每条约 190。省在四处：
- 相邻位置合成一个请求（1149 个问题、241 次请求）：判定约 2.4 倍。
- 判定用 `low` 档思考：再约 3 倍。同一设置重问一遍，标签本身就有约两成出入；换成 `low` 的出入只略多于此，多出来和少掉的标签抽查下来对错比例相当。
  例外是 B1，`low` 档会多标出只恢复了一部分的情况，所以它单独用 `high`。
- 规则能排除的不问：C1 只在通读报了意外结果的出分处问（其余 111 次提问没有产生过一个标签），B1 不在刷新最高分处问。
- `ideas`、`surprises` 两遍通读用 `low` 档：通读从 19 降到 9。这两遍只给 C1、D2、D3 找位置，位置少了标签就少，不会标错：
  回答不变时标签从 165 个变成 155 个，少的是 C1（26 → 20）、D2（6 → 3）和 A3（7 → 6），其余 pattern 一个不变。
  `high` 档下一遍通读要想 1.5–6 万 token，和日志长短无关，有时超过上限被截断重来（`ideas` 那遍最常见），`low` 档没有这个问题。
  `timeline` 那遍保持 `high`：D4、C2、A3、B1 只靠它，它用 `low` 时这几类位置少四到七成。

试过没有采用的：让 Kimi 在第一个否定答案处停止（思考量不减）；三遍通读全用 `low`（D4 的位置 21 个只剩 13 个）。

## 现状

**标签质量**（2026-10-12 抽查 128 个标签，逐个对照日志；对 / 存疑 / 错）：

| pattern | 4 条 rollout 的标签数 | 抽查 | |
|---|---|---|---|
| A4 一次提交只改一处 | 71 | 16 / 5 / 0 | 可用 |
| D4 两次失败后离开 | 6 | 5 / 2 / 1 | 可用 |
| L1 提交前先本地测 | 13 | 5 / 7 / 1 | 可用，存疑的多是同一次提交的重试被重复标注 |
| A1 尽早首次提交 | 1 | 1 / 0 / 0 | 样本太少 |
| C1 意外结果先查原因 | 26 | 8 / 8 / 12 | 一半不可靠：Kimi 把事先的猜测当成检查得出的结论 |
| B1 回到最佳版本 | 26 | 4 / 5 / 8 | 一半不可靠：只恢复了一部分也被标上（抽查的多是 `low` 档标的；改用 `high` 后留下的是 3 / 3 / 3） |
| D2 重拾搁置的想法 | 6 | 2 / 5 / 2 | 一半不可靠：“停滞”判得太松 |
| A2、A3、C2 | 5、7、4 | 2/2/6、2/2/11、1/1/3 | 大多是错的 |
| D1、D3 | 0、0 | - | 标不出来 |

两次标注都标到的标签约一半是对的、两成是错的；只有一次标到的，对的不到四分之一。

这张表查的是标签对不对，不是 pattern 对训练有没有用；后者 12 个都还没验证。标得准的是位置由规则确定、问题能直接对照日志核实的
（A4、L1、A1，其中 A4 占这几个标签的近八成）；要 Kimi 自己判断“算不算意外”“查没查到原因”“算不算同一方向”的，它判得松。

上面 4 条是旧桥接下的 rollout，只有 3–7% 的步带思考文字。在唯一一条新桥接的 rollout（147 步、40 分钟）上又抽查了
三次标注给出的 23 个标签：6 / 6 / 11。C1 2/1/2，D4 3/3/3，D1 1/1/0，D2 0/0/4，A3、C2 各 1 个都错，L1 1 个存疑（这条提交很少，A4 没有样本）。
样本小，但日志里有了思考文字并没有让这些 pattern 明显好标，D4 也不如旧数据上稳：把相隔几十步的两次失败凑成“同一方向”，把发起尝试的那一步当成离开。

第一轮训练建议只用 A1、A4、L1 和 D4（策略文件里写 `patterns: [A1, A4, D4, L1]`，对照组写同样的列表）；D4 只占这四个标签的 7%，去留影响不大。

**训练**（学生模型 Qwen3.8-Flash-Next，177B 参数；代码从 physicalagent-sft-pipeline 分支 `feat/flashnext-sft` 复制后改过）：

2026-10-12 第一次训起来，数据是开发用的 1 条 rollout（112 行，1.1 万–5 万 token）。单机 8 卡（WebIDE）冻结专家
跑了 3 步；32 卡 Job 345586471（4 节点 × 8 B300）全参跑完 1 个 epoch 共 4 步并存下模型，四个节点退出码都是 0，
loss 0.723、0.785、0.713、0.517，第 1 步 296 秒（含编译），之后每步 82–149 秒，加载约 6 分钟。

    # WebIDE 上 8 卡调试（冻结 MoE 专家，只为走通路径和量显存）
    source jobs/tools/ide_env.sh
    PRINT_PEAK_MEM=1 python3 -m torch.distributed.run --standalone --nproc_per_node=8 training/train_sft.py \
        --recipe configs/recipes/flashnext_debug_8gpu.yaml --model <模型目录> --data-manifest <清单>.json --output <新目录>

    # 32 卡 Job：生成配置（代码要先提交并 push），dry-run 后提交
    python3 jobs/tools/make_flashnext_job.py --recipe <recipe 文件名> --model ... --manifest ... --manifest-sha ... \
        --data-root ... --output ... --logs ... --run-name ... --stack-root ... --hf-overlay ... --job-file jobs/generated/<名字>.yml
    LAUNCH_CONTAINER_MODE=dev_container robbyctl submit --dry-run jobs/generated/<名字>.yml

| recipe | 用途 |
|---|---|
| `qwen38_flashnext_sft_experimental.yaml` | 正式训练：32 卡，5 个 epoch |
| `flashnext_first_job_mn4.yaml` | 验证 32 卡能训起来：1 个 epoch，只在结束时存一份模型 |
| `flashnext_debug_8gpu.yaml` | 单机 8 卡调试：冻结 MoE 专家，跑 3 步 |

训练的是除两块之外的全部参数（125.7B，占 70.9%）：视觉塔冻结；n-gram embedding 冻结并放在主机内存里
（recipe 的 `resident_parameters`）。后者是一整张 51.2B 的表（bf16 下 95 GiB），ZeRO-3 每步都要把它整张聚合到
每张卡上，之前三次 32 卡 Job 第一步就 OOM 是这个原因。

实测显存（每卡，GiB，B300 共 268）：

| | 加载后常驻 | 一步的峰值 |
|---|---|---|
| 8 卡，冻结专家，4.5–5 万 token | 32.8 | 118–126 |
| 32 卡，全部专家训练，3.6–5 万 token，第 1 步 | 29.3（Adam 状态第一步后才分配，再加约 29） | 122 |

激活约占 85–93 GiB（5 万 token），随长度增加（每千 token 约 1.7 GiB，只有 4.2–5 万两档数据）。照这个比例
196608 的上限装不下，哪一段代码占的还没查。8 卡装不下全参的优化器状态，全参只能用 Job。

存盘：训练结束时 Trainer 总会存一个带优化器状态的 checkpoint，全参时约 4.6 TB（模型本身 354 GB）。正式 recipe
现在写的是每个 epoch 存一次、留 5 个，照这样会写出二十多 TB，开正式训练前要先改成只存模型。

还没做：长于 5 万 token 的行；训练出的模型能否被评测用的推理服务加载；按最终成绩筛选 rollout、验证集划分；
`training/` 里 4 个测试文件和 1 个测试用例依赖原仓库的选数据模块，在这里跑不了。
