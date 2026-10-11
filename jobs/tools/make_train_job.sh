#!/usr/bin/env bash
# 套模板生成训练 job:一份 pod 的 yaml(jobs/templates/train_sft_mn4.yml.tmpl,多 node),
# 加一份能在 s3 dev container 直跑的脚本(jobs/templates/train_sft_s3.sh.tmpl,单机 8 卡)。
# 两份是同一个配方:s3 那份用 *_1node.yaml 的 recipe —— gbs 32 不变、gradient_accumulation_steps
# 1 → 4,步数 / lr schedule / 存点全一样,只是 32 条 decision 由 8 卡各累 4 个 micro-batch。
# 用不用 s3 那份另说,先都导出来。
#
# 用法:
#   bash jobs/tools/make_train_job.sh --model 27b                      # 50 task,10 epoch,4 node
#   bash jobs/tools/make_train_job.sh --model 4b --smoke               # 16 行 pipeline smoke,1 node,3 步
#   bash jobs/tools/make_train_job.sh --model 9b --dataset <id> --dm-sha <sha256> --recipe <file>
#   bash jobs/tools/make_train_job.sh --model 27b --stdout > /tmp/看一眼.yml
#
#   --model    4b | 9b | 27b。三个尺寸共用一份 recipe(hjy 的先例),只换基座
#   --dataset  数据集 id,默认 20260918_25175rows_50task1201unit_h40e76994
#   --dm-sha   该 manifest 的 sha256;preflight 会校
#   --recipe   configs/recipes/ 下的文件名,默认 pod 用 sft_<dataset>_mn4.yaml、s3 用 sft_<dataset>_1node.yaml;
#              显式给了就两份都用它(注意 gbs / 卡数的守卫)
#   --nodes    默认 4;--smoke 时 1。1 时 pod 也用 *_1node 的 recipe(accum 4),run 名带 _1node 而不是 _mn1
#   --variant  配方变体名(小写一段,如 turnuniform):recipe 默认名和 run 名末尾都加 _<变体>,
#              tag 里加 variant=<变体>。给对照组用,正式配方不带
#   --manifest pod 上 manifest 的整条路径。默认读取仓库内
#              $REPO/configs/datasets/<dataset>/train.manifest.json。
#              manifest 在 3D 盘上时给 /robby/share/s7/3D/... —— s3 那份脚本会自动换成 /robby/share/3D/...
#   --data-root pod 上 manifest source 的挂载根；默认 /robby/share/s7/3D，s3 自动换成 /robby/share/3D。
#   --out-root pod 上 checkpoint 的父目录(默认 $P/checkpoints);--log-root 同理(默认 $P/logs)。
#              落到 3D 就给 /robby/share/s7/3D/lingcong.hyb/harness
#   --ds / --ds-sha  单一 source 的老数据集才需要;多 source 数据直接按 manifest 读取
#   --smoke    pipeline smoke 预设:manifest 为数据集目录内 smoke.manifest.json、recipe smoke_pipeline_1node、1 node。
#              配 --dataset <id> --manifest <路径> --dm-sha <sha> 就烧别的 smoke manifest,run 名跟着换
#   --resume   续训:checkpoint 目录的绝对路径,或 latest(取 $OUT 下 global_step 最大的那个,
#              在生成时就解析好写进脚本)。run 名不变,preflight 放行已存在的 $OUT,W&B 续同一个 run
#   --branch   pod 从 AntCode clone 哪个分支;默认 = 生成时所在的 git 分支。不用为了换分支改模板
#   --commit   钉死 commit_id(默认空 = 取 origin/<branch> 的 HEAD)
#   WANDB_ENTITY / WANDB_PROJECT 环境变量可覆盖模板里的 lingcong-hyb / physicalagent-sft(别人用不用改文件)
#   --stdout   yaml 打到 stdout 不落盘(看一眼)
#   --out-dir  输出目录;默认 jobs/generated/(在 .gitignore 里,落盘但不 commit)
#
# W&B key 通过生成时的 WANDB_API_KEY 环境变量注入；生成产物不提交 Git。
#
# 提交就在本机,从 repo 根目录、干净的 checkout:
#   LAUNCH_CONTAINER_MODE=dev_container robbyctl submit --dry-run <yaml>
#   LAUNCH_CONTAINER_MODE=dev_container robbyctl submit -y <yaml>
set -euo pipefail

die() { echo "ABORT: $*" >&2 ; exit 2 ; }

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
TMPL=$ROOT/jobs/templates/train_sft_mn4.yml.tmpl
[ -f "$TMPL" ] || die "模板不在 $TMPL"

MODEL= ; DATASET=20260918_25175rows_50task1201unit_h40e76994 ; DATASET_ARG= ; DM_SHA= ; RECIPE= ; NODES=4
SMOKE=0 ; STDOUT=0 ; OUTDIR= ; RESUME= ; BRANCH= ; COMMIT= ; VARIANT=
DM_PATH= ; DM_PATH_ARG= ; DATA_ROOT= ; OUT_ROOT= ; LOG_ROOT= ; DS_PATH= ; DS_SHA_ARG=
POD_P=/robby/share/s7/lingcong.hyb/ppc_overfit_20260810
# 老数据集的单一 source(pod 视角);新数据集不给,数据门核对 manifest
known_ds() {
  case $1 in
    20260911_20646rows_h40e76994|20260916_2158rows_6task90unit_h40e76994|\
    smoke_pipeline16)
      echo "/robby/share/s7/lingcong.hyb/lcy_data/lcy_full_20260911/sft/lcy_full_20260911/train_le196608.jsonl f3499c080842e5289785ffb3be1e5567f27c70a6824a0872750484216c96fa35" ;;
    *) echo "" ;;
  esac
}
# pod 路径 → s3 交互式容器里的路径:同一块盘,两个等价点
to_s3() { printf '%s' "$1" | sed -e "s#^/robby/share/s7/3D#/robby/share/3D#" -e "s#^/robby/share/s7/lingcong.hyb#/personal#" ; }
while [ $# -gt 0 ]; do
  case $1 in
    --model)    MODEL=$2 ; shift 2 ;;
    --dataset)  DATASET=$2 ; DATASET_ARG=1 ; shift 2 ;;
    --dm-sha)   DM_SHA=$2 ; shift 2 ;;
    --recipe)   RECIPE=$2 ; shift 2 ;;
    --nodes)    NODES=$2 ; shift 2 ;;
    --smoke)    SMOKE=1 ; shift ;;
    --resume)   RESUME=$2 ; shift 2 ;;
    --branch)   BRANCH=$2 ; shift 2 ;;
    --commit)   COMMIT=$2 ; shift 2 ;;
    --variant)  VARIANT=$2 ; shift 2 ;;
    --manifest) DM_PATH=$2 ; DM_PATH_ARG=1 ; shift 2 ;;
    --data-root) DATA_ROOT=$2 ; shift 2 ;;
    --out-root) OUT_ROOT=$2 ; shift 2 ;;
    --log-root) LOG_ROOT=$2 ; shift 2 ;;
    --ds)       DS_PATH=$2 ; shift 2 ;;
    --ds-sha)   DS_SHA_ARG=$2 ; shift 2 ;;
    --stdout)   STDOUT=1 ; shift ;;
    --out-dir)  OUTDIR=$2 ; shift 2 ;;
    *) die "不认识的参数 $1(看文件头的用法)" ;;
  esac
done

case $MODEL in
  4b)  BASE=Qwen3.5-4B ;;
  9b)  BASE=base_models/Qwen3.5-9B ;;
  27b) BASE=base_models/Qwen3.8-27B ;;
  *)   die "--model 要是 4b | 9b | 27b,给的是 '${MODEL:-空}'" ;;
esac

# 已知 manifest 的 sha,免得每次手抄;新数据集用 --dm-sha 给。
# 用 case 不用 declare -A —— macOS 自带的 bash 3.2 没有关联数组,提交在本机跑。
known_sha() {
  case $1 in
    20260916_2158rows_6task90unit_h40e76994) echo 05d676ac95fc48b12580edeec70449dc2bd5667b8f29ed186ac1c22efe1875aa ;;
    smoke_pipeline16)                        echo 13dccef9cda7f55ca074c48b0f9bee4a31eaeb9f15dec8c2ac030d60f3dde498 ;;
    20260918_19016rows_32task1015unit_h40e76994)     echo 38280dd637265362fbb78e26692bc151947004c448cc5acd561dadc280e27462 ;;
    20260918_25175rows_50task1201unit_h40e76994)     echo d3941b95a259802b9405a2fb0e6c54073b047bdef69d3f48cbf7d0d60171d174 ;;
    *) echo "" ;;
  esac
}
if [ $SMOKE -eq 1 ]; then
  # 默认烧现成的 smoke_pipeline16;显式给了 --dataset 就用它 —— 换了数据就该换名字,
  # 不然 run 名和 tag 里写的还是别人的数据集,登记和翻日志时会对错。
  DM_NAME=$DATASET ; NODES=1
  if [ -z "$DM_PATH" ]; then
    DM_PATH='$REPO/configs/datasets/'"$DM_NAME/smoke.manifest.json"
    DM_PATH_ARG=1
    DM_SHA=${DM_SHA:-$(sha256sum "$ROOT/configs/datasets/$DM_NAME/smoke.manifest.json" | cut -d' ' -f1)}
  fi
  RECIPE_POD=${RECIPE:-smoke_pipeline_1node.yaml} ; RECIPE_S3=$RECIPE_POD
  # smoke 会反复跑,run 名带时间戳,不然 preflight 会因为 $OUT 已存在拒开(refusing non-fresh output);
  # 正式 run 名字固定,撞了就是该拒。
  RUN_SHORT=${DM_NAME}_${MODEL}_1node_$(date +%m%d%H%M) ; RUN_SHORT_S3=$RUN_SHORT
  HEADLINE="#### pipeline smoke:${MODEL} 走通 manifest -> 训 3 步 -> 存 checkpoint -> 传 OSS -> W&B(1 node)"
else
  DM_NAME=$DATASET
  case ${VARIANT:-} in ""|[a-z0-9]*) ;; *) die "--variant 要是小写字母数字一段,给的是 '$VARIANT'" ;; esac
  V=${VARIANT:+_$VARIANT}
  TOPO=$([ "$NODES" -eq 1 ] && echo 1node || echo mn${NODES})
  RECIPE_POD=${RECIPE:-sft_${DATASET}_${TOPO}${V}.yaml} ; RECIPE_S3=${RECIPE:-sft_${DATASET}_1node${V}.yaml}
  RUN_SHORT=${DATASET}_${MODEL}_${TOPO}${V} ; RUN_SHORT_S3=${DATASET}_${MODEL}_1node${V}
  HEADLINE="#### ${MODEL} SFT,数据集 ${DATASET}"
fi
DM_SHA=${DM_SHA:-$(known_sha "$DM_NAME")}
DM_PATH=${DM_PATH:-\$REPO/configs/datasets/$DM_NAME/train.manifest.json}
DATA_ROOT=${DATA_ROOT:-/robby/share/s7/3D}
OUT_ROOT=${OUT_ROOT:-$POD_P/checkpoints}
LOG_ROOT=${LOG_ROOT:-$POD_P/logs}
# known_ds 是按 dataset id 查老数据集的单一 source。--manifest 指到别处时这张表就不作数了:
# 那是另一份 manifest,它的源文件由自己的 source_digests 说了算,套上别人的 DS 只会白算一遍 sha。
if [ -z "$DS_PATH" ] && [ -z "$DS_SHA_ARG" ] && [ -z "$DM_PATH_ARG" ]; then
  read -r DS_PATH DS_SHA_ARG <<< "$(known_ds "$DM_NAME")"
fi
[ -n "$DM_SHA" ] || die "不知道 $DM_NAME 的 manifest sha,用 --dm-sha 给"
for r in "$RECIPE_POD" "$RECIPE_S3"; do [ -f "$ROOT/configs/recipes/$r" ] || die "recipe 不存在:configs/recipes/$r" ; done
RUN=sft_${RUN_SHORT} ; RUN_S3=sft_${RUN_SHORT_S3}
JOB_NAME=$(echo "$RUN" | tr '_' '-')
TAG="type=$([ $SMOKE -eq 1 ] && echo smoke || echo train),ckpt=${JOB_NAME},model=${MODEL},data=${DM_NAME},recipe=${RECIPE_POD%.yaml},cluster=robbys7,nodes=${NODES},${VARIANT:+variant=$VARIANT,}image=19900157,entry=recipe,code=repo_url"

# pod clone 哪个分支:默认就是现在所在的分支。pod 拉的是 origin 上的 HEAD,本地领先没 push 的话
# 跑的就不是你看到的代码 —— 这里核一下,不一致只警告不拦(diag / 看一眼的时候不必 push)。
BRANCH=${BRANCH:-$(git -C "$ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)}
if [ -z "$COMMIT" ]; then
  LOCAL=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || true)
  REMOTE=$(git -C "$ROOT" rev-parse "origin/$BRANCH" 2>/dev/null || true)
  if [ -z "$REMOTE" ]; then echo "WARN: origin/$BRANCH 本地没有记录,pod clone 时可能不存在这个分支" >&2
  elif [ "$LOCAL" != "$REMOTE" ]; then echo "WARN: 本地 HEAD ${LOCAL:0:8} != origin/$BRANCH ${REMOTE:0:8},pod 跑的是远端那份;先 push" >&2
  fi
  [ -n "$(git -C "$ROOT" status --porcelain 2>/dev/null)" ] && echo "WARN: 工作树有未提交改动,pod 看不到它们" >&2
fi
: "${WANDB_API_KEY:?生成 Job 前请设置 WANDB_API_KEY}"
WANDB_ENTITY=${WANDB_ENTITY:-lingcong-hyb} ; WANDB_PROJECT=${WANDB_PROJECT:-physicalagent-sft}

# 续训:latest 就在 $OUT 下找 global_step 最大的 checkpoint-N。这一步在本机跑不了(目录在 CPFS),
# 所以只在 s3 上生成时才解析;本机生成给 latest 会直接报错。
P_CPFS=/personal/ppc_overfit_20260810
RESUME_FROM=
if [ -n "$RESUME" ]; then
  if [ "$RESUME" = "latest" ]; then
    OUTDIR_CK=$P_CPFS/checkpoints/$RUN_S3
    [ -d "$OUTDIR_CK" ] || die "--resume latest 要在 s3 上跑(要看 $OUTDIR_CK)"
    RESUME_FROM=$(ls -d "$OUTDIR_CK"/checkpoint-* 2>/dev/null | sed -E 's/.*checkpoint-//' | sort -n | tail -1 | sed "s#^#$OUTDIR_CK/checkpoint-#")
    [ -n "$RESUME_FROM" ] && [ -f "$RESUME_FROM/trainer_state.json" ] || die "$OUTDIR_CK 下没有带 trainer_state.json 的 checkpoint"
  else
    RESUME_FROM=$RESUME
  fi
fi

# sed 的替换串里 & 表示"匹配到的原文"、| 是分隔符,值里有这两个字符就会替换错(W&B 就撞过)。
esc() { printf '%s' "$1" | sed -e 's/[&|\\]/\\&/g' ; }
# render <模板> <recipe> <run> <run_short> <nodes>
render() {
  sed -e "s|@HEADLINE@|$(esc "$HEADLINE(recipe $2)")|" -e "s|@MODEL@|$MODEL|g" -e "s|@BASE@|$BASE|g" \
      -e "s|@RUN@|$3|g" -e "s|@RUN_SHORT@|$4|g" -e "s|@JOB_NAME@|$JOB_NAME|g" \
      -e "s|@TAG@|$(esc "$TAG")|g" -e "s|@DM_NAME@|$DM_NAME|g" -e "s|@DM_SHA@|$DM_SHA|g" \
      -e "s|@RECIPE_FILE@|$2|g" -e "s|@NODES@|$5|g" -e "s|@GPUS@|8|g" -e "s|@WANDB_PROJECT@|$WANDB_PROJECT|g" \
      -e "s|@RESUME_FROM@|$RESUME_FROM|g" -e "s|@BRANCH@|$BRANCH|g" -e "s|@COMMIT_ID@|$COMMIT|g" \
      -e "s|@DM_PATH@|$(esc "$DM_PATH")|g" -e "s|@DM_PATH_S3@|$(esc "$(to_s3 "$DM_PATH")")|g" \
      -e "s|@DATA_ROOT@|$(esc "$DATA_ROOT")|g" -e "s|@DATA_ROOT_S3@|$(esc "$(to_s3 "$DATA_ROOT")")|g" \
      -e "s|@OUT_ROOT@|$(esc "$OUT_ROOT")|g" -e "s|@OUT_ROOT_S3@|$(esc "$(to_s3 "$OUT_ROOT")")|g" \
      -e "s|@LOG_ROOT@|$(esc "$LOG_ROOT")|g" -e "s|@LOG_ROOT_S3@|$(esc "$(to_s3 "$LOG_ROOT")")|g" \
      -e "s|@DS@|$(esc "$DS_PATH")|g" -e "s|@DS_S3@|$(esc "$(to_s3 "$DS_PATH")")|g" \
      -e "s|@DS_SHA@|$(esc "$DS_SHA_ARG")|g" \
      -e "s|@WANDB_API_KEY@|$(esc "$WANDB_API_KEY")|g" \
      -e "s|@WANDB_ENTITY@|$WANDB_ENTITY|g" \
      "$1"
}
TMPL_S3=$ROOT/jobs/templates/train_sft_s3.sh.tmpl
[ -f "$TMPL_S3" ] || die "模板不在 $TMPL_S3"
for t in "$TMPL:$RECIPE_POD:$RUN:$RUN_SHORT:$NODES" "$TMPL_S3:$RECIPE_S3:$RUN_S3:$RUN_SHORT_S3:1"; do
  IFS=: read -r tt rr rn rs nn <<< "$t"
  # grep 没匹配到会返回 1,pipefail 下整条算失败、set -e 直接退出 —— 恰恰是模板干净时触发。
  left=$(render "$tt" "$rr" "$rn" "$rs" "$nn" | { grep -o '@[A-Z_]*@' || true ; } | sort -u | tr '\n' ' ')
  [ -z "$left" ] || die "$(basename $tt) 里还有没替换的占位符:$left"
done

if [ $STDOUT -eq 1 ]; then render "$TMPL" "$RECIPE_POD" "$RUN" "$RUN_SHORT" "$NODES" ; exit 0 ; fi
OUTDIR=${OUTDIR:-$ROOT/jobs/generated} ; mkdir -p "$OUTDIR"
OUT=$OUTDIR/${RUN}_robbys7.yml ; OUT_S3=$OUTDIR/${RUN_S3}_s3.sh
render "$TMPL" "$RECIPE_POD" "$RUN" "$RUN_SHORT" "$NODES" > "$OUT"
render "$TMPL_S3" "$RECIPE_S3" "$RUN_S3" "$RUN_SHORT_S3" 1 > "$OUT_S3" ; chmod +x "$OUT_S3"

echo "生成两份(同一配方):"
echo "  pod : $OUT"
echo "        model=$MODEL base=\$P/$BASE nodes=$NODES recipe=$RECIPE_POD run=$RUN"
echo "  s3  : $OUT_S3"
echo "        单机 8 卡 recipe=$RECIPE_S3 run=$RUN_S3"
echo "  manifest=$DM_PATH(sha ${DM_SHA:0:8}…)  OUT=$OUT_ROOT/<run>  LOG=$LOG_ROOT/<run>"
echo "  data_root=$DATA_ROOT"
echo "  数据门:${DS_PATH:-manifest SHA 与首图可见性}"
echo "  manifest_name=$DM_NAME  pod clone: $BRANCH${COMMIT:+ @ $COMMIT}  W&B: $WANDB_ENTITY/$WANDB_PROJECT"
[ -n "$RESUME_FROM" ] && echo "  续训自 $RESUME_FROM(pod 那份也写了同一个路径,两边 /personal 与 /robby/share/s7 挂的是同一块盘)" || true
echo "提交 pod(本机,repo 根目录,干净 checkout):"
echo "  LAUNCH_CONTAINER_MODE=dev_container robbyctl submit --dry-run $OUT"
echo "  LAUNCH_CONTAINER_MODE=dev_container robbyctl submit -y $OUT"
echo "s3 直跑(先把 s3 的 repo_sft 拉到要跑的 commit,再在 s3 上):"
echo "  tmux new-session -d -s $RUN_S3 \"bash \$P/repo_sft/jobs/generated/$(basename $OUT_S3)\""
