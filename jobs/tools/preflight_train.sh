#!/usr/bin/env bash
# 训练发射前的校验门。任何一条不过就非零退出,别带着错配置起 4 node × 8 卡。
# 每一条都对应踩过的坑,不要因为"看着啰嗦"删掉。
#
# 入参全部走环境变量,由 job yaml 的第 1 段定义:
#   DS      可选。单一 source 的旧 manifest 才使用
#   DS_SHA  可选。给了 DS 时必须同时给出
#   DM      selection manifest
#   DM_SHA  manifest 的 sha256
#   DATA_ROOT 可选。覆盖 manifest 的 base_root；batch pod 与交互容器挂载前缀不同时使用
#   RECIPE  训练配方 yaml
#   BASE    基座目录
#   REPO    训练代码的 checkout
#   OUT     输出目录,必须不存在
# 用法:bash jobs/tools/preflight_train.sh || exit $?
set -u
PY=${PY:-/usr/bin/python3}
fail() { echo "preflight: $1" >&2 ; exit "${2:-2}" ; }

for v in DM DM_SHA RECIPE BASE REPO OUT ; do
  [ -n "${!v:-}" ] || fail "环境变量 $v 没设"
done

# --- 数据 ---
# pod 的挂载名与交互式容器的 /personal 不同,存在性与可读性都实测,不靠推断。
if [ -n "${DS:-}" ] ; then
  ls -d "$DS" >/dev/null 2>&1 || fail "数据集在本机不可见:$DS"
  [ -n "${DS_SHA:-}" ] || fail "给了 DS 就要给 DS_SHA"
  GOT=$(sha256sum "$DS" | cut -d' ' -f1)
  echo "preflight: dataset sha $GOT"
  [ "$GOT" = "$DS_SHA" ] || fail "数据集 sha 不符,期望 $DS_SHA"
fi

# --- manifest ---
ls -d "$DM" >/dev/null 2>&1 || fail "manifest 在本机不可见:$DM"
DM="$DM" "$PY" -c "
import json,os,sys
m=json.load(open(os.environ['DM']))
sys.exit(0 if m.get('version')==2 and m.get('train') else 9)" || fail "manifest 不是 v2 选择清单或没有 train 行"
GOTM=$(sha256sum "$DM" | cut -d' ' -f1)
echo "preflight: manifest sha $GOTM"
[ "$GOTM" = "$DM_SHA" ] || fail "manifest sha 不符,期望 $DM_SHA"

# Trust source offsets as the training loader does; do not rescan source files.
DM="$DM" DATA_ROOT="${DATA_ROOT:-}" "$PY" -c "
import json,os,sys
manifest=json.load(open(os.environ['DM']))
root=os.environ.get('DATA_ROOT') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(os.environ['DM'])),manifest['base_root']))
first=os.path.join(root,manifest['sources'][0])
row=json.loads(open(first).readline())
img=os.path.join(os.path.dirname(first),row['images'][0])
print('preflight: 首图',img)
sys.exit(0 if os.path.isfile(img) else 9)" || fail "首图不可见"

# --- 配方与基座 ---
ls "$RECIPE" >/dev/null 2>&1 || fail "recipe 不存在 $RECIPE"
ls -d "$BASE" >/dev/null 2>&1 || fail "基座不存在 $BASE"
BASE="$BASE" EXPECTED_MODEL_TYPE="${EXPECTED_MODEL_TYPE:-qwen3_5}" "$PY" -c "
import json,os,sys
c=json.load(open(os.environ['BASE']+'/config.json'))
sys.exit(0 if c['model_type']==os.environ['EXPECTED_MODEL_TYPE'] else 9)" || fail "基座 model_type 与指定训练架构不符"

# --- 训练代码的血统 ---
# recipe 里 deterministic: true 只有配上这两处修复才成立;repo 副本过旧会安静地跑出
# 不可复现的权重,所以宁可拒绝开训。
ls "$REPO/training/deterministic.py" >/dev/null 2>&1 || fail "$REPO/training/deterministic.py 不存在"
grep -q _patch_causal_conv1d_torch_fallback "$REPO/training/deterministic.py" \
  || fail "deterministic.py 不是 torch-conv 版(parity 血统),拒绝开训"
grep -q _hf_set_seed "$REPO/training/train_sft.py" \
  || fail "train_sft.py 缺 peft 前置 seed 修复(repo 副本过旧)"

# --- 输出目录必须是新的,别覆盖已有 checkpoint ---
[ -e "$OUT" ] && fail "refusing non-fresh output: $OUT" 3

"$PY" -c "import torch;print('preflight: cuda avail',torch.cuda.is_available(),torch.cuda.device_count())"
echo "preflight: 全部通过"
