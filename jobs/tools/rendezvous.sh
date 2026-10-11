#!/usr/bin/env bash
# 多节点 rendezvous:定出本 pod 的 node_rank 与 master IP,把三个赋值打到 stdout 供 eval。
# 诊断信息一律走 stderr,stdout 只有赋值,否则 eval 会吃到杂音。
#
#   eval "$(NODES=4 L=<日志目录> bash jobs/tools/rendezvous.sh)" || exit $?
#   之后可用:$NR(node_rank)、$MA(master IP)、$RDZV_SRC(ais | cpfs)
#
# 两条路:
#   ais  —— 调度器给了 NODE_RANK 与 MASTER_IP/MASTER_ADDR,直接用。
#   cpfs —— 拿不到就退回目录登记法:每个 pod 把自己 IP 写进 $L/nodes/<hostname>,
#           等齐 NODES 个,再按 hostname 排序定 rank、第一个当 master。
#           这条兜底是有由来的:robbyctl 默认把 master IP 写到 /personal/job_id_cache,
#           而 robbys7 的 pod 里 /personal 是本地空目录,worker 看不到 master 写的文件,
#           连着五次 1/N joined。现在 job yaml 的 job_id_cache_dir 已指到 CPFS,
#           ais 那条通常能走通,这里只是保险。
set -u
: "${NODES:?NODES 没设}" "${L:?L(日志目录)没设}"
WAIT_SECONDS=${WAIT_SECONDS:-900}   # 等齐所有 node 的上限,默认 15 分钟
H=$(hostname)

MYIP=$(ip -4 -o addr show eth0 2>/dev/null | awk '{print $4}' | cut -d/ -f1)
[ -n "$MYIP" ] || MYIP=$(hostname -I 2>/dev/null | awk '{print $1}')

if [ -n "${NODE_RANK:-}" ] && [ -n "${MASTER_IP:-${MASTER_ADDR:-}}" ] ; then
  # 调度器已经给了 rank 与 master,本机 IP 只用于回显,取不到也不拦。
  NR=$NODE_RANK ; MA=${MASTER_IP:-$MASTER_ADDR} ; SRC=ais
else
  # 只有走目录登记法时才真的需要本机 IP —— 它就是要写给别人看的那个地址。
  [ -n "$MYIP" ] || { echo "rendezvous: 拿不到本机 IP,无法用目录登记法" >&2 ; exit 4 ; }
  mkdir -p "$L/nodes" ; echo "$MYIP" > "$L/nodes/$H"
  n=0
  for _ in $(seq 1 $((WAIT_SECONDS / 5))) ; do
    n=$(ls "$L/nodes" | wc -l)
    [ "$n" -ge "$NODES" ] && break
    sleep 5
  done
  [ "$n" -ge "$NODES" ] || {
    echo "rendezvous: $((WAIT_SECONDS / 60)) 分钟只登记了 $n/$NODES 个 node:$(ls "$L/nodes" | tr '\n' ' ')" >&2
    exit 4 ; }
  NR=$(( $(ls "$L/nodes" | sort | grep -n "^$H\$" | cut -d: -f1) - 1 ))
  MA=$(cat "$L/nodes/$(ls "$L/nodes" | sort | head -1)")
  SRC=cpfs
fi

echo "rendezvous: node_rank=$NR hostname=$H ip=$MYIP master=$MA src=$SRC nodes=$(ls "$L/nodes" 2>/dev/null | sort | tr '\n' ' ')" >&2
printf 'NR=%s\nMA=%s\nRDZV_SRC=%s\n' "$NR" "$MA" "$SRC"
