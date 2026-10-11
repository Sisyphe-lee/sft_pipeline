"""Render an explicit, pinned 32-B300 experimental FlashNext job; never submit."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('model', 'manifest', 'manifest-sha', 'data-root', 'output', 'logs',
                'run-name', 'stack-root', 'hf-overlay', 'job-file'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    def git(*argv):
        return subprocess.check_output(['git', '-C', str(root), *argv], text=True).strip()
    if git('status', '--porcelain', '--untracked-files=no'):
        parser.error('Commit changes before rendering a pinned job')
    key = os.environ.get('WANDB_API_KEY')
    if not key:
        parser.error('WANDB_API_KEY must be supplied via the environment')
    branch, commit = git('branch', '--show-current'), git('rev-parse', 'HEAD')
    if commit != git('rev-parse', 'origin/' + branch):
        parser.error('Push this commit before rendering the job')
    env = dict(BASE=args.model, DM=args.manifest, DM_SHA=args.manifest_sha,
               DATA_ROOT=args.data_root, OUT=args.output, L=args.logs, RUN=args.run_name,
               SFT_STACK_ROOT=args.stack_root, SFT_HF_OVERLAY=args.hf_overlay,
               CODE_COMMIT=commit, WANDB_API_KEY=key)
    command = '\n'.join('export ' + k + '=' + shlex.quote(v) for k, v in env.items())
    command += '\nbash jobs/tools/run_flashnext_sft.sh\n'
    config = dict(
        k8s_app_name='robbys7', k8s_priority='high', gpu_type='b300', node_num=4,
        communication_network_card='eth0', gpu_per_node=8, cpu_per_gpu=12,
        memory_per_gpu='140G', disk_memory='200G', shared_memory='100G',
        data_stores=dict(store0=dict(mount_point='/robby/share/s7',
                                    store_name='et210-aidc-robby-cpfs', sub_path='/')),
        repo_url='https://code.alipay.com/lingcong.hyb/physicalagent-sft-pipeline',
        branch=branch, commit_id=commit,
        image='acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/aii/aistudio:19900157-20260915210256',
        runtime='pytorch', job_id_cache_dir='/robby/share/s7/lingcong.hyb/.robbyctl_job_cache',
        job_name=args.run_name.replace('_', '-'), tag='type=train,model=flashnext,experimental=true,nodes=4',
        command=command,
    )
    path = Path(args.job_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as output:
        yaml.safe_dump(config, output, sort_keys=False)
    print(f'{path}: 32 B300, {branch}@{commit}')


if __name__ == '__main__':
    main()
