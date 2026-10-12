"""Render an explicit, pinned 32-B300 FlashNext job (4 nodes x 8); never submit.

    python3 jobs/tools/make_flashnext_job.py --recipe flashnext_first_job_mn4.yaml --model ... --job-file jobs/generated/<run>.yml
    LAUNCH_CONTAINER_MODE=dev_container robbyctl submit --dry-run jobs/generated/<run>.yml     then: submit --yes

Paths are the pod's: the CPFS root is mounted at /robby/share/s7, so /robby/share/3D/... of a WebIDE container is
/robby/share/s7/3D/... here. WANDB_API_KEY is needed only when the recipe reports to wandb."""

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
    parser.add_argument('--recipe', default='qwen38_flashnext_sft_experimental.yaml', help='file name under configs/recipes')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    def git(*argv):
        return subprocess.check_output(['git', '-C', str(root), *argv], text=True).strip()
    if git('status', '--porcelain', '--untracked-files=no'):
        parser.error('Commit changes before rendering a pinned job')
    recipe = yaml.safe_load((root / 'configs/recipes' / args.recipe).read_text())
    key = os.environ.get('WANDB_API_KEY')
    if recipe.get('report_to') == 'wandb' and not key:
        parser.error('WANDB_API_KEY must be supplied via the environment')
    branch, commit = git('branch', '--show-current'), git('rev-parse', 'HEAD')
    if commit != git('rev-parse', 'origin/' + branch):
        parser.error('Push this commit before rendering the job')
    env = dict(BASE=args.model, DM=args.manifest, DM_SHA=args.manifest_sha,
               DATA_ROOT=args.data_root, OUT=args.output, L=args.logs, RUN=args.run_name,
               SFT_STACK_ROOT=args.stack_root, SFT_HF_OVERLAY=args.hf_overlay,
               CODE_COMMIT=commit, RECIPE_FILE=args.recipe,
               PRINT_PEAK_MEM='1')  # per-rank memory after setup and at every step, in the training log
    if key:
        env['WANDB_API_KEY'] = key
    command = '\n'.join('export ' + k + '=' + shlex.quote(v) for k, v in env.items())
    command += '\nbash jobs/tools/run_flashnext_sft.sh\n'
    config = dict(
        k8s_app_name='robbys7', k8s_priority='high', gpu_type='b300', node_num=4,
        communication_network_card='eth0', gpu_per_node=8, cpu_per_gpu=12,
        memory_per_gpu='140G', disk_memory='200G',
        shared_memory='200G',  # holds the 103 GB n-gram embedding that the ranks of a node share
        data_stores=dict(store0=dict(mount_point='/robby/share/s7',
                                    store_name='et210-aidc-robby-cpfs', sub_path='/')),
        repo_url='https://code.alipay.com/lcy554440/sft_pipeline',
        branch=branch, commit_id=commit,
        image='acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/aii/aistudio:19900157-20260915210256',
        runtime='pytorch', job_id_cache_dir='/robby/share/s7/3D/lcy/sft_pipeline/.robbyctl_job_cache',
        job_name=args.run_name.replace('_', '-'), tag=f'type=train,model=flashnext,recipe={args.recipe.removesuffix(".yaml")},nodes=4',
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
