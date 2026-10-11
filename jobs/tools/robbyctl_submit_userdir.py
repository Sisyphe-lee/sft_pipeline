#!/usr/bin/env python3
"""robbyctl submit 的共享工作站包装:把 pypai 的打包目录从 /tmp/pypai/target 改到
自己家目录,不碰 site-packages 原文件、不碰别人的 /tmp/pypai。

背景:pypai 在 base_job.py 把 target_dir_name 写死为 /tmp/pypai/target;共享
工作站(vilab42)上那个目录属于别的用户、不可写,submit 报
`[Errno 13] Permission denied: '/tmp/pypai/target/<job>'`。
用法(与 robbyctl.cli 完全同参):
    LAUNCH_CONTAINER_MODE=dev_container PYTHONPATH=$HOME/rbctl \
      python3 jobs/tools/robbyctl_submit_userdir.py submit -y jobs/<yml>
"""
import os
import sys

_target = os.path.join(os.path.expanduser("~"), ".pypai_target")
os.makedirs(_target, exist_ok=True)
from pypai.job.execution_job import base_job  # noqa: E402
base_job.BaseJob.target_dir_name = _target if hasattr(base_job, "BaseJob") else _target
# 有的版本类名不同:把模块里所有定义了 target_dir_name 的类都改掉
for _name in dir(base_job):
    _obj = getattr(base_job, _name)
    if isinstance(_obj, type) and getattr(_obj, "target_dir_name", None) == "/tmp/pypai/target":
        _obj.target_dir_name = _target

from robbyctl.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
