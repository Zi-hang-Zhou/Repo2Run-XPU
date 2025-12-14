#!/usr/bin/env python
import argparse
import os
import sys
from typing import Any, Dict, List

import jsonlines

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from env_setup_utils.process_trajectories_to_scripts import (  # noqa: E402
    parse_script_from_trajectory,
    parse_installamatic_trajectory,
)


def extract_scripts(traj_dir: str, output_path: str) -> str:
    """从包含多个 trajectory jsonl 的目录中批量抽取脚本并写出 scripts.jsonl。

    期望文件名格式: <repo_name_with__instead_of_/>@<revision>.jsonl
    例如: psf__requests@abcdef1234567890.jsonl
    """
    traj_dir = os.path.abspath(traj_dir)
    if not os.path.isdir(traj_dir):
        raise SystemExit(f"Trajectory directory does not exist: {traj_dir}")

    # 只处理名字里带 '@' 的 jsonl 文件，避免误读其它 jsonl（比如已有的 scripts.jsonl）
    # 排除 .llm.jsonl 和 .xpu.jsonl 文件
    traj_files = [
        f for f in os.listdir(traj_dir)
        if f.endswith(".jsonl") and "@" in f
        and not (f.endswith(".llm.jsonl") or f.endswith(".xpu.jsonl"))
    ]
    if not traj_files:
        raise SystemExit(f"No trajectory *.jsonl files with '@' found in: {traj_dir}")

    traj_files.sort()

    scripts: List[Dict[str, Any]] = []

    for fname in traj_files:
        traj_path = os.path.join(traj_dir, fname)
        name_no_ext = os.path.splitext(fname)[0]
        try:
            repository_part, revision = name_no_ext.split("@", 1)
        except ValueError as e:
            raise SystemExit(f"Unexpected trajectory filename format: {fname}") from e

        repository = repository_part.replace("__", "/")

        with jsonlines.open(traj_path, "r") as reader:
            trajectory = [line for line in reader]

        script = parse_script_from_trajectory(trajectory)
        if not script:
            # 回退到 Installamatic 风格解析，以防万一
            script = parse_installamatic_trajectory(trajectory)

        scripts.append(
            {
                "repository": repository,
                "revision": revision,
                "script": script,
            }
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with jsonlines.open(output_path, "w") as writer:
        for row in scripts:
            writer.write(row)

    print(f"Wrote {len(scripts)} scripts to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从包含多个 trajectory jsonl 的目录批量抽取脚本，生成 scripts.jsonl",
    )
    parser.add_argument(
        "traj_dir",
        type=str,
        help="包含多个 <repo__name@sha>.jsonl 轨迹文件的目录",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 scripts.jsonl 路径，默认写到 <traj_dir>/scripts.jsonl",
    )

    args = parser.parse_args()

    traj_dir = os.path.abspath(args.traj_dir)
    if args.output is None:
        output_path = os.path.join(traj_dir, "scripts.jsonl")
    else:
        output_path = os.path.abspath(args.output)

    extract_scripts(traj_dir, output_path)


if __name__ == "__main__":
    main()
