#!/usr/bin/env python
import argparse
import pathlib
import sys
from typing import Any, Dict, Iterable, List

import jsonlines

# 确保可以导入仓库内模块
ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from env_setup_utils.data_sources.hf import HFDataSource  # noqa: E402


def take_first_n(it: Iterable[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    """从迭代器中取前 n 条样本。"""
    items: List[Dict[str, Any]] = []
    for idx, row in enumerate(it):
        if idx >= n:
            break
        # 确保是普通 dict，避免 Dataset 自定义类型
        items.append(dict(row))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 HF split 中截取前 N 条样本，写成本地 JSONL 供 LocalFileDataSource 使用。",
    )
    parser.add_argument(
        "--hub-name",
        default="JetBrains-Research/EnvBench",
        help="HF 数据集 hub 名称 (默认: JetBrains-Research/EnvBench)",
    )
    parser.add_argument(
        "--split",
        default="python_baseline_failure",
        help="HF split 名称 (默认: python_baseline_failure)",
    )
    parser.add_argument(
        "--config",
        default="splits",
        help="HF config 名称 (默认: splits)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=50,
        help="截取的样本数量 (默认: 50)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 JSONL 路径，默认写到 data/python_baseline_failure_topN.jsonl",
    )

    args = parser.parse_args()

    if args.n <= 0:
        raise ValueError("--n 必须为正整数")

    if args.output is None:
        out_name = f"python_baseline_failure_top{args.n}.jsonl"
        output_path = ROOT_DIR / "data" / out_name
    else:
        output_path = pathlib.Path(args.output)
        if not output_path.is_absolute():
            # 允许相对路径，默认相对项目根目录
            output_path = ROOT_DIR / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 使用项目里已经定义好的 HFDataSource，以保证样本结构与 inference 一致
    data_source = HFDataSource(
        hub_name=args.hub_name,
        configs=[args.config],
        split=args.split,
        cache_dir=None,
    )

    samples = take_first_n(data_source, args.n)

    with jsonlines.open(str(output_path), mode="w") as writer:
        for row in samples:
            writer.write(row)

    print(f"写出 {len(samples)} 条样本到 {output_path}")


if __name__ == "__main__":
    main()
