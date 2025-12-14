import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


# 例如：
# [2025-11-19 16:23:18,214][root][INFO] - [psf/requests@7029... ] Downloading repository.
_RE_REPO_REV = re.compile(r"\[(?P<repo>[^@\]]+)@(?P<rev>[^\]]+)\]")

# 例如：
# [XPU] Selected candidates: ['xpu_4869205243', 'xpu_1199876338', 'xpu_2052583608']
_RE_XPU_LINE = re.compile(r"Selected candidates:\s*(\[.*\])")


def _iter_log_files(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    if root.is_dir():
        # 简单起见，遍历目录下所有文件
        return sorted(p for p in root.iterdir() if p.is_file())
    raise FileNotFoundError(str(root))


def _update_hits_from_log(path: Path, hits: Dict[Tuple[str, str], List[str]]) -> None:
    current_repo: str | None = None
    current_rev: str | None = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # 尝试从行中解析 repository@revision
            m_repo = _RE_REPO_REV.search(line)
            if m_repo:
                current_repo = m_repo.group("repo")
                current_rev = m_repo.group("rev")

            # 只关心包含 XPU 选择信息的行
            if "[XPU] Selected candidates:" not in line:
                continue

            m_xpu = _RE_XPU_LINE.search(line)
            if not m_xpu:
                continue

            if current_repo is None or current_rev is None:
                # 没有当前 repo@rev，上下文不完整，跳过
                continue

            list_text = m_xpu.group(1)
            try:
                parsed = ast.literal_eval(list_text)
            except Exception:
                # 解析失败就跳过本行
                continue

            if not isinstance(parsed, list):
                continue

            key = (current_repo, current_rev)
            dst = hits.setdefault(key, [])
            for x in parsed:
                if isinstance(x, str) and x.startswith("xpu_") and x not in dst:
                    dst.append(x)


def analyze_xpu_hits_from_log(root: Path) -> List[Dict[str, Any]]:
    hits: Dict[Tuple[str, str], List[str]] = {}

    for log_file in _iter_log_files(root):
        _update_hits_from_log(log_file, hits)

    results: List[Dict[str, Any]] = []
    for (repo, rev), ids in sorted(hits.items()):
        results.append(
            {
                "repository": repo,
                "revision": rev,
                "xpu_ids": ids,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse inference logs to extract which XPU ids were selected per repository.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="infer.log",
        help="Path to a log file or a directory of log files (default: infer.log)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Optional output JSONL file. If omitted, results are printed to stdout.",
    )

    args = parser.parse_args()
    root = Path(args.path)

    results = analyze_xpu_hits_from_log(root)

    if args.output:
        out_path = Path(args.output)
        with out_path.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        for r in results:
            print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
