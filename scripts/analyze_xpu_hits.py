import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


_ID_PATTERN = re.compile(r"id=(xpu_[^):\s]+)")


def _extract_ids_from_text(text: str) -> List[str]:
    ids = _ID_PATTERN.findall(text)
    # deduplicate while preserving order
    seen = set()
    result: List[str] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            result.append(i)
    return result


def _parse_repo_revision(path: Path) -> Dict[str, Any]:
    name = path.name
    repo = None
    rev = None
    if name.endswith(".jsonl") and "@" in name:
        base = name[: -len(".jsonl")]
        try:
            repo_part, rev = base.rsplit("@", 1)
            repo = repo_part.replace("__", "/")
        except ValueError:
            pass
    return {"repository": repo, "revision": rev}


def extract_xpu_hits_from_trajectory(path: Path) -> Dict[str, Any]:
    xpu_ids: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("node") != "agent":
                continue
            messages = obj.get("messages") or []
            for msg in messages:
                # 当前 EnvBench 里，msg 是 MessageInfo 结构：
                # {"message_type": ..., "message_content": {"content": ...}, ...}
                if not isinstance(msg, dict):
                    continue
                content = None
                mc = msg.get("message_content")
                if isinstance(mc, dict):
                    content = mc.get("content")
                # 兜底：如果结构有变化，仍然遍历所有值找字符串
                if isinstance(content, str):
                    texts = [content]
                else:
                    texts = []
                    for v in msg.values():
                        if isinstance(v, str):
                            texts.append(v)
                for text in texts:
                    if "Candidate Fixes from XPU" in text:
                        ids = _extract_ids_from_text(text)
                        if ids:
                            xpu_ids.extend(ids)
            # assume candidate block at most once per trajectory
            if xpu_ids:
                break

    # deduplicate again across messages
    seen = set()
    unique_ids: List[str] = []
    for i in xpu_ids:
        if i not in seen:
            seen.add(i)
            unique_ids.append(i)

    info = _parse_repo_revision(path)
    info.update({"file": str(path), "xpu_ids": unique_ids})
    return info


def iter_trajectory_files(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    if root.is_dir():
        return sorted(root.glob("*.jsonl"))
    raise FileNotFoundError(str(root))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze EnvBench trajectories and report which XPU ids were included in the prompt toolbox.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="tmp/single_repo_traj",
        help="Path to a trajectory JSONL file or a directory containing such files (default: tmp/single_repo_traj)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Optional output JSONL file. If omitted, results are printed to stdout, one JSON object per line.",
    )

    args = parser.parse_args()
    root = Path(args.path)
    files = iter_trajectory_files(root)

    results: List[Dict[str, Any]] = []
    for p in files:
        info = extract_xpu_hits_from_trajectory(p)
        results.append(info)

    if args.output:
        out_path = Path(args.output)
        with out_path.open("w", encoding="utf-8") as out_f:
            for r in results:
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        for r in results:
            print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
