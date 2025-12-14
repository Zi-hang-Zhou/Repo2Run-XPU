"""从 EnvBench 轨迹中抽取环境经验 XPU 的 MVP 脚本 (Repo2Run 适配版)。"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import requests
from dotenv import load_dotenv
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TRAJ_DIR = ROOT_DIR / "tmp" / "traj_py_subset_50_kimi"
DEFAULT_OUTPUT = ROOT_DIR / "xpuExtract" / "outputs" / "traj_xpu_mvp.jsonl"

# LLM 调用相关默认配置
DEFAULT_LLM_MODEL = os.environ.get("XPU_EXTRACT_MODEL", os.environ.get("MOONSHOT_MODEL", "gpt-4o-2024-05-13"))
DEFAULT_API_KEY_ENV = os.environ.get("XPU_EXTRACT_API_KEY_ENV", "OPENAI_API_KEY")
DEFAULT_BASE_URL_ENV = os.environ.get("XPU_EXTRACT_BASE_URL_ENV", "OPENAI_BASE_URL")
DEFAULT_TIMEOUT_SEC = int(os.environ.get("XPU_EXTRACT_TIMEOUT", "60"))

# 启发式关键词
ERROR_KEYWORDS = [
    "ModuleNotFoundError",
    "ImportError",
    "No module named",
    "cannot import name",
    "Could not find a version",
    "command not found",
    "Permission denied",
    "error:",
    "Error:",
    "Traceback",
    "failed with exit code"
]

ENV_CMD_KEYWORDS = [
    "pip install",
    "poetry install",
    "apt-get install",
    "conda install",
    "python setup.py"
]


def get_env_or_raise(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        # 降级尝试：如果是 Kimi 的key没找到，试试通用的
        if name == "MOONSHOT_API_KEY":
            val = os.environ.get("OPENAI_API_KEY")
    if not val:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


def openai_compatible_chat_completions(
    model: str,
    messages: List[Dict[str, str]],
    api_key: str,
    base_url: str,
    timeout_sec: int,
    response_format_json: bool = True,
) -> Dict[str, Any]:
    # 修正 URL 拼接逻辑
    if "v1" not in base_url and not base_url.endswith("/"):
        base_url += "/v1"
    url = base_url.rstrip("/") + "/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "stream": False,
    }
    if response_format_json:
        payload["response_format"] = {"type": "json_object"}
    
    # 打印调试信息（因为之前有配置错误）
    # print(f"DEBUG: Requesting {url} with model {model}")
    
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout_sec)
    if resp.status_code >= 400:
        raise RuntimeError(f"LLM HTTP {resp.status_code}: {resp.text[:500]}")
    return resp.json()


def parse_llm_json(s: str) -> Dict[str, Any]:
    s = s.strip()
    if s.startswith("\ufeff"):
        s = s.lstrip("\ufeff")
    if s.startswith("```"):
        if s.startswith("```json"):
            s = s[len("```json"):].strip()
        else:
            s = s[3:].strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    return json.loads(s)


def truncate(text: Any, max_len: int) -> str:
    if text is None:
        return ""
    text = str(text)
    if len(text) <= max_len:
        return text
    keep = max_len // 2
    return text[:keep] + "\n... [TRUNCATED] ...\n" + text[-keep:]


def load_llm_config_from_env() -> Dict[str, Any]:
    return {
        "llm_model": DEFAULT_LLM_MODEL,
        "api_key_env_var": DEFAULT_API_KEY_ENV,
        "base_url_env_var": DEFAULT_BASE_URL_ENV,
        "timeout_sec": DEFAULT_TIMEOUT_SEC,
        "llm_language": "zh",
    }


def iter_traj_files(traj_path: Path) -> List[Path]:
    if traj_path.is_file():
        return [traj_path]
    if traj_path.is_dir():
        return sorted([p for p in traj_path.glob("*.jsonl") if "@" in p.name])
    raise FileNotFoundError(str(traj_path))


def parse_repo_revision_from_name(path: Path) -> Tuple[str, str]:
    name = path.name
    if not (name.endswith(".jsonl") and "@" in name):
        return "unknown/repo", "unknown"
    base = name[: -len(".jsonl")]
    try:
        repo_part, rev = base.rsplit("@", 1)
    except ValueError:
        return base, "unknown"
    repo = repo_part.replace("__", "/")
    return repo, rev


def load_traj(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _iter_strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_strings(v)


def extract_commands_history(traj: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    [Repo2Run 适配版]
    从轨迹中提取命令。
    Repo2Run 的轨迹是 [{"role": "assistant", "content": "... ```bash\ncmd\n``` ..."}, ...]
    """
    cmds = []
    
    # 正则匹配 Markdown 代码块
    bash_pattern = re.compile(r"```bash\s+(.*?)\s+```", re.DOTALL)
    
    for item in traj:
        # 如果有 explicit nodes (EnvBench style)
        if item.get("node") == "commands_history":
            raw = item.get("commands") or []
            if isinstance(raw, list):
                return raw
        
        # Repo2Run style: Parse content
        content = item.get("content", "")
        role = item.get("role", "")
        
        # 只看 assistant 的输出
        if role == "assistant" and content:
            matches = bash_pattern.findall(content)
            for match in matches:
                # 简单的命令提取，假设每行是一个命令
                clean_cmd = match.strip()
                cmds.append({"command": clean_cmd, "exit_code": 0}) # 暂时假设为0，或者不重要
                
    return cmds


def heuristic_stats_for_traj(traj: List[Dict[str, Any]]) -> Dict[str, Any]:
    num_agent_steps = 0
    num_error_keywords = 0

    # 1. 扫描全文找错误
    for item in traj:
        # 兼容 role/node
        if item.get("role") == "assistant" or item.get("node") == "agent":
            num_agent_steps += 1
            
        for text in _iter_strings(item):
            t_low = text.lower()
            if any(kw.lower() in t_low for kw in ERROR_KEYWORDS):
                num_error_keywords += 1
                # break # 不要 break，统计所有

    # 2. 扫描命令
    cmds = extract_commands_history(traj)
    num_commands = len(cmds)
    num_env_commands = 0

    for c in cmds:
        cmd_str = str(c.get("command", ""))
        cmd_low = cmd_str.lower()
        if any(kw.lower() in cmd_low for kw in ENV_CMD_KEYWORDS):
            num_env_commands += 1

    return {
        "num_agent_steps": num_agent_steps,
        "num_commands": num_commands,
        "num_env_commands": num_env_commands,
        "num_error_keywords": num_error_keywords,
    }


def heuristic_is_candidate(stats: Dict[str, Any]) -> Tuple[bool, float]:
    """
    [Repo2Run 调整版] 放宽筛选条件。
    只要执行过命令，或者出现了错误关键词，就认为是候选。
    """
    score = 0.0
    
    if stats.get("num_env_commands", 0) >= 1:
        score += 5.0 # 大幅加分
    if stats.get("num_error_keywords", 0) >= 1:
        score += 5.0 # 大幅加分
    if stats.get("num_commands", 0) >= 1:
        score += 1.0

    # 只要有分就过
    return score > 0, score


def build_traj_prompt(
    repo: str,
    rev: str,
    traj: List[Dict[str, Any]],
    stats: Dict[str, Any],
    cfg: Dict[str, Any],
) -> List[Dict[str, str]]:
    cmds = extract_commands_history(traj)
    lines_cmds: List[str] = []
    for c in cmds:
        cmd_str = str(c.get("command", ""))
        lines_cmds.append(f"$ {cmd_str}")
    commands_text = truncate("\n".join(lines_cmds), 4000)

    error_lines: List[str] = []
    for item in traj:
        # 只提取 system (Observation) 里的错误
        if item.get("role") == "system":
            text = item.get("content", "")
            if any(kw.lower() in text.lower() for kw in ERROR_KEYWORDS):
                error_lines.append(text)
                
    if len(error_lines) > 30:
        error_lines = error_lines[:15] + ["... [TRUNCATED] ..."] + error_lines[-10:]
    errors_text = truncate("\n".join(error_lines), 4000)

    system_text = (
        "你是一名资深 Python 项目环境配置与依赖问题专家。"\
        "\n现在给你一个仓库在自动环境搭建时的完整 agent 轨迹（含执行的命令和报错日志）。"\
        "\n你的任务是判断：1）此案例是否值得提炼成可复用的环境经验 XPU；2）如果值得，请给出结构化的 XPU 条目。"\
        "\n回答必须是严格的 JSON 对象，不包含任何多余文字。"
    )

    user_payload: Dict[str, Any] = {
        "repository": repo,
        "revision": rev,
        "stats": stats,
        "commands_history_text": commands_text,
        "error_snippets_text": errors_text,
        "xpu_schema": {
            "id": "string，唯一标识，如 xpu_env_py_xxx",
            "context": {
                "lang": "例如 python",
                "os": ["相关操作系统，如 linux 等"],
                "python": ["相关 Python 版本前缀，如 3.8"],
                "tools": ["相关工具列表，如 pytest, pip 等"],
            },
            "signals": {
                "regex": ["匹配该错误的正则表达式"],
                "keywords": ["用于粗略检索的关键词"],
            },
            "advice_nl": ["1-5 条中文建议，解释问题根因和修复思路"],
            "atoms": [
                {
                    "name": "原子操作类型，如 pip_install / pip_pin / or_upgrade_pkg / set_env / set_umask 等",
                    "args": "一个字典，包含该原子需要的参数",
                }
            ],
        },
        "output_requirement": (
            "你必须输出一个 JSON 对象，形如：{decision, reason, xpu}。"\
            "decision 只能是 'skip' 或 'xpu'；当 decision='skip' 时，xpu 字段可以省略或为 null；"\
            "当 decision='xpu' 时，必须给出一个与 xpu_schema 兼容的 xpu 对象。所有说明性文字使用简体中文。"
        ),
        "language": cfg.get("llm_language", "zh"),
    }

    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def extract_xpu_from_trajs(
    traj_path: Path,
    output_jsonl: Path,
) -> None:
    load_dotenv()
    cfg = load_llm_config_from_env()

    api_key = get_env_or_raise(cfg["api_key_env_var"])
    # 默认 Base URL 处理
    base_url = os.environ.get(cfg["base_url_env_var"]) or "https://api.openai.com/v1"

    files = iter_traj_files(traj_path)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with output_jsonl.open("w", encoding="utf-8") as f_out:
        for path in tqdm(files, total=len(files), desc="Extracting XPU from trajs"):
            repo, rev = parse_repo_revision_from_name(path)
            traj = load_traj(path)
            stats = heuristic_stats_for_traj(traj)
            is_candidate, score = heuristic_is_candidate(stats)
            stats["heuristic_score"] = score
            stats["heuristic_is_candidate"] = is_candidate

            llm_decision: str = "heuristic_skip"
            llm_reason: str | None = None
            xpu_obj: Dict[str, Any] | None = None
            usage: Dict[str, Any] = {}
            error_info: str | None = None

            if is_candidate:
                try:
                    messages = build_traj_prompt(repo, rev, traj, stats, cfg)
                    raw = openai_compatible_chat_completions(
                        model=cfg["llm_model"],
                        messages=messages,
                        api_key=api_key,
                        base_url=base_url,
                        timeout_sec=cfg["timeout_sec"],
                        response_format_json=True,
                    )
                    content = raw["choices"][0]["message"]["content"]
                    usage = raw.get("usage") or {}
                    parsed = parse_llm_json(content)
                    llm_decision = str(parsed.get("decision") or "error")
                    llm_reason = parsed.get("reason")
                    if llm_decision == "xpu":
                        xpu_obj = parsed.get("xpu") or None
                    elif llm_decision not in {"skip", "xpu"}:
                        llm_decision = "error"
                        error_info = f"unexpected decision: {parsed!r}"
                except Exception as e:
                    llm_decision = "error"
                    error_info = str(e)

            out_obj = {
                "repository": repo,
                "revision": rev,
                "traj_path": str(path),
                "heuristics": stats,
                "llm_decision": llm_decision,
                "llm_reason": llm_reason,
                "xpu": xpu_obj,
                "llm_model": cfg.get("llm_model"),
                "usage": usage,
                "error": error_info,
            }
            f_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="从 EnvBench 轨迹中启发式筛选并通过 LLM 抽取 XPU 经验")
    parser.add_argument("--traj", type=Path, default=DEFAULT_TRAJ_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    extract_xpu_from_trajs(Path(args.traj), Path(args.output))


if __name__ == "__main__":
    main()