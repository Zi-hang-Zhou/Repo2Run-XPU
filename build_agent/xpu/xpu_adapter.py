import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


@dataclass
class XpuAtom:
    name: str
    args: Dict[str, Any]


@dataclass
class XpuEntry:
    id: str
    context: Dict[str, Any]
    signals: Dict[str, Any]
    advice_nl: List[str]
    atoms: List[XpuAtom] = field(default_factory=list)
    telemetry: Dict[str, Any] = field(default_factory=lambda: {"hits": 0, "successes": 0, "failures": 0})

@dataclass
class XpuContext:
    lang: Optional[str] = None
    os: Optional[str] = None
    python: Optional[str] = None
    tools: Sequence[str] = tuple()


def _parse_xpu_line(obj: Dict[str, Any]) -> XpuEntry:
    atoms_raw = obj.get("atoms") or []
    atoms = [XpuAtom(name=a.get("name", ""), args=a.get("args", {})) for a in atoms_raw]
    return XpuEntry(
        id=obj.get("id", ""),
        context=obj.get("context", {}),
        signals=obj.get("signals", {}),
        advice_nl=list(obj.get("advice_nl") or []),
        atoms=atoms,
        telemetry=obj.get("telemetry", {"hits": 0, "successes": 0, "failures": 0})
    )


def load_xpu_entries(jsonl_path: Path) -> List[XpuEntry]:
    """Load all XPU entries from a JSONL file.

    The file is expected to have one JSON object per line, following the schema
    similar to exp/xpu_v0.jsonl.
    """
    entries: List[XpuEntry] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            entries.append(_parse_xpu_line(obj))
    return entries


def _match_regex(log_snippet: str, patterns: Iterable[str]) -> bool:
    for p in patterns:
        try:
            if re.search(p, log_snippet):
                return True
        except re.error:
            # ignore invalid patterns
            continue
    return False


def _keyword_score(log_snippet: str, keywords: Iterable[str]) -> int:
    """A simple keyword overlap score.

    Counts how many keywords appear as substrings in the log snippet.
    """
    text = log_snippet.lower()
    score = 0
    for kw in keywords:
        if not kw:
            continue
        if kw.lower() in text:
            score += 1
    return score


def _context_match_score(entry: XpuEntry, ctx: XpuContext) -> int:
    score = 0
    ectx = entry.context

    # language
    if ctx.lang and ectx.get("lang") == ctx.lang:
        score += 2

    # tools intersection
    tools_entry = set(ectx.get("tools") or [])
    tools_ctx = set(ctx.tools or [])
    if tools_entry and tools_ctx and tools_entry & tools_ctx:
        score += 2

    # python version presence (loose match: major.minor prefix)
    if ctx.python:
        py_list = ectx.get("python") or []
        for py in py_list:
            if str(py).startswith(str(ctx.python)):
                score += 1
                break

    # os presence
    if ctx.os:
        os_list = ectx.get("os") or []
        if ctx.os in os_list:
            score += 1

    return score


def score_xpu(entry: XpuEntry, log_snippet: str, ctx: XpuContext) -> float:
    signals = entry.signals or {}
    regexes = signals.get("regex") or []
    keywords = signals.get("keywords") or []

    score = 0.0

    # regex gets a large bonus if any pattern matches
    if regexes and _match_regex(log_snippet, regexes):
        score += 10.0

    # keyword overlap
    score += 1.0 * _keyword_score(log_snippet, keywords)

    # context match
    score += 1.5 * _context_match_score(entry, ctx)

    # small bias towards entries that have atoms at all
    if entry.atoms:
        score += 0.5

    return score


def retrieve_xpu_candidates(
    entries: Sequence[XpuEntry],
    log_snippet: str,
    ctx: XpuContext,
    *,
    k: int = 3,
    prefer_atoms: bool = True,
) -> List[XpuEntry]:
    """Select top-k XPU entries for the given log/context.

    Scoring is done via regex + keyword + context; optionally prioritises
    entries that contain atoms.
    """
    if not entries:
        return []

    # compute base scores
    scored: List[tuple[float, XpuEntry]] = []
    for e in entries:
        s = score_xpu(e, log_snippet=log_snippet, ctx=ctx)
        scored.append((s, e))

    if not scored:
        return []

    # optionally prefer entries with atoms
    if prefer_atoms:
        with_atoms = [(s, e) for s, e in scored if e.atoms]
        without_atoms = [(s, e) for s, e in scored if not e.atoms]

        with_atoms.sort(key=lambda x: x[0], reverse=True)
        without_atoms.sort(key=lambda x: x[0], reverse=True)

        result: List[XpuEntry] = [e for _, e in with_atoms[:k]]
        if len(result) < k:
            result.extend(e for _, e in without_atoms[: k - len(result)])
        return result

    # no special preference
    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:k]]


def render_atom_to_commands(atom: XpuAtom) -> List[str]:
    """Render a single atom into one or more bash commands.

    Commands are returned as raw strings without comments.
    """
    name = atom.name
    args = atom.args or {}

    if name == "pip_pin":
        pkg = args.get("name")
        spec = args.get("spec", "")
        if pkg is None:
            return []
        return [f"pip install '{pkg}{spec}'"]

    if name == "pip_install":
        pkg = args.get("name")
        spec = args.get("spec", "")
        if pkg is None:
            return []
        return [f"pip install '{pkg}{spec}'"]

    if name == "set_pytest_flag":
        flag_name = args.get("name")
        value = args.get("value")
        if not flag_name or value is None:
            return []
        return [f"pytest {flag_name}={value}"]

    if name == "set_env":
        key = args.get("key")
        value = args.get("value")
        if not key or value is None:
            return []
        return [f"export {key}={value}"]

    if name == "set_umask":
        value = args.get("value")
        if value is None:
            return []
        return [f"umask {value}"]

    if name == "set_django_setting":
        key = args.get("key")
        value = args.get("value")
        if not key:
            return []
        # keep it minimal; let the caller decide exact quoting
        return [
            "python - <<'PY'",
            "from django.conf import settings",
            f"settings.{key} = {repr(value)}",
            "PY",
        ]

    if name == "or_upgrade_pkg":
        pkg = args.get("name")
        min_version = args.get("min_version")
        if not pkg or not min_version:
            return []
        return [f"pip install '{pkg}>={min_version}'"]

    # unknown atom type: do not emit anything for now
    return []


def render_entry_commands(entry: XpuEntry) -> List[str]:
    """Render all atoms of an entry into a flat list of bash commands."""
    commands: List[str] = []
    for atom in entry.atoms:
        commands.extend(render_atom_to_commands(atom))
    return commands


def render_candidates_block(entries: Sequence[XpuEntry]) -> str:
    """Render a human-readable "Candidate Fixes" block for inclusion in a prompt.

    The block is intended to be placed inside a human message, describing both
    the high-level advice and the corresponding bash snippets.
    """
    if not entries:
        return ""

    lines: List[str] = []
    lines.append("Candidate Fixes from XPU (choose only what you need):")
    for e in entries:
        lines.append(f"- Fix (id={e.id}):")
        if e.advice_nl:
            lines.append("  Advice:")
            for adv in e.advice_nl:
                lines.append(f"    - {adv}")
        cmds = render_entry_commands(e)
        if cmds:
            lines.append("  Bash snippet:")
            for c in cmds:
                lines.append(f"    {c}")
    return "\n".join(lines)
