#!/usr/bin/env python
"""Extract XPU entries from extraction results and save to xpu_v1.jsonl."""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))


def extract_xpu_entries(input_path: Path, output_path: Path) -> None:
    """Extract XPU entries from extraction results."""
    xpu_entries = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            
            # Only extract entries where LLM decided it's an XPU (not heuristic_skip)
            if entry.get('llm_decision') == 'xpu':
                xpu = entry.get('xpu')
                if xpu:
                    xpu_entries.append(xpu)
    
    # Write to output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for xpu in xpu_entries:
            f.write(json.dumps(xpu, ensure_ascii=False) + '\n')
    
    print(f"Extracted {len(xpu_entries)} XPU entries from {input_path}")
    print(f"Saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract XPU entries to xpu_v1.jsonl")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parents[2] / "xpuExtract" / "outputs" / "xpu_from_329repos.jsonl",
        help="Path to extraction results JSONL file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parents[2] / "exp" / "xpu_v1.jsonl",
        help="Path to output XPU JSONL file",
    )
    args = parser.parse_args()
    
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    extract_xpu_entries(args.input, args.output)


if __name__ == "__main__":
    main()

