#!/usr/bin/env python
"""统一的pipeline脚本：从traj目录 -> 抽取scripts -> eval -> (可选) XPU抽取

用法:
    python exp/scripts/pipeline_traj_to_eval_xpu.py \\
        --traj-dir tmp/traj_32repos_deepseek \\
        [--enable-xpu] \\
        [--eval-workers 4] \\
        [--output-dir tmp/pipeline_output]
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from tqdm import tqdm

# 项目根目录
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from exp.scripts.extract_scripts_from_traj_dir import extract_scripts  # noqa: E402


class PipelineLogger:
    """将日志同时输出到文件和控制台（仅显示关键信息）"""
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.file_handle = open(log_file, 'w', encoding='utf-8')
    
    def log(self, message: str, to_console: bool = False):
        """记录日志，可选择是否输出到控制台"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_line = f"[{timestamp}] {message}\n"
        self.file_handle.write(log_line)
        self.file_handle.flush()
        if to_console:
            print(message)
    
    def close(self):
        self.file_handle.close()


def run_script_extraction(
    traj_dir: Path,
    scripts_output: Path,
    logger: PipelineLogger
) -> bool:
    """步骤1: 从traj目录抽取scripts"""
    logger.log(f"步骤1: 开始从 {traj_dir} 抽取scripts...", to_console=True)
    
    try:
        extract_scripts(str(traj_dir), str(scripts_output))
        logger.log(f"步骤1完成: 已生成 {scripts_output}")
        logger.log(f"步骤1完成", to_console=True)
        return True
    except Exception as e:
        logger.log(f"步骤1失败: {e}", to_console=True)
        return False


def run_evaluation(
    scripts_file: Path,
    output_dir: Path,
    workers: int,
    logger: PipelineLogger
) -> tuple[bool, Optional[Path]]:
    """步骤2: 执行eval"""
    logger.log(f"步骤2: 开始执行eval (workers={workers})...", to_console=True)
    
    # 准备eval输出目录
    eval_tmp_dir = output_dir / "eval"
    eval_repos_dir = eval_tmp_dir / "repos"
    eval_json_results_dir = eval_tmp_dir / "results" / "json"
    eval_eval_results_dir = eval_tmp_dir / "results" / "eval"
    
    eval_log_file = output_dir / "logs" / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    eval_log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 构建eval命令
    cmd = [
        sys.executable, "-m", "evaluation.main",
        "language=python",
        "do_dry_run=false",
        "input.use_scripts=true",
        "input.mode=local",
        f"input.local={scripts_file.absolute()}",
        "output.mode=local",
        "output.keep_local_archives=false",
        "output.keep_local_jsonl=true",
        f"operation.dirs.tmp={eval_tmp_dir.absolute()}",
        f"operation.dirs.repo_data={eval_repos_dir.absolute()}",
        f"operation.dirs.json_results={eval_json_results_dir.absolute()}",
        f"operation.dirs.eval_results={eval_eval_results_dir.absolute()}",
        f"operation.pool_config.max_workers={workers}",
        "operation.pool_config.chunksize=1",
        "+operation.rewrite_results=true",
    ]
    
    logger.log(f"执行命令: {' '.join(cmd)}")
    
    # 读取scripts文件以获取仓库数量
    import jsonlines
    total_repos = 0
    with jsonlines.open(scripts_file, 'r') as reader:
        total_repos = sum(1 for _ in reader)
    
    logger.log(f"待eval的仓库数量: {total_repos}")
    
    # 启动eval进程，实时监控进度
    try:
        with open(eval_log_file, 'w', encoding='utf-8') as log_f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(ROOT_DIR),
                env={**os.environ, "HYDRA_FULL_ERROR": "1"},
            )
            
            # 实时读取输出并显示进度
            last_count = 0
            with tqdm(total=total_repos, desc="Eval进度", unit="repo") as pbar:
                for line in process.stdout:
                    log_f.write(line)
                    log_f.flush()
                    logger.log(line.rstrip())
                    
                    # 定期检查结果文件数量来更新进度
                    if eval_json_results_dir.exists():
                        results_dir = eval_json_results_dir / "results"
                        if results_dir.exists():
                            current_count = len(list(results_dir.glob("*.json")))
                            if current_count > last_count:
                                pbar.update(current_count - last_count)
                                last_count = current_count
            
            process.wait()
            
            # 最终检查结果文件数量
            if eval_json_results_dir.exists():
                results_dir = eval_json_results_dir / "results"
                if results_dir.exists():
                    final_count = len(list(results_dir.glob("*.json")))
                    if final_count > last_count:
                        pbar.update(final_count - last_count)
                    pbar.n = final_count
                    pbar.refresh()
            
            if process.returncode == 0:
                logger.log(f"步骤2完成: eval成功完成", to_console=True)
                return True, eval_json_results_dir
            else:
                logger.log(f"步骤2失败: eval退出码 {process.returncode}", to_console=True)
                return False, None
                
    except Exception as e:
        logger.log(f"步骤2失败: {e}", to_console=True)
        import traceback
        logger.log(traceback.format_exc())
        return False, None


def run_xpu_extraction(
    traj_dir: Path,
    output_dir: Path,
    logger: PipelineLogger
) -> tuple[bool, Optional[Path]]:
    """步骤3: 执行XPU抽取"""
    logger.log(f"步骤3: 开始执行XPU抽取...", to_console=True)
    
    xpu_output = output_dir / "xpu_extraction.jsonl"
    xpu_log_file = output_dir / "logs" / f"xpu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    xpu_log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 统计traj文件数量
    traj_files = [
        f for f in os.listdir(traj_dir)
        if f.endswith(".jsonl") and "@" in f
        and not (f.endswith(".llm.jsonl") or f.endswith(".xpu.jsonl"))
    ]
    total_trajs = len(traj_files)
    
    logger.log(f"待抽取XPU的traj数量: {total_trajs}")
    
    # 构建XPU抽取命令
    xpu_script = ROOT_DIR / "xpuExtract" / "extract_xpu_from_trajs_mvp.py"
    cmd = [
        sys.executable,
        str(xpu_script),
        "--traj", str(traj_dir),
        "--output", str(xpu_output),
    ]
    
    logger.log(f"执行命令: {' '.join(cmd)}")
    
    try:
        with open(xpu_log_file, 'w', encoding='utf-8') as log_f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(ROOT_DIR),
            )
            
            # 实时读取输出并显示进度
            import re
            last_progress = 0
            with tqdm(total=total_trajs, desc="XPU抽取进度", unit="traj") as pbar:
                for line in process.stdout:
                    log_f.write(line)
                    log_f.flush()
                    logger.log(line.rstrip())
                    
                    # 尝试从tqdm输出中解析进度
                    # tqdm格式通常是: "Extracting XPU from trajs: 50%|█████     | 25/50 [00:10<00:10, 2.5it/s]"
                    # 或者: "25/50 [00:10<00:10, 2.5it/s]"
                    match = re.search(r'(\d+)/(\d+)', line)
                    if match:
                        current = int(match.group(1))
                        total_from_line = int(match.group(2))
                        if current > last_progress and total_from_line == total_trajs:
                            pbar.update(current - last_progress)
                            last_progress = current
                    
                    # 也检查输出文件的行数（作为备用进度指示）
                    if xpu_output.exists():
                        try:
                            with open(xpu_output, 'r') as f:
                                line_count = sum(1 for _ in f)
                            if line_count > last_progress:
                                pbar.update(line_count - last_progress)
                                last_progress = line_count
                        except:
                            pass
            
            process.wait()
            
            if process.returncode == 0 and xpu_output.exists():
                logger.log(f"步骤3完成: XPU抽取成功，输出到 {xpu_output}", to_console=True)
                return True, xpu_output
            else:
                logger.log(f"步骤3失败: 退出码 {process.returncode} 或未生成输出文件", to_console=True)
                return False, None
                
    except Exception as e:
        logger.log(f"步骤3失败: {e}", to_console=True)
        import traceback
        logger.log(traceback.format_exc())
        return False, None


def main():
    parser = argparse.ArgumentParser(
        description="统一的pipeline: traj -> scripts -> eval -> (可选) XPU抽取",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 只执行scripts抽取和eval
  python exp/scripts/pipeline_traj_to_eval_xpu.py \\
      --traj-dir tmp/traj_32repos_deepseek

  # 执行完整pipeline（包括XPU抽取）
  python exp/scripts/pipeline_traj_to_eval_xpu.py \\
      --traj-dir tmp/traj_32repos_deepseek \\
      --enable-xpu

  # 自定义输出目录和worker数量
  python exp/scripts/pipeline_traj_to_eval_xpu.py \\
      --traj-dir tmp/traj_32repos_deepseek \\
      --enable-xpu \\
      --eval-workers 8 \\
      --output-dir tmp/my_pipeline_output
        """
    )
    
    parser.add_argument(
        "--traj-dir",
        type=Path,
        required=True,
        help="包含traj文件的目录路径"
    )
    parser.add_argument(
        "--enable-xpu",
        action="store_true",
        help="是否执行XPU抽取步骤"
    )
    parser.add_argument(
        "--eval-workers",
        type=int,
        default=4,
        help="eval阶段的worker数量 (默认: 4)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录，默认使用 <traj-dir>_pipeline_output"
    )
    
    args = parser.parse_args()
    
    # 验证输入目录
    traj_dir = args.traj_dir.resolve()
    if not traj_dir.exists() or not traj_dir.is_dir():
        print(f"错误: traj目录不存在: {traj_dir}")
        sys.exit(1)
    
    # 设置输出目录
    if args.output_dir is None:
        output_dir = traj_dir.parent / f"{traj_dir.name}_pipeline_output"
    else:
        output_dir = args.output_dir.resolve()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化日志
    log_file = output_dir / "logs" / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = PipelineLogger(log_file)
    
    logger.log(f"Pipeline开始执行", to_console=True)
    logger.log(f"输入目录: {traj_dir}")
    logger.log(f"输出目录: {output_dir}")
    logger.log(f"启用XPU抽取: {args.enable_xpu}")
    logger.log(f"Eval workers: {args.eval_workers}")
    
    # 步骤1: 抽取scripts
    scripts_output = output_dir / "scripts.jsonl"
    if not run_script_extraction(traj_dir, scripts_output, logger):
        logger.log("Pipeline失败: 步骤1 (script抽取) 失败", to_console=True)
        logger.close()
        sys.exit(1)
    
    # 步骤2: 执行eval
    success, eval_results_dir = run_evaluation(scripts_output, output_dir, args.eval_workers, logger)
    if not success:
        logger.log("Pipeline失败: 步骤2 (eval) 失败", to_console=True)
        logger.close()
        sys.exit(1)
    
    # 步骤3: (可选) XPU抽取
    if args.enable_xpu:
        success, xpu_output = run_xpu_extraction(traj_dir, output_dir, logger)
        if not success:
            logger.log("Pipeline失败: 步骤3 (XPU抽取) 失败", to_console=True)
            logger.close()
            sys.exit(1)
    else:
        logger.log("跳过步骤3: XPU抽取 (未启用)", to_console=True)
    
    logger.log("Pipeline完成！", to_console=True)
    logger.log(f"所有日志保存在: {log_file}")
    logger.close()


if __name__ == "__main__":
    main()

