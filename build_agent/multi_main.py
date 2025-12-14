# Copyright (2025) Bytedance Ltd. and/or its affiliates 

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 


import multiprocessing
import subprocess
import os
import sys
import random
import time
import signal




def run_command(command):
    # 忽略子进程的中断信号，防止 Ctrl+C 导致混乱
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    current_time = time.time()
    local_time = time.localtime(current_time)
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    
    try:
        # 尝试提取仓库名用于日志显示
        full_name = command.split('--full_name "')[1].split('"')[0]
    except:
        full_name = "Unknown-Repo"

    print(f'[{full_name}] Start time: {formatted_time}')

    # --- 修复磁盘检查逻辑 ---
    try:
        # 修改为检查根目录 / 的使用率，而不是特定的 /dev/vdb
        # awk 'NR==2 {print $5}' 提取第二行的第五列（使用率百分比）
        df_cmd = "df -h / | awk 'NR==2 {print $5}'"
        disk_check = subprocess.run(df_cmd, shell=True, capture_output=True, text=True)
        
        # 去掉百分号并转换为浮点数
        usage_str = disk_check.stdout.strip().replace('%', '')
        
        if usage_str and float(usage_str) > 99:
            print(f'[{full_name}] Warning! Disk usage is critical ({usage_str}%). Skipping task to protect server.')
            return
    except Exception as e:
        # 如果检查失败，打印警告但不要崩溃，继续执行任务
        print(f"[{full_name}] Warning: Could not check disk usage ({e}). Proceeding anyway.")
    # -----------------------

    try:
        print(f'[{full_name}] Begin execution...')
        
        # 执行 main.py
        subprocess.run(command, shell=True, check=True)
        
        print(f'[{full_name}] Finish: Success')
        
        # 尝试清理该仓库相关的 Docker 容器 (使用更安全的清理方式)
        repo_tag = full_name.lower().replace('/', '_').replace('-', '_')
        rm_cmd = f'docker ps -a --filter ancestor={repo_tag}:tmp -q | xargs -r docker rm'
        subprocess.run(rm_cmd, shell=True, capture_output=True)

    except subprocess.CalledProcessError as e:
        print(f"[{full_name}] Error: Task failed with exit code {e.returncode}")
    except Exception as e:
        print(f"[{full_name}] Error: Unexpected exception: {e}")

if __name__ == '__main__':
    # 初始清理
    os.system('docker rm -f $(docker ps -aq) > /dev/null 2>&1')

    if len(sys.argv) != 2:
        print('Usage: python multi_main.py <script_path>')
        sys.exit(1)
    script_path = sys.argv[1]

    try:
        with open(script_path, 'r') as r1:
            commands = r1.readlines()
        # 过滤掉空行和注释
        commands = [cmd.strip() for cmd in commands if cmd.strip() and not cmd.strip().startswith('#')]
    except Exception as e:
        print(f'Error reading script file: {e}')
        sys.exit(1)

    random.shuffle(commands)

    print(f"Loaded {len(commands)} tasks. Starting multiprocessing pool (3 processes)...")

    # 创建进程池，最多同时运行 1 个进程
    with multiprocessing.Pool(processes=1) as pool:
        pool.map(run_command, commands)
    
    print("All tasks completed.")