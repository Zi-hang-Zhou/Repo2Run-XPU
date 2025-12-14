import os
import json
import glob

SOURCE_DIR = "output"
TARGET_DIR = "data/raw_trajs_for_xpu"

if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)

print(f"正在从 {SOURCE_DIR} 收集 track.json ...")

# 遍历 output 下所有的 track.json

cnt = 0
for root, dirs, files in os.walk(SOURCE_DIR):
    if "track.json" in files:
        file_path = os.path.join(root, "track.json")
        
        try:
            # 1. 读取原始 JSON (List)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 2. 符合 EnvBench 格式的文件名
            parts = root.split(os.sep)
            if len(parts) >= 3:
                user_name = parts[-2]
                repo_name = parts[-1]

                target_name = f"{user_name}__{repo_name}@latest.jsonl"
                target_path = os.path.join(TARGET_DIR, target_name)
                
                # 3. 转换为 JSONL (每行一个对象)
                with open(target_path, 'w', encoding='utf-8') as f_out:
                    for step in data:
                        f_out.write(json.dumps(step, ensure_ascii=False) + "\n")
                
                cnt += 1
                print(f"  [OK] Converted: {target_name}")
        except Exception as e:
            print(f"  [Error] Failed to process {file_path}: {e}")

print(f"处理完成！共准备了 {cnt} 个轨迹文件在 {TARGET_DIR}")