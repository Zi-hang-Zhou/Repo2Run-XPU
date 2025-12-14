import os
import json
import glob

# 源目录：Repo2Run 的输出目录
SOURCE_DIR = "output"
# 目标目录：准备喂给 XPU 提取器的临时目录
TARGET_DIR = "data/raw_trajs_for_xpu"

if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)

print(f"正在从 {SOURCE_DIR} 收集 track.json ...")

# 遍历 output 下所有的 track.json
# 假设结构是 output/user/repo/track.json
cnt = 0
for root, dirs, files in os.walk(SOURCE_DIR):
    if "track.json" in files:
        file_path = os.path.join(root, "track.json")
        
        try:
            # 1. 读取原始 JSON (List)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 2. 生成符合 EnvBench 格式的文件名: user__repo@fake_sha.jsonl
            # 路径类似 output/user/repo/track.json
            parts = root.split(os.sep)
            if len(parts) >= 3:
                user_name = parts[-2]
                repo_name = parts[-1]
                # 这里我们伪造一个 SHA，或者如果你能从 track.json 里找到 revision 更好
                # 假设文件名格式: user__repo@latest.jsonl
                target_name = f"{user_name}__{repo_name}@latest.jsonl"
                target_path = os.path.join(TARGET_DIR, target_name)
                
                # 3. 转换为 JSONL (每行一个对象)
                # Repo2Run 的 track.json 通常是一个 list，里面每一步是一个 dict
                with open(target_path, 'w', encoding='utf-8') as f_out:
                    for step in data:
                        f_out.write(json.dumps(step, ensure_ascii=False) + "\n")
                
                cnt += 1
                print(f"  [OK] Converted: {target_name}")
        except Exception as e:
            print(f"  [Error] Failed to process {file_path}: {e}")

print(f"处理完成！共准备了 {cnt} 个轨迹文件在 {TARGET_DIR}")