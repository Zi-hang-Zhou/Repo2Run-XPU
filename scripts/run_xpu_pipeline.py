import os
import json
import shutil
from pathlib import Path


ROOT_DIR = Path.cwd()
OUTPUT_DIR = ROOT_DIR / "output"
DATA_DIR = ROOT_DIR / "data"
TEMP_TRAJ_DIR = DATA_DIR / "raw_trajs_for_xpu"
EXTRACTED_FILE = DATA_DIR / "extracted_experience.jsonl"
FINAL_KNOWLEDGE_FILE = DATA_DIR / "new_knowledge.jsonl"


def convert_tracks():
    print(f" 扫描 {OUTPUT_DIR} 下的 track.json ...")
    if TEMP_TRAJ_DIR.exists():
        shutil.rmtree(TEMP_TRAJ_DIR)
    TEMP_TRAJ_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    # 递归查找 output 下所有的 track.json
    for track_file in OUTPUT_DIR.glob("**/track.json"):
        try:
            # 解析目录结构 output/User/Repo/track.json
            parts = track_file.parts
            # 找到 output 后面跟着的那两层作为 user 和 repo
            try:
                output_idx = parts.index("output")
                user_name = parts[output_idx + 1]
                repo_name = parts[output_idx + 2]
            except IndexError:
                print(f"跳过路径异常文件: {track_file}")
                continue

            target_name = f"{user_name}__{repo_name}@latest.jsonl"
            target_path = TEMP_TRAJ_DIR / target_name


            with open(track_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 写入 JSONL 格式 (Flatten)
            with open(target_path, 'w', encoding='utf-8') as f_out:
                for step in data:
                    f_out.write(json.dumps(step, ensure_ascii=False) + "\n")
            
            count += 1
            print(f"  转换: {user_name}/{repo_name}")
        except Exception as e:
            print(f"  转换失败 {track_file}: {e}")
    
    print(f"转换完成，共 {count} 个轨迹准备就绪。\n")


def run_pipeline():
    if not os.environ.get("OPENAI_API_KEY"):
        print("错误: 请先 export OPENAI_API_KEY")
        return


    print("调用 LLM 提取经验...")
    extract_script = ROOT_DIR / "build_agent/xpu/extract_xpu_from_trajs_mvp.py"
    cmd_extract = f"python {extract_script} --traj {TEMP_TRAJ_DIR} --output {EXTRACTED_FILE}"
    if os.system(cmd_extract) != 0: return


    print("过滤有效经验...")
    filter_script = ROOT_DIR / "scripts/extract_xpu_to_v1.py"
    cmd_filter = f"python {filter_script} --input {EXTRACTED_FILE} --output {FINAL_KNOWLEDGE_FILE}"
    if os.system(cmd_filter) != 0: return


    print("存入向量数据库...")
    index_script = ROOT_DIR / "scripts/index_xpu_to_vector_db_enhanced.py"
    cmd_index = f"python {index_script} index --input {FINAL_KNOWLEDGE_FILE}"
    os.system(cmd_index)

if __name__ == "__main__":
    convert_tracks()
    run_pipeline()