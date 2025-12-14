import json

# 配置
input_file = "python329.jsonl"
output_file = "tasks.txt"
top_n = 50

print(f"正在读取 {input_file} 并生成前 {top_n} 个任务...")

with open(input_file, 'r') as f, open(output_file, 'w') as out:
    lines = f.readlines()
    for i in range(min(top_n, len(lines))):
        line = lines[i].strip()
        if not line: continue
        
        data = json.loads(line)
        repo = data['repository']
        sha = data['revision']
        
        # 构造符合 multi_main.py 解析规则的命令 (注意 full_name 必须带双引号)
        # 假设你的 root_path 是当前目录 .
        cmd = f'python build_agent/main.py --full_name "{repo}" --sha "{sha}" --root_path . --llm "gpt-4o-2024-05-13"'
        
        out.write(cmd + "\n")

print(f"生成完毕！请查看 {output_file}")