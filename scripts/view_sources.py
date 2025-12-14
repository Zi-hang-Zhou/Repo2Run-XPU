import os
import psycopg2
import json

DB_URL = os.environ.get("DATABASE_URL", "postgresql://zihang:123456@localhost:5432/xpu_db")

def view_sources():
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        
        # 尝试查询 source 字段，如果报错说明没加这个字段
        try:
            cur.execute("SELECT id, source, telemetry FROM xpu_entries ORDER BY created_at DESC;")
            has_source_col = True
        except psycopg2.errors.UndefinedColumn:
            conn.rollback() # 回滚事务
            print("提示：数据库中没有 source 字段，将尝试从 ID 解析仓库名。")
            cur.execute("SELECT id, NULL, telemetry FROM xpu_entries ORDER BY created_at DESC;")
            has_source_col = False

        rows = cur.fetchall()
        
        print(f"\n{'ID (推测仓库)':<40} | {'Hits':<5} | {'来源 (Source)'}")
        print("-" * 80)
        
        for row in rows:
            xpu_id, source_json, telemetry = row
            hits = telemetry.get('hits', 0) if telemetry else 0
            
            # 1. 尝试从 source 字段拿
            repo_name = "未知"
            if source_json and 'repo' in source_json:
                repo_name = source_json['repo']
            
            # 2. 如果没有 source，从 ID 解析 (去掉 xpu_env_py_ 前缀)
            if repo_name == "未知" and xpu_id.startswith("xpu_env_py_"):
                # xpu_env_py_scikit_learn -> scikit_learn
                repo_name = xpu_id.replace("xpu_env_py_", "")
            
            # 打印
            print(f"{repo_name:<40} | {hits:<5} | {source_json if source_json else 'N/A'}")

        conn.close()
        
    except Exception as e:
        print(f"查询失败: {e}")

if __name__ == "__main__":
    view_sources()