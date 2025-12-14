import os
import psycopg2
import json

# 配置
DB_URL = os.environ.get("DATABASE_URL", "postgresql://zihang:123456@localhost:5432/xpu_db")

def view_latest_xpu(limit=20):
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        
        # 查询最近入库的 N 条
        cur.execute("""
            SELECT id, context, advice_nl, atoms, created_at 
            FROM xpu_entries 
            ORDER BY created_at DESC 
            LIMIT %s;
        """, (limit,))
        
        rows = cur.fetchall()
        
        print(f"\n📚 === 最新入库的 {len(rows)} 条经验 === 📚\n")
        
        for row in rows:
            xpu_id, context, advice, atoms, created_at = row
            
            print(f"🆔 ID: {xpu_id}")
            print(f"⏰ 时间: {created_at}")
            
            # 显示上下文 (针对哪个库/Python版本)
            ctx_str = json.dumps(context, ensure_ascii=False)
            print(f"🌍 上下文: {ctx_str}")
            
            # 显示建议 (Agent 看到的提示)
            print("💡 建议 (Advice):")
            if isinstance(advice, list):
                for i, line in enumerate(advice, 1):
                    print(f"   {i}. {line}")
            else:
                print(f"   {advice}")
                
            # 显示原子操作 (具体的命令)
            print("🛠️  原子操作 (Atoms):")
            # atoms 通常存为 json 对象列表
            if isinstance(atoms, list):
                for atom in atoms:
                    name = atom.get('name')
                    args = atom.get('args')
                    print(f"   - {name}: {args}")
            
            print("-" * 60)
            
        conn.close()
        
    except Exception as e:
        print(f"❌ 查询失败: {e}")

if __name__ == "__main__":
    view_latest_xpu()