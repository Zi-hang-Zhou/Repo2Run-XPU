#!/usr/bin/env python
"""测试XPU数据库连接和功能是否正常 (Repo2Run适配版)"""

import os
import sys
from pathlib import Path

# 1. 确保项目根目录 (Repo2Run) 在 sys.path 中
# 当前文件在 scripts/ 下，父目录的父目录就是根目录
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

# 加载 .env
load_dotenv(ROOT_DIR / ".env")


def test_database_connection():
    """测试数据库连接"""
    print("=" * 80)
    print("测试1: 数据库连接 (PostgreSQL)")
    print("=" * 80)
    
    # 优先读取 DATABASE_URL，兼容 dns
    dns = os.environ.get('DATABASE_URL') or os.environ.get('dns') or os.environ.get('DNS')
    
    if not dns:
        print("❌ 未设置数据库连接字符串 (请在 .env 中设置 DATABASE_URL 或 dns)")
        return False
    
    # 隐藏密码打印
    safe_dns = dns.split('@')[-1] if '@' in dns else '...'
    print(f"✅ 数据库连接字符串找到: ...@{safe_dns}")
    
    try:
        import psycopg2
        
        print("  尝试连接数据库...")
        conn = psycopg2.connect(dns)
        
        with conn.cursor() as cur:
            # 检查pgvector扩展
            cur.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector');")
            has_vector = cur.fetchone()[0]
            print(f"  pgvector扩展: {'✅ 已安装' if has_vector else '❌ 未安装'}")
            
            if not has_vector:
                print("  ⚠️  需要安装pgvector扩展: CREATE EXTENSION vector;")
                conn.close()
                return False
            
            # 检查xpu_entries表
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'xpu_entries'
                );
            """)
            has_table = cur.fetchone()[0]
            print(f"  xpu_entries表: {'✅ 已存在' if has_table else '❌ 不存在'}")
            
            if has_table:
                cur.execute("SELECT COUNT(*) FROM xpu_entries;")
                count = cur.fetchone()[0]
                print(f"  xpu_entries记录数: {count}")
                
                if count == 0:
                    print("  ⚠️  表中没有数据，请运行: python scripts/index_xpu_to_vector_db_enhanced.py index ...")
            else:
                print("  ⚠️  表不存在，会在首次运行 index 脚本时自动创建")
        
        conn.close()
        print("✅ 数据库连接测试通过！")
        return True
        
    except ImportError:
        print("❌ 缺少psycopg2库，请安装: pip install psycopg2-binary")
        return False
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        return False


def test_openai_api():
    """测试OpenAI API（用于生成embeddings）"""
    print("\n" + "=" * 80)
    print("测试2: OpenAI API (用于生成embeddings)")
    print("=" * 80)
    
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ 未设置 OPENAI_API_KEY")
        return False
    
    print(f"✅ OPENAI_API_KEY: 已设置")
    
    try:
        # --- 修改点：指向 build_agent.xpu ---
        from build_agent.xpu.xpu_vector_store import text_to_embedding
        
        print("  测试生成embedding (请求API)...")
        embedding = text_to_embedding("test query")
        print(f"  ✅ Embedding生成成功 (维度: {len(embedding)})")
        return True
        
    except Exception as e:
        print(f"❌ Embedding生成失败: {e}")
        return False


def test_xpu_vector_store():
    """测试XPU Vector Store类初始化"""
    print("\n" + "=" * 80)
    print("测试3: XPU Vector Store 模块加载与初始化")
    print("=" * 80)
    
    try:
        # --- 修改点：指向 build_agent.xpu ---
        from build_agent.xpu.xpu_vector_store import XpuVectorStore, EMBEDDING_DIM
        
        dns = os.environ.get('DATABASE_URL') or os.environ.get('dns')
        if not dns:
            return False
        
        print("  初始化XpuVectorStore...")
        store = XpuVectorStore(connection_string=dns)
        print("  ✅ XpuVectorStore初始化成功")
        
        # 测试搜索功能
        print("  测试搜索功能 (Dummy Search)...")
        test_embedding = [0.0] * EMBEDDING_DIM 
        results = store.search(test_embedding, k=1)
        print(f"  ✅ 搜索功能正常 (返回 {len(results)} 条结果)")
        
        store.close()
        return True
        
    except ImportError as e:
        print(f"❌ 导入错误 (路径不对?): {e}")
        return False
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def main():
    print(f"项目根目录: {ROOT_DIR}")
    
    results = []
    results.append(("数据库连接", test_database_connection()))
    results.append(("OpenAI API", test_openai_api()))
    results.append(("Vector Store", test_xpu_vector_store()))
    
    print("\n" + "=" * 80)
    print("测试结果总结")
    print("=" * 80)
    
    all_passed = True
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 80)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())