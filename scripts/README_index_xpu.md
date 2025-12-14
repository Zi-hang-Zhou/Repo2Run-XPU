# XPU 向量数据库索引脚本使用说明

## 概述

`index_xpu_to_vector_db_enhanced.py` 是一个增强版的 XPU 索引脚本，用于将 XPU 条目从 JSONL 文件索引到 PostgreSQL 向量数据库中。

## 功能特性

1. **索引功能**：将 XPU 条目索引到向量数据库，自动生成 embedding
2. **验证功能**：检查数据库连接、表结构、数据统计
3. **查询功能**：根据 ID 查询特定 XPU 条目
4. **搜索功能**：使用向量相似度搜索相似的 XPU 条目

## Embedding 策略

脚本使用 `build_xpu_text()` 函数构建用于 embedding 的文本，包括：

- **Context（上下文）**：
  - Language（语言）
  - Tools（工具列表）
  - Python versions（Python 版本）
  - OS（操作系统）

- **Signals（信号）**：
  - Keywords（关键词）
  - Error patterns（错误模式，regex）

- **Advice（建议）**：
  - advice_nl 文本内容

这与推理阶段的检索策略完全一致：使用工具输出的错误日志作为查询文本，通过向量相似度搜索匹配的 XPU 条目。

## 使用方法

### 1. 索引 XPU 条目

```bash
# 基本用法（使用环境变量中的 dns）
python exp/scripts/index_xpu_to_vector_db_enhanced.py index --input exp/xpu_v0.jsonl

# 指定数据库连接字符串
python exp/scripts/index_xpu_to_vector_db_enhanced.py \
    --dns "postgresql://user:password@localhost:5433/dbname" \
    index --input exp/xpu_v0.jsonl

# Dry run（预览，不实际索引）
python exp/scripts/index_xpu_to_vector_db_enhanced.py \
    --dns "postgresql://user:password@localhost:5433/dbname" \
    index --input exp/xpu_v0.jsonl --dry-run

# 自定义批次大小（每 N 条记录输出一次进度）
python exp/scripts/index_xpu_to_vector_db_enhanced.py \
    index --input exp/xpu_v0.jsonl --batch-size 20
```

### 2. 验证数据库

```bash
# 检查数据库连接、表结构、数据统计
python exp/scripts/index_xpu_to_vector_db_enhanced.py \
    --dns "postgresql://user:password@localhost:5433/dbname" \
    verify
```

输出示例：
```
======================================================================
Database Verification Results
======================================================================
✅ Table exists

Statistics:
  Total entries: 53
  Entries with embedding: 53

Table structure:
  - id: text
  - context: jsonb
  - signals: jsonb
  - advice_nl: jsonb
  - atoms: jsonb
  - embedding: USER-DEFINED (vector)
  - created_at: timestamp without time zone
======================================================================
```

### 3. 查询特定条目

```bash
# 根据 ID 查询 XPU 条目
python exp/scripts/index_xpu_to_vector_db_enhanced.py \
    --dns "postgresql://user:password@localhost:5433/dbname" \
    query --id xpu_5458157951
```

### 4. 搜索相似条目

```bash
# 使用向量相似度搜索
python exp/scripts/index_xpu_to_vector_db_enhanced.py \
    --dns "postgresql://user:password@localhost:5433/dbname" \
    search --query "numpy error ufunc"

# 自定义返回数量和相似度阈值
python exp/scripts/index_xpu_to_vector_db_enhanced.py \
    --dns "postgresql://user:password@localhost:5433/dbname" \
    search --query "django sqlite error" --k 10 --min-similarity 0.4
```

## 环境变量

可以通过环境变量设置数据库连接：

```bash
export dns="postgresql://user:password@localhost:5433/dbname"
```

或者使用 `.env` 文件（脚本会自动加载）。

## 数据库表结构

表名：`xpu_entries`

| 字段 | 类型 | 说明 |
|------|------|------|
| id | TEXT (PRIMARY KEY) | XPU 条目 ID |
| context | JSONB | 上下文信息（语言、工具、Python版本等） |
| signals | JSONB | 信号信息（关键词、错误模式） |
| advice_nl | JSONB | 建议文本列表 |
| atoms | JSONB | 可执行原子列表 |
| embedding | VECTOR(1024) | 向量嵌入（维度由 EMBEDDING_DIM 决定） |
| created_at | TIMESTAMP | 创建时间 |

## 注意事项

1. **Embedding 维度**：默认使用 1024 维（可通过 `EMBEDDING_DIM` 环境变量配置）
2. **API 配置**：需要配置 `EMBEDDING_API_KEY` 和 `EMBEDDING_BASE_URL`（或使用 `OPENAI_API_KEY`）
3. **数据库扩展**：需要 PostgreSQL 安装 `pgvector` 扩展
4. **索引策略**：使用 `ivfflat` 索引加速向量搜索（lists=100）

## 故障排查

### 连接失败
- 检查数据库连接字符串格式
- 确认数据库服务正在运行
- 检查网络连接和防火墙设置

### Embedding 生成失败
- 检查 `EMBEDDING_API_KEY` 和 `EMBEDDING_BASE_URL` 配置
- 确认 API 服务可访问
- 检查 API 配额和限制

### 维度不匹配
- 确认 `EMBEDDING_DIM` 环境变量与 embedding 模型输出维度一致
- 检查数据库表结构中的 embedding 列维度

## 相关文件

- `exp/Adapter/xpu_vector_store.py`：向量存储实现
- `exp/Adapter/xpu_adapter.py`：XPU 数据适配器
- `inference/src/agents/python/prompts.py`：XPU 检索逻辑

