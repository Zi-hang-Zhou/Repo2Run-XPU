# XPU 向量检索系统使用指南

## 概述

本系统实现了基于 PostgreSQL + pgvector 的 XPU 经验向量检索机制，用于在推理过程中动态检索和注入相关经验。

## 前置要求

1. **PostgreSQL 数据库**：需要安装 `pgvector` 扩展
2. **环境变量**：在 `.env` 中配置数据库连接
   ```
   dns=postgresql://user@host/dbname
   OPENAI_API_KEY=your_key  # 用于生成 embeddings
   ```

## 使用流程

### 1. 索引 XPU 条目到向量数据库

将 XPU JSONL 文件中的条目向量化并存储到数据库：

```bash
python exp/scripts/index_xpu_to_vector_db.py \
    --input exp/xpu_v0.jsonl \
    --batch-size 10
```

**说明**：
- 脚本会为每个 XPU 条目生成 embedding（使用 OpenAI text-embedding-3-small）
- 自动创建 `xpu_entries` 表和向量索引
- 支持增量更新（相同 ID 会覆盖）

### 2. 运行推理（自动使用向量检索）

在推理过程中，系统会自动：
1. 检测工具输出中的错误信号
2. 提取错误上下文并生成 query embedding
3. 从向量数据库检索相似 XPU 条目
4. 将候选经验注入到 prompt 中

**无需额外配置**，只要：
- `.env` 中配置了 `dns` 连接字符串
- 数据库已索引 XPU 条目
- 推理时会自动使用向量检索

## 架构说明

### 核心模块

- **`exp/Adapter/xpu_vector_store.py`**：向量存储接口
  - `XpuVectorStore`：管理数据库连接和检索
  - `text_to_embedding()`：生成文本 embedding
  - `build_xpu_text()`：构建 XPU 条目的可检索文本

- **`inference/src/agents/python/prompts.py`**：推理时的检索逻辑
  - `_should_query_xpu()`：判断是否需要检索
  - `_build_contextual_xpu_block()`：执行向量检索并构建 prompt 块

### 检索策略

- **触发条件**：检测到工具输出包含错误关键词
- **检索方式**：基于 query embedding 的余弦相似度搜索
- **过滤条件**：支持按 `lang`、`python`、`tools` 等上下文字段过滤
- **相似度阈值**：默认 `min_similarity=0.3`（可调整）

## 故障排查

如果遇到问题，系统会直接抛出异常（无降级策略）：

- **数据库连接失败**：检查 `dns` 环境变量
- **pgvector 扩展未安装**：联系管理员安装 `CREATE EXTENSION vector;`
- **embedding 生成失败**：检查 `OPENAI_API_KEY` 是否有效
- **检索无结果**：可能是相似度阈值过高，或数据库中没有相关条目

## 后续扩展

- 支持其他 embedding 模型（如本地模型）
- 添加检索结果的反馈机制
- 实现更细粒度的上下文过滤

