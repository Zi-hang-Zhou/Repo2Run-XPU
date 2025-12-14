# Pipeline脚本使用说明

## 概述

`pipeline_traj_to_eval_xpu.py` 是一个统一的pipeline脚本，将以下步骤整合在一起：

1. **Script抽取**: 从traj目录中抽取bash脚本
2. **Eval执行**: 对抽取的脚本进行evaluation
3. **XPU抽取** (可选): 从traj中抽取XPU经验

## 使用方法

### 基本用法（只执行script抽取和eval）

```bash
python exp/scripts/pipeline_traj_to_eval_xpu.py \
    --traj-dir tmp/traj_32repos_deepseek
```

### 完整pipeline（包括XPU抽取）

```bash
python exp/scripts/pipeline_traj_to_eval_xpu.py \
    --traj-dir tmp/traj_32repos_deepseek \
    --enable-xpu
```

### 自定义配置

```bash
python exp/scripts/pipeline_traj_to_eval_xpu.py \
    --traj-dir tmp/traj_32repos_deepseek \
    --enable-xpu \
    --eval-workers 8 \
    --output-dir tmp/my_pipeline_output
```

## 参数说明

- `--traj-dir`: **必需**。包含traj文件的目录路径
- `--enable-xpu`: **可选**。是否执行XPU抽取步骤
- `--eval-workers`: **可选**。eval阶段的worker数量（默认: 4）
- `--output-dir`: **可选**。输出目录，默认使用 `<traj-dir>_pipeline_output`

## 输出结构

执行后会在输出目录生成以下结构：

```
<output-dir>/
├── scripts.jsonl                    # 抽取的脚本文件
├── logs/
│   ├── pipeline_YYYYMMDD_HHMMSS.log  # 主日志文件
│   ├── eval_YYYYMMDD_HHMMSS.log      # Eval日志
│   └── xpu_YYYYMMDD_HHMMSS.log       # XPU抽取日志（如果启用）
├── eval/
│   ├── repos/                        # Eval过程中下载的仓库
│   └── results/
│       └── json/
│           └── results/              # Eval结果JSON文件
└── xpu_extraction.jsonl              # XPU抽取结果（如果启用）
```

## 日志和进度

- **日志文件**: 所有详细日志保存在 `logs/` 目录下
- **控制台输出**: 只显示当前步骤和进度条
  - Eval进度: 显示已完成的仓库数量
  - XPU抽取进度: 显示已处理的traj数量

## 注意事项

1. 确保traj目录中包含格式为 `<repo__name@sha>.jsonl` 的文件
2. 脚本会自动排除 `.llm.jsonl` 和 `.xpu.jsonl` 文件
3. 如果某个步骤失败，pipeline会立即停止并报告错误
4. 所有日志都会保存到文件，方便后续分析

## 示例

假设你有一个traj目录 `tmp/traj_32repos_deepseek`，包含32个仓库的traj文件：

```bash
# 只执行script抽取和eval
python exp/scripts/pipeline_traj_to_eval_xpu.py \
    --traj-dir tmp/traj_32repos_deepseek \
    --eval-workers 4

# 执行完整pipeline
python exp/scripts/pipeline_traj_to_eval_xpu.py \
    --traj-dir tmp/traj_32repos_deepseek \
    --enable-xpu \
    --eval-workers 4
```

执行后，控制台会显示类似：

```
Pipeline开始执行
步骤1: 开始从 tmp/traj_32repos_deepseek 抽取scripts...
步骤1完成
步骤2: 开始执行eval (workers=4)...
Eval进度: 100%|████████████| 32/32 [05:23<00:00, 0.10repo/s]
步骤2完成: eval成功完成
步骤3: 开始执行XPU抽取...
XPU抽取进度: 100%|████████████| 32/32 [12:45<00:00, 0.04traj/s]
步骤3完成: XPU抽取成功
Pipeline完成！
所有日志保存在: <output-dir>/logs/pipeline_YYYYMMDD_HHMMSS.log
```
