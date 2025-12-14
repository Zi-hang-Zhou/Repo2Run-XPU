#!/usr/bin/env bash
set -e

# 多仓库推理流水线（仅 inference）：
# 1) 从 HF split python_baseline_failure 截取前 N 条样本，写成本地 JSONL
# 2) 使用 LocalFileDataSource + ReAct+XPU+Kimi 对该 JSONL 中的所有仓库做并发推理
#
# 可用环境变量：
#   N_REPOS          要截取的仓库数量，默认 50
#   MAX_CONCURRENT   并发仓库数，默认 5
#   MODEL_NAME       LLM 模型名，默认 kimi-k2-0905-preview
#   OUTPUT_JSONL     子集 JSONL 路径，默认 data/python_baseline_failure_top${N_REPOS}.jsonl
#   DOCKER_OUTPUT_DIR Docker 容器中代码 checkout 的输出目录（宿主机路径），
#                    默认 ${STORAGE_BASE:-/storage/ziqian/envbench}/repos_py_subset_${N_REPOS}
#   LOGGING_DIR      轨迹输出目录，默认 tmp/traj_py_subset_${N_REPOS}_kimi
#
# 注意：
# - 需要在仓库根目录下已有 .env 配置好 OpenAI 兼容的 Kimi endpoint 和 API key；
# - pip 镜像相关的环境变量（PIP_INDEX_URL / PIP_TIMEOUT）已经在 Hydra 配置中
#   通过 docker.env_vars 统一下发到容器中，本脚本无需额外处理。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${ROOT_DIR}"

# 激活根目录下的本地虚拟环境（如果存在）
if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

N_REPOS="${N_REPOS:-50}"
MAX_CONCURRENT="${MAX_CONCURRENT:-5}"
MODEL_NAME="${MODEL_NAME:-kimi-k2-0905-preview}"

# 计算默认输出路径（允许用户通过 OUTPUT_JSONL 覆盖）
if [ -z "${OUTPUT_JSONL:-}" ]; then
  OUTPUT_JSONL="${ROOT_DIR}/data/python_baseline_failure_top${N_REPOS}.jsonl"
fi

if [ -z "${DOCKER_OUTPUT_DIR:-}" ]; then
  STORAGE_BASE="${STORAGE_BASE:-/storage/ziqian/envbench}"
  DOCKER_OUTPUT_DIR="${STORAGE_BASE}/repos_py_subset_${N_REPOS}"
fi

if [ -z "${LOGGING_DIR:-}" ]; then
  LOGGING_DIR="${ROOT_DIR}/tmp/traj_py_subset_${N_REPOS}_kimi"
fi

echo "[Step 1] 从 HF 数据集截取前 ${N_REPOS} 条样本到 ${OUTPUT_JSONL}..."
python "${SCRIPT_DIR}/make_hf_subset.py" \
  --n "${N_REPOS}" \
  --output "${OUTPUT_JSONL}"

echo "[Step 2] 使用 LocalFileDataSource + Kimi (${MODEL_NAME}) 对 ${N_REPOS} 个仓库做并发推理 (max_concurrent=${MAX_CONCURRENT})..."
python -m inference.main \
  agent.model.model="${MODEL_NAME}" \
  data_source.type=local \
  data_source.local.path="${OUTPUT_JSONL}" \
  docker.output_dir="${DOCKER_OUTPUT_DIR}" \
  logging_dir="${LOGGING_DIR}" \
  hf.upload=false \
  max_concurrent="${MAX_CONCURRENT}"

echo "多仓库 inference 结束。"
echo "- 轨迹保存在: ${LOGGING_DIR}"
echo "- 仓库 checkout 根目录在: ${DOCKER_OUTPUT_DIR}"
