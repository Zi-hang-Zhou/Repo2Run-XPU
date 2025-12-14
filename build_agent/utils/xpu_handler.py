import os
import sys
import logging
from typing import List, Tuple, Optional

# --- [修复 Import 路径] ---
# 获取当前文件 (xpu_handler.py) 所在的目录: .../Repo2Run/build_agent/utils
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取 build_agent 目录
build_agent_dir = os.path.dirname(current_dir)
# 获取项目根目录 (Repo2Run)
project_root = os.path.dirname(build_agent_dir)

# 将根目录加入 sys.path，确保能 import build_agent.xpu
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# -------------------------

# 适配 Repo2Run 的目录结构引入 XPU 模块
try:
    from build_agent.xpu.xpu_adapter import (
        XpuAtom,
        XpuContext,
        XpuEntry,
        render_candidates_block,
    )
    from build_agent.xpu.xpu_vector_store import XpuVectorStore, text_to_embedding
except ImportError as e:
    # 把具体错误打印出来方便调试
    print(f"[XPU Handler] Critical Import Error: {e}")
    print(f"[XPU Handler] Current sys.path: {sys.path}")
    XpuVectorStore = None
    text_to_embedding = None

logger = logging.getLogger(__name__)


# 1. 保留原本的关键词列表
XPU_ERROR_KEYWORDS = [
    "module not found",
    "modulenotfounderror",
    "importerror",
    "no module named",
    "could not find a version",
    "command not found",
    "permission denied",
    "error:",
    "failed",
    "traceback",  # 新增
    "exception",  # 新增
]

class XpuHandler:
    def __init__(self):
        self.vector_store = None
        self.last_query = None  # 用来模拟 state["xpu_trace"] 的去重逻辑
        self._init_vector_store()
        self.session_used_ids = set()

    def _init_vector_store(self):
        """Get or create vector store instance. (对应原代码 _get_vector_store)"""
        if XpuVectorStore is None:
            return

        # 优先读取 dns (原代码逻辑)，兼容 DATABASE_URL
        dns = os.environ.get("dns") or os.environ.get("DATABASE_URL")
        
        if not dns:
            logger.error("Missing required environment variable: dns (PostgreSQL connection string)")
            return

        try:
            self.vector_store = XpuVectorStore(connection_string=dns)
            logger.info("[XPU] Initialized vector store")
        except Exception as e:
            logger.error(f"[XPU] Failed to initialize vector store: {e}")
            self.vector_store = None

    def _check_has_error(self, text: str) -> bool:
        """对应原代码中 _extract_recent_tool_context 的关键词检查部分"""
        if not text:
            return False
        lowered = text.lower()
        return any(keyword in lowered for keyword in XPU_ERROR_KEYWORDS)

    def _should_query_xpu(self, log_snippet: str, has_error: bool) -> bool:
        """对应原代码 _should_query_xpu"""
        if not self.vector_store:
            return False
        if not log_snippet:
            return False
        if not has_error:
            return False
        
        # 对应原代码: if last_trace.get("query") == log_snippet: return False
        # 防止对同一个报错重复查询
        if self.last_query == log_snippet:
            return False
            
        return True

    def retrieve_hints(self, sandbox_res: str) -> Tuple[str, List[str]]:
        """
        主入口函数。
        Returns:
            (prompt_text, list_of_xpu_ids)
        """
        # 1. 预处理文本
        max_chars = 4000
        log_snippet = sandbox_res.strip()
        if len(log_snippet) > max_chars:
            log_snippet = log_snippet[: max_chars // 2] + "\n... [TRUNCATED] ...\n" + log_snippet[-max_chars // 2 :]

        # 2. 检查是否有错误
        has_error = self._check_has_error(log_snippet)

        # 3. 判断是否需要查询
        # 注意：如果不查询，返回空字符串和空列表
        if not self._should_query_xpu(log_snippet, has_error):
            return "", []

        # 4. 执行查询
        try:
            ctx = XpuContext(lang="python")
            
            query_embedding = text_to_embedding(log_snippet)
            
            results = self.vector_store.search(
                query_embedding=query_embedding,
                ctx=ctx,
                k=3,
                min_similarity=0.3, 
            )
            
            if not results:
                logger.info("[XPU] No similar entries found (similarity < 0.3)")
                return "", []

            # Convert to XpuEntry objects
            candidates = []
            for result in results:
                atoms_data = result.get("atoms", [])
                atoms = [XpuAtom(name=a["name"], args=a["args"]) for a in atoms_data]
                
                entry = XpuEntry(
                    id=result["id"],
                    context=result.get("context", ""),
                    signals=result.get("signals", []),
                    advice_nl=result["advice_nl"],
                    atoms=atoms,
                    # 如果你的 XpuEntry 定义里加了 telemetry 字段，这里也要加上，没有就不加
                    # telemetry=result.get("telemetry", {}) 
                )
                candidates.append(entry)

            # 5. 更新 Trace & Telemetry
            self.last_query = log_snippet
            
            candidate_ids = [e.id for e in candidates]
            logger.info(f"[XPU] Selected candidates: {candidate_ids}")

            # [新增] 实时命中计数 (Hits +1)
            if self.vector_store:
                self.vector_store.increment_telemetry(candidate_ids, 'hits')
            
            # [新增] 加入本局缓存
            self.session_used_ids.update(candidate_ids)

            # 6. 渲染
            return render_candidates_block(candidates), candidate_ids

        except Exception as e:
            logger.error(f"[XPU] Error during retrieval: {e}")
            return "", []
    
    def update_realtime_feedback(self, last_ids: List[str], current_ids: List[str]):
        """
        [新增] 实时反馈逻辑：比较上一轮和这一轮的 ID
        """
        if not self.vector_store or not last_ids:
            return

        last_set = set(last_ids)
        curr_set = set(current_ids)

        # 1. 失败 (Failures): 上一轮有，这一轮还有 -> 说明没解决
        failed_ids = list(last_set.intersection(curr_set))
        if failed_ids:
            self.vector_store.increment_telemetry(failed_ids, 'failures')
            logger.info(f"[XPU] Realtime Feedback: {len(failed_ids)} entries marked as FAILURES (issue persisted).")

        # 2. 成功 (Successes): 上一轮有，这一轮没了 -> 说明报错消失
        succeeded_ids = list(last_set - curr_set)
        if succeeded_ids:
            self.vector_store.increment_telemetry(succeeded_ids, 'successes')
            logger.info(f"[XPU] Realtime Feedback: {len(succeeded_ids)} entries marked as SUCCESSES (issue resolved).")
    
    def finalize_session(self, is_task_success: bool):
        """
        [新增] 全局结算：任务结束时调用。
        如果任务最终成功了，把本局用过的所有经验再额外记一次 Success（可选）。
        """
        if not self.vector_store or not self.session_used_ids:
            return
            
        # 这里是一个策略选择：
        # 策略 A: 既然有了实时反馈，就不做全局结算了 (避免重复计数)。
        # 策略 B: 只有当任务最终 Success 时，给所有参与过的经验加分。
        
        # 目前推荐仅打印日志，依靠实时反馈即可。
        # 如果你想强化“最终成功”的权重，可以取消下面代码的注释：
        
        # if is_task_success:
        #     logger.info(f"[XPU] Session Final: Marking {len(self.session_used_ids)} entries as contributor to success.")
        #     self.vector_store.increment_telemetry(list(self.session_used_ids), 'successes')
        
        self.session_used_ids.clear()