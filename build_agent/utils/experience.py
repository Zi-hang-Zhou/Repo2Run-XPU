import json
import re
import os
from typing import List

class ExperienceRetriever:
    def __init__(self, knowledge_base_path: str):
        self.knowledge_base = []
        if os.path.exists(knowledge_base_path):
            try:
                with open(knowledge_base_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            self.knowledge_base.append(json.loads(line))
                print(f"[ExperienceRetriever] Loaded {len(self.knowledge_base)} entries from {knowledge_base_path}")
            except Exception as e:
                print(f"[ExperienceRetriever] Error loading knowledge base: {e}")
        else:
            print(f"[ExperienceRetriever] Warning: File not found at {knowledge_base_path}")
    
    def retrieve(self, observation: str, current_files: List[str] = None) -> List[str]:
        """
        数据驱动的检索逻辑：
        1. 反应式 : 检查报错日志 (observation) 是否匹配 regex 或 keywords
        2. 预判式 : 检查当前文件列表 (current_files) 是否包含 keywords 中的关键文件
        """
        matched_advices = []
        hit_ids = set()
        
        if current_files is None:
            current_files = []
        
        current_files_set = set(current_files)

        for exp in self.knowledge_base:
            signals = exp.get("signals", {})
            regex_list = signals.get("regex", [])
            keywords = signals.get("keywords", []) # e.g., ["pyproject.toml", "missing build tool"]
            
            is_match = False
            

            if observation:
                for pattern in regex_list:
                    try:
                        if re.search(pattern, observation, re.IGNORECASE):
                            is_match = True
                            break
                    except:
                        continue 
                
                if not is_match and keywords:
                    for kw in keywords:
                        if kw in observation:
                            is_match = True
                            break
            

            if not is_match and current_files_set and keywords:
                common = current_files_set.intersection(set(keywords))
                if common:
                    is_match = True


            if is_match and exp['id'] not in hit_ids:
                advice_list = exp.get("advice_nl", [])
                for advice in advice_list:
                    formatted_advice = f"[Knowledge Base Hint]: {advice}"
                    matched_advices.append(formatted_advice)
                hit_ids.add(exp['id'])

        return matched_advices