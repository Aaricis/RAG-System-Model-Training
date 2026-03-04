from typing import List
import re


# # Basic
# def get_inference_system_prompt() -> str:
#     """get system prompt for generation"""
#     prompt = "Answer the question based on the context below."
#     return prompt
#
# def get_inference_user_prompt(query : str, context_list : List[str]) -> str:
#     """Create the user prompt for generation given a query and a list of context passages."""
#     prompt = f"""
#       Context: {context_list}
#       Question: {query}
#       Answer:
#     """
#     return prompt

# CoT
# def get_inference_system_prompt() -> str:
#     """
#     System prompt with controlled Chain-of-Thought reasoning.
#     """
#     return (
#         "You are a careful and knowledgeable assistant.\n"
#         "Use the provided context to answer the question.\n"
#         "Follow these steps internally:\n"
#         "1. Identify the key information from the context relevant to the question.\n"
#         "2. Reason step by step to derive the answer.\n"
#         "3. Provide a concise and accurate final answer.\n\n"
#         "Do NOT mention your reasoning steps or the context explicitly.\n"
#         "Only output the final answer."
#     )


def get_inference_user_prompt(query: str, context_list: List[str]) -> str:
    """
    Create user prompt with structured context for RAG + COT.
    """
    context_str = "\n".join(
        [f"[{i + 1}] {ctx}" for i, ctx in enumerate(context_list)]
    )

    return (
        "Here is the retrieved context:\n"
        f"{context_str}\n\n"
        "Question:\n"
        f"{query}\n\n"
        "Answer:"
    )


# RAG-Aware No-COT Prompt
# def get_inference_system_prompt() -> str:
#     return (
#         "You are an assistant answering questions strictly based on the given context.\n"
#         "Your answer should closely follow the wording and facts in the context.\n"
#         "Do not add new information or rephrase unnecessarily.\n"
#         "Provide a short, direct answer."
#     )

# Basic + 约束
def get_inference_system_prompt():
    return "Answer the user concisely based on the context passages."


def parse_generated_answer(pred_ans: str) -> str:
    """解析模型生成的答案，提取 assistant\\n<think>\\n\\n</think>\\n\\n 後面的內容"""

    # 方法1: 尋找 </think> 後的內容
    think_pattern = r'</think>\s*\n\s*(.+?)(?:\n|$)'
    match = re.search(think_pattern, pred_ans, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        return answer

    # 方法2: 尋找 assistant 後的內容（如果沒有 think 標籤）
    assistant_pattern = r'assistant\s*\n\s*(.+?)(?:\n|$)'
    match = re.search(assistant_pattern, pred_ans, re.DOTALL)
    if match:
        content = match.group(1).strip()
        # 如果內容中沒有 <think>，直接返回
        if '<think>' not in content:
            return content

    # 方法3: 如果都找不到，返回最後一行非空內容
    lines = [line.strip() for line in pred_ans.split('\n') if line.strip()]
    if lines:
        return lines[-1]

    # 方法4: 如果以上都失敗，返回原始答案
    return pred_ans.strip()
