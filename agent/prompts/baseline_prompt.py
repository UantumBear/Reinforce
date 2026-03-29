"""
TextGrad Baseline 실험에서 사용할 데이터셋별 초기 프롬프트 정의.
논문에서 제시한 task-specific instruction을 참고하여 작성.
"""

# GSM8k: Grade School Math (수학 문제)
GSM8K_INIT_PROMPT = """You will answer a mathematical reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value."""

# GPQA: Graduate-level science questions (객관식)
GPQA_INIT_PROMPT = """Answer the following multiple choice question. Think through the problem carefully and explain your reasoning. End your response with 'The answer is (X)' where X is one of the given options."""

# MMLU: Massive Multitask Language Understanding (객관식)
MMLU_INIT_PROMPT = """Answer the following multiple choice question. Provide your reasoning and conclude with 'The answer is (X)' where X is the letter of your chosen option."""

# NASA/일반 RAG: 문맥 기반 질의응답
DEFAULT_INIT_PROMPT = """Answer the question based on the given context. Be precise and factual."""