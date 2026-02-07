from datasets import load_dataset

# 1. 데이터 로드 (금융 RAG 데이터)
# 만약 'rag-ko'가 안 되면 'RAG-Evaluation-Dataset-KO'를 시도해보세요.
try:
    dataset = load_dataset("allganize/RAG-Evaluation-Dataset-KO", split="test")
except:
    dataset = load_dataset("allganize/rag-ko", split="train")

# 2. 데이터 한 건 까보기
sample = dataset[0]

print("=== [1. RAG의 핵심: 문서(Context)] ===")
# 어떤 문서에서 답을 찾아야 하는지 (파일명/페이지 등)
print(f"출처 문서: {sample.get('target_file_name', 'N/A')} (p.{sample.get('target_page_no', 'N/A')})")

print("\n=== [2. 사용자 질문] ===")
print(f"Q: {sample['question']}")

print("\n=== [3. 우리가 목표로 하는 모범 답안] ===")
print(f"A: {sample.get('target_answer', sample.get('answer', ''))}")