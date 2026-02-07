"""
@경로: utils/datasets/rag_cot_en_ko.py
@설명: HuggingFace의 RAG-COT-En_KO 데이터셋을 로드하고 샘플 데이터를 저장/확인할 수 있는 클래스
@명령어: python utils/datasets/rag_cot_en_ko.py
"""
import random
import json
import os
import sys
from pathlib import Path
from datasets import load_dataset

# 직접 실행될 때만 경로 설정 (import 시에는 실행 안 됨)
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    os.environ['PYTHONPATH'] = str(project_root)

from utils.log.logging import logger

class RagCotEnKoDataset:
    """RAG-COT-En_KO 데이터셋 관리 클래스 (영어 컨텍스트, 한국어 질문/답변)"""
    
    def __init__(self, split="train", max_samples=None, streaming=False):
        """
        RAG-COT-En_KO 데이터셋 로드
        split: 'train' (학습용), 'validation' (평가용), 'test' (테스트용)
        max_samples: 최대 샘플 수 제한 (None이면 전체)
        streaming: 스트리밍 모드 사용 여부 (큰 데이터셋용)
        """
        logger.info(f"Loading RAG-COT-En_KO dataset ({split})...")
        if max_samples:
            logger.info(f"Dataset size limited to: {max_samples} samples")
        if streaming:
            logger.info("Using streaming mode for large dataset")
        
        # HuggingFace 다운로드 경로 정보 로깅
        dataset_name = "jaeyong2/RAG-COT-En_KO"
        logger.info(f"Dataset identifier: {dataset_name}")
        logger.info(f"HuggingFace URL: https://huggingface.co/datasets/{dataset_name}")

        try:
            # 데이터셋 로드 전 캐시 경로 확인 (datasets 라이브러리 버전별 호환성)
            try:
                from datasets.utils.file_utils import HF_CACHE_HOME
                cache_dir = HF_CACHE_HOME
            except ImportError:
                try:
                    from datasets import config
                    cache_dir = config.HF_DATASETS_CACHE
                except (ImportError, AttributeError):
                    cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
            
            logger.info(f"Local cache directory: {cache_dir}")
            
            #  ----------------------------- 1. 데이터셋 로드 (최적화) ----------------------------- 
            if streaming:
                # 스트리밍 모드: 메모리 효율적이지만 일부 기능 제한
                self.dataset = load_dataset(dataset_name, split=split, streaming=True)
                logger.info("Dataset loaded in streaming mode")
                
                # 스트리밍에서는 샘플 수를 제한하기 위해 take() 사용
                if max_samples:
                    self.dataset = self.dataset.take(max_samples)
                    logger.info(f"Limited to {max_samples} samples in streaming mode")
                    
            else:
                # 일반 모드: 전체 로드 후 필요시 샘플링
                self.dataset = load_dataset(dataset_name, split=split)
                original_size = len(self.dataset)
                logger.info(f"[CHECK] RAG-COT-En_KO dataset loaded successfully. Original size: {original_size}")
                
                # 크기 제한이 있으면 미리 샘플링
                if max_samples and max_samples < original_size:
                    indices = random.sample(range(original_size), max_samples)
                    self.dataset = self.dataset.select(indices)
                    logger.info(f"[CHECK] Pre-sampled to {max_samples} samples from {original_size}")
            
            # 스트리밍이 아닌 경우에만 필터링 적용 (스트리밍에서는 메모리 효율을 위해 스킵)
            if not streaming:
                #  ---------- 2. 전처리: 민감한 주제 필터링 (Azure Content Filter 방지) ---------- 
                logger.info("Starting Preprocessing: Filtering sensitive topics...")
                
                # 영어와 한국어 모두 고려한 필터링 단어
                forbidden_words_ko = ["정치", "선거", "대통령", "시위", "폭력", "살인", "범죄", "전쟁", "사망", "피해", "북한", "미사일", "정치적", "전후민주주의"]
                forbidden_words_en = ["politics", "election", "president", "protest", "violence", "murder", "crime", "war", "death", "damage", "nuclear", "missile", "political", "democracy"]
                forbidden_words = forbidden_words_ko + forbidden_words_en
                
                def is_safe_content(example):
                    # RAG-COT-En_KO 데이터 구조에 맞게 필드 추출
                    text_sources = [
                        example.get('context', ''),  # 영어 컨텍스트
                        example.get('question', ''),  # 한국어 질문
                        example.get('answer', ''),   # 한국어 답변
                        example.get('ko_question', ''),  # 가능한 필드명
                        example.get('ko_answer', ''),    # 가능한 필드명
                        example.get('en_context', ''),   # 가능한 필드명
                        str(example.get('reasoning', ''))  # COT reasoning 부분
                    ]
                    combined_text = " ".join([str(t) for t in text_sources if t])
                    
                    for word in forbidden_words:
                        if word in combined_text.lower():
                            return False
                    return True

                # 필터링 적용
                pre_filter_size = len(self.dataset) if hasattr(self.dataset, '__len__') else 0
                self.dataset = self.dataset.filter(is_safe_content)
                post_filter_size = len(self.dataset) if hasattr(self.dataset, '__len__') else 0
                if pre_filter_size > 0:
                    logger.info(f"[CHECK] Filtered dataset size: {post_filter_size} (removed {pre_filter_size - post_filter_size} items)")
                else:
                    logger.info(f"[CHECK] Filtered dataset completed")
            else:
                logger.info("Skipping filtering in streaming mode for memory efficiency")

            # 데이터셋 메타데이터 정보 로깅
            if hasattr(self.dataset, 'info'):
                logger.info(f"Dataset info: {self.dataset.info}")
            if hasattr(self.dataset, 'builder_name'):
                logger.info(f"Builder name: {self.dataset.builder_name}")
            if hasattr(self.dataset, 'config_name'):
                logger.info(f"Config name: {self.dataset.config_name}")
                
        except Exception as e:
            logger.error(f"Failed to load RAG-COT-En_KO dataset: {e}")
            raise

    def get_cache_info(self):
        """데이터셋 캐시 정보를 반환"""
        try:
            # datasets 라이브러리 버전별 호환성 처리
            try:
                from datasets.utils.file_utils import HF_CACHE_HOME
                cache_home = HF_CACHE_HOME
            except ImportError:
                try:
                    from datasets import config
                    cache_home = config.HF_DATASETS_CACHE
                except (ImportError, AttributeError):
                    cache_home = os.path.expanduser("~/.cache/huggingface/datasets")
            
            cache_info = {
                "cache_home": cache_home,
                "cache_exists": os.path.exists(cache_home),
                "dataset_name": "jaeyong2/RAG-COT-En_KO"
            }
            
            if hasattr(self.dataset, 'cache_files'):
                cache_info["cache_files"] = self.dataset.cache_files
            
            logger.info(f"Cache info: {cache_info}")
            return cache_info
            
        except Exception as e:
            logger.error(f"Failed to get cache info: {e}")
            return {}

    def get_random_samples(self, n=1):
        """
        랜덤하게 n개의 (질문, 정답, 지문) 샘플을 반환
        스트리밍 모드에서는 순차적으로 샘플 추출
        
        Returns:
            list of tuple: [(question, answer, context), ...]
        """
        if n <= 0:
            raise ValueError("n must be positive")
        
        results = []
        
        # 스트리밍 모드인지 확인
        is_streaming = hasattr(self.dataset, 'take')
        
        if is_streaming:
            # 스트리밍 모드: 순차적으로 n개 추출
            logger.info(f"Extracting {n} samples from streaming dataset...")
            count = 0
            for item in self.dataset:
                if count >= n:
                    break
                    
                try:
                    question, answer, context = self._extract_fields(item)
                    results.append((question, answer, context))
                    count += 1
                except Exception as e:
                    logger.warning(f"Error processing streaming sample {count}: {e}")
                    continue
                    
        else:
            # 일반 모드: 랜덤 샘플링
            dataset_size = len(self.dataset)
            if n > dataset_size:
                logger.warning(f"Requested {n} samples but dataset has only {dataset_size} items. Using all available.")
                n = dataset_size
                
            indices = random.sample(range(dataset_size), n)
            
            for idx in indices:
                item = self.dataset[idx]
                try:
                    question, answer, context = self._extract_fields(item)
                    results.append((question, answer, context))
                except Exception as e:
                    logger.warning(f"Error processing sample {idx}: {e}")
                    results.append(("", "", ""))
        
        return results

    def _extract_fields(self, item):
        """데이터 항목에서 question, answer, context 필드 추출"""
        # 실제 데이터 구조에 맞는 필드명 사용
        question_candidates = ['Question', 'question', 'ko_question', 'query', 'input']
        answer_candidates = ['Final Answer', 'answer', 'ko_answer', 'output', 'response']
        context_candidates = ['context', 'en_context', 'passage', 'document']
        thinking_candidates = ['Thinking', 'thinking', 'reasoning', 'cot', 'chain_of_thought']
        
        question = ""
        answer = ""
        context = ""
        thinking = ""
        
        # 질문 필드 찾기
        for q_field in question_candidates:
            if q_field in item and item[q_field]:
                question = str(item[q_field])
                break
        
        # 답변 필드 찾기 (Final Answer 우선)
        for a_field in answer_candidates:
            if a_field in item and item[a_field]:
                answer = str(item[a_field])
                break
        
        # 컨텍스트 필드 찾기
        for c_field in context_candidates:
            if c_field in item and item[c_field]:
                context = str(item[c_field])
                break
        
        # Thinking(COT reasoning) 필드 찾기
        for t_field in thinking_candidates:
            if t_field in item and item[t_field]:
                thinking = str(item[t_field])
                break
        
        # Chain-of-Thought가 있다면 답변에 포함 (더 풍부한 학습을 위해)
        if thinking and thinking not in answer:
            # Thinking이 매우 길면 요약된 부분만 포함
            if len(thinking) > 200:
                thinking_summary = thinking[:200] + "..."
                answer = f"[사고 과정: {thinking_summary}]\n\n{answer}" if answer else f"[사고 과정: {thinking_summary}]"
            else:
                answer = f"[사고 과정: {thinking}]\n\n{answer}" if answer else f"[사고 과정: {thinking}]"
        
        # 필드를 찾지 못한 경우 로깅
        if not question and not answer and not context:
            logger.warning(f"Could not extract fields from item. Available keys: {list(item.keys())}")
            
        return question, answer, context

    def get_fixed_sample(self, index: int = 0) -> tuple:
        """
        특정 인덱스의 샘플 반환 (디버깅 및 테스트용)
        스트리밍 모드에서는 지원하지 않음
        
        Args:
            index (int): 데이터셋 내 특정 인덱스 (기본값: 0)
            
        Returns:
            tuple: (질문, 정답, 지문)
        """
        # 스트리밍 모드인지 확인
        is_streaming = hasattr(self.dataset, 'take')
        if is_streaming:
            logger.error("get_fixed_sample is not supported in streaming mode")
            return "", "", ""
            
        if index >= len(self.dataset):
            logger.warning(f"Index {index} is out of range. Dataset size: {len(self.dataset)}")
            return "", "", ""
        
        sample = self.dataset[index]
        
        try:
            question, answer, context = self._extract_fields(sample)
            return question, answer, context
            
        except Exception as e:
            logger.warning(f"Error processing fixed sample {index}: {e}")
            logger.warning(f"Sample keys: {list(sample.keys())}")
            return "", "", ""

    def save_random_samples(self, num_samples: int = 10, output_format: str = "json") -> str:
        """
        랜덤 샘플들을 파일로 저장하여 데이터 구조 확인
        
        Args:
            num_samples (int): 저장할 샘플 개수 (기본: 10개)
            output_format (str): "json" 또는 "csv"
            
        Returns:
            str: 저장된 파일 경로
        """
        if not self.dataset:
            raise ValueError("Dataset not loaded.")
        
        # 스트리밍 모드인지 확인
        is_streaming = hasattr(self.dataset, 'take')
        
        # 프로젝트 루트 기준으로 samples 폴더 생성
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        samples_dir = Path(current_dir) / "datasets" / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        # 랜덤 샘플 수집
        samples = []
        
        if is_streaming:
            # 스트리밍 모드: 순차적으로 샘플 수집
            logger.info(f"Collecting {num_samples} samples from streaming dataset...")
            count = 0
            for sample in self.dataset:
                if count >= num_samples:
                    break
                
                processed_sample = {
                    "index": count,
                    "original_index": count,  # 스트리밍에서는 순차적
                }
                
                # 모든 필드를 동적으로 추가
                for key, value in sample.items():
                    processed_sample[key] = value
                
                samples.append({
                    "index": count,
                    "processed_sample": processed_sample,
                    "raw_keys": list(sample.keys())
                })
                count += 1
        else:
            # 일반 모드: 랜덤 샘플링
            dataset_size = len(self.dataset)
            indices = random.sample(range(dataset_size), min(num_samples, dataset_size))
            
            for i, idx in enumerate(indices):
                sample = self.dataset[idx]
                
                processed_sample = {
                    "index": i,
                    "original_index": idx,
                }
                
                # 모든 필드를 동적으로 추가
                for key, value in sample.items():
                    processed_sample[key] = value
                
                samples.append({
                    "index": i,
                    "processed_sample": processed_sample,
                    "raw_keys": list(sample.keys())
                })
        
        if output_format.lower() == "json":
            output_file = samples_dir / "rag_cot_en_ko_samples.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(samples, f, ensure_ascii=False, indent=2)
                
        elif output_format.lower() == "csv":
            import csv
            output_file = samples_dir / "rag_cot_en_ko_samples.csv"
            
            if samples:
                # 첫 번째 샘플의 키들을 기반으로 헤더 생성
                first_sample = samples[0]['processed_sample']
                fieldnames = ['index'] + [k for k in first_sample.keys() if k != 'index']
                
                with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for s in samples:
                        row_data = {}
                        p = s['processed_sample']
                        for field in fieldnames:
                            value = p.get(field, '')
                            # 복잡한 객체는 문자열로 변환
                            if isinstance(value, (list, dict)):
                                value = str(value)
                            row_data[field] = value
                        writer.writerow(row_data)
        else:
            raise ValueError("output_format must be 'json' or 'csv'")
        
        logger.info(f"Saved {len(samples)} samples to {output_file}")
        
        # 데이터 구조 로그 출력
        if samples:
            logger.info(f"Sample structure analysis:")
            first_sample = samples[0]['processed_sample']
            for key, value in first_sample.items():
                val_str = str(value)
                if len(val_str) > 50: 
                    val_str = val_str[:50] + "..."
                logger.info(f"  {key}: {type(value).__name__} - {val_str}")
        
        return str(output_file)

if __name__ == "__main__":
    """
    스크립트를 직접 실행할 때 샘플 데이터 저장
    사용법: 
    - 빠른 테스트: python utils/datasets/rag_cot_en_ko.py
    - 스트리밍 모드: python utils/datasets/rag_cot_en_ko.py --streaming
    """
    import sys
    
    try:
        print("RAG-COT-En_KO 데이터셋 샘플 저장 시작...")
        
        # 명령행 인수 확인
        use_streaming = "--streaming" in sys.argv
        use_small_sample = "--small" in sys.argv
        
        if use_streaming:
            print("🚀 스트리밍 모드 사용 (메모리 효율적)")
            # 스트리밍 모드: 메모리 효율적이지만 필터링 생략
            dataset = RagCotEnKoDataset(split="train", streaming=True, max_samples=50)
        elif use_small_sample:
            print("⚡ 소규모 샘플 모드 (빠른 테스트)")
            # 작은 샘플로 빠른 테스트
            dataset = RagCotEnKoDataset(split="train", max_samples=1000)
        else:
            print("📊 기본 모드 (전체 데이터셋)")
            # 기본 모드
            dataset = RagCotEnKoDataset(split="train", max_samples=5000)  # 5천개로 제한
        
        # JSON 저장
        json_file = dataset.save_random_samples(num_samples=10, output_format="json")
        print(f"[SUCCESS] JSON 샘플 저장 완료: {json_file}")
        
        # CSV 저장
        csv_file = dataset.save_random_samples(num_samples=5, output_format="csv")
        print(f"[SUCCESS] CSV 샘플 저장 완료: {csv_file}")
        
        # 데이터 구조 확인용 샘플 출력
        print("\n=== 샘플 데이터 확인 ===")
        samples = dataset.get_random_samples(n=3)
        for i, (question, answer, context) in enumerate(samples, 1):
            print(f"\n[샘플 {i}]")
            print(f"질문: {question[:100]}...")
            print(f"답변: {answer[:100]}...")  
            print(f"컨텍스트: {context[:100]}...")
        
        print(f"\n✅ 데이터 확인:")
        print("   - datasets/samples 폴더를 확인하세요.")
        print("\n💡 사용법:")
        print("   - 스트리밍 모드: python utils/datasets/rag_cot_en_ko.py --streaming")
        print("   - 소규모 테스트: python utils/datasets/rag_cot_en_ko.py --small")
        
    except Exception as e:
        print(f"[ERROR] 에러 발생: {e}")
        logger.error(f"Failed to save samples: {e}")