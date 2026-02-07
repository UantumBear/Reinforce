"""
@경로: scripts/data_prep/download_klue_rag.py
@설명: HuggingFace의 KLUE MRC KO RAG 데이터셋을 로드하고 샘플 데이터를 저장/확인할 수 있는 클래스
@명령어: python scripts/data_prep/download_klue_rag.py

- 26.02.07 구조 변경 후 테스트 해보지 않아서 테스트 필요

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

class KlueMrcKoRagDataset:
    """KLUE MRC KO RAG 데이터셋 관리 클래스"""
    
    def __init__(self, split="train"):
        """
        KLUE MRC KO RAG 데이터셋 로드
        split: 'train' (학습용), 'validation' (평가용), 'test' (테스트용)
        """
        logger.info(f"Loading KLUE MRC KO RAG dataset ({split})...")
        
        # HuggingFace 다운로드 경로 정보 로깅
        dataset_name = "iamjoon/klue-mrc-ko-rag-dataset"
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
                    import os
                    cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
            
            logger.info(f"Local cache directory: {cache_dir}")
            
            #  ----------------------------- 1. 데이터셋 로드 ----------------------------- 
            self.dataset = load_dataset(dataset_name, split=split)
            logger.info(f"[CHECK] KLUE MRC KO RAG dataset loaded successfully. Size: {len(self.dataset)}")


            #  ---------- 2 전처리: 민감한 주제 필터링 (Azure Content Filter 방지) ---------- 
            logger.info("Starting Preprocessing: Filtering sensitive topics...")
            
            forbidden_words = ["정치", "선거", "대통령", "시위", "폭력", "살인", "범죄", "전쟁", "사망", "피해", "북한", "미사일", "정치적", "전후민주주의"]
            
            def is_safe_content(example):
                # 데이터셋 구조에 따라 필드명이 다를 수 있으므로 안전하게 가져옴
                # 'context', 'document', 'search_result' 중 하나일 가능성 높음
                text_sources = [
                    example.get('context', ''),
                    example.get('document', ''),
                    example.get('question', ''),
                    example.get('answer', ''),
                    str(example.get('search_result', '')) # 리스트일 경우 문자열 변환
                ]
                combined_text = " ".join([str(t) for t in text_sources if t])
                
                for word in forbidden_words:
                    if word in combined_text:
                        return False
                return True

            # 필터링 적용
            self.dataset = self.dataset.filter(is_safe_content)
            logger.info(f"[CHECK] Filtered dataset size: {len(self.dataset)}")

            
            # 데이터셋 메타데이터 정보 로깅
            if hasattr(self.dataset, 'info'):
                logger.info(f"Dataset info: {self.dataset.info}")
            if hasattr(self.dataset, 'builder_name'):
                logger.info(f"Builder name: {self.dataset.builder_name}")
            if hasattr(self.dataset, 'config_name'):
                logger.info(f"Config name: {self.dataset.config_name}")
        except Exception as e:
            logger.error(f"Failed to load KLUE MRC KO RAG dataset: {e}")
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
                "dataset_name": "iamjoon/klue-mrc-ko-rag-dataset"
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
        
        Returns:
            list of tuple: [(question, answer, context), ...]
        """
        if n <= 0:
            raise ValueError("n must be positive")
        if n > len(self.dataset):
            logger.warning(f"Requested {n} samples but dataset has only {len(self.dataset)} items. Using all available.")
            n = len(self.dataset)
            
        indices = random.sample(range(len(self.dataset)), n)
        results = []
        
        for idx in indices:
            item = self.dataset[idx]
            
            # KLUE MRC KO RAG 실제 데이터 구조에 맞게 필드 추출
            try:
                # question 필드: 'question' 키 사용
                question = item['question']
                
                # answer 필드: 'answer' 키 사용 (참조 번호 포함된 형태)
                answer = item['answer']
                
                # context 필드: 'search_result' 키의 배열을 하나로 결합
                search_results = item['search_result']
                if isinstance(search_results, list):
                    # 여러 문서를 하나의 context로 결합 (각 문서 사이에 구분자 추가)
                    context = '\n\n--- Document Separator ---\n\n'.join(search_results)
                else:
                    context = str(search_results)
                
                results.append((str(question), str(answer), str(context)))
                
            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                logger.warning(f"Sample keys: {list(item.keys())}")
                # 기본값으로 처리
                results.append(("", "", ""))
            
        return results

    def get_fixed_sample(self, index: int = 0) -> tuple:
        """
        특정 인덱스의 샘플 반환 (디버깅 및 테스트용)
        
        Args:
            index (int): 데이터셋 내 특정 인덱스 (기본값: 0)
            
        Returns:
            tuple: (질문, 정답, 지문)
        """
        sample = self.dataset[index]
        
        # get_random_samples와 동일한 로직 사용
        try:
            # 실제 데이터 구조에 맞게 필드 추출
            question = sample['question']
            answer = sample['answer']
            
            # search_result 배열을 하나의 context로 결합
            search_results = sample['search_result']
            if isinstance(search_results, list):
                context = '\n\n--- Document Separator ---\n\n'.join(search_results)
            else:
                context = str(search_results)
            
            return str(question), str(answer), str(context)
            
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
        
        # samples 폴더 생성
        samples_dir = Path("datasets/samples")
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        # 랜덤 샘플 수집
        samples = []
        indices = random.sample(range(len(self.dataset)), min(num_samples, len(self.dataset)))
        
        for i, idx in enumerate(indices):
            sample = self.dataset[idx]
            
            # 보기 좋게 가공
            processed_sample = {
                "index": i,
                "original_index": idx,
            }
            
            # 모든 필드를 동적으로 추가 (데이터 구조 파악용)
            for key, value in sample.items():
                processed_sample[key] = value
            
            samples.append({
                "index": i,
                "processed_sample": processed_sample,
                "raw_keys": list(sample.keys())
            })
        
        if output_format.lower() == "json":
            output_file = samples_dir / "klue_rag_samples.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(samples, f, ensure_ascii=False, indent=2)
                
        elif output_format.lower() == "csv":
            import csv
            output_file = samples_dir / "klue_rag_samples.csv"
            
            if samples:
                # 첫 번째 샘플의 키들로 헤더 생성
                first_sample = samples[0]['processed_sample']
                fieldnames = ['index'] + [k for k in first_sample.keys() if k != 'index']
                
                with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for s in samples:
                        p = s['processed_sample']
                        row = {}
                        for field in fieldnames:
                            value = p.get(field, '')
                            # 복잡한 객체는 문자열로 변환
                            if isinstance(value, (list, dict)):
                                value = str(value)
                            row[field] = value
                        writer.writerow(row)
        else:
            raise ValueError("output_format must be 'json' or 'csv'")
        
        logger.info(f"Saved {len(samples)} samples to {output_file}")
        
        # 데이터 구조 로그 출력
        if samples:
            logger.info(f"Sample structure analysis:")
            first_sample = samples[0]['processed_sample']
            for key, value in first_sample.items():
                val_str = str(value)
                if len(val_str) > 50: val_str = val_str[:50] + "..."
                logger.info(f"  {key}: {type(value).__name__} - {val_str}")
        
        return str(output_file)

if __name__ == "__main__":
    """
    스크립트를 직접 실행할 때 샘플 데이터 저장
    사용법: python utils/datasets/klue_rag.py
    """
    try:
        print("KLUE MRC KO RAG 데이터셋 샘플 저장 시작...")
        
        # 데이터셋 로드 (train 셋 사용)
        dataset = KlueMrcKoRagDataset(split="train")
        
        # JSON 저장
        json_file = dataset.save_random_samples(num_samples=10, output_format="json")
        print(f"[SUCCESS] JSON 샘플 저장 완료: {json_file}")
        
        # CSV 저장
        csv_file = dataset.save_random_samples(num_samples=5, output_format="csv")
        print(f"[SUCCESS] CSV 샘플 저장 완료: {csv_file}")
        
        print("\n 데이터 확인:")
        print("   - datasets/samples 폴더를 확인하세요.")
        
        # 캐시 정보 출력
        cache_info = dataset.get_cache_info()
        
    except Exception as e:
        print(f"[ERROR] 에러 발생: {e}")
        logger.error(f"Failed to save samples: {e}")