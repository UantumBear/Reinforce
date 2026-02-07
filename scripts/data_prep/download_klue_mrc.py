"""
@경로 : scripts/data_prep/download_klue_mrc.py
@설명 : Hugging Face에서 KLUE MRC 데이터를 로드하고 랜덤 샘플을 제공하는 클래스
@명령어 : python scripts/data_prep/download_klue_mrc.py

- 26.02.07 구조 변경 후 테스트 해보지 않아서 테스트 필요
"""
import random
import json
import os
import sys
from pathlib import Path
from typing import Tuple, List
from datasets import load_dataset

# 직접 실행될 때만 경로 설정 (import 시에는 실행 안 됨)
if __name__ == "__main__":
    # Python 경로 설정 (PYTHONPATH=. 효과)
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    # 환경변수도 설정 (선택사항)
    os.environ['PYTHONPATH'] = str(project_root)
from utils.log.logging import logger

class KlueMrcDataset:
    """KLUE MRC 데이터셋 관리 클래스"""
    
    def __init__(self, split: str = "train"):
        """
        데이터셋 초기화 및 로드
        
        Args:
            split (str): 사용할 데이터 분할 ('train', 'validation' 등)
        """
        logger.info(f"Loading KLUE MRC dataset (split={split})...")
        
        # HuggingFace 다운로드 경로 정보 로깅
        dataset_name = "klue"
        config_name = "mrc"
        logger.info(f"Dataset identifier: {dataset_name}")
        logger.info(f"Dataset config: {config_name}")
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
            
            # lmqg/qg_koquad 데이터셋 사용 (질문 생성/답변용으로 정제된 버전)
            # self.dataset = load_dataset("lmqg/qg_koquad", split=split, trust_remote_code=True)
            self.dataset = load_dataset(dataset_name, config_name, split=split)
            #  trust_remote_code=True : 데이터셋 내부 스크립트 실행 허용 (보안 에러 해결)

            logger.info(f"Successfully loaded {len(self.dataset)} samples.")
            
            # 데이터셋 메타데이터 정보 로깅
            if hasattr(self.dataset, 'info'):
                logger.info(f"Dataset info: {self.dataset.info}")
            if hasattr(self.dataset, 'builder_name'):
                logger.info(f"Builder name: {self.dataset.builder_name}")
            if hasattr(self.dataset, 'config_name'):
                logger.info(f"Config name: {self.dataset.config_name}")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise e
    
    def get_cache_info(self):
        """데이터셋 캐시 정보를 반환"""
        try:
            from datasets.utils.file_utils import HF_CACHE_HOME
            import os
            
            cache_info = {
                "cache_home": HF_CACHE_HOME,
                "cache_exists": os.path.exists(HF_CACHE_HOME),
                "dataset_name": "klue",
                "config_name": "mrc"
            }
            
            if hasattr(self.dataset, 'cache_files'):
                cache_info["cache_files"] = self.dataset.cache_files
            
            logger.info(f"Cache info: {cache_info}")
            return cache_info
            
        except Exception as e:
            logger.error(f"Failed to get cache info: {e}")
            return {}
    
    def get_cache_info(self):
        """데이터셋 캐시 정보를 반환"""
        try:
            from datasets.utils.file_utils import HF_CACHE_HOME
            import os
            
            cache_info = {
                "cache_home": HF_CACHE_HOME,
                "cache_exists": os.path.exists(HF_CACHE_HOME),
                "dataset_name": "klue",
                "config_name": "mrc"
            }
            
            if hasattr(self.dataset, 'cache_files'):
                cache_info["cache_files"] = self.dataset.cache_files
            
            logger.info(f"Cache info: {cache_info}")
            return cache_info
            
        except Exception as e:
            logger.error(f"Failed to get cache info: {e}")
            return {}

    def get_random_samples(self, batch_size: int = 1) -> list:
        """
        데이터셋에서 랜덤하게 질문, 정답, 지문 쌍을 반환
        
        Args:
            batch_size (int): 반환할 샘플 개수
        
        Returns:
            list: [(질문1, 정답1, 지문1), (질문2, 정답2, 지문2), ...] 형태의 튜플 리스트
        """
        if not self.dataset:
            raise ValueError("Dataset not loaded.")
        
        samples = []
        
        for _ in range(batch_size):
            sample = random.choice(self.dataset)
            
            # KLUE MRC 실제 구조: {'question': ..., 'answers': {'text': [...], 'answer_start': [...]}, 'context': ...}
            question = sample['question']
            context = sample['context']  # 지문 추가
            
            # answers 필드에서 첫 번째 답변 추출
            answers = sample['answers']
            if isinstance(answers['text'], list) and len(answers['text']) > 0:
                answer = answers['text'][0]
            else:
                answer = str(answers['text'])
            
            samples.append((question, answer, context))
        
        return samples

    def get_fixed_sample(self, index: int = 0) -> Tuple[str, str, str]:
        """
        특정 인덱스의 샘플 반환 (디버깅 및 테스트용)
        
        Args:
            index (int): 데이터셋 내 특정 인덱스 (기본값: 0)
            
        Returns:
            Tuple[str, str, str]: (질문, 정답, 지문)
        """
        sample = self.dataset[index]
        
        # get_random_samples와 동일한 로직 사용
        question = sample['question']
        context = sample['context']  # 지문 추가
        
        # answers 필드에서 첫 번째 답변 추출
        answers = sample['answers']
        if isinstance(answers['text'], list) and len(answers['text']) > 0:
            answer = answers['text'][0]
        else:
            answer = str(answers['text'])
        
        return question, answer, context

    def save_random_samples(self, num_samples: int = 10, output_format: str = "json") -> str:
        """
        랜덤 샘플들을 파일로 저장하여 데이터 구조 확인
        
        Args:
            num_samples (int): 저장할 샘플 개수 (기본: 10개)
            output_format (str): 출력 형식 ("json" 또는 "csv", 기본: "json")
            
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
        for i in range(min(num_samples, len(self.dataset))):
            sample = random.choice(self.dataset)
            # 원본 구조 그대로 저장
            samples.append({
                "index": i,
                "raw_sample": dict(sample),  # 원본 데이터 구조
                "sample_keys": list(sample.keys()),  # 사용 가능한 키들
            })
        
        if output_format.lower() == "json":
            # JSON 형식으로 저장
            output_file = samples_dir / "klue_mrc_samples.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(samples, f, ensure_ascii=False, indent=2)
        
        elif output_format.lower() == "csv":
            # CSV 형식으로 저장
            import csv
            output_file = samples_dir / "klue_mrc_samples.csv"
            
            if samples:
                # 첫 번째 샘플의 키들로 헤더 생성
                fieldnames = ['index', 'sample_keys'] + list(samples[0]['raw_sample'].keys())
                
                with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for sample in samples:
                        row = {
                            'index': sample['index'],
                            'sample_keys': str(sample['sample_keys'])
                        }
                        # raw_sample의 모든 필드 추가
                        for key, value in sample['raw_sample'].items():
                            row[key] = str(value) if not isinstance(value, (str, int, float)) else value
                        writer.writerow(row)
        else:
            raise ValueError("output_format must be 'json' or 'csv'")
        
        logger.info(f"Saved {len(samples)} samples to {output_file}")
        logger.info(f"Sample structure analysis:")
        if samples:
            logger.info(f"  Available keys: {samples[0]['sample_keys']}")
            logger.info(f"  Sample data types:")
            for key, value in samples[0]['raw_sample'].items():
                logger.info(f"    {key}: {type(value).__name__} - {str(value)[:100]}...")
        
        return str(output_file)


if __name__ == "__main__":
    """
    스크립트를 직접 실행할 때 샘플 데이터 저장
    사용법: python utils/datasets/klue_mrc.py
    """
    try:
        print("KLUE MRC 데이터셋 샘플 저장 시작...")
        
        # 데이터셋 로드
        dataset = KlueMrcDataset(split="train")
        
        # JSON 형식으로 10개 샘플 저장
        json_file = dataset.save_random_samples(num_samples=10, output_format="json")
        print(f"✅ JSON 샘플 저장 완료: {json_file}")
        
        # CSV 형식으로도 저장 (선택사항)
        csv_file = dataset.save_random_samples(num_samples=5, output_format="csv")
        print(f"✅ CSV 샘플 저장 완료: {csv_file}")
        
        print("\n📋 데이터 구조 확인을 위해 저장된 파일을 확인하세요!")
        print("   - JSON 파일: 상세한 구조 분석용")
        print("   - CSV 파일: 엑셀에서 쉽게 확인 가능")
        
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        logger.error(f"Failed to save samples: {e}")