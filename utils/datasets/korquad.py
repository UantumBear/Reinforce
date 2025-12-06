"""
@경로: utils/datasets/korquad.py
@설명: HuggingFace의 KorQuAD v1.0 데이터셋을 로드하고 샘플 데이터를 저장/확인할 수 있는 클래스
@명령어: python utils/datasets/korquad.py
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

class KorQuADDataset:
    """KorQuAD/squad_kor_v1 데이터셋 관리 클래스"""
    
    def __init__(self, split="validation"):
        """
        KorQuAD 데이터셋 로드
        split: 'train' (학습용) 또는 'validation' (평가용)
        """
        logger.info(f"Loading KorQuAD v1.0 dataset ({split})...")
        
        # HuggingFace 다운로드 경로 정보 로깅
        dataset_name = "squad_kor_v1"
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
                    cache_dir = config.HF_DATASETS_CACHE # datasets 라이브러리 2.x 이상
                except (ImportError, AttributeError):
                    import os
                    cache_dir = os.path.expanduser("~/.cache/huggingface/datasets") # 기본 경로
            
            logger.info(f"Local cache directory: {cache_dir}")
            
            self.dataset = load_dataset(dataset_name, split=split)
            logger.info(f"KorQuAD dataset loaded successfully. Size: {len(self.dataset)}")
            
            # 데이터셋 메타데이터 정보 로깅
            if hasattr(self.dataset, 'info'):
                logger.info(f"Dataset info: {self.dataset.info}")
            if hasattr(self.dataset, 'builder_name'):
                logger.info(f"Builder name: {self.dataset.builder_name}")
            if hasattr(self.dataset, 'config_name'):
                logger.info(f"Config name: {self.dataset.config_name}")
        except Exception as e:
            logger.error(f"Failed to load KorQuAD dataset: {e}")
            raise

    def get_random_samples(self, n=1):
        """
        랜덤하게 n개의 (질문, 정답, 지문) 샘플을 반환
        
        Returns:
            list of tuple: [(question, answer, context), ...]
        """
        indices = random.sample(range(len(self.dataset)), n)
        results = []
        
        for idx in indices:
            item = self.dataset[idx]
            context = item['context']
            question = item['question']
            answer = item['answers']['text'][0] # 첫 번째 정답 사용
            results.append((question, answer, context))
            
        return results

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
                "id": sample['id'],
                "title": sample['title'],
                "context": sample['context'],
                "question": sample['question'],
                "answers": sample['answers']['text'], # 정답 리스트 전체
                "answer_start": sample['answers']['answer_start']
            }
            
            samples.append({
                "index": i,
                "processed_sample": processed_sample,
                "raw_keys": list(sample.keys())
            })
        
        if output_format.lower() == "json":
            output_file = samples_dir / "korquad_samples.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(samples, f, ensure_ascii=False, indent=2)
                
        elif output_format.lower() == "csv":
            import csv
            output_file = samples_dir / "korquad_samples.csv"
            
            if samples:
                # 헤더 생성
                fieldnames = ['index', 'title', 'question', 'answers', 'context']
                
                with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for s in samples:
                        p = s['processed_sample']
                        writer.writerow({
                            'index': p['index'],
                            'title': p['title'],
                            'question': p['question'],
                            'answers': str(p['answers']), # 리스트를 문자열로 변환
                            'context': p['context']
                        })
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
    사용법: python utils/datasets/korquad.py
    """
    try:
        print("KorQuAD v1.0 데이터셋 샘플 저장 시작...")
        
        # 데이터셋 로드 (validation 셋이 가벼워서 테스트용으로 적합)
        dataset = KorQuADDataset(split="validation")
        
        # JSON 저장
        json_file = dataset.save_random_samples(num_samples=10, output_format="json")
        print(f"[SUCCESS] JSON 샘플 저장 완료: {json_file}")
        
        # CSV 저장
        csv_file = dataset.save_random_samples(num_samples=5, output_format="csv")
        print(f"[SUCCESS] CSV 샘플 저장 완료: {csv_file}")
        
        print("\n 데이터 확인:")
        print("   - datasets/samples 폴더를 확인하세요.")
        
    except Exception as e:
        print(f"[ERROR] 에러 발생: {e}")
        logger.error(f"Failed to save samples: {e}")