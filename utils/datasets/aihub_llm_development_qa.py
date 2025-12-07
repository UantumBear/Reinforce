"""
@경로 : utils/datasets/aihub_llm_development_qa.py
@설명 : Hugging Face에서 AIHub LLM Development QA 데이터를 로드하고 랜덤 샘플을 제공하는 클래스
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

class AihubLlmDevelopmentQaDataset:
    """AIHub LLM Development QA 데이터셋 관리 클래스
    --> 살펴본 결과, 데이터 셋이 부적합하다.
    RAG 챗봇을 테스트해보려면,
    유저 질문 : 그때의 모범답안 : 그때 참고하는 문서 데이터가 있어야 한다..
    그리고, 이 데이터셋은 문서정보는 항상 없는데도, '문서가 없습니다.' 와 같은 샘플응답이 있기에
    내가 사용하기엔 부적합한듯하다.
    """
    
    def __init__(self, split: str = "train"):
        """
        데이터셋 초기화 및 로드
        
        Args:
            split (str): 사용할 데이터 분할 ('train', 'validation' 등)
        """
        logger.info(f"Loading AIHub LLM Development QA dataset (split={split})...")
        try:
            # lmqg/qg_koquad 데이터셋 사용 (질문 생성/답변용으로 정제된 버전)
            self.dataset = load_dataset("AtwoM/aihub_llm_development_qa", split=split)

            logger.info(f"Successfully loaded {len(self.dataset)} samples.")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise e

    def get_random_samples(self, batch_size: int = 1) -> list:
        """
        데이터셋에서 랜덤하게 질문과 정답 쌍을 반환
        
        Args:
            batch_size (int): 반환할 샘플 개수
        
        Returns:
            list: [(질문1, 정답1), (질문2, 정답2), ...] 형태의 튜플 리스트
        """
        if not self.dataset:
            raise ValueError("Dataset not loaded.")
        
        samples = []
        
        for _ in range(batch_size):
            sample = random.choice(self.dataset)
            
            # AI Hub LLM Development QA 실제 구조: {'prompt': ..., 'completion': ..., ...}
            question = sample['prompt']
            answer = sample['completion']
            
            samples.append((question, answer))
        
        return samples

    def get_fixed_sample(self, index: int = 0) -> Tuple[str, str]:
        """
        특정 인덱스의 샘플 반환 (재현성 확인용)
        """
        sample = self.dataset[index]
        
        # AI Hub LLM Development QA 실제 구조에 맞게 처리
        question = sample['prompt']
        answer = sample['completion']
        
        return question, answer

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
            output_file = samples_dir / "aihub_llm_development_qa_samples.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(samples, f, ensure_ascii=False, indent=2)
        
        elif output_format.lower() == "csv":
            # CSV 형식으로 저장
            import csv
            output_file = samples_dir / "aihub_llm_development_qa_samples.csv"
            
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
    사용법: python utils/datasets/aihub_llm_development_qa.py
    """
    try:
        print("AIHub LLM Development QA 데이터셋 샘플 저장 시작...")
        
        # 데이터셋 로드
        dataset = AihubLlmDevelopmentQaDataset(split="train")
        
        # JSON 형식으로 10개 샘플 저장
        json_file = dataset.save_random_samples(num_samples=10, output_format="json")
        print(f"[SUCCESS] JSON 샘플 저장 완료: {json_file}")
        
        # CSV 형식으로도 저장 (선택사항)
        csv_file = dataset.save_random_samples(num_samples=5, output_format="csv")
        print(f"[SUCCESS] CSV 샘플 저장 완료: {csv_file}")
        
        print("\n 데이터 구조 확인을 위해 저장된 파일을 확인하세요!")
        print("   - JSON 파일: 상세한 구조 분석용")
        print("   - CSV 파일: 엑셀에서 쉽게 확인 가능")
        
    except Exception as e:
        print(f"[ERROR] 에러 발생: {e}")
        logger.error(f"Failed to save samples: {e}")