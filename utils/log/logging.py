"""
@경로: utils/log/logging.py
@설명: 로깅 설정 및 전역 logger 객체
@사용법: 
from utils.log.logging import logger
logger.info("메시지")
"""

import logging
import sys
from pathlib import Path

class CustomFormatter(logging.Formatter):
    def format(self, record):
        record.filename = Path(record.pathname).name
        return super().format(record)

def _create_logger(name: str = "app_logger") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:  # 중복 핸들러 방지
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.flush = sys.stdout.flush # 강제 flush        
        # flush=True: Uvicorn 로그 버퍼링 문제 해결용
        # 파이썬은 상황에 따라 출력이 버퍼링된다. 
        # print가 출력한 내용을 버퍼에 쌓아두지 않고, 즉시 출력 스트림(보통 stdout)에 흘려보내라는 뜻으로, flush 옵션을 사용한다.
        formatter = CustomFormatter(
            fmt="%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) ↓\n %(message)s \n",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

# 전역 logger 객체 생성 
logger = _create_logger()
