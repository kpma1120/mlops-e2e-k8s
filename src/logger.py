import logging
import os
import sys
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR,exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")
LOG_STR = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"


def get_logger(name):
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_STR,
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(name)
    return logger