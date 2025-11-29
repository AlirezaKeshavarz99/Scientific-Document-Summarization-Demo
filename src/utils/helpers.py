# src/utils/helpers.py
import os
import json
import yaml
import logging
from typing import Dict, Any
import torch

def setup_logging(log_file: str = "summarization.log"):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

def save_json(data: Dict, file_path: str):
    folder = os.path.dirname(file_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(file_path: str) -> Dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def ensure_directory(path: str):
    os.makedirs(path, exist_ok=True)
