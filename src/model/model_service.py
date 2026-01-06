from pathlib import Path
import os
import time
from config.config import MODEL_DIR
import json

def get_model_status():
        
    filename = os.getenv('HF_FILENAME', 'model.cb')
    model_path = MODEL_DIR / filename

    exists = model_path.exists()
    status = {
        "model_name": filename,
        "exists": exists,
        "path": str(model_path),
    }

    if exists:
        stats = model_path.stat()
        status.update({
            "size_kb": round(stats.st_size / 1024, 2),
            "last_modified": time.ctime(stats.st_mtime)
        })

    return status

def get_model_signature():

    signature_path = MODEL_DIR / "input_example.json"

    exists = signature_path.exists()

    with open(signature_path,'r', encoding='utf-8') as f:
        signature = json.load(f)

    if exists:
        result = {  "exists" : exists,
                    "signature": signature,
                    "nb_features" : len(signature['columns'])
                }
        
    else:
        result = {  "exists" : exists,
                    "signature": {}
                }
        
    return result