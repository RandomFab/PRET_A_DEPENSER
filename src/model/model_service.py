from pathlib import Path
import os
import time
from config.config import MODEL_DIR
import yaml
from catboost import CatBoostClassifier
from config.logger import logger

def get_model_status() -> dict:
        
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

def get_model_signature() -> dict:
    MLmodel_path = MODEL_DIR / "MLmodel"

    if not MLmodel_path.exists():
        return {
            "exists": False,
            "columns": [],
            "nb_features": 0,
            "best_threshold": None
        }

    with open(MLmodel_path, 'r') as f:
        config = yaml.safe_load(f)

    # MLflow can store signature as a JSON string inside the YAML
    raw_inputs = config.get('signature', {}).get('inputs', [])
    if isinstance(raw_inputs, str):
        import json
        columns = json.loads(raw_inputs)
    else:
        columns = raw_inputs

    threshold = config.get('metadata', {}).get('best_threshold', None)

    return {
        "exists": True,
        "columns": columns,
        "nb_features": len(columns),
        "best_threshold": threshold
    }

def get_model_info():
    MLmodel_path = MODEL_DIR / "MLmodel"

    if not MLmodel_path.exists():
        return {
            "exists": False,
            "model_type": None,
            "model_name": None,
            "mlflow_model_id": None,
            "created_on": None,
            "nb_feature": 0
        }

    with open(MLmodel_path, 'r') as f:
        config = yaml.safe_load(f)

    # On récupère les infos de la flavor catboost
    catboost_info = config.get('flavors', {}).get('catboost', {})

    return {
        "exists": True,
        "model_type": catboost_info.get('model_type'),
        "model_name": catboost_info.get('data'),
        "mlflow_model_id": config.get('model_id'),
        "created_on": config.get('utc_time_created'),
        "nb_feature": len(config.get('signature', {}).get('inputs', [])),
        "best_threshold" : config.get('metadata',{}).get('best_threshold',None)
    }

def load_model_instance():
    """Charge le modèle CatBoost en mémoire."""
    filename = os.getenv('HF_FILENAME', 'model.cb')
    model_path = MODEL_DIR / filename
    
    if not model_path.exists():
        logger.warning(f"⚠️ Model file not found at {model_path}")
        return None
        
    try:
        logger.info(f"ℹ️ Loading model from {model_path}...")
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        logger.info(f"✅ Model loaded successfully from {filename}")
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return None

def get_prediction(model: CatBoostClassifier, data_dict: dict):
    if model is None:
        logger.error("❌ Prediction failed: No model instance provided")
        return {'error': 'Model instance is missing'}
    
    signature = get_model_signature()

    if not signature['exists']:
        logger.error("❌ Prediction failed: Signature file 'MLmodel' not found")
        return {'error' : 'Signature file not found'}
    
    # Get columns to ordered values for the model
    columns = signature['columns']
    
    ordered_values = []
    for col in columns:
        val = data_dict.get(col['name'])
        ordered_values.append(val)
    
    # Get best threshold for prediction

    infos =  get_model_info()
    best_threshold = infos.get('best_threshold')
    threshold_state = "Best for this model"
    if best_threshold == None:
        best_threshold = 0.5
        threshold_state = "0.5 by default"
        logger.warning("⚠️ No threshold recommended for this model. A 0.5 threshold has been attributed by default")

    try:
        proba_array = model.predict_proba([ordered_values]) 
        proba = float(proba_array[0][1])
        
        # 1 = Défaut/Refusé, 0 = Accordé
        is_refused = int(proba >= best_threshold)
        decision = "Refusé" if is_refused == 1 else "Accordé"

        result = {
            "score": round(proba, 4),
            "prediction": is_refused,
            "threshold": best_threshold,
            "decision": decision
        }
        
        logger.info(f"✅ Prediction successful: Decision={decision}, Score={round(proba, 4)}, Threshold={best_threshold}")
        return result

    except Exception as e:
        logger.error(f"❌ Prediction computation error: {e}")
        return {"error": str(e)}
