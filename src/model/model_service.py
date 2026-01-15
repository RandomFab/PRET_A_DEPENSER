from pathlib import Path
import os
import time
from config.config import MODEL_DIR
import yaml
from catboost import CatBoostClassifier
from config.logger import logger
from src.api.schemas import ScoringData
import functools
import onnxruntime as ort
import numpy as np

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
# --- Chargement de la signature au démarrage ---
@functools.lru_cache(maxsize=1)
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

    # Enrichir avec les descriptions du ScoringData
    enriched_columns = []
    pydantic_fields = ScoringData.model_fields
    
    for col in columns:
        if isinstance(col, dict):
            name = col.get("name")
            # On récupère la description Pydantic si elle existe
            field_info = pydantic_fields.get(name)
            if field_info and field_info.description:
                col["description"] = field_info.description
            enriched_columns.append(col)
        else:
            # Si c'est juste une liste de strings
            field_info = pydantic_fields.get(col)
            desc = field_info.description if field_info else None
            enriched_columns.append({"name": col, "type": "double", "description": desc})

    threshold = config.get('metadata', {}).get('best_threshold', None)

    return {
        "exists": True,
        "columns": enriched_columns,
        "nb_features": len(enriched_columns),
        "best_threshold": threshold
    }

@functools.lru_cache(maxsize=1)
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
    """Charge le modèle ONNX (prioritaire) ou CatBoost en mémoire."""
    onnx_path = MODEL_DIR / "model.onnx"
    cb_filename = os.getenv('HF_FILENAME', 'model.cb')
    cb_path = MODEL_DIR / cb_filename
    
    # 1. Essayer de charger ONNX
    if onnx_path.exists():
        try:
            logger.info(f"ℹ️ Loading ONNX model from {onnx_path}...")
            # On utilise le CPU par défaut pour la portabilité
            session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
            logger.info("✅ ONNX model loaded successfully")
            return session
        except Exception as e:
            logger.error(f"⚠️ Failed to load ONNX model, falling back to CatBoost: {e}")

    # 2. Fallback sur CatBoost
    if not cb_path.exists():
        logger.warning(f"⚠️ Model file not found at {cb_path}")
        return None
        
    try:
        logger.info(f"ℹ️ Loading CatBoost model from {cb_path}...")
        model = CatBoostClassifier()
        model.load_model(str(cb_path))
        logger.info(f"✅ CatBoost model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load CatBoost model: {e}")
        return None

def convert_catboost_to_onnx(cb_path: Path, onnx_path: Path):
    """Convertit un modèle CatBoost existant au format ONNX."""
    try:
        logger.info(f"🔄 Converting CatBoost model {cb_path.name} to ONNX...")
        model = CatBoostClassifier()
        model.load_model(str(cb_path))
        
        model.save_model(
            str(onnx_path),
            format="onnx",
            export_parameters={
                'onnx_domain': 'ai.catboost',
                'onnx_model_version': 1,
                'onnx_doc_string': 'Model for credit scoring',
                'onnx_graph_name': 'CatBoostModel'
            }
        )
        logger.info(f"✅ Model successfully converted to {onnx_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Conversion to ONNX failed: {e}")
        return False

def get_prediction(model, data_dict: dict):
    if model is None:
        logger.error("❌ Prediction failed: No model instance provided")
        return {'error': 'Model instance is missing'}
    
    signature = get_model_signature()
    if not signature['exists']:
        logger.error("❌ Prediction failed: Signature file 'MLmodel' not found")
        return {'error' : 'Signature file not found'}
    
    columns = signature['columns']
    ordered_values = [data_dict.get(col['name']) for col in columns]
    
    best_threshold = signature.get('best_threshold') or 0.5

    try:
        # --- Cas 1 : Modèle ONNX ---
        if isinstance(model, ort.InferenceSession):
            input_name = model.get_inputs()[0].name
            # ONNX attend un array numpy 2D (batch_size, n_features)
            # On convertit les None éventuels en 0.0 pour éviter les crashs ONNX
            clean_values = [v if v is not None else 0.0 for v in ordered_values]
            input_data = np.array([clean_values], dtype=np.float32)
            
            # session.run renvoie [labels, probabilities]
            _, probas = model.run(None, {input_name: input_data})
            # probas est de forme (1, 2) -> [[p0, p1]]
            proba = float(probas[0][1])

        # --- Cas 2 : Modèle CatBoost Classique ---
        else:
            proba_array = model.predict_proba([ordered_values]) 
            proba = float(proba_array[0][1])
        
        is_refused = int(proba >= best_threshold)
        decision = "Refusé" if is_refused == 1 else "Accordé"

        result = {
            "score": round(proba, 4),
            "prediction": is_refused,
            "threshold": best_threshold,
            "decision": decision
        }
        
        logger.info(f"✅ Prediction successful: Decision={decision}, Score={round(proba, 4)}, (Mode={'ONNX' if isinstance(model, ort.InferenceSession) else 'CatBoost'})")
        return result

    except Exception as e:
        logger.error(f"❌ Prediction computation error: {e}")
        return {"error": str(e)}
