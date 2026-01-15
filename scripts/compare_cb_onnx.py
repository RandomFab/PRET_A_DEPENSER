import time
import os
import numpy as np
import onnxruntime as ort
from catboost import CatBoostClassifier
from config.config import MODEL_DIR
from src.model.model_service import get_model_signature, convert_catboost_to_onnx
from src.api.database.database import SessionLocal
from src.api.database.table_models import PredictionLog

def get_real_data(n_features, columns_order):
    """Récupère les données réelles de la base de données."""
    db = SessionLocal()
    try:
        logs = db.query(PredictionLog).limit(1000).all()
        if not logs:
            print("⚠️ Base de données vide, repli sur des données aléatoires.")
            return np.random.rand(1, n_features).astype(np.float32)

        data_list = []
        for log in logs:
            row = []
            for col in columns_order:
                val = log.inputs.get(col['name'], 0.0) # 0.0 par défaut pour les manquant
                row.append(val if val is not None else 0.0)
            data_list.append(row)
        
        print(f"✅ {len(data_list)} lignes réelles récupérées depuis la DB.")
        return np.array(data_list).astype(np.float32)
    except Exception as e:
        print(f"⚠️ Erreur DB : {e}. Repli sur des données aléatoires.")
        return np.random.rand(1, n_features).astype(np.float32)
    finally:
        db.close()

def run_benchmark():
    cb_path = MODEL_DIR / "model.cb"
    onnx_path = MODEL_DIR / "model.onnx"

    # 1. Vérification / Conversion
    if not cb_path.exists():
        print(f"❌ Erreur : Modèle CatBoost introuvable à {cb_path}")
        return

    if not onnx_path.exists():
        print("🔄 Le modèle ONNX n'existe pas. Tentative de conversion...")
        convert_catboost_to_onnx(cb_path, onnx_path)

    # 2. Chargement des modèles
    print("ℹ️ Chargement des modèles...")
    model_cat = CatBoostClassifier()
    model_cat.load_model(str(cb_path))
    
    session_onnx = None
    if onnx_path.exists():
        try:
            session_onnx = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
            print("✅ Modèle ONNX chargé.")
        except Exception as e:
            print(f"⚠️ Impossible de charger ONNX : {e}")

    # 3. Préparation des données de test
    signature = get_model_signature()
    n_features = signature['nb_features']
    if n_features == 0:
        n_features = len(model_cat.feature_names_)
    
    # Récupération des données réelles
    data = get_real_data(n_features, signature['columns'])
    num_samples = len(data)

    print(f"📊 Benchmark sur {n_features} features ({num_samples} samples)")

    # 4. Benchmark CatBoost
    start = time.time()
    for row in data:
        _ = model_cat.predict_proba([row])
    cb_time = (time.time() - start) / num_samples
    print(f"✅ Temps moyen CatBoost : {cb_time:.6f}s")

    # 5. Benchmark ONNX
    if session_onnx:
        input_name = session_onnx.get_inputs()[0].name
        start = time.time()
        for row in data:
            # Reshape car ONNX attend (1, n_features)
            row_reshaped = row.reshape(1, -1)
            _ = session_onnx.run(None, {input_name: row_reshaped})
        onnx_time = (time.time() - start) / num_samples
        print(f"✅ Temps moyen ONNX     : {onnx_time:.6f}s")
        if onnx_time > 0:
            print(f"🚀 Gain de vitesse      : {cb_time / onnx_time:.1f}x")
    else:
        print("❌ Benchmark ONNX impossible (problème de conversion ou de fichier)")

if __name__ == "__main__":
    run_benchmark()
