import pytest
from src.api.database.database import init_db, SessionLocal
from src.api.database.table_models import PredictionLog

def test_database_insertion():
    """
    Objectif : Vérifier qu'on peut créer la table, insérer une ligne 
    et la relire correctement.
    """
    # 1. Initialisation (Création des tables)
    init_db()
    
    db = SessionLocal()
    try:
        # 2. Préparation d'une donnée de test
        test_model_version = "v1_test"
        new_log = PredictionLog(
            model_version=test_model_version,
            latency_ms=10.0,
            status_code=200,
            inputs={"feature": 1},
            outputs={"pred": 0}
        )

        # 3. Action : Insertion
        db.add(new_log)
        db.commit()
        db.refresh(new_log) # On récupère l'ID généré par la DB

        # 4. ASSERTIONS : C'est ici que pytest travaille vraiment
        # On vérifie que l'objet a bien un ID (donc qu'il est en base)
        assert new_log.id is not None
        
        # On vérifie que ce qu'on relit est bien ce qu'on a envoyé
        queried_log = db.query(PredictionLog).filter(PredictionLog.id == new_log.id).first()
        assert queried_log.model_version == test_model_version
        assert queried_log.inputs["feature"] == 1

    finally:
        # On nettoie la session à la fin
        db.close()