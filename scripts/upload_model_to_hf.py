import os
from pathlib import Path
from dotenv import load_dotenv
from config.config import BASE_DIR, MODEL_DIR
from config.logger import logger
from src.model.hf_interaction import upload_model_to_hf
from src.model.mlflow_interaction import set_tracking_uri, download_model_artifacts, find_model_file

load_dotenv()

if __name__ == '__main__':
    repo_id = os.getenv('HF_REPO_ID')
    if not repo_id:
        logger.error('HF_REPO_ID n\'est pas défini dans les variables d\'environnement')
        raise SystemExit(1)

    token = os.getenv('HUGGINGFACE_TOKEN', None)

    # Set MLflow tracking if provided
    set_tracking_uri()

    # Download model artifacts from MLflow into MODEL_DIR
    model_uri = os.getenv('MLFLOW_MODEL_URI')
    if not model_uri:
        logger.error('MLFLOW_MODEL_URI non défini dans les variables d\'environnement')
        raise SystemExit(1)

    try:
        download_model_artifacts(model_uri, dst_dir=MODEL_DIR)
    except Exception as e:
        logger.error(f"Impossible de télécharger les artifacts MLflow: {e}")
        raise

    # Find model file
    # model_file = find_model_file(MODEL_DIR)
    # if not model_file:
    #     logger.error('Aucun fichier de modèle trouvé après téléchargement des artifacts')
    #     raise SystemExit(1)

    # path_in_repo = os.getenv('HF_PATH_IN_REPO', model_file.name)

    # logger.info(f"Upload: local_file={model_file}, repo_id={repo_id}, path_in_repo={path_in_repo}")
    # result = upload_model_to_hf(str(model_file), repo_id, path_in_repo, token=token)
    
    # Upload de tout le dossier MODEL_DIR
    logger.info(f"Upload du dossier complet: {MODEL_DIR} -> {repo_id}")
    result = upload_model_to_hf(MODEL_DIR, repo_id, path_in_repo=".", token=token)
    
    logger.info(f"Upload result: {result}")
