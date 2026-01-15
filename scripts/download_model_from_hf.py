import os
from dotenv import load_dotenv
from config.config import BASE_DIR, MODEL_DIR
from config.logger import logger
from src.model.hf_interaction import download_repo_from_hf

load_dotenv(dotenv_path=BASE_DIR / ".env.dev")

if __name__ == '__main__':
    repo_id = os.getenv('HF_REPO_ID')
    token = os.getenv('HUGGINGFACE_TOKEN', None)

    if not repo_id:
        logger.error('HF_REPO_ID n\'est pas défini dans les variables d\'environnement')
        raise SystemExit(1)

    logger.info(f"Downloading entire repo from {repo_id}...")
    local_path = download_repo_from_hf(repo_id, token=token, local_dir=MODEL_DIR)
    logger.info(f"Model saved to: {local_path}")
