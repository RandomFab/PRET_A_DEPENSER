import os
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, hf_hub_download
from config.config import BASE_DIR, MODEL_DIR
from config.logger import logger
from dotenv import load_dotenv

load_dotenv(dotenv_path=BASE_DIR / ".devenv")


def _get_token(token: Optional[str]) -> Optional[str]:
    return token or os.getenv("HUGGINGFACE_TOKEN")


def upload_model_to_hf(local_file_path: str | Path, repo_id: str, path_in_repo: Optional[str] = None, *, token: Optional[str] = None, repo_type: str = "model", commit_message: str = "Upload model") -> dict:
    """Upload a local file or folder to a Hugging Face model repo.
    
    If local_file_path is a directory, it uploads the entire folder content.
    Returns the API response dict on success.
    """
    local_path = Path(local_file_path)
    token = _get_token(token)
    api = HfApi(token=token)
    
    # Si path_in_repo n'est pas spécifié :
    # - pour un fichier : on garde le nom du fichier
    # - pour un dossier : on upload à la racine du repo (path_in_repo=".")
    if path_in_repo is None:
        path_in_repo = "." if local_path.is_dir() else local_path.name

    logger.info(f"Uploading {local_path} to HF repo {repo_id} (path: {path_in_repo})")
    
    try:
        if local_path.is_dir():
            # Upload de dossier complet
            result = api.upload_folder(
                folder_path=str(local_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=commit_message,
                token=token,
                delete_patterns="*" # Écrase les anciennes versions (supprime tout avant upload)
            )
        else:
            # Upload de fichier unique
            result = api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=commit_message,
                token=token,
            )
            
        logger.info(f"Upload successful: {result}")
        return result
    except Exception as e:
        logger.error(f"Failed to upload to HF: {e}")
        raise


def download_model_from_hf(repo_id: str, filename: str, repo_type: str = "model", *, token: Optional[str] = None, cache_dir: Optional[str | Path] = None) -> str:
    """Download a file from a Hugging Face repo and return local path.

    Example: download_model_from_hf('username/model', 'model.cb')
    """
    token = _get_token(token)
    cache_dir = Path(cache_dir or MODEL_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {filename} from {repo_id} to {cache_dir}")
    try:
        local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type, token=token, cache_dir=str(cache_dir))
        logger.info(f"Downloaded to {local_path}")
        return local_path
    except Exception as e:
        logger.error(f"Failed to download from HF: {e}")
        raise
