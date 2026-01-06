import os
from pathlib import Path
from typing import Optional
import mlflow
from config.config import BASE_DIR, MODEL_DIR
from config.logger import logger
from dotenv import load_dotenv

load_dotenv()


def set_tracking_uri(uri: Optional[str] = None) -> None:
    """Set MLflow tracking URI from argument or env var."""
    uri = uri or os.getenv('MLFLOW_TRACKING_URI')
    if uri:
        mlflow.set_tracking_uri(uri)
        logger.info(f"MLflow tracking URI set to: {uri}")
    else:
        logger.info("No MLFLOW_TRACKING_URI provided; using default MLflow settings")


def download_model_artifacts(model_uri: str, dst_dir: Optional[Path] = None) -> Path:
    """Download MLflow model artifacts to dst_dir (defaults to MODEL_DIR).

    Returns the local path where artifacts were downloaded (Path).
    """
    dst = Path(dst_dir or MODEL_DIR)
    dst.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading MLflow artifacts for '{model_uri}' into {dst}")
    try:
        local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=str(dst))
        logger.info(f"Artifacts downloaded to: {local_path}")
        return Path(local_path)
    except Exception as e:
        logger.error(f"Failed to download MLflow artifacts: {e}")
        raise


def find_model_file(base_dir: Optional[Path] = None, candidates=None) -> Optional[Path]:
    """Search recursively for common model filenames inside base_dir.

    Returns the Path to the first match or None.
    """
    base = Path(base_dir or MODEL_DIR)
    if candidates is None:
        candidates = ('model.cb', 'model.cbm', 'model.pkl', 'model.bin', 'model')

    for p in base.rglob('*'):
        if p.is_file() and p.name in candidates:
            logger.info(f"Found model file: {p}")
            return p
    logger.warning(f"No model file found in {base} among candidates {candidates}")
    return None
