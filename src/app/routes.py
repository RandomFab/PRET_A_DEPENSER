from fastapi import APIRouter
import os

from config.config import MODEL_DIR
from config.logger import logger

router = APIRouter()

@router.get("/router_health")
async def router_health():
    return {'message':'Router is well linked to the app'}

@router.post("/individual_score")
async def individual_score():
    return {'message':'You have well landed on "individual_score" endpoint...'}

@router.post("/multiple_score")
async def multiple_score():
    return {'message':'You have well landed on "multiple_score" endpoint...'}

@router.post("/reload_model")
async def reload_model():
    from src.model.hf_interaction import download_model_from_hf
    from fastapi import HTTPException
    
    repo_id = os.getenv('HF_REPO_ID')
    filename = os.getenv('HF_FILENAME', 'model.cb')
    token = os.getenv('HUGGINGFACE_TOKEN', None)

    if not repo_id:
        logger.error('HF_REPO_ID has not been declared in your environment.')
        raise HTTPException(status_code=500, detail="HF_REPO_ID is not configured")

    try:
        logger.info(f"Downloading {filename} from {repo_id}...")
        local_path = download_model_from_hf(repo_id, filename, token=token, cache_dir=MODEL_DIR)
        logger.info(f"Model saved to: {local_path}")

        return {'message':'Last version model has been well retrieved from HF model space.'}
    except Exception as e :
        logger.error(f'An error occured : {e}')
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")
