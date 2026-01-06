from fastapi import APIRouter, HTTPException, Request
import os
from dotenv import load_dotenv

from config.config import MODEL_DIR, BASE_DIR
from config.logger import logger

from src.model.hf_interaction import download_model_from_hf
from src.model.model_service import (
    get_model_signature, 
    get_model_status, 
    get_model_info, 
    get_prediction,
    load_model_instance
)

router = APIRouter()
load_dotenv(dotenv_path=BASE_DIR / ".devenv")

@router.get("/router_health")
async def router_health():
    return {'message':'Router is well linked to the app'}

@router.get("/model_status")
async def model_status():
    
    status = get_model_status()

    if status["exists"]:
        return {'message' : f'''Model ({status['model_name']}) is ready.\n
                {status['model_name']} is saved at this location : {status['path']}\n
                Model stats : size = {status['size_kb']}kb ; last modification = {status['last_modified']}'''}
    else:
        return {'message' : 'Model is not loaded and not ready for predictions.'}

@router.get("/model_signature")
async def model_signature():
    
    signature = get_model_signature()

    if signature["exists"]:
        return {'message' : 'Model signature :',
                'columns':signature['columns'],
                'nb_features':signature['nb_features']}
    else:
        return {'message' : 'No signature found'}

@router.get("/model_info")
async def model_info():
    
    infos = get_model_info()

    if infos["exists"]:
        return {'message': 'Model info retrieved', 'info': infos}
    else:
        return {'message' : 'No infos found'}
    
@router.post("/individual_score")
async def individual_score(request: Request, data: dict):
    model = getattr(request.app.state, "model", None)
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = get_prediction(model, data)
        return results
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/multiple_score")
async def multiple_score(request: Request, data_list: list[dict]):
    model = getattr(request.app.state, "model", None)
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # On pourrait optimiser ici mais pour l'instant un par un
        results = [get_prediction(model, d) for d in data_list]
        return results
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reload_model")
async def reload_model(request: Request):
    
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

        # Recharge le modèle en mémoire
        request.app.state.model = load_model_instance()

        return {'message':'Last version model has been well retrieved from HF and reloaded.'}
    except Exception as e :
        logger.error(f'An error occured : {e}')
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")
