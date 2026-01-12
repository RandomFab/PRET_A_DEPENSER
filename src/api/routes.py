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
from src.api.schemas import (
    ScoringData, 
    PredictionResponse, 
    ModelStatusResponse, 
    ModelSignatureResponse, 
    ModelInfoResponse
)

router = APIRouter()
load_dotenv(dotenv_path=BASE_DIR / ".devenv")

@router.get("/router_health")
async def router_health():
    """Verifies the router is properly connected to the main application."""
    try:
        return {'message':'Router is well linked to the app'}
    except Exception as e:
        logger.error(f"❌ Router Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Router connection issue.")

@router.get("/model_status", response_model=ModelStatusResponse)
async def model_status():
    """
    Retrieve the status of the model file on disk.
    Returns name, path, size and last modification time.
    """
    try:
        status = get_model_status()
        if not status["exists"]:
            logger.warning("⚠️ Model status requested but model file is missing")
            raise HTTPException(status_code=404, detail="Model file not found on disk.")
        
        return {
            "message": f"Model ({status['model_name']}) is ready.",
            "status": status
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error checking model status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while checking model status.")

@router.get("/model_signature", response_model=ModelSignatureResponse)
async def model_signature():
    """
    Retrieve the model signature (expected input variables).
    Provides the list of columns and total number of features.
    """
    try:
        signature = get_model_signature()
        if not signature["exists"]:
            logger.warning("⚠️ Model signature requested but MLmodel file is missing")
            raise HTTPException(status_code=404, detail="Signature file (MLmodel) not found.")
        
        return {
            "message": "Model signature retrieved",
            "columns": signature["columns"],
            "nb_features": signature["nb_features"]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error retrieving signature: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while retrieving signature.")

@router.get("/model_info", response_model=ModelInfoResponse)
async def model_info():
    """
    Provide detailed information about the model.
    Includes CatBoost version, MLflow ID, creation date and decision threshold.
    """
    try:
        infos = get_model_info()
        if not infos["exists"]:
            logger.warning("⚠️ Model info requested but MLmodel metadata is missing")
            raise HTTPException(status_code=404, detail="Model information not found.")
        
        return {"message": "Model info retrieved", "info": infos}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error retrieving model info: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while retrieving model info.")

@router.post("/individual_score", response_model=PredictionResponse)
async def individual_score(
    request: Request, 
    data: ScoringData
):
    """
    Compute a creditworthiness score for an individual client.
    Accepts client features and returns probability and credit decision.
    """
    model = getattr(request.app.state, "model", None)
    if not model:
        logger.error("❌ Scoring failed: Model is not loaded in app state")
        raise HTTPException(status_code=503, detail="Model is currently not loaded on the server. Please reload it.")
    
    try:
        # On convertit l'objet Pydantic en dict pour le service
        data_dict = data.model_dump()
        results = get_prediction(model, data_dict)
        if "error" in results:
            raise HTTPException(status_code=400, detail=results["error"])
        return results
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred during prediction: {str(e)}")

@router.post("/multiple_score", response_model=list[PredictionResponse])
async def multiple_score(request: Request, data_list: list[ScoringData]):
    """
    Perform batch predictions for a list of clients.
    Returns a list of results with scores and decisions for each entry.
    """
    model = getattr(request.app.state, "model", None)
    if not model:
        logger.error("❌ Bulk scoring failed: Model is not loaded")
        raise HTTPException(status_code=503, detail="Model is not loaded")
    
    try:
        # Conversion du batch en liste de dicts
        batch_dicts = [d.model_dump() for d in data_list]
        results = []
        for d in batch_dicts:
            res = get_prediction(model, d)
            if "error" in res:
                # Si erreur dans un item, on stoppe ou on gère ? 
                # On va lever une exception pour simplifier.
                raise HTTPException(status_code=400, detail=f"Error in batch item: {res['error']}")
            results.append(res)
        return results
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Bulk prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reload_model")
async def reload_model(request: Request):
    """
    Trigger download of the latest model version from Hugging Face.
    After download, the model is reloaded into RAM immediately.
    """
    repo_id = os.getenv('HF_REPO_ID')
    filename = os.getenv('HF_FILENAME', 'model.cb')
    token = os.getenv('HUGGINGFACE_TOKEN', None)

    if not repo_id:
        logger.error('❌ HF_REPO_ID has not been declared in your environment.')
        raise HTTPException(status_code=500, detail="HF_REPO_ID is not configured")

    try:
        logger.info(f"ℹ️ Manual reload requested. Downloading {filename} from {repo_id}...")
        local_path = download_model_from_hf(repo_id, filename, token=token, cache_dir=MODEL_DIR)
        
        # Recharge le modèle en mémoire
        new_model = load_model_instance()
        if new_model:
            request.app.state.model = new_model
            logger.info("✅ Model reloaded and updated in app state")
            return {'message':'✅ Last version model has been well retrieved from HF and reloaded in memory.'}
        else:
            logger.error("❌ Reload failed: Failed to create model instance from downloaded file")
            raise HTTPException(status_code=500, detail="Failed to load the downloaded model into memory.")
            
    except Exception as e :
        logger.error(f'❌ Reload error: {e}')
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")
