from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager
from src.app.routes import router
from src.model.model_service import load_model_instance
from config.logger import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Démarrage : Charge le modèle dans l'état de l'application
    logger.info("ℹ️ Application starting up...")
    app.state.model = load_model_instance()
    if app.state.model:
        logger.info("✅ Model successfully injected into app state")
    else:
        logger.error("❌ Application started WITHOUT an active model")
    yield
    # Arrêt : On peut nettoyer ici si besoin
    logger.info("ℹ️ Application shutting down...")
    if hasattr(app.state, "model"):
        del app.state.model
    logger.info("✅ Cleanup completed")

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    """Redirects the user to the interactive documentation (Swagger UI)."""
    return RedirectResponse("/docs")

@app.get("/api_health")
async def router_health():
    """Checks that the API is operational and responding correctly."""
    try:
        return {'message':'API is running correctly'}
    except Exception as e:
        logger.error(f"❌ API Health check failed: {e}")
        raise HTTPException(status_code=500, detail="API is experiencing issues.")



app.include_router(router)