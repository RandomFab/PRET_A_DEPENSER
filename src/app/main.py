from fastapi import FastAPI
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
    return RedirectResponse("/docs")

@app.get("/api_health")
async def router_health():
    return {'message':'API is running correctly'}



app.include_router(router)