from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager
from src.app.routes import router
from src.model.model_service import load_model_instance

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Démarrage : Charge le modèle dans l'état de l'application
    app.state.model = load_model_instance()
    yield
    # Arrêt : On peut nettoyer ici si besoin
    del app.state.model

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return RedirectResponse("/docs")

@app.get("/api_health")
async def router_health():
    return {'message':'API is running correctly'}



app.include_router(router)