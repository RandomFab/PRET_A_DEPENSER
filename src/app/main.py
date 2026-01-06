from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from src.app.routes import router
app = FastAPI()

@app.get("/")
async def root():
    return RedirectResponse("/docs")

@app.get("/api_health")
async def router_health():
    return {'message':'API is running correctly'}



app.include_router(router)