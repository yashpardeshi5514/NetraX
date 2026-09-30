from contextlib import asynccontextmanager
from app.routes.auth import router as auth_router
from app.routes.analysis import router as analysis_router
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.services.inference import netrax_inference
from app.routes.prediction import router as prediction_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    netrax_inference.load()
    yield


app = FastAPI(
    title="NetraX API",
    description="AI-powered retinal image analysis API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(auth_router)
app.include_router(prediction_router)
app.include_router(analysis_router)


@app.get("/")
def root():
    return {
        "name": "NetraX API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }