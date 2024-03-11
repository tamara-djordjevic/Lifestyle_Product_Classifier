from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import image_classification

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*']
)

app.include_router(image_classification.router)
