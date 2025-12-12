from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

from src.api.v1.agent_router import router as agent_router
from src.config.settings import get_settings

app = FastAPI(title="TIA Agent API")

origins = [
    "https://tia-frontend.vercel.app",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(agent_router)

handler = Mangum(app)

@app.get('/')
def root():
    return {"message": "Welcome to TIA Agent API"}

@app.get("/health")
def health():
    return {"status": "active"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
