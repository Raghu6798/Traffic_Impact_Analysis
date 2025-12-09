from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.v1.agent_router import router as agent_router
from src.config.settings import get_settings

app = FastAPI(title="TIA Agent API")

app.include_router(agent_router)

origins = [
    "https://tia-frontend.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def root():
    return {"message": "Welcome to TIA Agent API"}

@app.get("/health")
def health():
    return {"status": "active"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)