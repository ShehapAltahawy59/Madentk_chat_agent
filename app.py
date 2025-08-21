from fastapi import FastAPI
import uvicorn

from routes.chat import router as chat_router

app = FastAPI()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Service is running"}

@app.get("/")
async def root():
    return {"message": "SmartFood Agent API"}

app.include_router(chat_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
