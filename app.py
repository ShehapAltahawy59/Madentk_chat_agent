from fastapi import FastAPI
import uvicorn
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from routes.chat import router as chat_router
    logger.info("Successfully imported chat router")
except Exception as e:
    logger.error(f"Failed to import chat router: {e}")
    chat_router = None

app = FastAPI()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Service is running"}

@app.get("/")
async def root():
    return {"message": "SmartFood Agent API"}

if chat_router:
    app.include_router(chat_router)
    logger.info("Chat router included in app")
else:
    logger.warning("Chat router not available")

@app.on_event("startup")
async def startup_event():
    logger.info("Application starting up...")
    logger.info(f"Python version: {sys.version}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
