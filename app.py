from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)
for name in ["httpx","autogen_core","autogen","autogen_agentchat","openai","google.generativeai"]:
  logging.getLogger(name).disabled = True
try:
    from routes.chat import router as chat_router
    logger.info("Successfully imported chat router")
except Exception as e:
    logger.error(f"Failed to import chat router: {e}")
    chat_router = None

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Service is running"}

@app.get("/")
async def root():
    return {"message": "SmartFood Agent API"}

@app.get("/debug")
async def debug_info():
    """Debug endpoint to check imports and routes"""
    import sys
    import os
    
    # Check if chat router was imported
    chat_router_status = "imported" if chat_router is not None else "failed"
    
    # Get available routes
    routes = []
    for route in app.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            routes.append({
                "path": route.path,
                "methods": list(route.methods) if route.methods else []
            })
    
    # Check imports
    import_status = {}
    
    try:
        import autogen_agentchat
        import_status["autogen_agentchat"] = "✅ OK"
    except ImportError as e:
        import_status["autogen_agentchat"] = f"❌ {str(e)}"
    
    try:
        import autogen_core
        import_status["autogen_core"] = "✅ OK"
    except ImportError as e:
        import_status["autogen_core"] = f"❌ {str(e)}"
    
    try:
        from agent.agent import smart_agent
        import_status["agent.agent"] = "✅ OK"
    except ImportError as e:
        import_status["agent.agent"] = f"❌ {str(e)}"
    
    try:
        from routes.chat import router
        import_status["routes.chat"] = "✅ OK"
    except ImportError as e:
        import_status["routes.chat"] = f"❌ {str(e)}"
    
    return {
        "chat_router_status": chat_router_status,
        "routes": routes,
        "total_routes": len(routes),
        "import_status": import_status,
        "python_version": sys.version,
        "working_directory": os.getcwd()
    }

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
    uvicorn.run(app, host="0.0.0.0", port=8080)
