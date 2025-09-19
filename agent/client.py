import os
from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Azure/OpenAI Setup ---

api_key = os.environ.get("Gemini_API_KEY", "")

az_model_client = OpenAIChatCompletionClient(
    model="gemini-2.5-flash",
    api_key=api_key,
    model_info=ModelInfo(
        vision=True,
        function_calling=True,
        json_output=True,
        family="unknown",
        structured_output=True,
        multiple_system_messages=True  # Enable multiple system messages
    )
) 
