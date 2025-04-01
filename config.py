import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API keys
HUGGINGFACE_API_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN')
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  # If you want to use OpenAI later

def check_api_keys():
    """Verify that required API keys are set"""
    if not HUGGINGFACE_API_TOKEN:
        raise ValueError("HUGGINGFACE_API_TOKEN not found in .env file")
    # Add other API key checks as needed 