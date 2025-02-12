import os
import json

from pathlib import Path
from dotenv import load_dotenv

from utils.api_request import prompt_llm_single, prompt_llm_parallel

if __name__ == '__main__':

    # Get absolute path to .env file
    current_dir = Path(__file__).resolve().parent
    env_path = current_dir / '.env'

    # Load environment variables
    # Check if environment variables are set
    required_env_vars = ['OPENAI_API_KEY']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_vars:
        raise ValueError(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
    
    """ API request to OpenAI """
    
    messages = [
        [{"role": "user", "content": "Hello, how are you?"}],
        [{"role": "user", "content": "What is 1+1?"}],
    ]
    
    responses = prompt_llm_parallel(model="gpt-4o-mini", messages=messages)
    
    for r in responses:
        print(json.dumps(r['response'], indent=4))
    
