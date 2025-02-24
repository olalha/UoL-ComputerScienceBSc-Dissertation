import os
from pathlib import Path

from utils.settings_manager import get_setting
from chunk_manager.rulebook_parser import parse_rulebook_excel
from dataset_manager.generate_dataset import generate_dataset_outline

""" Environment setup """

# Get absolute path to .env file
current_dir = Path(__file__).resolve().parent
env_path = current_dir / '.env'

# Check if environment variables are set
required_env_vars = ['OPENAI_API_KEY']
missing_vars = [var for var in required_env_vars if not os.getenv(var)]

if missing_vars:
    raise ValueError(f"Error: Missing required environment variables: {', '.join(missing_vars)}")

""" Review content generation """

# Parse Excel rulebook
RULEBOOK_NAME = "Laptop---word_count.xlsx"
COLLECTION_MODE = "word"

rulebook = parse_rulebook_excel(rulebook_name=RULEBOOK_NAME, collection_mode=COLLECTION_MODE)
if not rulebook:
    raise ValueError(f"Error: Parsing rulebook {RULEBOOK_NAME} failed.")

SEARCH_TIME_S = 10
dataset_outline = generate_dataset_outline(rulebook=rulebook,
                                           collection_mode=COLLECTION_MODE, 
                                           solution_search_time_s=SEARCH_TIME_S)

import json
data_dir = current_dir / "_data"
data_dir.mkdir(parents=True, exist_ok=True)
output_path = data_dir / "dataset_outline.json"

with open(output_path, 'w') as f:
    json.dump(dataset_outline, f, indent=4)

MODEL = get_setting('OPENAI_LLM_MODELS', 'GPT4o-mini')

"""
So far the program will take the inputted rulebook, allocate the chunks, generate the dataset, and then print it to the console.
"""
