"""
This module provides functions for making API requests to OpenAI.
It includes utilities for sending requests and handling responses.
"""

import asyncio
import aiohttp
import os
import logging
from typing import List, Dict

from utils.settings_manager import get_setting

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load OpenAI API settings
OPENAI_RETRY_LIMIT = get_setting('OPENAI_API', 'retry_limit')
OPENAI_RATE_LIMIT_SLEEP_SEC = get_setting('OPENAI_API', 'rate_limit_sleep_sec')
OPENAI_MAX_CLIENT_TIMEOUT_SEC = get_setting('OPENAI_API', 'max_client_timeout_sec')

async def _send_openai_request(json_content: dict, retries = OPENAI_RETRY_LIMIT) -> Dict:
    """
    Send an asynchronous request to the OpenAI API.

    Args:
        json_content (dict): The request payload to send to the API.
        retries (int): The number of retry attempts for rate limiting.

    Returns:
        Dict: The JSON response from the API, or None if the request fails.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Initialize the client session with a timeout
    timeout = aiohttp.ClientTimeout(total=OPENAI_MAX_CLIENT_TIMEOUT_SEC)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            # Send the request
            async with session.post(url, headers=headers, json=json_content) as response:
                
                # Log error if response status is not 200 and retry if possible
                if response.status != 200:
                    if retries > 0:
                        if response.status == 429:
                            t = OPENAI_RATE_LIMIT_SLEEP_SEC
                            logger.warning(f"_send_openai_request: Rate limit hit - Waiting: {t}s - Retries left: {retries}")
                            await asyncio.sleep(t)
                        else: 
                            logger.warning(f"_send_openai_request: Request failed - Code {response.status} - Retries left: {retries}")
                        return await _send_openai_request(json_content, retries - 1)
                    
                    # Return none if max retries reached
                    logger.error(f"_send_openai_request: Request failed with max retries - Code {response.status}")
                    return None
                
                # Return the response as JSON
                return await response.json()
        
        # Handle client errors
        except aiohttp.ClientError as e:
            logger.error("Client error occurred: %s", e)
            return None

def _validate_openai_response(response: Dict, messages: List[Dict]) -> Dict:
    """
    Checks if the OpenAI API response indicates a failed request.
    
    Args:
        response (Dict): The API response to validate.
        messages (List[Dict]): The original message list that generated this response.
    
    Returns:
        Dict: A dictionary containing a success flag, original messages, and response data.
    """
    # Check if the response is empty
    if not response:
        return {'success': False, 'messages': messages, 'response': {}}
    
    # Check if the response did not succeed
    if 'error' in response:
        error_message = response.get('error', {}).get('message', 'Unknown error occurred')
        print(f"validate_response_choices: API request failed: \n{messages} \n\n{error_message}")
        return {'success': False, 'messages': messages, 'response': response}
    
    # Check if the response contains choices
    if 'choices' not in response or not response['choices']:
        print(f"validate_response_choices: 'choices' not found or empty in api response: \n{messages}")
        return {'success': False, 'messages': messages, 'response': response}
    
    return {'success': True, 'messages': messages, 'response': response}

def prompt_openai_llm_parallel(model: str, messages: List[List[Dict]]) -> List[Dict]:
    """
    Send prompts to the specified LLM model via OpenAI API in parallel.
    
    Args:
        model (str): The name of the LLM model to use.
        messages (List[List[Dict]]): A list of message lists, each representing a separate request.
    
    Returns:
       List[Dict]: A list of processed API responses. Each is a dictionary containing a success flag,
            the original message list, an index and the response dictionary.
    """
    # Asynchronous wrapper function
    async def async_wrapper():
        
        # Asynchronous request processing function
        async def process_request(msg, idx):
            response = await _send_openai_request({'model': model, 'messages': msg})
            result = _validate_openai_response(response, msg)
            result['idx'] = idx
            return result
        
        # Asynchronously process all requests
        tasks = [process_request(msg, idx) for idx, msg in enumerate(messages)]
        return await asyncio.gather(*tasks)
    
    return asyncio.run(async_wrapper())

def prompt_openai_llm_single(model: str, messages: List[Dict]) -> Dict:
    """
    Send a prompt to the specified LLM model via OpenAI API.
    
    Args:
        model (str): The name of the LLM model to use.
        messages (List[Dict]): A list of message dictionaries for one prompt.
    
    Returns:
        Dict: A processed API response containing a success flag,
              the original message list, and the response dictionary.
    """
    # Send the request and validate the response
    response = asyncio.run(_send_openai_request({'model': model, 'messages': messages}))
    return _validate_openai_response(response, messages)
