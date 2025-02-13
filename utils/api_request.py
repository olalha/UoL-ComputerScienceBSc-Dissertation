"""
This module provides functions for making API requests to OpenAI's LLM service.
It includes utilities for sending requests and handling responses.
"""

import asyncio
import aiohttp
import os
from typing import List, Dict

async def _send_openai_request(json_content: dict) -> Dict:
    """
    Send an asynchronous request to the OpenAI API.
    
    Args:
        json_content (dict): The request payload to send to the API.
    
    Returns:
        Dict: The JSON response from the API.
    """
    # Get the API key from the environment
    api_key = os.getenv('OPENAI_API_KEY')
    
    # Set the request URL and headers
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Send the request
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, headers=headers, json=json_content) as response:
                # Check if the request was successful
                if response.status != 200:
                    print(f"send_openai_request: API request failed with status code {response.status}")
                    return None
                return await response.json()
        # Handle client errors
        except aiohttp.ClientError as e:
            print(f"send_openai_request: Client error occurred: {e}")
            return None

def _validate_response_choices(response: Dict, messages: List[Dict]) -> Dict:
    """
    Checks if an API response indicates a failed request.
    
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

def prompt_llm_parallel(model: str, messages: List[List[Dict]]) -> List[Dict]:
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
            result = _validate_response_choices(response, msg)
            result['idx'] = idx
            return result
        
        # Asynchronously process all requests
        tasks = [process_request(msg, idx) for idx, msg in enumerate(messages)]
        return await asyncio.gather(*tasks)
    
    return asyncio.run(async_wrapper())

def prompt_llm_single(model: str, messages: List[Dict]) -> Dict:
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
    return _validate_response_choices(response, messages)
