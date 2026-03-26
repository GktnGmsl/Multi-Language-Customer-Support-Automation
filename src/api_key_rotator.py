import os
import itertools
import logging
from typing import Iterator
from dotenv import load_dotenv

# Load env variables before trying to read them
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

def get_api_key_rotator() -> Iterator[str]:
    """
    Finds all GEMINI_API_KEY* environment variables and returns a round-robin cycle iterator.
    """
    keys = []
    for k, v in os.environ.items():
        if k.startswith("GEMINI_API_KEY") and v.strip():
            keys.append(v.strip())
            
    if not keys:
        raise ValueError("No GEMINI_API_KEY found in environment variables.")
        
    logging.info(f"Initialized API Key Rotator with {len(keys)} keys.")
    return itertools.cycle(keys)

# Global rotator instance
API_KEY_ROTATOR = get_api_key_rotator()

def get_next_api_key() -> str:
    """Gets the next API key from the rotator."""
    return next(API_KEY_ROTATOR)
