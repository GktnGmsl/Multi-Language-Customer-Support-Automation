import os
import time
import asyncio
import itertools
from typing import Any, List, Optional, Dict
from pydantic import Field, PrivateAttr

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration

from google import genai
from google.genai import errors
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

# Global lock to completely enforce sequential processing across all async tasks
_evaluation_lock = asyncio.Lock()

def get_api_key_rotator():
    keys = []
    for k, v in os.environ.items():
        if k.startswith("GEMINI_API_KEY") and v.strip():
            keys.append(v.strip())
    if not keys:
        raise ValueError("No GEMINI_API_KEY found")
    return itertools.cycle(keys)

class RotatingGeminiLLM(BaseChatModel):
    model_name: str = "gemini-2.5-flash"
    rotator: Any = Field(default_factory=get_api_key_rotator, alias="gemini_rotator")

    @property
    def _llm_type(self) -> str:
        return "rotating-gemini"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Exponential backoff parameters
        max_retries = 30 # Time-based break will stop it earlier if needed
        base_delay = 2
        start_time = time.time()
        
        # Build prompt from Langchain messages
        prompt = ""
        for msg in messages:
            prompt += f"{msg.type}: {msg.content}\n"
            
        for attempt in range(max_retries):
            current_key = next(self.rotator)
            try:
                client = genai.Client(api_key=current_key)
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                
                text = response.text if response.text else ""
                message = AIMessage(content=text)
                generation = ChatGeneration(message=message)
                return ChatResult(generations=[generation])
                
            except errors.APIError as e:
                # 429 Resource Exhausted: Wait exponentially mostly when all keys are exhausted
                delay = base_delay * (1.5 ** (attempt // 3)) # Increase delay after trying all 3 keys
                print(f"API Error with key ...{current_key[-4:]}: {e.message}. Retrying in {delay:.1f}s...")
                
                if time.time() - start_time > 60:
                    print(f"Timeout: 60 saniye sınırı aşıldı, tüm işlemler durduruluyor. Son Hata: {e.message}")
                    import sys
                    sys.exit(f"Kritik Hata: API'ler 60 saniye boyunca yanıt veremedi veya limitleri aştı: {e.message}")
                    
                time.sleep(delay)
            except Exception as e:
                delay = base_delay * (1.5 ** (attempt // 3))
                print(f"Unexpected error: {e}. Retrying in {delay:.1f}s...")
                
                if time.time() - start_time > 60:
                    print(f"Timeout: 60 saniye sınırı aşıldı, tüm işlemler durduruluyor. Son Beklenmeyen Hata: {e}")
                    import sys
                    sys.exit(f"Kritik Hata: Bilinmeyen bir sorun nedeniyle 60 saniyelik limit aşıldı: {e}")
                    
                time.sleep(delay)
                
        # Fallback
        message = AIMessage(content="Error generating response.")
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Enforce exactly one concurrent LLM call
        async with _evaluation_lock:
            # We can use asyncio.to_thread to run the sync _generate
            # but since we want synchronous blocking, we can just call it
            # inside to prevent blocking the entire event loop (which might crash Ragas)
            result = await asyncio.to_thread(self._generate, messages, stop, run_manager, **kwargs)
            # Add a small delay between requests to be safe
            await asyncio.sleep(4.0)
            return result
