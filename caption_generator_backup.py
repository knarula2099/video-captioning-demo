import os
import time
import random
import re
import base64
from typing import Dict, Optional
from PIL import Image as PIL_Image
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError
from anthropic import Anthropic
from openai import OpenAI
import mimetypes

# Pricing constants (if you still track cost)
PRICE_PER_IMAGE = 0.00025
PRICE_PER_INPUT_TOKEN = 0.00000035
PRICE_PER_OUTPUT_TOKEN = 0.00000105

# Retry settings
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 15

class CaptionGenerator:
    def __init__(self, 
                 google_api_key: Optional[str] = None,
                 anthropic_api_key: Optional[str] = None,
                 openai_api_key: Optional[str] = None,
                 default_prompt: Optional[str] = None):
        """
        Initialize caption generator with support for multiple models.
        """
        self.default_prompt = default_prompt or (
            "Describe this image in one concise sentence (present simple). "
            "Use two sentences only if truly needed (<10% of cases)."
        )
        
        # Initialize Gemini
        if google_api_key:
            genai.configure(api_key=google_api_key)
            self.gemini_model = genai.GenerativeModel(
                model_name="gemini-1.5-flash-latest",
                generation_config={
                    "temperature": 0.4,
                    "top_p": 1.0,
                    "top_k": 32,
                    "max_output_tokens": 150
                }
            )
        else:
            self.gemini_model = None
            
        # Initialize Claude
        if anthropic_api_key:
            self.claude_client = Anthropic(api_key=anthropic_api_key)
        else:
            self.claude_client = None
            
        # Initialize GPT-4
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
        else:
            self.openai_client = None

    def _retry_generate(self, img: PIL_Image.Image, prompt: str) -> Dict:
        retries = 0
        backoff = INITIAL_BACKOFF_SECONDS
        last_exc = None

        while retries <= MAX_RETRIES:
            try:
                return self.gemini_model.generate_content([prompt, img])
            except GoogleAPIError as e:
                last_exc = e
                # detect 429
                is_rate_limit = "429" in str(e) or getattr(e, "code", None) == 429
                if is_rate_limit and retries < MAX_RETRIES:
                    retries += 1
                    m = re.search(r"retry_delay\s*{\s*seconds:\s*(\d+)\s*}", str(e))
                    delay = int(m.group(1)) if m else backoff
                    print(f"Rate limit, retrying in {delay}s ({retries}/{MAX_RETRIES})…")
                    time.sleep(delay + random.random())
                    backoff *= 2
                    continue
                raise

        raise last_exc

    def generate_gemini_caption(self, image_path: str, prompt: Optional[str] = None) -> str:
        """Generate caption using Gemini."""
        if not self.gemini_model:
            return "Gemini API key not configured"
            
        try:
            img = PIL_Image.open(image_path).convert("RGB")
            response = self.gemini_model.generate_content([prompt or self.default_prompt, img])
            return response.text.strip()
        except Exception as e:
            return f"Gemini Error: {str(e)}"

    def generate_claude_caption(self, image_path: str, prompt: Optional[str] = None) -> str:
        """Generate caption using Claude."""
        if not self.claude_client:
            return "Claude API key not configured"
            
        try:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Detect image format
            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type or not mime_type.startswith('image/'):
                # Fallback based on file extension
                ext = os.path.splitext(image_path)[1].lower()
                if ext == '.png':
                    mime_type = 'image/png'
                elif ext in ['.jpg', '.jpeg']:
                    mime_type = 'image/jpeg'
                elif ext == '.webp':
                    mime_type = 'image/webp'
                elif ext == '.gif':
                    mime_type = 'image/gif'
                else:
                    mime_type = 'image/jpeg'  # final fallback
                
            message = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20240620",  # Fixed model name
                max_tokens=150,
                temperature=0.4,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt or self.default_prompt
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime_type,
                                    "data": image_data
                                }
                            }
                        ]
                    }
                ]
            )
            return message.content[0].text.strip()
        except Exception as e:
            return f"Claude Error: {str(e)}"

    def generate_gpt4_caption(self, image_path: str, prompt: Optional[str] = None) -> str:
        """Generate caption using GPT-4 Vision."""
        if not self.openai_client:
            return "GPT-4 API key not configured"
            
        try:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')  # Fixed: added base64 encoding
                
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt or self.default_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                    ]
                }],
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"GPT-4 Error: {str(e)}"

    def generate_all_captions(self, image_path: str, prompt: Optional[str] = None) -> Dict[str, str]:
        """Generate captions using all available models."""
        captions = {}
        
        if self.gemini_model:
            captions["Gemini"] = self.generate_gemini_caption(image_path, prompt)
        if self.claude_client:
            captions["Claude"] = self.generate_claude_caption(image_path, prompt)
        if self.openai_client:
            captions["GPT-4"] = self.generate_gpt4_caption(image_path, prompt)
            
        return captions