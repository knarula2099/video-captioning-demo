import os
import time
import random
import re
import base64
import asyncio
from typing import Dict, Optional, List, Tuple
from PIL import Image as PIL_Image
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError
from anthropic import Anthropic
from openai import OpenAI
import mimetypes
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# Try importing PyTorch and related libraries
TORCH_AVAILABLE = False
try:
    import torch
    import torchvision
    from transformers import BlipProcessor, BlipForConditionalGeneration
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PyTorch or related libraries not available: {str(e)}")
    print("BLIP models will not be available. Please install required dependencies:")
    print("pip install torch>=2.7.1 torchvision>=0.18.1 torchaudio>=2.7.1 --extra-index-url https://download.pytorch.org/whl/cpu")
    print("pip install transformers>=4.38.0")

# Pricing constants (if you still track cost)
PRICE_PER_IMAGE = 0.00025
PRICE_PER_INPUT_TOKEN = 0.00000035
PRICE_PER_OUTPUT_TOKEN = 0.00000105

# Retry settings
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 15

@st.cache_resource(show_spinner=False)
def load_blip_models():
    """Load and cache BLIP models."""
    if not TORCH_AVAILABLE:
        return None, None
        
    try:
        # Check CUDA availability
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA for BLIP models")
        else:
            device = torch.device("cpu")
            print("Using CPU for BLIP models")
        
        print("Loading BLIP models...")
        
        # Load base model
        base_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            use_fast=True,
            local_files_only=False
        )
        base_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            local_files_only=False
        ).to(device).eval()
        
        # Load large model
        large_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-large",
            use_fast=True,
            local_files_only=False
        )
        large_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large",
            local_files_only=False
        ).to(device).eval()
        
        models = {
            "BLIP-Base": {
                "processor": base_processor,
                "model": base_model,
            },
            "BLIP-Large": {
                "processor": large_processor,
                "model": large_model,
            }
        }
        print("BLIP models loaded successfully!")
        return models, device
    except Exception as e:
        print(f"Error loading BLIP models: {str(e)}")
        return None, None

class CaptionGenerator:
    def __init__(self, 
                 google_api_key: Optional[str] = None,
                 anthropic_api_key: Optional[str] = None,
                 openai_api_key: Optional[str] = None,
                 default_prompt: Optional[str] = None,
                 max_workers: int = 5):
        """
        Initialize caption generator with support for multiple models.
        
        Args:
            max_workers: Maximum number of concurrent workers for batch processing
        """
        self.default_prompt = default_prompt or (
            "Describe this image in one concise sentence (present simple). "
            "Use two sentences only if truly needed (<10% of cases)."
        )
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
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

        # Initialize BLIP models using cached loader
        if TORCH_AVAILABLE:
            with st.spinner("Loading BLIP models..."):
                self.blip_models, self.device = load_blip_models()
        else:
            self.blip_models = None
            self.device = None

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

    async def generate_gemini_caption_async(self, image_path: str, prompt: Optional[str] = None) -> str:
        """Generate caption using Gemini asynchronously."""
        if not self.gemini_model:
            return "Gemini API key not configured"
            
        try:
            img = PIL_Image.open(image_path).convert("RGB")
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                lambda: self.gemini_model.generate_content([prompt or self.default_prompt, img])
            )
            return response.text.strip()
        except Exception as e:
            return f"Gemini Error: {str(e)}"

    async def generate_claude_caption_async(self, image_path: str, prompt: Optional[str] = None) -> str:
        """Generate caption using Claude asynchronously."""
        if not self.claude_client:
            return "Claude API key not configured"
            
        try:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type or not mime_type.startswith('image/'):
                ext = os.path.splitext(image_path)[1].lower()
                mime_type = {
                    '.png': 'image/png',
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.webp': 'image/webp',
                    '.gif': 'image/gif'
                }.get(ext, 'image/jpeg')
            
            loop = asyncio.get_event_loop()
            message = await loop.run_in_executor(
                self.executor,
                lambda: self.claude_client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    max_tokens=150,
                    temperature=0.4,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt or self.default_prompt},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime_type,
                                    "data": image_data
                                }
                            }
                        ]
                    }]
                )
            )
            return message.content[0].text.strip()
        except Exception as e:
            return f"Claude Error: {str(e)}"

    async def generate_gpt4_caption_async(self, image_path: str, prompt: Optional[str] = None) -> str:
        """Generate caption using GPT-4 Vision asynchronously."""
        if not self.openai_client:
            return "GPT-4 API key not configured"
            
        try:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                lambda: self.openai_client.chat.completions.create(
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
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"GPT-4 Error: {str(e)}"

    async def generate_blip_caption_async(self, image_path: str, model_name: str = "BLIP-Base") -> str:
        """Generate caption using BLIP model asynchronously."""
        if not TORCH_AVAILABLE or not self.blip_models:
            return "BLIP models not available. Please install required dependencies."
            
        if model_name not in self.blip_models:
            return f"BLIP model {model_name} not found"
            
        try:
            img = PIL_Image.open(image_path).convert("RGB")
            cfg = self.blip_models[model_name]
            
            prompt = (
                "Give a succinct 1-sentence description for most cases, and 2 sentences in a minority of cases (target <10%) but only if/when 2 sentences are more optimal for describing the image. "
                "We don't need flowery descriptions. Whenever possible, give the descriptions in present simple verb tense. "
                "If the image includes text on the screen, then the description should include the full text, even if it is 2 sentences or more sentences."
            )
            
            loop = asyncio.get_event_loop()
            inputs = await loop.run_in_executor(
                self.executor,
                lambda: cfg["processor"](images=img, text=prompt, return_tensors="pt").to(self.device)
            )
            
            with torch.no_grad():
                out_ids = cfg["model"].generate(
                    pixel_values=inputs.pixel_values,
                    max_length=100,
                    num_beams=5,
                    early_stopping=True
                )
            caption = cfg["processor"].decode(out_ids[0], skip_special_tokens=True).strip()
            return caption
        except Exception as e:
            return f"BLIP Error: {str(e)}"

    async def generate_all_captions_async(self, image_path: str, prompt: Optional[str] = None) -> Dict[str, str]:
        """Generate captions using all available models asynchronously."""
        tasks = []
        
        if self.gemini_model:
            tasks.append(self.generate_gemini_caption_async(image_path, prompt))
        if self.claude_client:
            tasks.append(self.generate_claude_caption_async(image_path, prompt))
        if self.openai_client:
            tasks.append(self.generate_gpt4_caption_async(image_path, prompt))
        if TORCH_AVAILABLE and self.blip_models:
            tasks.append(self.generate_blip_caption_async(image_path, "BLIP-Base"))
            tasks.append(self.generate_blip_caption_async(image_path, "BLIP-Large"))
        
        results = await asyncio.gather(*tasks)
        
        captions = {}
        model_names = []
        if self.gemini_model:
            model_names.append("Gemini")
        if self.claude_client:
            model_names.append("Claude")
        if self.openai_client:
            model_names.append("GPT-4")
        if TORCH_AVAILABLE and self.blip_models:
            model_names.extend(["BLIP-Base", "BLIP-Large"])
            
        for model_name, result in zip(model_names, results):
            captions[model_name] = result
            
        return captions

    async def process_batch_async(self, image_paths: List[str], prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """Process a batch of images asynchronously."""
        tasks = [self.generate_all_captions_async(path, prompt) for path in image_paths]
        return await asyncio.gather(*tasks)

    def process_batch(self, image_paths: List[str], prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """Process a batch of images using the event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.process_batch_async(image_paths, prompt))
        finally:
            loop.close()

    # Keep the original synchronous methods for backward compatibility
    def generate_all_captions(self, image_path: str, prompt: Optional[str] = None) -> Dict[str, str]:
        """Generate captions using all available models (synchronous version)."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.generate_all_captions_async(image_path, prompt))
        finally:
            loop.close()