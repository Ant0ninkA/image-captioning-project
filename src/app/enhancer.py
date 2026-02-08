"""
Module for enhancing image captions using Google Gemini AI.
This version automatically detects available models to prevent 404 errors.
"""

import os
import logging
import google.generativeai as genai
from dotenv import load_dotenv

from src.app.errors import (
    EnhancementError,
    APIConfigurationError,
    ResourceLimitError,
    ModelNetworkError
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CaptionEnhancer:
    """
    Enhances short, literal captions into descriptive, cinematic sentences.
    Uses Google Gemini API for natural language processing.
    """

    def __init__(self, model_type: str = "cloud", creativity: float = 0.8):
        """
        Initializes the enhancer and discovers available models.
        """
        self.model_type = model_type
        self.creativity = creativity
        self.model = None

        if self.model_type == "cloud":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.error("GEMINI_API_KEY not found in .env file.")
                raise APIConfigurationError("API Key missing. Please check your .env file.",
                                        details="GEMINI_API_KEY environment variable is required.")

            try:
                genai.configure(api_key=api_key)

                available_models = [
                    m.name for m in genai.list_models()
                    if 'generateContent' in m.supported_generation_methods
                ]

                if not available_models:
                    raise EnhancementError("No generative models available for this API key.",
                                details="Check your API key permissions and model availability.")

                priority_list = [
                    'models/gemini-1.5-flash',
                    'models/gemini-1.5-pro',
                    'models/gemini-pro'
                ]

                selected_model_name = None
                for model_name in priority_list:
                    if model_name in available_models:
                        selected_model_name = model_name
                        break

                if not selected_model_name:
                    selected_model_name = available_models[0]

                logger.info("Using Gemini model: %s", selected_model_name)

                self.model = genai.GenerativeModel(
                    model_name=selected_model_name,
                    system_instruction=(
                        "You are a creative storyteller. Your task is to transform simple "
                        "image captions into vivid, cinematic, and emotional descriptions. "
                        "Output only one sentence in English."
                    )
                )
            except Exception as e:
                logger.error("Initialization failed: %s", str(e))
                raise APIConfigurationError(f"API initialization failed: {e}",
                                             details=str(e)) from e

    def enhance(self, caption: str) -> str:
        """
        Main method to enhance the caption.
        """
        if not caption or len(caption.strip()) < 3 or not self.model:
            return caption

        if self.model_type == "cloud":
            return self._run_cloud_inference(caption)
        return caption

    def _run_cloud_inference(self, caption: str) -> str:
        prompt = (
            f"Context: {caption}\n"
            "Task: Rewrite this into a single, long, cinematic, and vivid sentence. "
            "Focus on the atmosphere and textures. Output only the enhanced sentence. "
            "Do not truncate the sentence."
        )

        try:
            config = genai.types.GenerationConfig(
                temperature=self.creativity,
                top_p=0.9
            )

            response = self.model.generate_content(prompt, generation_config=config)

            if response and response.text:
                text = response.text.strip().replace('"', '')

                return text

            return caption
        except Exception as e:
            err_msg = str(e).lower()
            if "api_key" in err_msg or "403" in err_msg:
                raise APIConfigurationError("Invalid Gemini API Key",
                                             details=str(e)) from e
            if "quota" in err_msg or "429" in err_msg:
                raise ResourceLimitError("Gemini API quota exhausted",
                                          details=str(e)) from e
            if "network" in err_msg or "connection" in err_msg:
                raise ModelNetworkError("Please check your internet connection",
                                        details=str(e)) from e
            raise EnhancementError("Enhancement failed", details=str(e)) from e
