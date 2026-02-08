"""Module for image captioning using the BLIP model."""

import logging
import os
from typing import List
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from src.app.errors import (
    CaptionGenerationError,
    ImageNotFoundError,
    InvalidImageError,
    ResourceLimitError
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["HF_HUB_READ_TIMEOUT"] = "300"

class CaptionModel:
    """Generates captions using BLIP model."""

    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        device: str | None = None,
        max_length: int = 50
    ):
        """
        Initialize the BLIP captioning model.

        Args:
            model_name: The HuggingFace model identifier.
            device: Computing device (cuda, mps, or cpu).
            max_length: Maximum length of the generated caption.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or self._detect_device()

        self.processor = None
        self.model = None

    def _detect_device(self) -> str:
        """Detect the best available computing device."""
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self) -> None:
        """Load the model and processor from HuggingFace."""
        if self.model is not None:
            return

        try:
            logger.info("Loading caption model...")
            self.processor = BlipProcessor.from_pretrained(self.model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(self.model_name)

            self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            raise CaptionGenerationError(f"Failed to load caption model: {e}",
                                          details=str(e)) from e

    def generate(self, image_path: str) -> str:
        """
        Generate a caption for a single image.

        Args:
            image_path: Path to the image file.

        Returns:
            str: The generated caption.
        """
        self._load_model()

        if not os.path.exists(image_path):
            raise ImageNotFoundError(image_path)

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise InvalidImageError(str(e), details=str(e)) from e

        try:
            inputs = self.processor(image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                output = self.model.generate(
                   **inputs,
                    max_length=self.max_length,
                    num_beams=5,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=2
                )

            return self.processor.decode(output[0], skip_special_tokens=True)

        except torch.cuda.OutOfMemoryError as exc:
            raise ResourceLimitError("Model inference failed due to GPU memory limits. " \
                                    "Try reducing the batch size or using a smaller model.",
                                    details="Out of GPU memory during caption generation.") from exc
        except Exception as e:
            raise CaptionGenerationError("Caption generation failed.", details=str(e)) from e

    def generate_batch(self, image_paths: List[str]) -> List[str]:
        """
        Generate captions for a list of images.

        Args:
            image_paths: A list of paths to image files.

        Returns:
            List[str]: A list of generated captions.
        """
        return [self.generate(p) for p in image_paths]
