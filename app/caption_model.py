# app/caption_model.py

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from app.errors import CaptionGenerationError


class CaptionModel:
    def __init__(self):
        try:
            self.device = "cpu"
            self.processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            self.model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )

            self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            raise CaptionGenerationError(f"Failed to load caption model: {e}")

    def generate(self, image_path: str) -> str:
        try:
            image = Image.open(image_path).convert("RGB")

            inputs = self.processor(image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                output = self.model.generate(**inputs, max_length=20)

            caption = self.processor.decode(
                output[0],
                skip_special_tokens=True
            )

            return caption

        except Exception as e:
            raise CaptionGenerationError(f"Caption generation failed: {e}")
