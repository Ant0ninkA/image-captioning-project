import os

from app.errors import ImageNotFoundError
from app.caption_model import CaptionModel
from app.enhancer import CaptionEnhancer
    
class CaptionPipeline:
    def __init__(self):
        self.caption_model = CaptionModel()
        self.caption_enhancer = CaptionEnhancer()
        
    def generate_caption(self, image_path: str) -> str:
        """Generates a caption for the given image."""
        if not os.path.exists(image_path):
            raise ImageNotFoundError(f"Image not found: {image_path}")

        caption = self.caption_model.generate(image_path)
        enhanced_caption = self.caption_enhancer.enhance_caption(caption)
        return enhanced_caption