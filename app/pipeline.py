import os

from app.caption_model import CaptionModel
from app.enhancer import CaptionEnhancer
    
class CaptionPipeline:
    def __init__(self):
        self.caption_model = CaptionModel()
        self.caption_enhancer = CaptionEnhancer()
        
    def generate_caption(self, image_path: str, enhance: bool = True) -> str:
        caption = self.caption_model.generate(image_path)

        if enhance:
            caption = self.caption_enhancer.enhance(caption)

        return caption