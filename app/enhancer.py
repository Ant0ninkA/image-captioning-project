from app.errors import EnhancementError

class CaptionEnhancer:
    def enhance_caption(self, caption: str) -> str:
        """Enhances the given caption."""
        if not caption:
            raise EnhancementError("Caption is empty and cannot be enhanced.")
        
        # Dummy implementation for caption enhancement (testing purposes)
        # Logic to enhance the caption
        enhanced_caption = caption + " This caption has been enhanced."
        return enhanced_caption