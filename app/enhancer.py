# app/enhancer.py

from app.errors import EnhancementError


class CaptionEnhancer:
    def enhance_caption(self, caption: str) -> str:
        if not caption:
            raise EnhancementError("Caption is empty")

        caption = caption.capitalize()

        if "dog" in caption.lower():
            caption += " The dog appears energetic and playful."
        elif "cat" in caption.lower():
            caption += " The cat looks calm and curious."
        else:
            caption += " The scene appears natural and detailed."

        return caption
