import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from app.errors import EnhancementError

class CaptionEnhancer:
    """Enhances image captions to be more descriptive and vivid."""
    def __init__(self, model_name: str = "google/flan-t5-base"):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model.eval()
        except Exception as e:
            raise EnhancementError(f"Failed to load enhancement model: {e}")

    def enhance(self, caption: str) -> str:
        if not caption or not caption.strip():
            raise EnhancementError("Caption is empty")

        try:
            prompt = (
                "Describe the image in one vivid, detailed sentence. "
                "Mention colors, environment, and visual details. "
                "Do not repeat the original wording:\n"
                f"{caption}"
            )

            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)

            with torch.no_grad():
                outputs = self.model.generate(
                     **inputs,
                    max_new_tokens=50
                )

            enhanced = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            return enhanced.strip()
        except Exception as e:
            raise EnhancementError(f"Enhancement failed: {e}")
