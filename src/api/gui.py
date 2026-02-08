"""
GUI for AI Image Captioning with Streamlit,
integrating BLIP и Gemini for generating and
enhancing image captioning.
"""
import os
from typing import Tuple
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

from src.app.caption import CaptionModel
from src.app.enhancer import CaptionEnhancer

from src.app.errors import (
    APIConfigurationError,
    ModelNetworkError,
    EnhancementError,
    CaptionGenerationError,
    ImageNotFoundError,
    InvalidImageError,
    ResourceLimitError
)

load_dotenv()

st.set_page_config(page_title="AI Image Captioner", layout="centered")

@st.cache_resource
def load_models() -> Tuple[CaptionModel, CaptionEnhancer]:
    """Load the captioning and enhancement models with caching to improve performance."""
    captioner = CaptionModel()
    enhancer = CaptionEnhancer(model_type="cloud")
    return captioner, enhancer


def main():
    """Main function to run the Streamlit UI."""
    st.title("AI Image Captioning")
    st.write("Upload an image and let the AI generate a caption, \
              then enhance it with cinematic flair!")

    captioner = None
    enhancer = None

    try:
        captioner, enhancer = load_models()
    except Exception as e:
        st.error(f"Failed to initialize models: {e}")
        st.stop()

    if captioner is None or enhancer is None:
        st.error("Models failed to load properly")
        st.stop()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        process_uploaded_file(uploaded_file, captioner, enhancer)

def process_uploaded_file(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile,
                          captioner: CaptionModel,
                          enhancer: CaptionEnhancer) -> None:
    """Handle image processing and display results."""
    image: Image.Image = Image.open(uploaded_file, formats=["JPEG", "PNG"]).convert("RGB")
    st.image(image, caption='Your uploaded image', use_container_width=True)

    temp_path: str = "temp_image.jpg"
    image.save(temp_path)

    try:
        with st.spinner('Generating captions...'):
            base_caption: str = captioner.generate(temp_path)
            enhanced_caption: str = enhancer.enhance(base_caption)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Base model (BLIP)")
            st.info(base_caption)
        with col2:
            st.subheader("Enhanced description")
            st.success(enhanced_caption)

    except (APIConfigurationError, ModelNetworkError, EnhancementError,
            CaptionGenerationError, ImageNotFoundError, InvalidImageError,
            ResourceLimitError) as e:
        st.error(f"Error: {e.message}")
    except Exception as e: # pylint: disable=broad-exception-caught
        st.error(f"Unexpected error occurred: {e}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    main()
