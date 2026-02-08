"""
GUI for AI Image Captioning with Streamlit,
integrating BLIP и Gemini for generating and
enhancing image captioning.
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from PIL import Image
from dotenv import load_dotenv

from app.caption import CaptionModel
from app.enhancer import CaptionEnhancer

from app.errors import (
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
def load_models():
    """Load the captioning and enhancement models with caching to improve performance."""
    captioner = CaptionModel()
    enhancer = CaptionEnhancer(model_type="cloud")
    return captioner, enhancer

st.title("AI Image Captioning")
st.write("Upload an image and let the AI generate a caption, then enhance it with cinematic flair!")

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
    image = Image.open(uploaded_file)
    st.image(image, caption='Your uploaded image', use_container_width=True)

    temp_path = "temp_image.jpg"
    image.save(temp_path)

    try:
        with st.spinner('BLIP is generating a caption...'):
            base_caption = captioner.generate(temp_path)
    
        with st.spinner('Gemini is enhancing the caption...'):
            enhanced_caption = enhancer.enhance(base_caption)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Base model (BLIP)")
            st.info(base_caption)

        with col2:
            st.subheader("Enhanced description")
            st.success(enhanced_caption)
    except APIConfigurationError as e:
        st.error(f"API Configuration Error: {e.message}")
        if e.details:
            st.write(f"Details: {e.details}")
    except ModelNetworkError as e:
        st.error(f"Network Error: {e.message}")
        if e.details:
            st.write(f"Details: {e.details}")
    except EnhancementError as e:
        st.error(f"Enhancement Error: {e.message}")
        if e.details:
            st.write(f"Details: {e.details}")
    except CaptionGenerationError as e:
        st.error(f"Caption Generation Error: {e.message}")
        if e.details:
            st.write(f"Details: {e.details}")
    except ImageNotFoundError as e:
        st.error(f"Image Not Found Error: {e.message}")
        if e.details:
            st.write(f"Details: {e.details}")
    except InvalidImageError as e:
        st.error(f"Invalid Image Error: {e.message}")
        if e.details:
            st.write(f"Details: {e.details}")
    except ResourceLimitError as e:
        st.error(f"Resource Limit Error: {e.message}")
        if e.details:
            st.write(f"Details: {e.details}")
    except Exception as e:
        st.error(f"Unexpected error occurred: {e}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
