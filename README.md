# 📸 AI Image Captioner

A hybrid image captioning system that leverages local computer vision (BLIP) and cloud-based Large Language Models (Gemini) to transform visual data into rich, cinematic narratives.


## Features

* **Dual-Layer Pipeline:**
* **Interactive UI:**
* **Performance Evaluation:**
* **Cloud Integration:**
* **Robust Architecture:**

## Technologies Used

* **Python 3.12+**
* **HuggingFace Transformers:** BLIP model for visual feature extraction.
* **Google Generative AI:** Gemini-2.0-flash for natural language processing.
* **PyTorch:** Backend for local AI inference.
* **Streamlit:** For the web-based user interface.
* **Pandas / Matplotlib / Seaborn:** For data analysis and visualization.

## Project Structure

```text
image-captioning/
├── src
    ├── api/
    |   └── __init__.py 
    │   └── gui.py            
    ├── app/
    │   ├── __init__.py
    │   ├── caption.py
    │   ├── enhancer.py
    │   └── errors.py
├── main.py
├── examples/             # Dataset for evaluation and testing
├── tests/                # Unit tests for core components
├── .env                  # Environment variables (API Keys)
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
└── INSTALL.md            # Install guide
```

## Installation
    For detailed setup instructions and environment configuration, please refer to INSTALL.md.
