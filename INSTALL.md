# Installation Guide

This guide provides step-by-step instructions to set up the **AI Image Captioner** on your local machine.

---

## Prerequisites

Make sure you have installed:

- Python **3.12 or newer**
- Git
- Internet connection (required for first model download)

Check Python version:

```bash
python --version
```

## Clone the repository
```bash
git clone https://github.com/Ant0ninkA/image-captioning-project.git
cd image-captioning
```

## Create and activate Virtual Environment

### Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows
```bash
python -m venv venv
source venv/Scripts/activate
```

## Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Configure environment variables
1. Create a file named .env in the root directory.
2. Add your API key in the following format:
```
GEMINI_API_KEY=<your_api_key>
```

## Run the application
```bash
streamlit run api/gui.py
```

## Veryfication
To ensure everything is installed correctly, run the automated tests:
```bash
pytest -v
```

## Troubleshooting
1. First-Run Delay
During the first execution, the system will download the Salesforce/blip-image-captioning-base model (approx. 990MB). This happens only once.

2. Gemini API Errors
Ensure your .env file is in the root folder.

Verify that your API key is active and has remaining quota.

3. Hardware Requirements
RAM: Minimum 4GB (8GB recommended).

GPU: Optional. The system automatically detects CUDA-enabled GPUs; otherwise, it defaults to CPU.