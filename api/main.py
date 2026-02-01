from fastapi import FastAPI, UploadFile, File, HTTPException
from app.pipeline import CaptionPipeline
from app.errors import ImageNotFoundError, CaptionGenerationError
import tempfile
import shutil
import os

app = FastAPI(
    title="Image Captioning API",
    description="Generate captions for images using a pretrained vision-language model",
    version="1.0.0",
)

pipeline = CaptionPipeline()


@app.post("/caption")
async def generate_caption(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    try:
        # save uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name

        caption = pipeline.generate_caption(temp_path)
        return {"caption": caption}

    except ImageNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except CaptionGenerationError as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if "temp_path" in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
