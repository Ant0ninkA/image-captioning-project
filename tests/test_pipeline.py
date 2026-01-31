import pytest

from app.pipeline import CaptionPipeline
from app.errors import ImageNotFoundError


def test_pipeline_image_not_found():
    pipeline = CaptionPipeline()

    with pytest.raises(ImageNotFoundError):
        pipeline.generate_caption("non_existent_image.jpg")

    