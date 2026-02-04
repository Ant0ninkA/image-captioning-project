import argparse
import os
import sys

from app.pipeline import CaptionPipeline
from app.enhancer import CaptionEnhancer
from app.errors import ImageNotFoundError, CaptionGenerationError

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate image captions using a pretrained model"
    )

    parser.add_argument(
        "--image",
        "-i",
        required=True,
        help="Path to the input image"
    )

    parser.add_argument(
        "--enhance",
        action="store_true",
        help="Enhance the generated caption"
    )

    return parser.parse_args()

def main():
    args = parse_args()

    image_path = args.image

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        sys.exit(1)

    try:
        pipeline = CaptionPipeline()

        print("\nImage caption:")
        caption = pipeline.generate_caption(image_path, enhance=False)
        print(caption)

        print("\nWith enhancement:")
        print(pipeline.generate_caption(image_path, enhance=True))

    except ImageNotFoundError as e:
        print(f"{e}")
        sys.exit(1)
    except CaptionGenerationError as e:
        print(f"Caption generation failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
