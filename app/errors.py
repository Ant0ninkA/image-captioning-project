class ImageCaptioningError(Exception):
    """Base class for exceptions in the image captioning application."""
    pass

class ImageNotFoundError(ImageCaptioningError):
    """Exception raised when an image file is not found."""
    pass

class InvalidImageError(ImageCaptioningError):
    """Exception raised when an image file is invalid or cannot be opened."""
    pass

class CaptionGenerationError(ImageCaptioningError):
    """Exception raised when caption generation fails."""
    pass

class EnhancementError(ImageCaptioningError):
    """Exception raised when caption enhancement fails."""
    pass