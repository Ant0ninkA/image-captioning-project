"""Module for defining custom exceptions used in the image captioning application."""

class ImageCaptioningError(Exception):
    """Base class for exceptions in the image captioning application."""
    def __init__(self, message: str, details: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

class APIConfigurationError(ImageCaptioningError):
    """Exception raised for errors in API configuration."""

class ModelNetworkError(ImageCaptioningError):
    """Exception raised when there is a network error while loading models."""

class ModelTimeoutError(ImageCaptioningError):
    """Exception raised when model loading or generation times out."""

class ResourceLimitError(ImageCaptioningError):
    """Exception raised when system resources are insufficient for model operations."""

class ImageNotFoundError(ImageCaptioningError):
    """Exception raised when an image file is not found."""

class InvalidImageError(ImageCaptioningError):
    """Exception raised when an image file is invalid or cannot be opened."""

class CaptionGenerationError(ImageCaptioningError):
    """Exception raised when caption generation fails."""

class EnhancementError(ImageCaptioningError):
    """Exception raised when caption enhancement fails."""
