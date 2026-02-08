"""Unit tests for the CaptionModel class using mocks."""

import os
from typing import Any
import pytest
from unittest.mock import MagicMock, patch
import torch
from app.caption import CaptionModel
from app.errors import (
    CaptionGenerationError,
    ImageNotFoundError, 
    InvalidImageError,
    ResourceLimitError
)

class DummyProcessor:
    """Mock for the BLIP processor."""
    def __call__(self, image: Any, return_tensors: Any = None) -> 'DummyInputs':
        """Simulate processor call."""
        return DummyInputs()

    def decode(self, output: Any, skip_special_tokens: bool = True) -> str:
        """Simulate decoding of model tokens."""
        _ = (output, skip_special_tokens)
        return "dummy caption"


class DummyInputs(dict):
    """Mock for model input tensors."""
    def to(self, device: str) -> 'DummyInputs':
        """Simulate moving tensors to device."""
        _ = device
        return self

class DummyModel:
    """Mock for the BLIP model."""
    def generate(self, **kwargs: Any) -> list[list[int]]:
        """Simulate text generation."""
        _ = kwargs
        return [[1, 2, 3]]

    def to(self, device: str) -> 'DummyModel':
        """Simulate moving model to device."""
        _ = device
        return self

    def eval(self) -> 'DummyModel':
        """Simulate setting eval mode and return self."""
        self.is_eval = True
        return self

def create_fake_caption_model() -> CaptionModel:
    """Create a CaptionModel instance with mocked components."""
    model = CaptionModel.__new__(CaptionModel)
    model.device = "cpu"
    model.processor = DummyProcessor()
    model.model = DummyModel()
    model.max_length = 20
    model._load_model = lambda: None
    return model

class FakeImage:
    """Mock for PIL Image object."""
    def convert(self, *args: Any) -> 'FakeImage':
        """Simulate image mode conversion."""
        _ = args
        return self

def test_caption_model_image_not_found() -> None:
    """Test that ImageNotFoundError is raised when file does not exist."""
    model = create_fake_caption_model()
    with pytest.raises(ImageNotFoundError):
        model.generate("missing.jpg")


def test_caption_model_invalid_image(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that InvalidImageError is raised for corrupted image files."""
    model = create_fake_caption_model()
    monkeypatch.setattr(os.path, "exists", lambda _: True)

    def fake_open(*args: Any, **kwargs: Any) -> None:
        """Simulate a corrupted image file."""
        _ = (args, kwargs)
        raise RuntimeError("bad image")

    monkeypatch.setattr("PIL.Image.open", fake_open)
    with pytest.raises(InvalidImageError):
        model.generate("fake.jpg")


def test_caption_model_resource_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that ResourceLimitError is raised when GPU memory is exceeded."""
    model = create_fake_caption_model()
    monkeypatch.setattr(os.path, "exists", lambda _: True)

    monkeypatch.setattr("PIL.Image.open", lambda _: FakeImage())

    def fake_generate(*args: Any, **kwargs: Any) -> None:
        """Simulate GPU out of memory error."""
        _ = (args, kwargs)
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(model.model, "generate", fake_generate)
    with pytest.raises(ResourceLimitError) as excinfo:
        model.generate("fake.jpg")
    assert "GPU memory limits" in str(excinfo.value)

def test_caption_model_general_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that CaptionGenerationError is raised for general exceptions."""
    model = create_fake_caption_model()
    monkeypatch.setattr(os.path, "exists", lambda _: True)

    monkeypatch.setattr("PIL.Image.open", lambda _: FakeImage())

    def fake_generate_fail(*args: Any, **kwargs: Any) -> None:
        """Simulate a generic failure during generation."""
        raise RuntimeError("Something went wrong")

    monkeypatch.setattr(model.model, "generate", fake_generate_fail)

    with pytest.raises(CaptionGenerationError) as excinfo:
        model.generate("fake.jpg")
    assert "Caption generation failed" in str(excinfo.value)

def test_load_model_already_loaded():
    """Tests that _load_model returns early if model is already present"""
    model = create_fake_caption_model()
    assert model._load_model() is None

def test_load_model_error():
    """Test that CaptionGenerationError is raised if model loading fails."""
    with patch("transformers.BlipForConditionalGeneration.from_pretrained", 
               side_effect=Exception("Download failed")):
        model = CaptionModel()
        with pytest.raises(CaptionGenerationError):
            model._load_model()

def test_load_model_success():
    """Verify that model is moved to device and set to eval mode."""
    model_inst = CaptionModel()
    
    mock_processor = MagicMock()
    mock_model = MagicMock()
    
    mock_model.to.return_value = mock_model

    with patch("transformers.BlipProcessor.from_pretrained", return_value=mock_processor), \
         patch("transformers.BlipForConditionalGeneration.from_pretrained", return_value=mock_model):
        
        model_inst._load_model()
        
        mock_model.to.assert_called_with(model_inst.device)
        mock_model.eval.assert_called_once()
        assert model_inst.model == mock_model
        assert model_inst.processor == mock_processor

def test_caption_model_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test successful caption generation."""
    model = create_fake_caption_model()
    monkeypatch.setattr(os.path, "exists", lambda _: True)


    monkeypatch.setattr("PIL.Image.open", lambda _: FakeImage())
    result = model.generate("fake.jpg")
    assert result == "dummy caption"

def test_detect_device_cuda(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    model = CaptionModel()
    assert model.device == "cuda"

def test_detect_device_cpu(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    model = CaptionModel()
    assert model.device == "cpu"

def test_detect_device_mps(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    model = CaptionModel()
    assert model.device == "mps"

def test_generate_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test batch generation calls generate for each path."""
    model = create_fake_caption_model()
    monkeypatch.setattr(os.path, "exists", lambda _: True)
    monkeypatch.setattr("PIL.Image.open", lambda _: FakeImage())
    
    paths = ["img1.jpg", "img2.jpg"]
    results = model.generate_batch(paths)
    
    assert len(results) == 2
    assert all(res == "dummy caption" for res in results)
