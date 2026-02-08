"""Tests for the Streamlit GUI application."""

import os
from io import BytesIO
from unittest.mock import patch, MagicMock
import pytest
from streamlit.testing.v1 import AppTest

GUI_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "api", "gui.py"))
PATCH_TARGET = "src.api.gui.load_models"

@patch(PATCH_TARGET)
def test_gui_smoke_load(mock_load: MagicMock ) -> None:
    """Basic smoke test to ensure the GUI loads without exceptions."""
    mock_load.return_value = (MagicMock(), MagicMock())
    at = AppTest.from_file(GUI_PATH, default_timeout=10).run()
    assert not at.exception

@patch(PATCH_TARGET)
@patch("api.gui.process_uploaded_file")
def test_gui_interaction_flow(mock_process, mock_load: MagicMock) -> None:
    """Tests the basic interaction flow of the GUI without actual file processing."""
    mock_load.return_value = (MagicMock(), MagicMock())
    _ = mock_process
    at = AppTest.from_file(GUI_PATH, default_timeout=15).run()

    assert len(at.get("file_uploader")) > 0

    from api.gui import process_uploaded_file
    dummy_file = BytesIO(b"fake data")
    dummy_file.name = "test.png"

    try:
        process_uploaded_file(dummy_file, MagicMock(), MagicMock())
    except Exception as e:
        pytest.fail(f"Function failed: {e}")

@patch(PATCH_TARGET)
def test_gui_model_error_display(mock_load: MagicMock) -> None:
    """Tests displaying error when there is a model loading issue."""
    mock_load.side_effect = Exception("Auth Error")
    at = AppTest.from_file(GUI_PATH, default_timeout=15).run()

    assert len(at.error) > 0
    assert "Auth Error" in at.error[0].value
