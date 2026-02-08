"""Unit tests for the CaptionEnhancer class in app.enhancer."""

from unittest.mock import MagicMock, patch
import pytest
from src.app.enhancer import CaptionEnhancer
from src.app.errors import (
    APIConfigurationError,
    ModelNetworkError,
    EnhancementError,
    ResourceLimitError
)

@pytest.fixture
def enhancer_instance() -> CaptionEnhancer:
    """Fixture to create a CaptionEnhancer instance with mocked API interactions."""
    mock_model_info = MagicMock()
    mock_model_info.name = 'models/gemini-1.5-flash'
    mock_model_info.supported_generation_methods = ['generateContent']

    with patch('os.getenv', return_value="fake_key"), \
         patch('google.generativeai.configure'), \
         patch('google.generativeai.list_models', return_value=[mock_model_info]):
        return CaptionEnhancer(model_type="cloud")

def test_creativity_parameter_passing() -> None:
    """Checks that the creativity parameter changes temperature in API requests."""
    mock_model_info = MagicMock()
    mock_model_info.name = 'models/gemini-1.5-flash'
    mock_model_info.supported_generation_methods = ['generateContent']

    with patch('google.generativeai.configure'), \
         patch('google.generativeai.list_models', return_value=[mock_model_info]), \
         patch('os.getenv', return_value="fake_key"):

        custom_creativity = 0.95
        enhancer = CaptionEnhancer(creativity=custom_creativity)

        with patch.object(enhancer.model, 'generate_content') as mock_generate:
            mock_generate.return_value = MagicMock(text="Cinematic description.")

            enhancer.enhance("test sentence")

            _, kwargs = mock_generate.call_args
            config = kwargs['generation_config']
            assert config.temperature == custom_creativity

def test_enhance_removes_quotes(enhancer_instance) -> None:
    """Checks that the enhance method removes quotes from the generated caption."""
    mock_response = MagicMock()
    mock_response.text = '"A beautiful landscape."'

    with patch.object(enhancer_instance.model, 'generate_content', return_value=mock_response):
        result = enhancer_instance.enhance("landscape")
        assert result == "A beautiful landscape."
        assert '"' not in result

@pytest.mark.parametrize("input_text, expected_output", [
    ("", ""),
    ("ab", "ab"),
    (None, None),
])
def test_enhance_edge_cases(enhancer_instance, input_text, expected_output):
    """Checks that the enhance method returns the input unchanged for invalid inputs."""
    assert enhancer_instance.enhance(input_text) == expected_output

def test_init_missing_api_key() -> None:
    """Tests that APIConfigurationError is raised when GEMINI_API_KEY is missing."""
    with patch('os.getenv', return_value=None):
        with pytest.raises(APIConfigurationError) as excinfo:
            CaptionEnhancer(model_type="cloud")

        assert "API Key missing" in str(excinfo.value)
        assert "GEMINI_API_KEY environment variable is required" in excinfo.value.details

def test_init_no_models_found() -> None:
    """Tests the case where API returns an empty list of models."""
    with patch('os.getenv', return_value="fake_key"), \
         patch('google.generativeai.configure'), \
         patch('google.generativeai.list_models', return_value=[]):

        with pytest.raises(APIConfigurationError) as excinfo:
            CaptionEnhancer(model_type="cloud")
        assert "No generative models available" in str(excinfo.value)

def test_enhance_general_error_handling(enhancer_instance) -> None:
    """Checks that EnhancementError is raised for general failures."""
    with patch.object(enhancer_instance.model, 'generate_content', side_effect=Exception("Random Error")):
        with pytest.raises(EnhancementError):
            enhancer_instance.enhance("a red car")

def test_network_error_handling(enhancer_instance) -> None:
    """Checks that ModelNetworkError is raised when there is no internet connection."""
    with patch.object(enhancer_instance.model, 'generate_content',
                      side_effect=Exception("Failed to establish a network connection")):
        with pytest.raises(ModelNetworkError) as excinfo:
            enhancer_instance.enhance("a dog")
        assert "internet connection" in str(excinfo.value).lower()

def test_api_key_error_handling(enhancer_instance) -> None:
    """Checks that APIConfigurationError is raised for invalid API key."""
    with patch.object(enhancer_instance.model, 'generate_content',
                      side_effect=Exception("403 Forbidden: api_key is invalid")):
        with pytest.raises(APIConfigurationError) as excinfo:
            enhancer_instance.enhance("a cat")
        assert "Invalid Gemini API Key" in str(excinfo.value)

def test_init_unexpected_api_failure() -> None:
    """Tests the generic exception block in __init__"""
    with patch('os.getenv', return_value="fake_key"), \
         patch('google.generativeai.configure') as mock_conf:

        mock_conf.side_effect = RuntimeError("Something went critically wrong")

        with pytest.raises(APIConfigurationError) as excinfo:
            CaptionEnhancer(model_type="cloud")

        assert "API initialization failed" in str(excinfo.value)
        assert "Something went critically wrong" in excinfo.value.details

def test_resource_limit_error_handling(enhancer_instance) -> None:
    """Checks that ResourceLimitError is raised when API quota is exhausted."""
    with patch.object(enhancer_instance.model, 'generate_content',
                      side_effect=Exception("429 Too Many Requests: quota exceeded")):
        with pytest.raises(ResourceLimitError) as excinfo:
            enhancer_instance.enhance("a sunset")
        assert "Gemini API quota exhausted" in str(excinfo.value)
