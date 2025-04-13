import pytest
from unittest.mock import patch, MagicMock

from fabricai_inference_server.engine import LlamaEngine, load_default_engine


@pytest.fixture
def mock_model_path(tmp_path):
    """
    Creates a temporary file to simulate a model file.
    """
    fake_model_file = tmp_path / "fake-model.gguf"
    fake_model_file.write_text("not a real model data")
    return str(fake_model_file)


def test_llama_engine_init_file_not_found():
    """
    Ensure engine raises FileNotFoundError if the model file doesn't exist.
    """
    with pytest.raises(FileNotFoundError):
        LlamaEngine(model_path="non_existent_file.gguf")


@patch("fabricai_inference_server.engine.Llama")
def test_llama_engine_init_success(mock_llama, mock_model_path):
    """
    If the model file exists, LlamaEngine shouldn't raise an error (mock the Llama init).
    """
    # Arrange
    mock_instance = MagicMock()
    mock_llama.return_value = mock_instance

    # Act
    engine = LlamaEngine(model_path=mock_model_path)

    # Assert
    assert engine.model_path == mock_model_path
    mock_llama.assert_called_once()


@patch("fabricai_inference_server.engine.LlamaEngine.__init__", return_value=None)
def test_load_default_engine_no_init(mock_init, monkeypatch, mock_model_path):
    # Arrange
    monkeypatch.setenv("LLM_MODEL", mock_model_path)

    # Act
    engine = load_default_engine()

    # Assert
    mock_init.assert_called_once()
