import pytest
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path
import json
from src.model import model_service

class TestModelService:
    
    @patch('src.model.model_service.MODEL_DIR', Path('/fake/path'))
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.stat')
    def test_get_model_status_present(self, mock_stat, mock_exists):
        """Verifies the return value when the model file exists."""
        mock_exists.return_value = True
        mock_stat.return_value = MagicMock(st_size=1024*1024, st_mtime=1600000000)
        
        status = model_service.get_model_status()
        
        assert status["exists"] is True
        assert status["size_kb"] == 1024.0
    
    @patch('src.model.model_service.MODEL_DIR', Path('/fake/path'))
    @patch('pathlib.Path.exists')
    def test_get_model_status_missing(self, mock_exists):
        """Verifies the return value when the model file is missing."""
        mock_exists.return_value = False
        status = model_service.get_model_status()
        assert status["exists"] is False
        assert "size_kb" not in status

    @patch('src.model.model_service.MODEL_DIR', Path('/fake/path'))
    @patch('pathlib.Path.exists')
    def test_get_signature_missing(self, mock_exists):
        """If MLmodel is missing, returns an empty but consistent structure."""
        model_service.get_model_signature.cache_clear()
        mock_exists.return_value = False
        sig = model_service.get_model_signature()
        assert sig["exists"] is False
        assert sig["columns"] == []

    @patch('src.model.model_service.MODEL_DIR', Path('/fake/path'))
    @patch('pathlib.Path.exists', return_value=True)
    def test_get_signature_parsing(self, mock_exists):
        """Verifies standard YAML parsing."""
        model_service.get_model_signature.cache_clear()
        yaml_content = """
signature:
  inputs:
    - name: "AGE"
      type: "long"
    - name: "SALARY"
      type: "double"
metadata:
  best_threshold: 0.65
"""
        with patch("builtins.open", mock_open(read_data=yaml_content)):
             sig = model_service.get_model_signature()
             
        assert sig["exists"] is True
        assert len(sig["columns"]) == 2
        assert sig["best_threshold"] == 0.65

    @patch('src.model.model_service.MODEL_DIR', Path('/fake/path'))
    @patch('pathlib.Path.exists', return_value=True)
    def test_get_signature_with_json_string(self, mock_exists):
        """Verifies parsing when MLflow stores the signature as a JSON string."""
        model_service.get_model_signature.cache_clear()
        # Tricky case where inputs is a string, not a list
        json_inputs = json.dumps(["A", "B", "C"])
        yaml_content = f"""
signature:
  inputs: '{json_inputs}'
metadata:
  best_threshold: 0.5
"""
        with patch("builtins.open", mock_open(read_data=yaml_content)):
             sig = model_service.get_model_signature()
             
        assert len(sig["columns"]) == 3
        assert sig["columns"] == [
             {"name": "A", "type": "double", "description": None},
             {"name": "B", "type": "double", "description": None},
             {"name": "C", "type": "double", "description": None}
        ]
