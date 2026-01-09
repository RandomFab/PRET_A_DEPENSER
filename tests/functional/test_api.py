import pytest
from unittest.mock import patch
from src.api.schemas import ScoringData
from src.api.main import app as fastapi_app

class TestApiRoutes:
    
    def test_health_check(self, client):
        """Verifies that the API is alive."""
        response = client.get("/api_health")
        assert response.status_code == 200
        assert response.json() == {'message': 'API is running correctly'}

    def test_router_health(self, client):
        """Verifies that the router is correctly mounted."""
        response = client.get("/router_health")
        assert response.status_code == 200

    def test_get_model_info(self, client):
        """Verifies the info endpoint. Note: uses the mocked context of the service if necessary,
           but 'model_info' reads from disk. Unit test mocked the disk,
           here it is a functional test.
           If the test environment does not have the 'MLmodel' files, it may 404.
           For this functional test, we expect the app to handle absence cleanly.
        """
        response = client.get("/model_info")
        # In test env without real files, we likely expect a 404
        # or a default result if we mocked everything.
        # Here we just verify it doesn't crash (500).
        assert response.status_code in [200, 404]

    def test_prediction_nominal(self, client, sample_payload):
        """Happy Path for a prediction."""
        response = client.post("/individual_score", json=sample_payload)
        assert response.status_code == 200
        
        json_resp = response.json()
        assert "score" in json_resp
        assert "decision" in json_resp
        assert 0.0 <= json_resp["score"] <= 1.0
        # Check DummyModel (fixed at 0.42 in conftest.py)
        assert json_resp["score"] == 0.42

    def test_prediction_validation_error(self, client, sample_payload):
        """Verifies rejection if data is invalid."""
        # We corrupt the payload (invalid age)
        bad_payload = sample_payload.copy()
        bad_payload["YEARS_BIRTH"] = 10 
        
        response = client.post("/individual_score", json=bad_payload)
        assert response.status_code == 422
        assert "detail" in response.json()

    def test_prediction_idempotency(self, client, sample_payload):
        """Verifies determinism: 2 identical calls = same result."""
        resp1 = client.post("/individual_score", json=sample_payload)
        resp2 = client.post("/individual_score", json=sample_payload)
        
        assert resp1.json() == resp2.json()

    def test_security_extra_fields(self, client, sample_payload):
        """Verifies behavior with unknown fields (security).
           By default Pydantic ignores extra fields unless 'extra=forbid'.
           Let's just verify it doesn't crash.
        """ 
        payload = sample_payload.copy()
        payload["SQL_INJECTION"] = "DROP TABLE users;"
        
        response = client.post("/individual_score", json=payload)
        # If configured to ignore, it should pass 200. If forbid, 422.
        # In current schemas.py, it is default (ignore), so 200 expected.
        assert response.status_code == 200 

    # --- Error Handling & Boundaries Tests ---

    def test_model_status_error(self, client):
        """Simulates an internal error when checking model status."""
        # We patch the function imported in routes.py
        with patch("src.app.routes.get_model_status", side_effect=Exception("Disk failure")):
            response = client.get("/model_status")
            assert response.status_code == 500
            assert "Internal server error" in response.json()["detail"]

    def test_model_signature_error(self, client):
        """Simulates an internal error when retrieving model signature."""
        with patch("src.app.routes.get_model_signature", side_effect=Exception("Corrupted file")):
            response = client.get("/model_signature")
            assert response.status_code == 500
            assert "Internal server error" in response.json()["detail"]

    def test_model_info_error(self, client):
        """Simulates an internal error when retrieving model info."""
        with patch("src.app.routes.get_model_info", side_effect=Exception("Unexpected structure")):
            response = client.get("/model_info")
            assert response.status_code == 500
            assert "Internal server error" in response.json()["detail"]

    def test_prediction_model_not_loaded(self, client, sample_payload):
        """Verifies 503 error if model is not loaded in app state."""
        # Temporarily remove model from app state
        original_model = getattr(fastapi_app.state, "model", None)
        fastapi_app.state.model = None
        
        try:
            response = client.post("/individual_score", json=sample_payload)
            assert response.status_code == 503
            # Checked actual msg: "Model is currently not loaded on the server. Please reload it."
            assert "not loaded" in response.json()["detail"]
        finally:
            # Restore model to avoid breaking other tests
            fastapi_app.state.model = original_model

    def test_prediction_internal_error(self, client, sample_payload):
        """Simulates a crash during value prediction."""
        with patch("src.app.routes.get_prediction", side_effect=Exception("CatBoost crash")):
            response = client.post("/individual_score", json=sample_payload)
            assert response.status_code == 500
            assert "CatBoost crash" in response.json()["detail"]

    # --- Batch Prediction Tests ---

    def test_batch_prediction_nominal(self, client, sample_payload):
        """Verifies batch scoring logic."""
        # Create a batch with 2 identical items
        batch = [sample_payload, sample_payload]
        
        response = client.post("/multiple_score", json=batch)
        assert response.status_code == 200
        
        results = response.json()
        assert isinstance(results, list)
        assert len(results) == 2
        # Check that both results are correct
        assert results[0]["score"] == 0.42
        assert results[1]["score"] == 0.42

    def test_batch_prediction_model_not_loaded(self, client, sample_payload):
        """Verifies 503 if model not loaded during batch."""
        original_model = getattr(fastapi_app.state, "model", None)
        fastapi_app.state.model = None
        try:
            response = client.post("/multiple_score", json=[sample_payload])
            assert response.status_code == 503
        finally:
             fastapi_app.state.model = original_model

    def test_batch_prediction_item_error(self, client, sample_payload):
        """Verifies handling of an error within one item of the batch."""
        # Simulate get_prediction returning an error dict for the first item
        # We need to preserve the real behavior for valid items if possible, 
        # but here we mock get_prediction entirely to test the route logic.
        
        # Scenario: 1st call -> error, 2nd call -> success
        mock_returns = [
            {"error": "Value too high"},
            {"score": 0.42, "decision": "Accepted"}
        ]
        
        with patch("src.app.routes.get_prediction", side_effect=mock_returns):
            batch = [sample_payload, sample_payload]
            # The route raises 400 immediately if ANY item fails
            response = client.post("/multiple_score", json=batch)
            assert response.status_code == 400
            assert "Value too high" in response.json()["detail"]

    # --- Reload Model Tests ---

    def test_reload_model_success(self, client):
        """Verifies model reload triggers download and update."""
        with patch("src.app.routes.download_model_from_hf") as mock_dl, \
             patch("src.app.routes.load_model_instance") as mock_load:
            
            # Mock load_model_instance to return a 'NewModel'
            mock_load.return_value = "NewModelInstance"
            
            response = client.post("/reload_model")
            
            assert response.status_code == 200
            assert "model has been well retrieved" in response.json()["message"]
            
            # Verify download and load were called
            mock_dl.assert_called_once()
            mock_load.assert_called_once()
            
            # Verify app state was updated
            assert fastapi_app.state.model == "NewModelInstance"

    def test_reload_model_failure_download(self, client):
        """Verifies behavior if download fails."""
        with patch("src.app.routes.download_model_from_hf", side_effect=Exception("HF Down")):
            response = client.post("/reload_model")
            assert response.status_code == 500
            assert "HF Down" in response.json()["detail"]

    def test_reload_model_failure_load(self, client):
        """Verifies behavior if load fails (returns None)."""
        with patch("src.app.routes.download_model_from_hf"), \
             patch("src.app.routes.load_model_instance", return_value=None):
            
            response = client.post("/reload_model")
            assert response.status_code == 500
            assert "Failed to reload" in response.json()["detail"]

    # --- Missing File / Warning Logs Scenarios ---

    def test_model_status_missing_file_warning(self, client):
        """Covers: ⚠️ Model status requested but model file is missing."""
        # We mock get_model_status directly from routes to return {'exists': False}
        with patch("src.app.routes.get_model_status", return_value={"exists": False, "model_name": "m.cb"}):
            response = client.get("/model_status")
            assert response.status_code == 404
            assert "Model file not found" in response.json()["detail"]

    def test_model_signature_missing_file_warning(self, client):
        """Covers: ⚠️ Model signature requested but MLmodel file is missing."""
        with patch("src.app.routes.get_model_signature", return_value={"exists": False}):
            response = client.get("/model_signature")
            assert response.status_code == 404
            assert "Signature file (MLmodel) not found" in response.json()["detail"]

    def test_model_info_missing_file_warning(self, client):
        """Covers: ⚠️ Model info requested but MLmodel metadata is missing."""
        with patch("src.app.routes.get_model_info", return_value={"exists": False}):
            response = client.get("/model_info")
            assert response.status_code == 404
            assert "Model information not found" in response.json()["detail"] 
