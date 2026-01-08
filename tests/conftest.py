import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.app.main import app
from src.model import model_service as model_service_module


class DummyModel:
    """Mock model for testing purposes to avoid loading the real heavy model."""
    def predict_proba(self, X):
        # returns a fixed probability for each row (class 0, class 1)
        # mimics CatBoost predict_proba output
        return [[0.58, 0.42] for _ in X]


@pytest.fixture(scope="session")
def client():
    """FastAPI TestClient with a patched model loader to avoid loading the real model.

    This fixture:
    - temporarily replaces `load_model_instance` in `src.app.main` with a function returning `DummyModel()`
    - starts the client (triggering startup/shutdown event handlers)
    - restores the original function after use
    """
    # We must patch where it is imported (src.app.main), because main.py does "from ... import ..."
    from src.app import main
    original_loader = getattr(main, "load_model_instance", None)
    
    try:
        main.load_model_instance = lambda: DummyModel()
        with TestClient(app) as c:
            yield c
    finally:
        if original_loader is not None:
            main.load_model_instance = original_loader


@pytest.fixture
def sample_payload():
    """Loads `tests/data/sample_payload.json` if present, otherwise returns a minimal valid payload.

    Useful for testing scoring endpoints.
    """
    data_path = Path(__file__).parent / "data" / "sample_payload.json"
    if data_path.exists():
        return json.loads(data_path.read_text())

    # Minimal payload based on `ScoringData` (plausible values)
    return {
        "FE_EXT_SOURCE_MEAN": 0.5,
        "BURO_MONTHS_BALANCE_SIZE_MEAN": 12.0,
        "CODE_GENDER": 1,
        "INSTAL_DPD_MEAN": 0.0,
        "BURO_MONTHS_BALANCE_MAX_MIN": 5.0,
        "FE_GOODS_CREDIT_RATE": 0.2,
        "APPROVED_CNT_PAYMENT_MEAN": 1.0,
        "YEARS_BIRTH": 35,
        "YEARS_EMPLOYED": 5,
        "AMT_ANNUITY": 1000.0,
        "NAME_FAMILY_STATUS_Married": False,
        "INSTAL_AMT_PAYMENT_SUM": 1000.0,
        "FE_EXT_SOURCE_MIN": 0.2,
        "PREV_CNT_PAYMENT_MEAN": 0.0,
        "FE_EXT_SOURCE_MAX": 0.7
    }
