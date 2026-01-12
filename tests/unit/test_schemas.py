from pydantic import ValidationError
import pytest
from src.api.schemas import ScoringData

class TestScoringDataValidation:
    
    def test_valid_payload(self, sample_payload):
        """Verifies that a valid payload passes."""
        # sample_payload est injecté automatiquement depuis conftest.py
        model = ScoringData(**sample_payload)
        
        # On vérifie juste que l'objet est bien créé avec les valeurs du sample
        if "YEARS_BIRTH" in sample_payload:
            assert model.YEARS_BIRTH == sample_payload["YEARS_BIRTH"]

    def test_employment_cannot_exceed_age(self, sample_payload):
        """Verifies that (Years employed > Age) raises an error."""
        # On copie pour ne pas modifier la fixture pour les autres tests
        data = sample_payload.copy()
        
        # On crée le cas d'erreur spécifique
        data["YEARS_BIRTH"] = 25
        data["YEARS_EMPLOYED"] = 30 

        with pytest.raises(ValidationError) as excinfo:
            ScoringData(**data)
        # Vérif message d'erreur (compatible FR/EN selon ton implémentation)
        assert "n emploi" in str(excinfo.value) or "mployment" in str(excinfo.value) or "d'emploi" in str(excinfo.value)

    def test_age_boundaries(self, sample_payload):
        """Verifies age boundaries (18-100)."""
        # Utilisation de l'unpacking pour surcharger un seul champ
        
        # Too young
        with pytest.raises(ValidationError):
            ScoringData(**{**sample_payload, "YEARS_BIRTH": 17})
            
        # Too old
        with pytest.raises(ValidationError):
            ScoringData(**{**sample_payload, "YEARS_BIRTH": 101})

    def test_numeric_cleaning_and_nan(self, sample_payload):
        """Verifies numeric type cleaning and NaN rejection."""
        
        # Valid string -> converted to float
        data_str = {**sample_payload, "AMT_ANNUITY": "5000.50"}
        model = ScoringData(**data_str)
        assert model.AMT_ANNUITY == 5000.50
        
        # NaN -> Error ("NaN forbidden")
        with pytest.raises(ValidationError) as exc:
            ScoringData(**{**sample_payload, "AMT_ANNUITY": float("nan")})
        assert "NaN" in str(exc.value)

    def test_ext_source_range(self, sample_payload):
        """Verifies that external sources are in [0, 1]."""
        
        # < 0
        with pytest.raises(ValidationError):
            ScoringData(**{**sample_payload, "FE_EXT_SOURCE_MEAN": -0.01})
            
        # > 1
        with pytest.raises(ValidationError):
            ScoringData(**{**sample_payload, "FE_EXT_SOURCE_MEAN": 1.01}) 
