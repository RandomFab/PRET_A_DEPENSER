import math
from pydantic import BaseModel, Field, field_validator

class ScoringData(BaseModel):
    """
    Schéma de données pour la prédiction, basé sur la signature MLflow.
    """
    FE_EXT_SOURCE_MEAN: float = Field(
        ..., ge=0.0, le=1.0, description="Moyenne des 3 sources externes"
    )

    BURO_MONTHS_BALANCE_SIZE_MEAN: float = Field(
        ..., ge=0.0, description="Nombre moyen de mois d'historique bureau"
    )

    CODE_GENDER: int = Field(
        ..., ge=0, le=1, description="Genre (0 = F, 1 = M)"
    )

    INSTAL_DPD_MEAN: float = Field(
        ..., ge=0.0, description="Moyenne des jours de retard sur les paiements"
    )

    BURO_MONTHS_BALANCE_MAX_MIN: float = Field(
        ..., description="Différence max-min des mois d'historique bureau"
    )

    FE_GOODS_CREDIT_RATE: float = Field(
        ..., ge=0.0, description="Ratio montant du bien / montant du crédit"
    )

    APPROVED_CNT_PAYMENT_MEAN: float = Field(
        ..., ge=0.0, description="Nombre moyen de paiements approuvés"
    )

    YEARS_BIRTH: int = Field(
        ..., ge=18, le=100, description="Âge du client en années"
    )

    YEARS_EMPLOYED: int = Field(
        ..., ge=0, le=80, description="Années d'expérience professionnelle"
    )

    AMT_ANNUITY: float = Field(
        ..., ge=0.0, description="Montant de l'annuité du crédit"
    )

    NAME_FAMILY_STATUS_Married: bool = Field(
        ..., description="Client marié"
    )

    INSTAL_AMT_PAYMENT_SUM: float = Field(
        ..., ge=0.0, description="Somme totale des montants payés en mensualités"
    )

    FE_EXT_SOURCE_MIN: float = Field(
        ..., ge=0.0, le=1.0, description="Minimum des 3 sources externes"
    )

    PREV_CNT_PAYMENT_MEAN: float = Field(
        ..., ge=0.0, description="Nombre moyen de paiements précédents"
    )

    FE_EXT_SOURCE_MAX: float = Field(
        ..., ge=0.0, le=1.0, description="Maximum des 3 sources externes"
    )

    # --- Validateurs personnalisés ---

    @field_validator(
        "INSTAL_DPD_MEAN",
        "AMT_ANNUITY",
        "INSTAL_AMT_PAYMENT_SUM",
        "FE_GOODS_CREDIT_RATE",
        mode="before"
    )
    @classmethod
    def clean_numeric(cls, v):
        if v is None:
            raise ValueError("Valeur manquante")

        try:
            v = float(v)
        except (ValueError, TypeError):
            raise ValueError("Valeur numérique attendue")

        if math.isnan(v):
            raise ValueError("NaN interdit")

        return v

    @field_validator("YEARS_EMPLOYED")
    @classmethod
    def check_employment_vs_age(cls, v, info):
        age = info.data.get("YEARS_BIRTH")
        if age is not None and v > age:
            raise ValueError(f"Années d'emploi ({v}) ne peuvent pas être supérieures à l'âge ({age})")
        return v

    @field_validator("FE_EXT_SOURCE_MAX")
    @classmethod
    def check_ext_sources(cls, v, info):
        min_val = info.data.get("FE_EXT_SOURCE_MIN")
        mean_val = info.data.get("FE_EXT_SOURCE_MEAN")

        if min_val is not None and v < min_val:
            raise ValueError(f"EXT_SOURCE_MAX ({v}) < EXT_SOURCE_MIN ({min_val})")

        if mean_val is not None and min_val is not None:
            if not (min_val <= mean_val <= v):
                raise ValueError(f"La moyenne ({mean_val}) doit être entre le min ({min_val}) et le max ({v})")

        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "FE_EXT_SOURCE_MEAN": 0.5892,
                "BURO_MONTHS_BALANCE_SIZE_MEAN": 0.0,
                "CODE_GENDER": 0,
                "INSTAL_DPD_MEAN": 0.0,
                "BURO_MONTHS_BALANCE_MAX_MIN": 0.0,
                "FE_GOODS_CREDIT_RATE": 1.0,
                "APPROVED_CNT_PAYMENT_MEAN": 0.0,
                "YEARS_BIRTH": 59,
                "YEARS_EMPLOYED": 0,
                "AMT_ANNUITY": 20952.0,
                "NAME_FAMILY_STATUS_Married": True,
                "INSTAL_AMT_PAYMENT_SUM": 0.0,
                "FE_EXT_SOURCE_MIN": 0.2635,
                "PREV_CNT_PAYMENT_MEAN": 0.0,
                "FE_EXT_SOURCE_MAX": 0.7992
            }
        }
    }

class PredictionResponse(BaseModel):
    """
    Format standard pour une réponse de prédiction individuelle.
    """
    score: float = Field(..., description="Probabilité brute (0 à 1)")
    prediction: int = Field(..., description="Classe prédite (0=Accordé, 1=Refusé)")
    threshold: float = Field(..., description="Seuil d'arbitrage utilisé")
    decision: str = Field(..., description="Décision textuelle")

class ModelStatusResponse(BaseModel):
    """
    État actuel du fichier modèle sur le disque.
    """
    message: str
    status: dict | None = None

class ModelSignatureResponse(BaseModel):
    """
    Signature technique (colonnes attendues).
    """
    message: str
    columns: list[str] | None = None
    nb_features: int | None = None

class ModelInfoResponse(BaseModel):
    """
    Métadonnées détaillées du modèle.
    """
    message: str
    info: dict | None = None
