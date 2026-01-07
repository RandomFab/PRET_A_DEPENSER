from pydantic import BaseModel, Field

class ScoringData(BaseModel):
    """
    Schéma de données pour la prédiction, basé sur la signature MLflow.
    """
    FE_EXT_SOURCE_MEAN: float = Field(..., description="Moyenne des sources externes")
    BURO_MONTHS_BALANCE_SIZE_MEAN: float
    CODE_GENDER: int = Field(..., description="Genre (0 ou 1)")
    INSTAL_DPD_MEAN: float
    BURO_MONTHS_BALANCE_MAX_MIN: float
    FE_GOODS_CREDIT_RATE: float
    APPROVED_CNT_PAYMENT_MEAN: float
    YEARS_BIRTH: int = Field(..., description="Âge du client en années")
    YEARS_EMPLOYED: int = Field(..., description="Années d'expérience")
    AMT_ANNUITY: float
    NAME_FAMILY_STATUS_Married: bool
    INSTAL_AMT_PAYMENT_SUM: float
    FE_EXT_SOURCE_MIN: float
    PREV_CNT_PAYMENT_MEAN: float
    FE_EXT_SOURCE_MAX: float

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
