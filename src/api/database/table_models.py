from sqlalchemy import Column, Integer, Float, String, DateTime, JSON
from sqlalchemy.orm import declarative_base
from datetime import datetime, timezone
Base = declarative_base()

class PredictionLog(Base):
    """
    Objectif : Mapper cette classe à une table 'prediction_logs' dans PostgreSQL.
    Cette table va stocker chaque appel fait à l'API de scoring.
    """

    __tablename__ = 'prediction_logs'

    # Déclaration des colonne de la table :

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    model_version = Column(String)
    latency_ms = Column(Float)
    inputs = Column(JSON)
    outputs = Column(JSON)
    status_code = Column(Integer)
