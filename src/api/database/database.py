from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .table_models import Base

import os
from dotenv import load_dotenv

load_dotenv()

user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")

SQLALCHEMY_DATABASE_URL = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"

engine = create_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False,bind=engine)

def init_db():
    """
    Objectif : Créer physiquement la table 'prediction_logs' dans Postgres
    si elle n'existe pas encore.
    """
    Base.metadata.createall(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()