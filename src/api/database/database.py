from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .table_models import Base

import os
from dotenv import load_dotenv

load_dotenv()
ENV = os.getenv("APP_ENV", "development")
# 2. On charge le fichier .env correspondant si on est en local
if ENV == "development":
    load_dotenv(".env.dev")
elif ENV == "production":
    # Sur HF, les variables sont déjà dans le système, pas besoin de fichier .env
    pass
SQLALCHEMY_DATABASE_URL = os.getenv('DATABASE_URL')

connect_args = {}
if SQLALCHEMY_DATABASE_URL and SQLALCHEMY_DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """
    Objectif : Créer physiquement la table 'prediction_logs' dans Postgres
    si elle n'existe pas encore.
    """
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()