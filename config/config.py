import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
EXTERNAL_DATA_DIR = BASE_DIR / 'data' / 'external'
RAW_DATA_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed'

VIOLET_CLAIR = '#99abf7'
VIOLET_FONCE = '#7451eb'
GRIS_CLAIR = '#3f3f3f'
JAUNE_FONCE = '#f9ab2d'
JAUNE_CLAIR = '#f7e096ff'

PALETTE = [VIOLET_CLAIR, VIOLET_FONCE, GRIS_CLAIR, JAUNE_FONCE, JAUNE_CLAIR]
