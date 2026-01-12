import pandas as pd
import requests
import streamlit as st
import concurrent.futures

def csv_to_json(csv_file):
    """Read an uploaded CSV (or path-like) and return a list of records.

    Returns a Python list of dicts suitable to pass to `requests.post(..., json=...)`.
    """
    df = pd.read_csv(csv_file)
    return df.to_dict(orient='records')

def reload_model():
    try:
        resp = requests.post("http://localhost:8000/reload_model", timeout=10)
        resp.raise_for_status()
        st.success("Modèle rechargé avec succès")
    except requests.RequestException as e:
        st.error(f"Erreur lors du reload : {e}")

def fetch_service_status(url):
    """Effectue l'appel réseau pour le statut."""
    try:
        # Augmentation du timeout pour éviter le rouge au premier chargement
        response = requests.get(url, timeout=5)
        return "#28a745" if response.status_code == 200 else "#dc3545"
    except:
        return "#dc3545"

@st.cache_data(ttl=300) # On cache la pastille HTML 5 minutes
def get_cached_status_html(url, color):
    return f"<span style='color: {color}; font-size: 15px;'>●</span>"

@st.cache_data(ttl=3600) # Cache la signature 1h
def get_model_signature_cached():
    try:
        response = requests.get("http://localhost:8000/model_signature", timeout=10)
        if response.status_code == 200:
            return response.json().get("columns", [])
    except:
        pass
    return []

# --- 2. GESTION DES APPELS ASYNCHRONES ---

@st.cache_resource
def get_executor():
    return concurrent.futures.ThreadPoolExecutor(max_workers=5)

def model_info():
    # Récupération sûre des informations du modèle
    try:
        resp = requests.get("http://localhost:8000/model_info", timeout=5)
        resp.raise_for_status()
        json_resp = resp.json()
        model_infos = json_resp.get("info", {}) if isinstance(json_resp, dict) else {}
       
    except requests.RequestException as e:
        st.error(f"Impossible de joindre l'API /model_info : {e}")
        model_infos = {}
    except ValueError:
        st.error("Réponse JSON invalide depuis /model_info")
        model_infos = {}
    
    return model_infos

def model_signature():

    try:
        resp = requests.get("http://localhost:8000/model_signature", timeout=5)
        resp.raise_for_status()
        json_resp = resp.json()
        columns = json_resp.get("columns", {}) if isinstance(json_resp, dict) else {}
        df = pd.DataFrame(columns)

    except requests.RequestException as e:
        st.error(f"Impossible de joindre l'API /model_signature : {e}")
        df=pd.DataFrame()
    except ValueError:
        st.error("Réponse JSON invalide depuis /model_signature")
        df=pd.DataFrame()
    
    return df