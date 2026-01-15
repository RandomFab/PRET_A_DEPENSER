---
title: 🚀 Pret-à-Dépenser
emoji: 💰
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

<!-- Project title and badges -->
# 🚀 Pret-à-Dépenser

[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128+-green.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.17+-black.svg?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16+-336791.svg?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![MLflow](https://img.shields.io/badge/MLflow-%3E%3D3.8.1-orange.svg?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-%3E%3D1.2.8-yellow.svg)](https://catboost.ai/)
[![Evidently](https://img.shields.io/badge/Evidently-%3E%3D0.7.20-red.svg)](https://www.evidentlyai.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Hub-purple.svg?logo=huggingface&logoColor=white)](https://huggingface.co/)

**API de Scoring Crédit : Déploiement, Monitoring et MLOps.**

---

## 🎯 Objectif

Automatiser une chaîne complète de MLOps pour l'octroi de crédits, assurant la reproductibilité et la surveillance du modèle en production :
- **Déploiement** d'un modèle CatBoost exporté au format **ONNX** via une API **FastAPI**.
- **Tracking** des expériences et centralisation des artifacts avec **MLflow**.
- **Historisation** des prédictions (inputs/outputs) dans **PostgreSQL**.
- **Monitoring** de la qualité des données et du *Data Drift* avec **Evidently**.
- **Interface Utilisateur** interactive avec **Streamlit**.

---

## ✨ Fonctionnalités

- ✅ **API RESTful (FastAPI)** : Endpoints pour le scoring unitaire et batch optimisés.
- ✅ **Inférence Accélérée** : Utilisation de **ONNX Runtime** pour réduire la latence de prédiction.
- ✅ **Base de Données (PostgreSQL)** : Logging asynchrone des requêtes et réponses pour constitution du dataset de production.
- ✅ **Analyses de Drift (Evidently)** : Notebook dédié pour comparer les données de production vs référence (Training).
- ✅ **Gestion de Modèle** : Chargement dynamique, versioning MLflow, et rechargement à chaud depuis Hugging Face Hub.
- ✅ **Interface Streamlit** : Dashboard simple pour tester le scoring manuellement.

---

## ⚡ Optimisation & Performance

### 🚀 Inférence (Latence)

Une attention particulière a été portée à l'optimisation du pipeline d'inférence, passant d'un modèle CatBoost natif à une exécution optimisée via **ONNX Runtime**.

| Version | Temps d'exécution (Moyen) | Gain de performance |
| :--- | :--- | :--- |
| **Baseline (Python)** | ~61 ms | - |
| **Optimisation Code** | ~20 ms | **x3.0** |
| **ONNX Runtime** | ~18 ms | **x3.4** |

> *Note : Les mesures incluent le pré-traitement et l'inférence pour une requête unitaire.*

### 🐳 Image Docker (Taille)

L'empreinte du container a été drastiquement réduite grâce à une stratégie **Multi-stage Build** combinée au gestionnaire de paquets **uv** :
1.  **Usage de `python:3.13-slim`** pour une base légère.
2.  **Exclusion des dépendances de dev** (pytest, black, jupyter... sont ignorés en prod).
3.  **Nettoyage des artifacts** de build via l'étape intermédiaire.

| Version | Taille de l'image | Réduction |
| :--- | :--- | :--- |
| **Standard Build** | ~2.30 Go | - |
| **Multi-Stage + uv** | **1.02 Go** | **-56%** |

---

## 📡 Endpoints exposés

L'application FastAPI se trouve dans `src/api/main.py` et expose les routes suivantes (sans préfixe). Le frontend Streamlit se trouve dans `src/app/main.py` et communique avec l'API pour afficher l'interface utilisateur :

- `GET /` → redirection vers la documentation interactive `/docs`.
- `GET /api_health` → état de santé global de l'API.

Routes du routeur (`src/api/routes.py`):
- `GET /router_health` → health du router.
- `GET /model_status` → état du fichier modèle sur disque (`model.onnx` ou `model.cb`).
- `GET /model_signature` → colonnes attendues (signature MLflow) et nombre de features.
- `GET /model_info` → métadonnées (version, date, threshold recommandé).
- `POST /individual_score` → prédiction pour un individu (Pydantic)
- `POST /multiple_score` → prédictions en batch (liste d'objets Pydantic)
- `POST /reload_model` → télécharge le fichier `HF_FILENAME` depuis `HF_REPO_ID` et recharge le modèle en mémoire.

Exemple de payload (utilisez l'exemple depuis le schema `ScoringData` dans `src/app/schemas.py`):

```json
{
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
	"NAME_FAMILY_STATUS_Married": true,
	"INSTAL_AMT_PAYMENT_SUM": 0.0,
	"FE_EXT_SOURCE_MIN": 0.2635,
	"PREV_CNT_PAYMENT_MEAN": 0.0,
	"FE_EXT_SOURCE_MAX": 0.7992
}
```

Réponse de prédiction (exemple):

```json
{
	"score": 0.1234,
	"prediction": 0,
	"threshold": 0.5,
	"decision": "Accordé"
}
```

---

## 📁 Architecture & diagrammes

**Arborescence principale**

```text
PRET_A_DEPENSER/
│
├── 📂 config/               # Configuration (chemins, logger, etc.)
├── 📂 data/                 # Données (raw, processed)
├── 📂 exported_model/       # Artifacts MLflow (model.onnx, model.cb, MLmodel)
├── 📂 notebooks/            # Notebooks (Drift Analysis, Training)
├── 📂 scripts/              # Utilitaires HF (upload/download)
│
├── 📂 src/
│   ├── 🎨 api/              # Backend FastAPI (Modèle, Routes, Schemas)
│   │   ├── database/        # Modèles SQLAlchemy & Connexion DB
│   │   ├── main.py
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── 🧠 app/              # Frontend Streamlit
│   └── ⚙️ model/            # Logique métier & Hubs (MLflow, HF)
├── 🧪 tests/                # Tests unitaires et fonctionnels
├── 🐳 Dockerfile            # Packaging Docker
├── 🐙 docker-compose.yml    # Orchestration locale
└── 🚀 start.sh             # Script de démarrage dual (API + App)
```

### Architecture Technique

```mermaid
graph TB
    subgraph Client
        Browser[🌍 Frontend Streamlit / User]
    end

    subgraph "Serveur API (FastAPI)"
        API[🚀 API Gateway]
        Routes[🛣️ Router]
        Mservice[🧠 Model Service]
        DB_Service[💾 Database Service]
    end

    subgraph "Stockage & Monitoring"
        HF[🤗 Hugging Face Hub]
        PG[(🐘 PostgreSQL)]
        MLflow[📈 MLflow Tracking]
        Evidently[📊 Evidently Reports]
    end

    Browser -- "POST /individual_score" --> API
    API --> Routes
    Routes -- "Predict (ONNX)" --> Mservice
    Routes -- "Log Prediction (Background)" --> DB_Service
    
    Mservice -- "Load Model" --> HF
    DB_Service -- "INSERT prediction_logs" --> PG
    
    subgraph "Offline Analysis"
        NB[📓 Notebook Drift]
        NB -- "Read Reference" --> MLflow
        NB -- "Read Production Logs" --> PG
        NB --> Evidently
    end
```

--- 

```mermaid
sequenceDiagram
	participant Client
	participant API
	participant ModelService
	participant DB as PostgreSQL

	Client->>API: POST /individual_score
	API->>ModelService: Request Prediction
	ModelService->>ModelService: Compute Score (ONNX Runtime)
	ModelService-->>API: Result (Score, Decision)
	API-->>Client: JSON Response (200 OK)
	
	Note right of API: Background Task
	API->>DB: INSERT INTO prediction_logs (inputs, outputs, latency)
```

---

## ⚙️ Variables d'environnement (importantes)

- `HF_REPO_ID` — identifiant du repo HF (ex: `username/model-repo`) requis pour `POST /reload_model`.
- `HUGGINGFACE_TOKEN` — token HF (ou `HF_TOKEN`) pour accéder au repo privé.
- `HF_FILENAME` — nom du fichier dans le repo HF (défaut `model.onnx`).
- `DATABASE_URL` — Connection string PostgreSQL (ex: `postgresql://user:pass@host:5432/db`).
- `MLFLOW_TRACKING_URI` — (optionnel) point vers le serveur MLflow.

Ces variables peuvent être mises dans `.env.dev` (utilisé par le projet).

---

## 🚀 Installation & Déploiement

### 🐳 Via Docker (Recommandé)

Le projet utilise **Docker Compose** pour orchestrer l'API, la Base de Données, et l'interface utilisateur.

1.  Créer le fichier `.env.dev` ou `.env` avec les variables ci-dessus.
2.  Lancer la stack :
    ```bash
    docker-compose up --build -d
    ```
3.  Accéder aux services :
    - **FastAPI (Docs)** : [http://localhost:8000/docs](http://localhost:8000/docs)
    - **Streamlit** : [http://localhost:7860](http://localhost:7860)

### 🐍 Installation Locale (Dev)

Préréquis : **Python 3.13+** et **uv** (recommandé).

1.  **Cloner le dépôt** :
    ```bash
    git clone https://github.com/RandomFab/PRET_A_DEPENSER.git
    cd PRET_A_DEPENSER
    ```
2.  **Installer les dépendances** :
    ```bash
    uv sync
    ```
3.  **Lancer les services** :
    - S'assurer qu'une base PostgreSQL tourne localement.
    - API : `uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload`
    - App : `uv run streamlit run src/app/main.py`

---

## 🧪 Monitoring & Drift Analysis

Le projet intègre une surveillance de la dérive des données (Data Drift) :

1. Les prédictions faites en production sont enregistrées dans PostgreSQL.
2. Un notebook d'analyse compare ces données avec le dataset de référence (Entraînement).
3. **Usage** : Ouvrir `notebooks/drift_analysis.ipynb` et exécuter toutes les cellules pour générer le rapport HTML `data_drift_report.html` (Evidently).

---

## 🧪 Tests & Qualité

La suite de tests utilise `pytest` et génère un rapport de couverture.

```bash
uv run pytest --cov=src --cov-report=html
```
Le rapport est généré dans `htmlcov/index.html`.

---

## 🤖 CI/CD (GitHub Actions)

Le projet intègre une pipeline automatisée (`.github/workflows/ci-cd.yml`) :
- **Test Job** : Exécuté sur `push/PR` (main & develop). Installe les dépendances, lance les tests et exporte le rapport de couverture.
- **Deploiement Job** : Déclenche automatiquement le déploiement vers **Hugging Face Spaces** lors d'un push sur `main`.

---

## 👤 Auteur & remerciements

**Fabien** - [RandomFab](https://github.com/RandomFab)

Merci aux bibliothèques et projets open-source utilisés : FastAPI, MLflow, CatBoost, ONNX, HuggingFace Hub.
