<!-- Project title and badges -->
# 🚀 Pret-à-Dépenser

[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128+-green.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-%3E%3D3.8.1-orange.svg)](https://mlflow.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-%3E%3D1.2.8-blue.svg)](https://catboost.ai/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace--hub-%3E%3D1.2.3-purple.svg)](https://huggingface.co/)

**Packaging et déploiement d'un modèle CatBoost avec MLflow et Hugging Face Hub.**

---

## 🎯 Objectif

Automatiser une chaîne reproductible pour déployer un modèle CatBoost :
- Enregistrement/versioning via MLflow
- Export des artifacts essentiels (`model.cb`, `MLmodel`, `input_example.json`)
- Publication et téléchargement depuis Hugging Face Hub
- Fournir une API HTTP via FastAPI pour la prédiction et la gestion du modèle

---

## ✨ Fonctionnalités

- ✅ API FastAPI minimale pour health, inspection et scoring
- ✅ Chargement automatique du modèle au démarrage (injection dans `app.state`)
- ✅ Endpoints pour signature, info, statut et reload depuis HF
- ✅ Prédiction individuelle et batch avec validation Pydantic
- ✅ Scripts d'upload/download vers/depuis Hugging Face Hub

---

## 📡 Endpoints exposés

L'application FastAPI se trouve dans `src/api/main.py` et expose les routes suivantes (sans préfixe). Le frontend Streamlit se trouve dans `src/app/main.py` et communique avec l'API pour afficher l'interface utilisateur :

- `GET /` → redirection vers la documentation interactive `/docs`.
- `GET /api_health` → état de santé global de l'API.

Routes du routeur (`src/api/routes.py`):
- `GET /router_health` → health du router.
- `GET /model_status` → état du fichier modèle sur disque (`model.cb`).
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
├── 📂 config/               # Configuration (chemins, logger, etc.)
│   ├── config.py
│   └── logger.py
├── 📂 data/                 # Données (raw, processed)
│   └── processed/
│       └── scoring_template_app.csv
├── 📂 scripts/              # Utilitaires HF (upload/download)
│   ├── download_model_from_hf.py
│   └── upload_model_to_hf.py
├── 📂 src/                  # Code source
│   ├── 🎨 api/              # Backend FastAPI (Modèle, Routes, Schemas)
│   │   ├── main.py
│   │   ├── model.py
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── 🧠 app/              # Frontend Streamlit
│   │   ├── main.py
│   │   └── utils.py
│   └── ⚙️ model/            # Logique métier & Hubs (MLflow, HF)
│       ├── hf_interaction.py
│       ├── mlflow_interaction.py
│       └── model_service.py
├── 🧪 tests/                # Tests unitaires et fonctionnels
├── 🐳 Dockerfile            # Packaging Docker
├── 🐙 docker-compose.yml    # Orchestration locale
└── 🚀 start.sh             # Script de démarrage dual (API + App)
```

**Architecture (flow principal)**

```mermaid
graph TB
	A[Start API] --> B[Load model via `load_model_instance`]
	B --> C{Model in app.state}
	C -->|Yes| D[Ready for predictions]
	C -->|No| E[API runs but returns 503 on scoring]
	F[Reload request POST /reload_model] --> G[download_model_from_hf]
	G --> H[Store file in `MODEL_DIR`]
	H --> I[load_model_instance and update `app.state`]
```
--- 

```mermaid
sequenceDiagram
	participant Client
	participant API
	participant ModelService
	participant HF

	Client->>API: POST /individual_score — JSON
	API->>ModelService: validate & reorder inputs
	ModelService->>ModelService: model.predict_proba
	ModelService-->>API: score, decision
	API-->>Client: JSON response

	Client->>API: POST /reload_model
	API->>HF: hf_hub_download repo_id, filename
	HF-->>API: local file path
	API->>ModelService: load_model_instance
	API-->>Client: reload confirmation
```

---

## ⚙️ Variables d'environnement (importantes)

- `HF_REPO_ID` — identifiant du repo HF (ex: `username/model-repo`) requis pour `POST /reload_model`.
- `HUGGINGFACE_TOKEN` — token HF (ou `HF_TOKEN`) pour accéder au repo privé.
- `HF_FILENAME` — nom du fichier dans le repo HF (défaut `model.cb`).
- `MLFLOW_TRACKING_URI` — (optionnel) point vers le serveur MLflow.

Ces variables peuvent être mises dans `.devenv` (utilisé par le projet) ou exportées dans votre CI.

---

## 🚀 Installation & Déploiement Local

### 🐍 Installation Développement (venv)
Préréquis : **Python 3.13+** et **uv** (recommandé).

1.  **Cloner le dépôt** :
    ```bash
    git clone https://github.com/RandomFab/PRET_A_DEPENSER.git
    cd PRET_A_DEPENSER
    ```
2.  **Installer les dépendances** :
    ```bash
    uv sync --frozen
    ```
3.  **Lancer séparément (Dev)** :
    - **API** : `uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload`
    - **App** : `uv run streamlit run src/app/main.py --server.port 7860`

---

## 🐳 Déploiement Local (Docker)

La méthode la plus simple pour reproduire l'environnement de production.

1.  **Configuration** : Créez un fichier `.devenv` avec vos tokens si nécessaire.
2.  **Lancement** :
    ```bash
    docker compose up --build -d
    ```
3.  **Accès** :
    - **Streamlit (UI)** : [http://localhost:7860](http://localhost:7860)
    - **FastAPI (Docs)** : [http://localhost:8000/docs](http://localhost:8000/docs)

Le conteneur utilise le script `start.sh` pour orchestrer le démarrage de l'API, attendre son initialisation, puis lancer l'interface Streamlit.  
HF ne proposant qu'un port (7870), c'est la méthode la plus simple pour déployer le couple API + APP.  
Un multi-conteners serait plus approprié sur un VPS par exemple.

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

Merci aux bibliothèques et projets open-source utilisés : FastAPI, MLflow, CatBoost, HuggingFace Hub.
