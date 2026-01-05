<!-- Project title and badges -->
# 🚀 Pret-à-Dépenser

[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-%3E%3D3.5-orange.svg)](https://mlflow.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-%3E%3D1.2-blue.svg)](https://catboost.ai/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace--hub-purple.svg)](https://huggingface.co/)

Description
-----------
Packaging et versioning via MLflow, extraction des artifacts essentiels et publication sur le Hub Hugging Face afin de faciliter le déploiement d'une API reproductible et intégrable dans une pipeline CI/CD.

## 🎯 Objectif

Automatiser une chaîne de déploiement reproductible pour modèles CatBoost :
- enregistrer et versionner les artifacts avec MLflow,
- extraire et conserver les fichiers indispensables (`model.cb`, `MLmodel`, `input_example.json`, `requirements.txt`),
- publier les artifacts sur Hugging Face Hub pour accès et distribution,
- fournir des scripts et outils permettant d'intégrer et charger le modèle dans une API prête pour la production (CI/CD friendly).

## ✨ Fonctionnalités principales

- ✅ Enregistrement et téléchargement d'artifacts MLflow
- ✅ Export du modèle CatBoost natif (`model.cb`) et métadonnées (`MLmodel`)
- ✅ Scripts pour exporter/importer depuis/vers Hugging Face Hub
- ✅ Utilities pour reproduire l'environnement (`requirements.txt`, `conda.yaml`)
- ✅ Upload automatique du dossier d'artifacts et remplacement (overwrite) sur HF

## 📁 Structure du dépôt

```
Pret-à-Dépenser/
│
├── ⚙️ config/                        # Paramètres, chemins et logger
│   ├── config.py
│   └── logger.py
│
├── 📦 exported_model/                # Artifacts exportés (à ignorer)
│   ├── MLmodel
│   ├── model.cb
│   ├── input_example.json
│   └── requirements.txt
│
├── 📂 data/
│   ├── external/
│   ├── raw/
│   └── processed/
│
├── 🧰 scripts/                       # CLI helpers pour MLflow & HF
│   ├── download_model.py
│   ├── upload_to_hf.py
│   └── download_from_hf.py
│
├── 💻 src/
│   ├── model/
│   │   ├── hf_interaction.py
│   │   └── mlflow_interaction.py
│   ├── utils/
│   └── app/
│       └── model.py
│
├── 🧪 tests/
├── 🧾 README.md                       # Ce fichier
├── 📜 pyproject.toml
├── 📦 requirements.txt
├── 🐳 Dockerfile
└── 🐳 docker-compose.yml

```

## 🤔 Quels fichiers gardez et pourquoi

- `model.cb` — votre binaire CatBoost (chargez-le directement en production avec `CatBoost.load_model`).
- `MLmodel` — métadonnées MLflow (indique flavors et loader).
- `input_example.json` — exemple d'entrée utile pour tests et construction de schemas.
- `requirements.txt`, `conda.yaml` — reproduire l'environnement d'exécution.

## ⚙️ Variables d'environnement importantes

Définissez-les dans `.devenv` ou votre environnement CI/CD :

- `MLFLOW_TRACKING_URI` — URL du serveur MLflow (optionnel)
- `MLFLOW_MODEL_URI` — ex: `models:/BEST CATABOOST - W FEATURE IMPORTANCE@champion`
- `HUGGINGFACE_TOKEN` — token pour le Hub HF
- `HF_REPO_ID` — ex: `username/your-model-repo`

## 🧰 Installation rapide

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
pip install huggingface_hub
```

## 🚀 Commandes utiles

- Télécharger artifacts MLflow localement :

```bash
python scripts/download_model.py
```

- Télécharger depuis HF Hub :

```bash
HF_REPO_ID=username/model-repo HF_FILENAME=model.cb HUGGINGFACE_TOKEN=hf_xxx python scripts/download_from_hf.py
```

- Télécharger depuis MLflow puis uploader vers HF (écrase le contenu distant) :

```bash
MLFLOW_MODEL_URI="models:/...@champion" HF_REPO_ID=username/model-repo HUGGINGFACE_TOKEN=hf_xxx python scripts/upload_to_hf.py
```

## 🔁 Chargement du modèle dans l'API

- Via CatBoost (léger, recommandé en production) :

```python
from catboost import CatBoostClassifier
from config.config import MODEL_DIR

model = CatBoostClassifier()
model.load_model(str(MODEL_DIR / 'model.cb'))
preds = model.predict(X)
```

- Via MLflow (si preprocessing enregistré dans le modèle) :

```python
import mlflow.pyfunc
model = mlflow.pyfunc.load_model(str(MODEL_DIR))
preds = model.predict(X)
```
---

## 👤 Auteur

**Fabien** - [RandomFab](https://github.com/RandomFab)

---

**⭐ Si ce projet vous est utile, donnez-lui une étoile !**
```