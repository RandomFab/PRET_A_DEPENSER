import streamlit as st
import requests
import sys
from pathlib import Path
from utils import *

# Ajouter la racine du projet au chemin de recherche de modules
root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from config.config import BASE_DIR

# --- 0. CONFIG PAGE (DOIT ÊTRE LE PREMIER) ---
# Utilisation du chemin correct pour l'icône
logo_path = BASE_DIR / "logo" / "PICTO.png"

st.set_page_config(
    page_title="Prêt à dépenser - Scoring",
    page_icon=str(logo_path) if logo_path.exists() else "💰",
    layout="wide"
)

# Limiter la largeur de l'app et styliser simplement les boutons
st.markdown(
    """
    <style>
    /* Conteneur principal : largeur limitée et centré */
    .block-container {
        max-width: 50vw !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    @media (max-width: 900px) {
        .block-container { max-width: 95vw !important; }
    }

    /* Boutons simples : vert, coins carrés, pas d'effet au survol */
    .stButton>button,
    .stDownloadButton>button,
    form button,
    div[data-testid="stForm"] button {
        background-color: #1fad91 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 0 !important;
        padding: 0.5rem 1rem !important;
        box-shadow: none !important;
        font-size: 1rem !important;
        min-height: 40px !important;
    }

    /* Pas d'effet au survol : garder la même couleur */
    .stButton>button:hover,
    .stDownloadButton>button:hover,
    form button:hover {
        background-color: #1fad91 !important;
        box-shadow: none !important;
        outline: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if logo_path.exists():
    st.logo(str(logo_path))
else:
    st.sidebar.warning("⚠️ Logo introuvable")

columns_header= [9,1,1,1]

# --- 1. FONCTIONS DE RÉCUPÉRATION (SANS CACHE POUR LE THREAD) ---

executor = get_executor()

# On lance les futurs en arrière-plan
f_api = executor.submit(fetch_service_status, "http://localhost:8000/api_health")
f_router = executor.submit(fetch_service_status, "http://localhost:8000/router_health")
f_model = executor.submit(fetch_service_status, "http://localhost:8000/model_status")

# Signature : on essaie de l'avoir proprement
fields = get_model_signature_cached()


# --- HEADER SECTION ---


col1, col2, col3, col4= st.columns(columns_header, vertical_alignment="center")

with col1:
    st.title("Prêt à dépenser")

with col2:
    api_status_ph = st.empty()
    api_status_ph.markdown("<p style='text-align: center; margin: 0;'><span style='color: gray; font-size: 15px;'>●</span> API</p>", unsafe_allow_html=True)

with col3:
    router_status_ph = st.empty()
    router_status_ph.markdown("<p style='text-align: center; margin: 0;'><span style='color: gray; font-size: 15px;'>●</span> Router</p>", unsafe_allow_html=True)

with col4:
    model_status_ph = st.empty()
    model_status_ph.markdown("<p style='text-align: center; margin: 0;'><span style='color: gray; font-size: 15px;'>●</span> Model</p>", unsafe_allow_html=True)


col_description,_,_,_ = st.columns(columns_header, vertical_alignment="center")

with col_description:
    st.markdown("""
            > Ce projet vise a mettre à disposition un modèle de scoring de prêt.
            > En renseignant différentes information d'un client, le modèle vous oriente sur l'acceptation ou non de son profile pour un prêt
                """)

# --- AJOUT DU CSS POUR LES TABS (à mettre en haut du fichier ou avant le container) ---
# st.markdown("""
#     <style>
#     /* Force la liste d'onglets à prendre toute la largeur et centre les boutons */
#     div[data-baseweb="tab-list"] {
#         width: 100% !important;
#         justify-content: center !important;
#     }
#     /* Force chaque bouton d'onglet à s'étendre (flex-grow) */
#     div[data-baseweb="tab-list"] button {
#         flex: 1 !important;
#     }
#     </style>
# """, unsafe_allow_html=True)
st.markdown("""
    <style>
    /* 1. On cache la barre rouge par défaut */
    div[data-baseweb="tab-highlight"] {
        display: none !important;
    }
    
    /* 2. On stylise le conteneur des onglets */
    div[data-baseweb="tab-list"] {
        width: 100% !important;
        justify-content: center !important;
        gap: 10px !important; /* Espace entre les boîtes */
    }
    
    /* 3. On transforme chaque onglet en "Boîte" */
    div[data-baseweb="tab-list"] button {
        flex: 1 !important;
        border: 1px solid #0e1117 !important; /* Bordure légère */
        border-radius: 8px !important;       /* Coins arrondis */
        padding: 10px !important;
        background-color: #0e1117 !important;
        transition: all 0.3s ease !important;
    }
    
    /* 4. Style de l'onglet ACTIF (la boîte sélectionnée) */
    div[data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #262730 !important; /* Rouge Streamlit ou ta couleur */
        color: white !important;             /* Texte en blanc */
        border-color: #262730 !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
    }

    /* 5. Effet au survol (hover) */
    div[data-baseweb="tab-list"] button:hover {
        border-color: #262730 !important;
        background-color: #262730 !important;
        color: white !important;  /* Texte en blanc au survol */
    }

          
    </style>
""", unsafe_allow_html=True)
with st.container(border=True,horizontal=True ,horizontal_alignment='center'):
    sco_indiv,sco_csv = st.tabs(['Scoring Manuel','Scoring CSV'])

with sco_indiv:
    with st.form("scoring"):
        st.markdown("## Scoring Manuel")
        st.markdown("<p style='color:#757575'>Renseignez les champs pour obtenir un score</p>",unsafe_allow_html=True)

        if fields:
            # Grille de 3 colonnes par défaut
            cols_per_row = 3
            for i in range(0, len(fields), cols_per_row):
                row_fields = fields[i:i+cols_per_row]
                cols = st.columns(cols_per_row)
                for j, field_info in enumerate(row_fields):
                    # Extraction des infos (supporte dict pour MLflow ou str pour simple liste)
                    if isinstance(field_info, dict):
                        field_name = field_info.get("name")
                        field_type = field_info.get("type", "double")
                        field_desc = field_info.get("description")
                    else:
                        field_name = field_info
                        field_type = "double"
                        field_desc = None
                    
                    # On concatène le type technique et la description métier
                    help_parts = []
                    if field_desc:
                        help_parts.append(f"**Description**: {field_desc}")
                    help_parts.append(f"*Type technique: {field_type}*")
                    help_text = "\n\n".join(help_parts)
                    
                    with cols[j]:
                        if field_type in ["integer", "long"]:
                            st.number_input(field_name, key=f"in_{field_name}", value=0, step=1, help=help_text)
                        elif field_type == "boolean":
                            st.selectbox(field_name, options=[True, False], key=f"in_{field_name}", help=help_text)
                        else:
                            st.number_input(field_name, key=f"in_{field_name}", value=0.0, help=help_text)
        else:
            st.error("Impossible de récupérer la signature du modèle (Vérifiez que l'API est lancée)")
        
        submitted_indiv = st.form_submit_button("Calculer mon score",use_container_width=True)

    if submitted_indiv:
        try:
            # Construction correcte du dictionnaire de paramètres
            params = {field.get('name'): st.session_state.get(f"in_{field.get('name')}") for field in fields}

            # Envoi synchrone à l'API (on peut rendre asynchrone si besoin)
            response = requests.post("http://localhost:8000/individual_score", json=params, timeout=10)
            if response.status_code == 200:
                result = response.json()
                if result.get('decision') == "Refusé" :
                    st.error(f"Décision : Ce profile est {result.get('decision').lower()}")
                else : 
                    st.success(f"Décision : Ce profile est {result.get('decision').lower()}")
                st.markdown(f"""**Informations complémentaires** :  
                        - Probabilité : {result.get('score')} ;  
                        - Threshold : {result.get('threshold')} ;  
                        - Prédiction : {result.get('prediction')} / {result.get('decision')}""")
            else:
                try:
                    error_detail = response.json().get("detail", [])
                    if response.status_code == 422 and isinstance(error_detail, list):
                        msg = "\n".join([f"- **{err.get('loc', ['?'])[-1]}** : {err.get('msg')}" for err in error_detail])
                        st.error(f"Données invalides :\n{msg}")
                    else:
                        st.error(f"Erreur API : {response.status_code} - {response.text}")
                except Exception:
                    st.error(f"Erreur API : {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Erreur lors de l'envoi : {e}")


with sco_csv:
    template_csv = BASE_DIR / 'src' / 'app' / "static" / "scoring_template_app.csv"
    if template_csv.exists():
        data_bytes = template_csv.read_bytes()
        template_button = st.download_button(
            label='Téléchargez le template',
            data=data_bytes,
            file_name='scoring_template.csv',
            mime='text/csv',
            icon=':material/download:'
        )
    else:
        st.warning(f"Template CSV introuvable : {template_csv}")

    uploader = st.file_uploader('Importez votre fichier "profile"...',accept_multiple_files=False, type='csv')

    submitted_multi = st.button("Calculer les scores", use_container_width=True)

    if submitted_multi:
        try:
            if uploader is None:
                st.error("Aucun fichier CSV fourni. Importez un fichier avant de lancer le calcul.")
            else:
                payload = csv_to_json(uploader)
                response = requests.post("http://localhost:8000/multiple_score", json=payload, timeout=30)
                if response.status_code == 200:
                    try:
                        result = response.json()
                        # Affiche le tableau des réponses renvoyées par l'endpoint
                        import pandas as pd

                        if isinstance(result, list):
                            df = pd.DataFrame(result)
                            st.markdown("**Résultats (tableau)**")
                            st.dataframe(df)
                        else:
                            st.write(result)
                    except Exception as exc:
                        st.error(f"Impossible de parser la réponse JSON : {exc}")
                else:
                    try:
                        error_detail = response.json().get("detail", [])
                        if response.status_code == 422 and isinstance(error_detail, list):
                            # Pour le batch, on affiche les premières erreurs pour ne pas flooder
                            msg = "\n".join([f"- Ligne {err.get('loc', ['?'])[1] if len(err.get('loc', []))>1 else '?'} | **{err.get('loc', ['?'])[-1]}** : {err.get('msg')}" for err in error_detail[:10]])
                            if len(error_detail) > 10: msg += "\n- ..."
                            st.error(f"Erreur de validation (batch) :\n{msg}")
                        else:
                            st.error(f"Erreur API : {response.status_code} - {response.text}")
                    except Exception:
                        st.error(f"Erreur API : {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Erreur lors de l'envoi : {e}")

with st.expander("📖 Information du modèle"):

    model_infos = model_info()

    # 2x2 grid : deux lignes, deux colonnes
    Model_type, nb_feature = st.columns(2)
    with Model_type:
        st.markdown("#### Type de modèle")
        st.write(model_infos.get('model_type', '—'))

    with nb_feature:
        st.markdown("#### Nombre de features")
        st.write(model_infos.get('nb_feature', '—'))

    Thershold, update = st.columns(2)
    with Thershold:
        st.markdown("#### Seuil de décision")
        st.write(model_infos.get('best_threshold', '—'))

    with update:
        st.markdown("#### Dernière MàJ")
        st.write(model_infos.get('created_on', '—'))

    st.divider()
    
    st.button('Reload modèle', use_container_width=True, on_click=reload_model)

with st.expander("✍🏻 Signature du modèle"):
    df = model_signature()
    st.dataframe(df)


# --- 4. MISE À JOUR DES STATUTS (EN FIN DE SCRIPT) ---
# Ici on récupère les résultats des futures lancées au début
try:
    # On récupère les couleurs (attente si pas encore fini)
    c_api = f_api.result()
    c_router = f_router.result()
    c_model = f_model.result()

    # On utilise le cache Streamlit pour l'affichage HTML
    api_status_ph.markdown(f"<p style='text-align: center; margin: 0;'>{get_cached_status_html('api', c_api)} API</p>", unsafe_allow_html=True)
    router_status_ph.markdown(f"<p style='text-align: center; margin: 0;'>{get_cached_status_html('router', c_router)} Router</p>", unsafe_allow_html=True)
    model_status_ph.markdown(f"<p style='text-align: center; margin: 0;'>{get_cached_status_html('model', c_model)} Model</p>", unsafe_allow_html=True)
except Exception as e:
    # En cas d'erreur de thread, on laisse les placeholders gris ou on log
    pass