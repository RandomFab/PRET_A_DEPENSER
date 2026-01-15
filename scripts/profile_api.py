import cProfile
import pstats
from src.model.model_service import get_prediction, load_model_instance
from src.api.schemas import ScoringData

# Accès direct à l'exemple via la configuration de la classe (Pydantic v2)
sample = ScoringData.model_config['json_schema_extra']['example']

# Instanciation et conversion en dict pour la fonction de prédiction
request = ScoringData(**sample)
data_dict = request.model_dump()

model = load_model_instance()

if model is None:
    print("❌ Erreur : Le modèle n'a pas pu être chargé.")
else:
    profiler = cProfile.Profile()
    profiler.enable()

    # Deuxième appel pour voir l'effet du cache (get_model_signature)
    print("\n🚀 Deuxième appel (avec cache)...")
    get_prediction(model, data_dict)

    profiler.disable()

    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats(20)
