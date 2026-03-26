from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. On initialise l'API
app = FastAPI(title="API de Prédiction de Churn", description="L'Inference Pipeline de notre projet MLOps")
# 2. On charge notre cerveau artificiel (Model Registry) et notre règle mathématique (Scaler)
# (Vérifie bien que les chemins correspondent à là où tu as sauvegardé tes fichiers .pkl !)
try:
    model = joblib.load('models/baseline_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    print("✅ Modèle et Scaler chargés avec succès depuis le stockage.")
except Exception as e:
    print(f"❌ Erreur lors du chargement : {e}")
# 3. On définit le format exact de la question que le client doit poser (Le "Schema")
class CustomerData(BaseModel):
    age: float
    city_Lyon: int
    city_Marseille: int
    city_Paris: int
# 4. On crée la route (Le guichet /predict)
@app.post("/predict")
def predict_churn(data: CustomerData):
    # Étape A : On transforme la question du client en tableau Pandas
    df = pd.DataFrame([data.model_dump()])
    # Étape B : On applique EXACTEMENT la même échelle mathématique qu'à l'entraînement
    df_scaled = scaler.transform(df)
    # Étape C : Le modèle fait sa prédiction
    prediction = model.predict(df_scaled)
    # Étape D : On renvoie la réponse au format lisible
    resultat = "Va se désabonner" if prediction[0] == 1 else "Va rester fidèle"
    return {
        "prediction_brute": int(prediction[0]),
        "message": resultat
    }
