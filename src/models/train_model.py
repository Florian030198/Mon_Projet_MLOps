import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# --- NOUVEAUTÉ : On importe MLflow ---
import mlflow
import mlflow.sklearn
from src.features.build_features import build_features

def train_model():
    print("--- 🚀 DÉMARRAGE DE L'ENTRAÎNEMENT AVEC MLFLOW ---")
    
    # LA NOUVELLE LIGNE MAGIQUE : On force l'écriture dans le dossier mlruns !
    mlflow.set_tracking_uri("file:./mlruns")

    # 1. On donne un nom à notre carnet de bord
    mlflow.set_experiment("Churn_Prediction_Experiment")
    
    # 2. On allume l'enregistreur MLflow pour cette session (start_run)
    with mlflow.start_run():
        
        df = build_features()
        df['churn'] = [0, 1, 0] 
        X = df.drop(columns=['churn'])
        y = df['churn']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression()
        model.fit(X_train_scaled, y_train)
        
        predictions = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, predictions)
        print(f"📊 Accuracy : {acc * 100}%")
        
        # --- 3. LE TRACKING (La magie opère ici) ---
        print("📝 Enregistrement dans MLflow...")
        
        # On enregistre les paramètres (La recette)
        mlflow.log_param("model_type", "Logistic Regression")
        mlflow.log_param("test_size", 0.33)
        
        # On enregistre les métriques (La note)
        mlflow.log_metric("accuracy", acc)
        
        # On enregistre l'artefact (Le modèle lui-même !)
        mlflow.sklearn.log_model(model, "model")
        
        # (On garde aussi notre petite sauvegarde locale classique)
        os.makedirs('artifacts/models', exist_ok=True) 
        joblib.dump(scaler, 'artifacts/models/scaler.pkl') 
        
        print("✅ Expérience enregistrée avec succès dans le Model Registry !")

if __name__ == "__main__":
    train_model()