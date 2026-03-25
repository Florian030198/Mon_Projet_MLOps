import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os
from src.features.build_features import build_features

def train_model():
    print("--- 🚀 DÉMARRAGE DE L'ENTRAÎNEMENT (SANS TRICHE) ---")
    
    # 1. On récupère nos caractéristiques
    df = build_features()
    
    # --- SIMULATION DE LA CIBLE (Target) ---
    df['churn'] = [0, 1, 0] 
    
    X = df.drop(columns=['churn'])
    y = df['churn']
    
    # 2. TRAIN-TEST SPLIT EN PREMIER ! (Comme sur le schéma)
    print("🪓 Séparation des données (Train/Test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    # 3. LE SCALING APRÈS LA SÉPARATION (La règle d'or)
    print("⚖️ Mise à l'échelle (Scaling) basée uniquement sur le Train...")
    scaler = StandardScaler()
    
    # Le secret est là : 
    # fit_transform() pour le Train (Calcule la moyenne ET applique)
    X_train_scaled = scaler.fit_transform(X_train)
    
    # transform() pour le Test (Applique JUSTE la règle du Train, sans recalculer)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Entraînement du Baseline Model
    print("🤖 Entraînement du modèle de base (Logistic Regression)...")
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    
    # 5. Évaluation
    predictions = model.predict(X_test_scaled)
    score = accuracy_score(y_test, predictions)
    print(f"📊 Précision (Accuracy) sur les données test : {score * 100}%")
    
    # 6. Sauvegarde du modèle ET du scaler
    os.makedirs('models', exist_ok=True) # Sécurité pour s'assurer que le dossier existe
    joblib.dump(model, 'models/baseline_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl') # Indispensable pour la suite !
    print("💾 Modèle ET Scaler sauvegardés avec succès dans 'models/'")

if __name__ == "__main__":
    train_model()