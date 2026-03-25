import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.data.load_data import load_data # On réutilise notre tuyau d'ingestion !

def build_features():
    print("🔧 Début du Feature Engineering...")
    
    # 1. On récupère les données propres
    df = load_data()
    
    # 2. On supprime les colonnes inutiles pour l'algorithme
    df_features = df.drop(columns=['id', 'name'])
    
    # 3. On traduit le texte en chiffres (One-Hot Encoding pour les villes)
    # L'option dtype=int permet d'avoir des 0 et des 1 au lieu de True/False
    df_features = pd.get_dummies(df_features, columns=['city'], dtype=int)
    
    # 4. On normalise les chiffres (L'âge)
    scaler = StandardScaler()
    # Le scaler va transformer les âges (28, 35, 42) en valeurs autour de zéro (-1.2, 0, 1.2)
    df_features['age'] = scaler.fit_transform(df_features[['age']])
    
    print("✅ Feature Engineering terminé !")
    return df_features

# L'interrupteur pour tester le script
if __name__ == "__main__":
    df_final = build_features()
    print("\nVoici les données prêtes pour le robot :")
    print(df_final)