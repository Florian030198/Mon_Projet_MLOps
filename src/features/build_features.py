import pandas as pd
from src.data.load_data import load_data

def build_features():
    print("🔧 Début du Feature Engineering (Catégories uniquement)...")
    
    # On récupère les données propres
    df = load_data()
    
    # On supprime l'inutile
    df_features = df.drop(columns=['id', 'name'])
    
    # On traduit le texte en chiffres (One-Hot Encoding pour les villes)
    df_features = pd.get_dummies(df_features, columns=['city'], dtype=int)
    
    return df_features

if __name__ == "__main__":
    print(build_features())