import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv() # Charge les variables cachées (on le configurera plus tard)

# Va chercher le chemin de la donnée, ou utilise le chemin par défaut
DATA_PATH = os.getenv("DATA_PATH", "data/raw/customers.csv")

def load_data(path=DATA_PATH):
    # 1. On lit le fichier
    df = pd.read_csv(path)

    # 2. Nettoyage express : colonnes en minuscules et sans espaces
    df.columns = df.columns.str.lower().str.strip()

    # 3. On supprime les lignes en double
    df.drop_duplicates(inplace=True)

    # 4. Nettoyer le contenu de la colonne 'name' (minuscules et sans espaces)
    df['name'] = df['name'].str.lower().str.strip()

    return df

if __name__ == "__main__":
    df_propre = load_data()
    print(df_propre)