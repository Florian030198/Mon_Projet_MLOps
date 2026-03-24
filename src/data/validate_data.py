import great_expectations as ge
from load_data import load_data  # 👈 Magie : on importe l'outil que tu as créé à la Phase 2 !

def validate_my_data():
    # 1. On charge tes données nettoyées
    df = load_data()
    
    # 2. On transforme le tableau Pandas en tableau Great Expectations (comme sur ton image)
    ge_df = ge.from_pandas(df)
    
    print("--- DÉBUT DE LA VALIDATION ---")
    
    # 3. TEST 1 : L'âge doit être entre 0 et 120 ans
    age_result = ge_df.expect_column_values_to_be_between(
        column="age",
        min_value=0,
        max_value=120
    )
    print(f"Test Âge logique (0-120) : {age_result['success']}")
    
    # 4. TEST 2 : La ville doit faire partie de notre liste d'opérations
    villes_autorisees = ["Paris", "Lyon", "Marseille"]
    city_result = ge_df.expect_column_values_to_be_in_set(
        column="city",
        value_set=villes_autorisees
    )
    print(f"Test Villes connues : {city_result['success']}")
    
    print("--- FIN DE LA VALIDATION ---")

# L'interrupteur de sécurité qu'on connaît bien maintenant !
if __name__ == "__main__":
    validate_my_data()