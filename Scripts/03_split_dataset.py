import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Percorsi
INPUT_PATH = "data/processed/billionaires_clean_numeric.csv"
OUTPUT_DIR = "data/splitted"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Caricamento dati
df = pd.read_csv(INPUT_PATH)

# Selezione feature numeriche e categoriche
numerical_cols = ['finalWorth', 'age', 'gdp_country', 'cpi_country',
                  'life_expectancy_country', 'gross_tertiary_education_enrollment',
                  'total_tax_rate_country', 'population_country']

categorical_cols = ['country', 'category', 'industries', 'source']

# Target
target = 'selfMade'

# Encoding variabili categoriche
categorical_encoded = pd.get_dummies(df[categorical_cols], drop_first=True)

# Costruzione del dataset finale
X = pd.concat([df[numerical_cols], categorical_encoded], axis=1)
y = df[target]

# Split train/test con stratificazione
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Salvataggio dei dataset
X_train.to_csv(os.path.join(OUTPUT_DIR, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(OUTPUT_DIR, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False)

print("\n✅ Dataset (completo) diviso e salvato in 'data/splitted/'")
