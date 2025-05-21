import pandas as pd
from sklearn.model_selection import train_test_split
import os

import numpy as np





# Percorsi
INPUT_PATH = "data/processed/billionaires_clean_numeric.csv"
OUTPUT_DIR = "data/splitted"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Caricamento dati
df = pd.read_csv(INPUT_PATH)

df["log_finalWorth"] = np.log1p(df["finalWorth"])
df["log_gdp_country"] = np.log1p(df["gdp_country"])

# Selezione feature numeriche e categoriche
numerical_cols = ['log_finalWorth', 'age', 'log_gdp_country', 'cpi_country',
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

# Rimozione dei record con target mancante 
mask = y.notna()
X = X[mask]
y = y[mask]

# Split train/test con stratificazione
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Salvataggio dei dataset
X_train.to_csv(os.path.join(OUTPUT_DIR, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(OUTPUT_DIR, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False)

print("\n✅ Dataset (completo) diviso e salvato in 'data/splitted/'")
