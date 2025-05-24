
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# === 1. Percorsi e Caricamento ===
INPUT_PATH = "../data/raw/Billionaires Statistics Dataset.csv"
OUTPUT_DIR = "../data/splitted_category"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_PATH)

df['gdp_country'] = df['gdp_country'].astype(str).str.replace("$", "", regex=False)
df['gdp_country'] = df['gdp_country'].str.replace(",", "").str.strip()
df['gdp_country'] = pd.to_numeric(df['gdp_country'], errors='coerce')



# === 2. Trasformazioni logaritmiche ===
df["log_finalWorth"] = np.log1p(df["finalWorth"])
df["log_gdp_country"] = np.log1p(df["gdp_country"])

# === 3. Selezione colonne utili ===
numerical_cols = ['log_finalWorth', 'age', 'log_gdp_country', 'cpi_country',
                  'life_expectancy_country', 'gross_tertiary_education_enrollment',
                  'total_tax_rate_country', 'population_country']

categorical_cols = ['country', 'category', 'industries', 'source']

# === 4. Gestione NaN ===
print("\n Valori mancanti prima del trattamento:")
print(df[numerical_cols].isnull().sum())

for col in numerical_cols:
    median_val = df[col].median()
    df[col].fillna(median_val)

# === 5. Identificazione outlier (IQR) ===
outlier_mask = pd.DataFrame(index=df.index)
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outlier_mask[col + '_outlier'] = ~df[col].between(lower, upper)

print("\n Numero di outlier per variabile:")
print(outlier_mask.sum())


# === 6. Encoding categoriche ===
categorical_encoded = pd.get_dummies(df[categorical_cols], drop_first=True)

# === 7. Target ===
target = 'category'
y = df[target]
le = LabelEncoder()
y_encoded = le.fit_transform(y) # trasformo tutte le variabili target in 0,1,2... più facili da gestire per fgli algoritmi di classificazione
joblib.dump(le, "../models/label_encoder_category.joblib")


# === 8. Costruzione dataset finale ===
X = pd.concat([df[numerical_cols], categorical_encoded], axis=1)

# === 9. Rimozione record con target mancante ===
mask = y.notna()
X = X[mask]
y_encoded = y_encoded[mask]

# === 10. Split train/test ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# === 11. Salvataggio ===
X_train.to_csv(os.path.join(OUTPUT_DIR, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(OUTPUT_DIR, "X_test.csv"), index=False)
pd.Series(y_train, name="category_encoded").to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False)
pd.Series(y_test, name="category_encoded").to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False)

print("\n Dataset pronto per la classificazione su 'category'. Salvato in '../data/splitted_category/'")