import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Impostazioni grafiche
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("pastel")

# Percorsi
DATA_PATH = "data/processed/billionaires_clean.csv"
RESULTS_PATH = "results/eda"
os.makedirs(RESULTS_PATH, exist_ok=True)

# Caricamento dataset
df = pd.read_csv(DATA_PATH)

# Analisi iniziale
with open(os.path.join(RESULTS_PATH, "summary.txt"), "w", encoding="utf-8") as f:
    f.write("🔍 SHAPE DEL DATASET:\n")
    f.write(str(df.shape) + "\n\n")

    f.write("🔍 COLONNE:\n")
    f.write(str(df.columns.tolist()) + "\n\n")

    f.write("🔍 INFORMAZIONI GENERALI:\n")
    df.info(buf=f)
    f.write("\n\n🔍 STATISTICHE DESCRITTIVE:\n")
    f.write(str(df.describe(include='all')))

# Distribuzione di alcune variabili chiave
plt.figure(figsize=(10, 6))
sns.histplot(df['age'], kde=True, bins=30)
plt.title('Distribuzione dell\'et\u00e0 dei miliardari')
plt.xlabel('Et\u00e0')
plt.ylabel('Frequenza')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, "age_distribution.png"))
plt.close()

plt.figure(figsize=(10, 6))
sns.histplot(df['finalWorth'], kde=True, bins=30)
plt.title('Distribuzione del patrimonio (in milioni)')
plt.xlabel('Patrimonio (milioni USD)')
plt.ylabel('Frequenza')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, "finalworth_distribution.png"))
plt.close()

# Correlazioni numeriche
plt.figure(figsize=(12, 8))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matrice di correlazione")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, "correlation_matrix.png"))
plt.close()

print("\n✅ Report EDA generato con successo. Risultati salvati in ../results/eda")
