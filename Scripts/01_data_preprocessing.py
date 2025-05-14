#!/usr/bin/env python3
# Scripts/01_data_preprocessing.py

import os
import pandas as pd

def main():
    # 1. Definisci i percorsi
    raw_path = os.path.join("data", "raw", "Billionaires Statistics Dataset.csv")
    processed_dir = os.path.join("data", "processed")
    processed_path = os.path.join(processed_dir, "billionaires_clean_numeric.csv")

    # 2. Crea la cartella processed se non esiste
    os.makedirs(processed_dir, exist_ok=True)

    # 3. Carica i dati grezzi
    print(f"🔄 Caricamento dati da: {raw_path}")
    df = pd.read_csv(raw_path)

    # 4. Primo controllo: shape e colonne
    print(f"📊 Shape originale: {df.shape}")
    print(f"📑 Colonne: {df.columns.tolist()}")

    # 5. Rimuovi i duplicati basandoti sul nome 
    if "Name" in df.columns:
        before = df.shape[0]
        df = df.drop_duplicates(subset="Name")
        after = df.shape[0]
        print(f"🧹 Duplicati rimossi (Name): {before - after} righe in meno")

    # 6. Gestione dei valori mancanti
    #   - Stampa la percentuale di null per colonna
    na_pct = df.isna().mean() * 100
    print("❓ Percentuale NA per colonna:")
    print(na_pct[na_pct > 0].sort_values(ascending=False))

    #   - scarto tutte le righe senza finalWorth o Country, Nel contesto dei miliardari, finalWorth è la variabile numerica primaria: tutte le statistiche descrittive (media, mediana, istogrammi, box‐plot, correlazioni) si basano su di essa.
    # - Senza un valore valido in Country, quel record non può essere collocato in alcuna categoria geografica
    required = []
    if "finalWorth" in df.columns:
        required.append("finalWorth")
    if "country" in df.columns:
        required.append("country")

    if required:
        before = df.shape[0]
        df = df.dropna(subset=["finalWorth", "country"])
        after = df.shape[0]
        print(f"🧹 Rimosse righe senza {required}: {before - after} righe in meno")

    # 7a. Trasforma colonne, rimuovi il sufix “ B” in NetWorth e converti in float
    if "finalWorth" in df.columns and df["finalWorth"].dtype == object:
        # es. “12.3 B” → 12.3
        df["finalWorth"] = (
            df["finalWorth"]
            .str.replace(" B", "", regex=False)
            .astype(float)
        )
        print("🔄 finalWorth convertito da stringa a float")

    # 7b. Pulizia simboli da altre colonne numeriche scritte come stringhe
    for col in df.select_dtypes(include='object').columns:
        if df[col].astype(str).str.contains(r'[\\$,]', na=False).any():
            print(f"🔍 Pulizia colonna {col} da simboli $, ,")
            df[col] = df[col].replace('[\\$,]', '', regex=True).replace(',', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce') 


    # 8. Salva il DataFrame pulito
    df.to_csv(processed_path, index=False)
    print(f"✅ Dati puliti salvati in: {processed_path}")
    print(f"📊 Shape finale: {df.shape}")

if __name__ == "__main__":
    main()
