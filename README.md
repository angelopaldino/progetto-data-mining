# Progetto di Data Mining — Analisi dei Miliardari

Questo progetto analizza un dataset contenente statistiche sui miliardari nel mondo, utilizzando tecniche di data mining e analisi esplorativa. Il dataset è stato preso da Kaggle: [Billionaires Statistics Dataset](https://www.kaggle.com/datasets/nelgiriyewithana/billionaires-statistics-dataset).

## Struttura del progetto

```
DataMiningProject/
├── data/
├── scripts/
├── results/
├── notebooks/
├── .venv/
└── README.md
```

## Obiettivi del progetto

* Caricamento e pulizia dei dati
* Analisi esplorativa e generazione di grafici descrittivi
* Esportazione di grafici e report testuali in una cartella dedicata (`results/`)
* Vengono testati diversi modelli tra cui: decision tree, svm, knn calcolando le prestazioni(accuracy, f1 score, precision recall) e matrice di confusione di ciascun modello al variare dei parametri, anche ottenuti tramite ricerca GridSearchCV. Infine viene fatto un confronto tra i modelli.
* Questo progetto esplora problemi di classificazione binaria e multiclasse rispettivamente usando selfMade come target per la classificazione binaria e category per la classificazione multiclasse

## Come eseguire il progetto

### 1. Clona il repository (se applicabile)

```bash
git clone https://github.com/tuo-utente/DataMiningProject.git
cd DataMiningProject
```

### 2. Crea e attiva un ambiente virtuale

```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 3. Installa le dipendenze

```bash
pip install pandas numpy matplotlib seaborn
```


## Output generato

Tutti i risultati sono salvati in `results/`:

*  Grafici: distribuzioni di età e patrimonio, matrice di correlazione
*  Riepilogo testuale: info sul dataset e statistiche descrittive
*  Confronto tra modelli con le metriche ottenute

## Dataset

* Fonte: [Kaggle - Billionaires Dataset](https://www.kaggle.com/datasets/nelgiriyewithana/billionaires-statistics-dataset)
* Dimensione: \~1MB
* Colonne principali: `name`, `age`, `finalWorth`, `country`, `industry`, `gender`, `source`, `state`, ecc.


## Deployment
Questo progetto include un'applicazione interattiva sviluppata con Streamlit che permette di testare e utilizzare i modelli di classificazione.
Funzionalità principali:
  - Selezione del tipo di classificazione: binaria o multiclasse
  - Scelta del modello da utilizzare
  - Valutazione su dati etichettati caricati manualmente (.csv)
  - Inserimento di nuovi dati personalizzati da parte dell'utente per ottenere una predizione
Come avviare l'interfaccia:
  - Assicurati di essere all'interno dell'ambiente virtuale .venv e poi esegui: streamlit run Scripts/app.py

## Autore

Angelo Paldino — Università della calabria Unical, facoltà ingegneria informatica indirizzo intelligenza artificiale e machine learning, corso di Data Mining

---

Per qualsiasi dubbio o proposta di miglioramento, apri una issue o contattami direttamente.

