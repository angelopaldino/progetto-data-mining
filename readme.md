
# Progetto di Data Mining - Classificazione `selfMade` e `category`

Questo progetto è stato realizzato per l’esame del corso di Data Mining. L’obiettivo principale è classificare la variabile binaria `selfMade` e la variabile multiclasse `category` all’interno di un dataset relativo ai miliardari.

## Struttura del progetto

Il progetto è organizzato nelle seguenti cartelle:

- `data/`: contiene i dati grezzi, i dati puliti e i dati splittati sia nel caso di classificazione binaria che multiclasse

- `models/`: contiene i modelli salvati in formato joblib

- `Notebooks/`: contiene i notebook Python suddivisi per modello. Ogni notebook è dedicato a un classificatore (Decision Tree, KNN, Random Forest, MLP, SVM...) applicato al problema binario o multiclasse. I notebook non indicizzati, sono notebook di confronto che sono stati implementati per ultimo dopo avere ottenuto le metriche dai vari modelli al fine di fare un confronto tra i diversi modelli.
- `Scripts/`: contiene le funzioni ausiliarie comuni utilizzate nei notebook.
- `results/`: raccoglie i risultati ottenuti dai modelli, come metriche di performance e grafici ROC. I file sono divisi in due sottocartelle: una per la classificazione binaria (`classification_selfMade`) e una per quella multiclasse (`classification_category`).


## Come funzionano i notebook

I notebook possono essere eseguiti in modo indipendente. Caricano i dati da `data/splitted/` e applicano i modelli direttamente, senza necessità di rieseguire il preprocessing ogni volta. Per ogni classificatore vengono calcolate e salvate le metriche (accuracy, F1-score, precision, recall, confusion matrix) e, nel caso della classificazione binaria, anche la curva ROC.

Tutti i grafici ROC sono salvati come immagini `.png` e si trovano nella cartella `results`, nella sottocartella corrispondente al modello utilizzato.

## Modelli sviluppati

Sono stati implementati i seguenti modelli per il problema binario (classificazione di `selfMade`):

- Decision Tree
- KNN 
- naive bayes
- MLP 
- ripper
- SVM 

Per il problema multiclasse (classificazione di `category`) sono stati testati:

- Decision Tree
- KNN 
- naive bayes
- MLP 
- ensemble MLP
- ripper
- SVM 
# Confronto modelli
Sono stati testati diversi modelli e per ogni modello sono state utilizzate diverse combinazioni dei parametri ottenuti tramite la ricerca GridSearchCV. I risultati che ogni modello ci ha fornito sono stati salvati opportunamente per effettuare un confronto delle prestazioni

## Dove trovare i risultati

Tutti i risultati delle classificazioni binarie sono salvati in `results/classification_selfMade/`, mentre quelli relativi alla classificazione multiclasse si trovano in `results/classification_category/`.

Le metriche sono salvate in formato testuale, e dove previsto, sono presenti anche i grafici ROC.
