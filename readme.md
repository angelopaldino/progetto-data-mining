
# Progetto di Data Mining - Classificazione `selfMade` e `category`

Questo progetto è stato realizzato per l’esame del corso di Data Mining. L’obiettivo principale è classificare la variabile binaria `selfMade` e la variabile multiclasse `category` all’interno di un dataset relativo ai miliardari.

## Struttura del progetto

Il progetto è organizzato nelle seguenti cartelle:

- `data/`: contiene i dati grezzi, i dati puliti e i dati splittati sia nel caso di classificazione binaria che multiclasse

- `models/`: contiene i modelli salvati in formato joblib

- `Notebooks/`: contiene i notebook Python suddivisi per modello. Ogni notebook è dedicato a un classificatore (Decision Tree, KNN, Random Forest, MLP, SVM...) applicato al problema binario o multiclasse.
- `Scripts/`: contiene le funzioni ausiliarie comuni utilizzate nei notebook, come ad esempio la generazione automatica della curva ROC o il salvataggio delle metriche.
- `results/`: raccoglie i risultati ottenuti dai modelli, come metriche di performance e grafici ROC. I file sono divisi in due sottocartelle: una per la classificazione binaria (`classification_selfMade`) e una per quella multiclasse (`classification_category`).


## Come funzionano i notebook

I notebook possono essere eseguiti in modo indipendente. Caricano i dati da `data/splitted/` e applicano i modelli direttamente, senza necessità di rieseguire il preprocessing ogni volta. Per ogni classificatore vengono calcolate e salvate le metriche (accuracy, F1-score, precision, recall, confusion matrix) e, nel caso della classificazione binaria, anche la curva ROC.

Tutti i grafici ROC sono salvati come immagini `.png` e si trovano nella cartella `results`, nella sottocartella corrispondente al modello utilizzato.

## Modelli sviluppati

Sono stati implementati i seguenti modelli per il problema binario (classificazione di `selfMade`):

- Decision Tree
- KNN (con k=5)
- naive bayes
- MLP 
- ripper
- SVM 

Per il problema multiclasse (classificazione di `category`) sono stati testati:

- decision tree
- Random Forest (con macro-averaging)
- KNN
- naive bayes
- balanced random forest
- MLP
- MLP dopo PCA
- ripper 
- svm 



## Dove trovare i risultati

Tutti i risultati delle classificazioni binarie sono salvati in `results/classification_selfMade/`, mentre quelli relativi alla classificazione multiclasse si trovano in `results/classification_category/`.

Le metriche sono salvate in formato testuale, e dove previsto, sono presenti anche i grafici ROC.
