
# Progetto di Data Mining - Classificazione `selfMade` e `category`

Questo progetto è stato realizzato per l’esame del corso di Data Mining. L’obiettivo principale è classificare la variabile binaria `selfMade` e la variabile multiclasse `category` all’interno di un dataset relativo ai miliardari.

## Fasi del progetto CRISP-DM

1. Comprensione del dominio applicativo
Sono stati definiti gli obiettivi del progetto, identificando i task principali: prevedere se un individuo è self made e classificare la variabile category, che rappresenta una suddivisione in 18 categorie generali riferite al tipo di attività o area di influenza del soggetto.

2. Comprensione dei dati
È stata condotta un'analisi esplorativa del dataset per comprenderne la struttura, identificare eventuali problemi(valori mancanti, outlier, classi sbilanciate)

3. Preparazione dei dati
Sono state svolte operazioni di pulizia, encoding, scaling, oversampling delle classi minoritarie e riduzione dimensionale PCA. I dati sono stati suddivisi in training set e test set e salvati in ../data/splitted

4. Creazione dei modelli
Sono stati sviluppati vari modelli di classificazione, testando più algoritmi su entrambi i target (selfMade e category). Per ogni modello sono state testate diverse combinazioni dei parametri, ed inoltre sono stati testati anche i parametri ottenuti tramite ricerca GridSearchCV

5. Valutazione del modello e dei risultati
I modelli sono stati confrontati utilizzando metriche standard (accuracy, f1 score, precision e recall) ed è stata calcolata anche la matrice di confusione; e per i problemi binari è stata calcolata anche la curva ROC. È stata valutata l'eventuale presenza di overfitting tramite analisi train/test

6. Deployment
I modelli e i risultati ottenuti sono stati organizzati e documentati all'interno del progetto. Tutti i modelli sono stati salvati in formato .joblib, mentre i risultati e le visualizzazioni sono disponibili nella cartella ../results/.
Il progetto include un'applicazione interattiva con Streamlit che consente di utilizzare i modelli di classificazione direttamente da interfaccia web. L'utente può:
- scegliere il tipo di classificazione
- selezionare il modello desiderato
- caricare dati esistenti (.csv)
- effettuare una valutazione
- inserire manualmente un nuovo dato, personalizzando ogni attributo, per ottenere una predizione in tempo reale
L'interfaccia può essere avviata con : streamlit run Scripts/app.py


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
