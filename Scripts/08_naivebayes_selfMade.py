import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import os

# Caricamento dati
X_train = pd.read_csv("data/splitted/X_train.csv")
X_test = pd.read_csv("data/splitted/X_test.csv")
y_train = pd.read_csv("data/splitted/y_train.csv").values.ravel()
y_test = pd.read_csv("data/splitted/y_test.csv").values.ravel()

# 🧼 Pulizia: rimuove colonne costanti o tutte NaN
X_train = X_train.drop(columns=['source'], errors='ignore')
X_train = X_train.loc[:, X_train.nunique() > 1]
X_test = X_test[X_train.columns]
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_train.median())

# Standardizzazione
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modello Naive Bayes
model = GaussianNB()
model.fit(X_train_scaled, y_train)

# Predizioni
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# Metriche
train_metrics = {
    "Accuracy": accuracy_score(y_train, y_pred_train),
    "Precision": precision_score(y_train, y_pred_train),
    "Recall": recall_score(y_train, y_pred_train),
    "F1-Score": f1_score(y_train, y_pred_train),
    "Confusion Matrix": confusion_matrix(y_train, y_pred_train).tolist()
}
test_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred_test),
    "Precision": precision_score(y_test, y_pred_test),
    "Recall": recall_score(y_test, y_pred_test),
    "F1-Score": f1_score(y_test, y_pred_test),
    "Confusion Matrix": confusion_matrix(y_test, y_pred_test).tolist()
}

# Salvataggio risultati
output_dir = "results/classification_selfMade/naive_bayes"
os.makedirs(output_dir, exist_ok=True)

filename = os.path.join(output_dir, "naive_bayes_metrics.txt")
with open(filename, "w") as f:
    f.write("Modello: GaussianNB (Naive Bayes)\n\n")

    f.write("TRAIN METRICS:\n")
    for k, v in train_metrics.items():
        if k == "Confusion Matrix":
            f.write("Confusion Matrix:\n")
            f.write(f"{v}\n")
        else:
            f.write(f"{k}: {v:.4f}\n")

    f.write("\nTEST METRICS:\n")
    for k, v in test_metrics.items():
        if k == "Confusion Matrix":
            f.write("Confusion Matrix:\n")
            f.write(f"{v}\n")
        else:
            f.write(f"{k}: {v:.4f}\n")

print(f"✔ Metriche salvate in: {filename}")
