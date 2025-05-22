import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import joblib
from sklearn.impute import SimpleImputer


# Caricamento dati
X_train = pd.read_csv("data/splitted/X_train.csv")
X_test = pd.read_csv("data/splitted/X_test.csv")
y_train = pd.read_csv("data/splitted/y_train.csv").values.ravel()
y_test = pd.read_csv("data/splitted/y_test.csv").values.ravel()
model_path = "models/ANN_selfMade.joblib"


# Pulizia dei dati
X_train = X_train.drop(columns=['source'], errors='ignore')
X_train = X_train.loc[:, X_train.nunique() > 1]
X_test = X_test[X_train.columns]

# Imputazione
#sostituisco i valori nan con la media

imputer = SimpleImputer(strategy="mean")
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Standardizzazione
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

model = MLPClassifier(
    activation='relu',
    alpha=0.01,
    hidden_layer_sizes=(100, 50),
    learning_rate='constant',
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=10,
    max_iter=300,
    random_state=42
)

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
output_dir = "../results/classification_selfMade/ann"
os.makedirs(output_dir, exist_ok=True)

filename = os.path.join(output_dir, "ann_metrics_gridsearch.txt")
with open(filename, "w") as f:
    f.write("Modello: MLPClassifier (2 hidden layer da 100, 50 neuroni, early stopping)\n\n")

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
joblib.dump(model, model_path)