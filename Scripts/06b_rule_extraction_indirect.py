import joblib
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

# Carica il modello già addestrato
model_path = "../models/decision_tree_selfmade_final.joblib"
model = joblib.load(model_path)

X_train = pd.read_csv("../data/splitted/X_train.csv")
X_test = pd.read_csv("../data/splitted/X_test.csv")
y_train = pd.read_csv("../data/splitted/y_train.csv").values.ravel()
y_test = pd.read_csv("../data/splitted/y_test.csv").values.ravel()

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

#  Metriche
train_metrics = {
    "Accuracy": accuracy_score(y_train, y_pred_train),
    "Precision": precision_score(y_train, y_pred_train),
    "Recall": recall_score(y_train, y_pred_train),
    "F1-Score": f1_score(y_train, y_pred_train)
}
test_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred_test),
    "Precision": precision_score(y_test, y_pred_test),
    "Recall": recall_score(y_test, y_pred_test),
    "F1-Score": f1_score(y_test, y_pred_test)
}

#  Salvataggio risultati
output_dir = "../results/classification_selfMade/rule_based/indirect"
os.makedirs(output_dir, exist_ok=True)

filename = os.path.join(output_dir, f"decision_tree_final_rules.txt")
with open(filename, "w") as f:
    f.write(f"Modello: DecisionTreeClassifier (entropy, depth={5})\n\n")
    f.write("TRAIN METRICS:\n")
    for k, v in train_metrics.items():
        f.write(f"{k}: {v:.4f}\n")
    f.write("\nTEST METRICS:\n")
    for k, v in test_metrics.items():
        f.write(f"{k}: {v:.4f}\n")

    # Estrazione regole testuali
    f.write("\nREGOLE ESTRATTE:\n")
    rules = export_text(model, feature_names=list(X_train.columns))
    f.write(rules)

print(f"✔ Regole e metriche salvate in: {filename}")