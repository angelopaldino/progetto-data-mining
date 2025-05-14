import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import joblib

# Percorsi
X_train_path = "data/splitted/X_train.csv"
X_test_path = "data/splitted/X_test.csv"
y_train_path = "data/splitted/y_train.csv"
y_test_path = "data/splitted/y_test.csv"
results_dir = "results/classification_selfMade/decision_tree/entropy"
model_path = "models/decision_tree_selfmade_entropy.joblib"
os.makedirs(results_dir, exist_ok=True)
os.makedirs("models", exist_ok=True)

# Caricamento dati
X_train = pd.read_csv(X_train_path)
X_test = pd.read_csv(X_test_path)
y_train = pd.read_csv(y_train_path).values.ravel()
y_test = pd.read_csv(y_test_path).values.ravel()

# Addestramento del modello
model = DecisionTreeClassifier(criterion="entropy", random_state=42)
model.fit(X_train, y_train)

# Valutazione
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Salvataggio metriche
with open(os.path.join(results_dir, "metrics_entropy.txt"), "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1-Score: {f1:.4f}\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm))

# Visualizzazione e salvataggio dell'albero
plt.figure(figsize=(20, 10))
plot_tree(model, filled=True, max_depth=3, feature_names=X_train.columns, class_names=["False", "True"])
plt.savefig(os.path.join(results_dir, "decision_tree_preview_entropy.png"))
plt.close()

# Salvataggio del modello addestrato
joblib.dump(model, model_path)
print(f"\n✅ Modello salvato in: {model_path}")
print(f"📊 Risultati salvati in: {results_dir}")
