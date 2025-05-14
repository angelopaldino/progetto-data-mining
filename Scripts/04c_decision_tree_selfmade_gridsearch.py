import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, make_scorer
import matplotlib.pyplot as plt
import joblib

# Percorsi
X_train_path = "data/splitted/X_train.csv"
X_test_path = "data/splitted/X_test.csv"
y_train_path = "data/splitted/y_train.csv"
y_test_path = "data/splitted/y_test.csv"
results_dir = "results/classification_selfMade/decision_tree/gridsearch"
model_path = "models/decision_tree_gridsearch.joblib"
os.makedirs(results_dir, exist_ok=True)
os.makedirs("models", exist_ok=True)

# Caricamento dati
X_train = pd.read_csv(X_train_path)
X_test = pd.read_csv(X_test_path)
y_train = pd.read_csv(y_train_path).values.ravel()
y_test = pd.read_csv(y_test_path).values.ravel()

# GridSearchCV per scegliere max_depth e criterion su più metriche
param_grid = {
    'max_depth': list(range(2, 20)),
    'criterion': ['gini', 'entropy']
}

scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1'
}

clf = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring=scoring, refit='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Miglior modello trovato secondo accuracy
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Valutazione
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Salvataggio metriche
with open(os.path.join(results_dir, "metrics_gridsearch.txt"), "w") as f:
    f.write(f"Migliori parametri trovati (in base a accuracy): {grid_search.best_params_}\n\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1-Score: {f1:.4f}\n")
    f.write("\nClassification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))

# Salvataggio immagine dell'albero (solo preview max_depth=3)
plt.figure(figsize=(20, 10))
plot_tree(best_model, filled=True, max_depth=3, feature_names=X_train.columns, class_names=["False", "True"])
plt.savefig(os.path.join(results_dir, "decision_tree_grid_preview.png"))
plt.close()

# Salvataggio del modello
joblib.dump(best_model, model_path)
print(f"\n✅ Modello ottimizzato salvato in: {model_path}")
print(f"📊 Risultati salvati in: {results_dir}")
