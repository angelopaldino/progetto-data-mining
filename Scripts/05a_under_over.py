import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os

# 🔹 Caricamento dati
X_train = pd.read_csv('data/splitted/X_train.csv')
X_test = pd.read_csv('data/splitted/X_test.csv')
y_train = pd.read_csv('data/splitted/y_train.csv').values.ravel()
y_test = pd.read_csv('data/splitted/y_test.csv').values.ravel()

# 🔹 Variabili per memorizzare metriche
depths = range(1, 21)
train_metrics = {'accuracy': [], 'f1': [], 'precision': [], 'recall': []}
test_metrics = {'accuracy': [], 'f1': [], 'precision': [], 'recall': []}

# 🔹 Addestramento e valutazione
for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=42)
    clf.fit(X_train, y_train)

    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    # Train metrics
    train_metrics['accuracy'].append(accuracy_score(y_train, y_pred_train))
    train_metrics['f1'].append(f1_score(y_train, y_pred_train))
    train_metrics['precision'].append(precision_score(y_train, y_pred_train))
    train_metrics['recall'].append(recall_score(y_train, y_pred_train))

    # Test metrics
    test_metrics['accuracy'].append(accuracy_score(y_test, y_pred_test))
    test_metrics['f1'].append(f1_score(y_test, y_pred_test))
    test_metrics['precision'].append(precision_score(y_test, y_pred_test))
    test_metrics['recall'].append(recall_score(y_test, y_pred_test))

# 🔹 Cartella output
output_dir = 'results/overfitting_underfitting'
os.makedirs(output_dir, exist_ok=True)

# 🔹 Funzione di plotting
def plot_metric(metric_name, train_values, test_values):
    plt.figure()
    plt.plot(depths, train_values, label=f'Training {metric_name.capitalize()}')
    plt.plot(depths, test_values, label=f'Test {metric_name.capitalize()}')
    plt.xlabel('Max Depth')
    plt.ylabel(metric_name.capitalize())
    plt.title(f'Overfitting vs Underfitting - {metric_name.capitalize()}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/{metric_name}_plot.png')
    plt.close()
# 🔹 Genera i 4 grafici
for metric in ['accuracy', 'f1', 'precision', 'recall']:
    plot_metric(metric, train_metrics[metric], test_metrics[metric])
