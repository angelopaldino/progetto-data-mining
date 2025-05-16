import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os

X_train = pd.read_csv('data/splitted/X_train.csv')
X_test = pd.read_csv('data/splitted/X_test.csv')
y_train = pd.read_csv('data/splitted/y_train.csv').values.ravel()
y_test = pd.read_csv('data/splitted/y_test.csv').values.ravel()

criterion = 'entropy'    
max_depth = 6            

clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=42)
clf.fit(X_train, y_train)

y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

train_metrics = {
    "accuracy": accuracy_score(y_train, y_pred_train),
    "precision": precision_score(y_train, y_pred_train),
    "recall": recall_score(y_train, y_pred_train),
    "f1": f1_score(y_train, y_pred_train)
}

test_metrics = {
    "accuracy": accuracy_score(y_test, y_pred_test),
    "precision": precision_score(y_test, y_pred_test),
    "recall": recall_score(y_test, y_pred_test),
    "f1": f1_score(y_test, y_pred_test)
}

output_dir = 'results/classification_selfMade/decision_tree/overfitting_underfitting_final'
os.makedirs(output_dir, exist_ok=True)

filename = f"{criterion}_depth{max_depth}.txt"
with open(os.path.join(output_dir, filename), "w") as f:
    f.write(f"Modello: DecisionTreeClassifier\n")
    f.write(f"Criterion: {criterion}\n")
    f.write(f"Max Depth: {max_depth}\n\n")

    f.write("TRAIN METRICS:\n")
    for metric, value in train_metrics.items():
        f.write(f"{metric}: {value:.4f}\n")

    f.write("\nTEST METRICS:\n")
    for metric, value in test_metrics.items():
        f.write(f"{metric}: {value:.4f}\n")

print(f"✔ Metriche salvate in: {os.path.join(output_dir, filename)}")
plot_dir = os.path.join(output_dir, 'plots')
os.makedirs(plot_dir, exist_ok=True)

def plot_single_metric(metric_name, train_value, test_value):
    plt.figure()
    plt.bar(['Train', 'Test'], [train_value, test_value], color=['#1f77b4', '#ff7f0e'])
    plt.title(f'{metric_name.capitalize()} - Train vs Test')
    plt.ylabel(metric_name.capitalize())
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{metric_name}_comparison.png'))
    plt.close()

for metric in ['accuracy', 'precision', 'recall', 'f1']:
    plot_single_metric(metric, train_metrics[metric], test_metrics[metric])
