from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix



X_train = pd.read_csv("data/splitted/X_train.csv")
X_test = pd.read_csv("data/splitted/X_test.csv")
y_train = pd.read_csv("data/splitted/y_train.csv").values.ravel()
y_test = pd.read_csv("data/splitted/y_test.csv").values.ravel()



model = DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=42)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("TEST METRICS:")

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("Confusion Matrix:")
print(cm)

y_pred_train = model.predict(X_train)

print("TRAIN METRICS:")
print("Accuracy:", accuracy_score(y_train, y_pred_train))
print("Precision:", precision_score(y_train, y_pred_train))
print("Recall:", recall_score(y_train, y_pred_train))
print("F1-Score:", f1_score(y_train, y_pred_train))
print("Confusion Matrix:")
print(confusion_matrix(y_train, y_pred_train))


plt.figure(figsize=(20, 10))
plot_tree(model, filled=True, max_depth=3, feature_names=X_train.columns, class_names=["False", "True"])
plt.savefig("results/classification_selfMade/decision_tree/final_metrics/preview_gini.png")
plt.close()

# Cartella di output
output_dir = "../results/classification_selfMade/decision_tree/final_metrics"
os.makedirs(output_dir, exist_ok=True)

def save_metrics_to_file(model_name, criterion, max_depth, 
                         train_metrics, test_metrics, 
                         train_cm, test_cm):
    
    # Crea nome file in base ai parametri del modello
    filename = f"{model_name}_{criterion}_depth{max_depth}.txt"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        f.write(f"Modello: {model_name}\n")
        f.write(f"Criterion: {criterion}\n")
        f.write(f"Max Depth: {max_depth}\n\n")

        f.write("TRAIN METRICS:\n")
        for k, v in train_metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{train_cm}\n\n")

        f.write("TEST METRICS:\n")
        for k, v in test_metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{test_cm}\n")

    print(f"✔ Output salvato: {filepath}")

# Metriche
train_metrics = {
    "Accuracy": accuracy_score(y_train, y_pred_train),
    "Precision": precision_score(y_train, y_pred_train),
    "Recall": recall_score(y_train, y_pred_train),
    "F1-Score": f1_score(y_train, y_pred_train)
}
test_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1-Score": f1_score(y_test, y_pred)
}

# Matrici di confusione
train_cm = confusion_matrix(y_train, y_pred_train)
test_cm = confusion_matrix(y_test, y_pred)

save_metrics_to_file("DecisionTree", "entropy", 6, train_metrics, test_metrics, train_cm, test_cm)

