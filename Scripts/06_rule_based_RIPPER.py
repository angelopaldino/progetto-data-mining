import pandas as pd
import wittgenstein as lw
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score,confusion_matrix

X_train = pd.read_csv("data/splitted/X_train.csv")
X_test = pd.read_csv("data/splitted/X_test.csv")
y_train = pd.read_csv("data/splitted/y_train.csv").values.ravel()
y_test = pd.read_csv("data/splitted/y_test.csv").values.ravel()

k=2
model = lw.RIPPER(k=k)
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("TRAIN METRICS")
print(classification_report(y_train, y_pred_train))
train_cm = confusion_matrix(y_train, y_pred_train)
print(train_cm)
print("\nTEST METRICS")
print(classification_report(y_test, y_pred_test))
test_cm = confusion_matrix(y_test, y_pred_test)
print(test_cm)

print("\nRegole generate:")
print(model.ruleset_)

import os 

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

output_dir = "results/classification_selfMade/rule_based/direct"
os.makedirs(output_dir, exist_ok=True)

filename = os.path.join(output_dir, "ripper_rule_based.txt")
with open(filename, "w") as f:
    f.write("Modello: RIPPER (Rule-Based) con livello di pruning "+str(k)+ "\n")
    f.write("\nTRAIN METRICS:\n")
    for k, v in train_metrics.items():
        f.write(f"{k}: {v:.4f}\n")
    f.write("Confusion Matrix:\n")
    f.write(f"{train_cm}\n\n")    
    f.write("\nTEST METRICS:\n")
    for k, v in test_metrics.items():
        f.write(f"{k}: {v:.4f}\n")
    f.write("Confusion Matrix:\n")
    f.write(f"{test_cm}\n")    

    f.write("\nREGOLE APPRESE:\n")
    f.write(str(model.ruleset_))

print(f"✔ Metriche salvate in: {filename}")