import os
import joblib
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

def evaluate_and_save_model(
    model,
    model_name: str,
    y_train, y_pred_train,
    y_test, y_pred_test,
    output_dir: str,
    model_path: str,
    extra_params: dict = None
):
    """
    Calcola metriche su train e test set, salva il modello e scrive i risultati in un file.
    
    Parameters:
    - model: modello sklearn addestrato
    - model_name: nome descrittivo del modello (es. "Bagging KNN")
    - y_train, y_pred_train: valori reali e predetti sul training set
    - y_test, y_pred_test: valori reali e predetti sul test set
    - output_dir: directory dove salvare il file delle metriche
    - model_path: percorso dove salvare il modello con joblib
    - extra_params: dizionario di parametri extra (es. {"k": 7, "maxf": 0.7})
    """

    # Crea la cartella di output se non esiste
    os.makedirs(output_dir, exist_ok=True)

    # Salva il modello
    joblib.dump(model, model_path)

    # Calcola metriche
    metrics = {
        "train": {
            "accuracy": accuracy_score(y_train, y_pred_train),
            "f1": f1_score(y_train, y_pred_train),
            "precision": precision_score(y_train, y_pred_train),
            "recall": recall_score(y_train, y_pred_train),
            "confusion": confusion_matrix(y_train, y_pred_train)
        },
        "test": {
            "accuracy": accuracy_score(y_test, y_pred_test),
            "f1": f1_score(y_test, y_pred_test),
            "precision": precision_score(y_test, y_pred_test),
            "recall": recall_score(y_test, y_pred_test),
            "confusion": confusion_matrix(y_test, y_pred_test)
        },
        "report": classification_report(y_test, y_pred_test)
    }

    # Crea filename con parametri extra se forniti
    filename_parts = [model_name.replace(" ", "_").lower()]
    if extra_params:
        filename_parts += [f"{k}{v}" for k, v in extra_params.items()]
    filename = "metrics_" + "_".join(filename_parts) + ".txt"

    # Scrive i risultati
    with open(os.path.join(output_dir, filename), "w") as f:
        f.write(f"=== {model_name}")
        if extra_params:
            param_str = ", ".join([f"{k}={v}" for k, v in extra_params.items()])
            f.write(f" ({param_str})")
        f.write(" ===\n\n")

        f.write(">>> TRAIN METRICS:\n")
        for metric in ["accuracy", "f1", "precision", "recall"]:
            f.write(f"{metric.capitalize()}: {metrics['train'][metric]:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(str(metrics["train"]["confusion"]) + "\n\n")

        f.write(">>> TEST METRICS:\n")
        for metric in ["accuracy", "f1", "precision", "recall"]:
            f.write(f"{metric.capitalize()}: {metrics['test'][metric]:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(str(metrics["test"]["confusion"]) + "\n\n")

        f.write(">>> CLASSIFICATION REPORT (TEST):\n")
        f.write(metrics["report"])




def evaluate_and_save_model_multiclass(
    model,
    model_name: str,
    y_train, y_pred_train,
    y_test, y_pred_test,
    output_dir: str,
    model_path: str,
    extra_params: dict = None
):

    # Crea la cartella di output se non esiste
    os.makedirs(output_dir, exist_ok=True)

    # Salva il modello
    joblib.dump(model, model_path)

    # Calcola metriche
    metrics = {
        "train": {
            "accuracy": accuracy_score(y_train, y_pred_train),
            "f1": f1_score(y_train, y_pred_train,average='macro',zero_division=0),
            "precision": precision_score(y_train, y_pred_train,average='macro',zero_division=0),
            "recall": recall_score(y_train, y_pred_train,average='macro',zero_division=0),
            "confusion": confusion_matrix(y_train, y_pred_train)
        },
        "test": {
            "accuracy": accuracy_score(y_test, y_pred_test),
            "f1": f1_score(y_test, y_pred_test,average='macro',zero_division=0),
            "precision": precision_score(y_test, y_pred_test,average='macro',zero_division=0),
            "recall": recall_score(y_test, y_pred_test,average='macro',zero_division=0),
            "confusion": confusion_matrix(y_test, y_pred_test)
        },
        "report": classification_report(y_test, y_pred_test, zero_division=0)
    }

    # Crea filename con parametri extra se forniti
    filename_parts = [model_name.replace(" ", "_").lower()]
    if extra_params:
        filename_parts += [f"{k}{v}" for k, v in extra_params.items()]
    filename = "metrics_" + "_".join(filename_parts) + ".txt"

    # Scrive i risultati
    with open(os.path.join(output_dir, filename), "w") as f:
        f.write(f"=== {model_name}")
        if extra_params:
            param_str = ", ".join([f"{k}={v}" for k, v in extra_params.items()])
            f.write(f" ({param_str})")
        f.write(" ===\n\n")

        f.write(">>> TRAIN METRICS:\n")
        for metric in ["accuracy", "f1", "precision", "recall"]:
            f.write(f"{metric.capitalize()}: {metrics['train'][metric]:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(str(metrics["train"]["confusion"]) + "\n\n")

        f.write(">>> TEST METRICS:\n")
        for metric in ["accuracy", "f1", "precision", "recall"]:
            f.write(f"{metric.capitalize()}: {metrics['test'][metric]:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(str(metrics["test"]["confusion"]) + "\n\n")

        f.write(">>> CLASSIFICATION REPORT (TEST):\n")
        f.write(metrics["report"])
