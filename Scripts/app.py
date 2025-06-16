import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report

# === Mappatura modelli ===

models_binari = {
    "Decision Tree": ("models/decision_tree_selfmade_final.joblib", None),
    "KNN": ("models/KNN_selfmade.joblib", "models/scaler_knn.joblib"),
    "RIPPER": ("models/rule_based_ripper.joblib", None),
    "SVM": ("models/svm_selfmade.joblib", "models/scaler_svm.joblib"),
    "ANN": ("models/ann_selfmade.joblib", "models/scaler_ann.joblib"),
    "Naive Bayes": ("models/naive_bayes_selfmade.joblib", "models/scaler_naivebayes.joblib")
}

models_multiclasse = {
    "Decision Tree": ("models/decision_tree_category.joblib", None),
    "Random forest" :("models/randomforest_final_model.joblib",None),
    "KNN": ("models/knn_category.joblib", "models/scaler_knn_category.joblib"),
    "SVM": ("models/svm_category.joblib", "models/scaler_svm_category.joblib"),
    "ANN": ("models/ann_category.joblib", "models/scaler_ann_category.joblib"),
    "Naive Bayes": ("models/naive_bayes_category.joblib", "models/scaler_naivebayes_category.joblib"),
}




# === Funzione per predizione ===
def predict(model, df):
    return model.predict(df)

# === Titolo ===
st.title("Classificatore Data Mining - selfMade / category")

# === Selezione tipo classificazione ===
classification_type = st.selectbox("Tipo di classificazione", ["Binaria (selfMade)", "Multiclasse (category)"])

# === Scelta del modello ===
model_dict = models_binari if classification_type == "Binaria (selfMade)" else models_multiclasse
model_name = st.selectbox("Seleziona il modello", list(model_dict.keys()))
model_path, scaler_path = model_dict[model_name]
model = joblib.load(model_path)
scaler = joblib.load(scaler_path) if scaler_path else None

# === Selezione modalità ===
mode = st.radio("Modalità", ["Valutazione su dati noti (test)", "Predizione su nuovo dato"])

# === Colonne codificate ===
if classification_type == "Binaria (selfMade)":
    input_columns = pd.read_csv("data/splitted/X_train.csv").columns.tolist()
    target_col = "selfMade"
else:
    label_encoder = joblib.load("models/label_encoder_category.joblib")
    input_columns = pd.read_csv("data/splitted_category/X_train.csv").columns.tolist()
    target_col = "category"

# === Modalità Test ===
if mode == "Valutazione su dati noti (test)":
    uploaded_file = st.file_uploader("Carica un file CSV contenente i dati codificati e la colonna target corretta")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if target_col not in df.columns:
            st.error(f"Colonna target '{target_col}' mancante nel file caricato.")
        else:
            X = df[input_columns]
            if scaler:
                X = scaler.transform(X)
            y_true = df[target_col]
            y_pred = predict(model, X)

            st.subheader("Report di classificazione")
            st.text(classification_report(y_true, y_pred))

            df["Predizione"] = y_pred
            st.subheader("Anteprima con confronto predizione-vero")
            st.dataframe(df[[*input_columns, target_col, "Predizione"]])

# === Modalità Nuovo Dato ===
else:
    st.subheader("Inserisci un nuovo dato da classificare")
    input_data = {col: 0.0 for col in input_columns}

    with st.expander("Variabili numeriche", expanded=True):
        for col in input_columns:
            if any(keyword in col for keyword in ["log", "age", "gdp", "cpi", "life_expectancy", "education_enrollment", "tax_rate", "population", "selfMade_encoded"]):
                input_data[col] = st.number_input(f"{col}", value=0.0)

    def multiselect_section(label, prefix):
        cols = [col for col in input_columns if col.startswith(prefix)]
        options = [col.replace(prefix, "") for col in cols]
        selected = st.multiselect(label, options)
        for opt in options:
            full_col = prefix + opt
            if full_col in input_columns:
                input_data[full_col] = 1.0 if opt in selected else 0.0

    with st.expander("Country"):
        multiselect_section("Seleziona paese", "country_")

    with st.expander("Categoria"):
        multiselect_section("Seleziona categoria", "category_")

    with st.expander("Industrie"):
        multiselect_section("Seleziona industria", "industries_")

    with st.expander("Fonte (source)"):
        multiselect_section("Seleziona fonte", "source_")

    with st.expander("Altri attributi (es. gender, status)"):
        multiselect_section("Genere", "gender_")
        multiselect_section("Status", "status_")

    input_df = pd.DataFrame([input_data]).fillna(0.0)

    if scaler:
        scaler_cols = scaler.feature_names_in_
        input_df = input_df.reindex(columns=scaler_cols, fill_value=0.0)
        input_df = scaler.transform(input_df)
    else:
        input_df = input_df.reindex(columns=input_columns, fill_value=0.0)



    if st.button("Predici"):
        try:
            prediction = predict(model, input_df)

            if label_encoder:
                try:
                    pred_label = label_encoder.inverse_transform(prediction)[0]
                except Exception as e:
                    st.warning(f"Errore nella decodifica dell'etichetta: {e}")
                    pred_label = prediction[0]
            else:
                pred_label = prediction[0]

            st.success(f"Classe predetta: {pred_label}")
        except Exception as e:
            st.error(f"Errore durante la predizione: {e}")


