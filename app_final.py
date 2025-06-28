import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import plotly.express as px

# === Load saved predictions and data ===
y_test = np.load("y_test.npy")
y_pred_svm = np.load("y_pred_svm.npy")
y_pred_rf = np.load("y_pred_rf.npy")
y_pred_ada = np.load("y_pred_ada.npy")
y_pred_xgb = np.load("y_pred_xgb.npy")
y_test_dl = np.load("y_test_dl.npy")
X_test_dl = np.load("X_test_dl.npy")

# === Load training data for CV ===
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

# === Scale training data ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# === Recreate ML models ===
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

svm_model = SVC()
rf_model = RandomForestClassifier()
ada_model = AdaBoostClassifier()
xgb_model = XGBClassifier(
    n_estimators=50,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    random_state=42
)

# === Load deep learning models ===
cnn_model = load_model("cnn_model.h5")
lstm_model = load_model("lstm_model.h5")

# === Streamlit App ===
st.set_page_config(layout="wide")
st.title("📊 Model Performance Dashboard")
st.markdown("Compare accuracy, precision, recall, F1-score, CV accuracy, and deep learning validation accuracy.")

# === Calculate metrics ===
def get_model_results():
    models = {
        "SVM": (svm_model, y_pred_svm),
        "Random Forest": (rf_model, y_pred_rf),
        "AdaBoost": (ada_model, y_pred_ada),
        "XGBoost": (xgb_model, y_pred_xgb)
    }

    results = []

    for name, (model, y_pred) in models.items():
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
        # Compute CV accuracy
        cv_score = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy').mean()
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "CV Accuracy": cv_score,
            "Val Accuracy": "-"
        })

    for name, model in [("CNN", cnn_model), ("LSTM", lstm_model)]:
        y_pred_dl = model.predict(X_test_dl).argmax(axis=1)
        acc = accuracy_score(y_test_dl, y_pred_dl)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test_dl, y_pred_dl, average='weighted', zero_division=0)
        _, val_acc = model.evaluate(X_test_dl, y_test_dl, verbose=0)
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "CV Accuracy": "-",
            "Val Accuracy": val_acc
        })

    return results

# === Run and display ===
model_results = get_model_results()

metrics_df = pd.DataFrame(model_results)
metrics_df = metrics_df.round(4).sort_values(by="Accuracy", ascending=False)

st.subheader("📈 Model Performance Table")
st.dataframe(metrics_df)

# === Bar chart ===
fig = px.bar(metrics_df, x='Model', y='Accuracy', color='Model', text='Accuracy', title="Model Accuracy")
fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
fig.update_layout(yaxis=dict(tickformat=".0%"))
st.plotly_chart(fig)
