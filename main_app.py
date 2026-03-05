import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ==============================
# Configuración inicial
# ==============================
st.set_page_config(page_title="Wine Classification", layout="wide")
st.title("🍷 Clasificación - Dataset Wine")

# ==============================
# Cargar datos
# ==============================
wine = load_wine()
X = wine.data
y = wine.target

df = pd.DataFrame(X, columns=wine.feature_names)
df["target"] = y

st.subheader("Vista del Dataset")
st.dataframe(df.head())

# ==============================
# Sidebar - Parámetros del modelo
# ==============================
st.sidebar.header("⚙️ Configuración del Modelo")

max_depth = st.sidebar.slider("Profundidad máxima del árbol",
                               min_value=1,
                               max_value=20,
                               value=3)

n_splits = st.sidebar.slider("Número de folds (Cross Validation)",
                             min_value=3,
                             max_value=10,
                             value=5)

criterion = st.sidebar.selectbox("Criterio",
                                  ["gini", "entropy", "log_loss"])

# ==============================
# Modelo
# ==============================
model = DecisionTreeClassifier(
    max_depth=max_depth,
    criterion=criterion,
    random_state=42
)

# ==============================
# Validación cruzada
# ==============================
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

scoring = {
    "accuracy": "accuracy",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro"
}

cv_results = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring=scoring,
    return_train_score=False
)

# ==============================
# Mostrar métricas promedio
# ==============================
st.subheader("📊 Resultados de Validación Cruzada")

metrics_df = pd.DataFrame({
    "Accuracy": cv_results["test_accuracy"],
    "Precision": cv_results["test_precision_macro"],
    "Recall": cv_results["test_recall_macro"],
    "F1 Score": cv_results["test_f1_macro"]
})

st.write("Promedios:")
st.write(metrics_df.mean())

# ==============================
# Visualización métricas por fold
# ==============================
st.subheader("📈 Métricas por Fold")

fig, ax = plt.subplots()
metrics_df.plot(kind="box", ax=ax)
st.pyplot(fig)

# ==============================
# Matriz de confusión final
# ==============================
st.subheader("🔎 Matriz de Confusión (Entrenamiento completo)")

model.fit(X, y)
y_pred = model.predict(X)

cm = confusion_matrix(y, y_pred)

fig2, ax2 = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2)
ax2.set_xlabel("Predicción")
ax2.set_ylabel("Real")

st.pyplot(fig2)
