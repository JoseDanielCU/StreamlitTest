import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# =========================
# CONFIGURACIÓN GENERAL
# =========================
st.set_page_config(page_title="Clasificador IRIS", layout="wide")

st.title("🌸 Clasificador Dinámico - Dataset IRIS")
st.markdown("""
Este aplicativo permite:
- Seleccionar diferentes modelos de clasificación
- Ajustar hiperparámetros
- Visualizar métricas y matriz de confusión
- Probar nuevas muestras manualmente
""")

# =========================
# CARGAR DATASET
# =========================
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

df = X.copy()
df["species"] = y.map(dict(enumerate(iris.target_names)))

# =========================
# SIDEBAR - CONFIGURACIÓN
# =========================
st.sidebar.header("⚙️ Configuración del Modelo")

model_option = st.sidebar.selectbox(
    "Seleccione el modelo",
    ("KNN", "SVM", "Random Forest")
)

test_size = st.sidebar.slider(
    "Porcentaje de datos para prueba",
    0.1, 0.5, 0.2
)

# Hiperparámetros dinámicos
if model_option == "KNN":
    n_neighbors = st.sidebar.slider("Número de vecinos (K)", 1, 15, 5)

elif model_option == "SVM":
    C = st.sidebar.slider("Parámetro C", 0.1, 10.0, 1.0)
    kernel = st.sidebar.selectbox("Kernel", ("linear", "rbf", "poly"))

elif model_option == "Random Forest":
    n_estimators = st.sidebar.slider("Número de árboles", 10, 200, 100)
    max_depth = st.sidebar.slider("Profundidad máxima", 1, 20, 5)


# =========================
# DIVISIÓN DE DATOS
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

# =========================
# SELECCIÓN DEL MODELO
# =========================
if model_option == "KNN":
    model = KNeighborsClassifier(n_neighbors=n_neighbors)

elif model_option == "SVM":
    model = SVC(C=C, kernel=kernel)

elif model_option == "Random Forest":
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

# =========================
# ENTRENAMIENTO
# =========================
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# =========================
# MÉTRICAS
# =========================
accuracy = accuracy_score(y_test, y_pred)

st.subheader("📊 Desempeño del Modelo")

col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy", f"{accuracy:.4f}")

with col2:
    st.text("Reporte de Clasificación")
    st.text(classification_report(y_test, y_pred))

# =========================
# MATRIZ DE CONFUSIÓN
# =========================
st.subheader("🔎 Matriz de Confusión")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names,
    ax=ax
)
plt.xlabel("Predicción")
plt.ylabel("Real")

st.pyplot(fig)

# =========================
# VISUALIZACIÓN INTERACTIVA
# =========================
st.subheader("🌺 Visualización Interactiva del Dataset")

fig2 = px.scatter(
    df,
    x=iris.feature_names[0],
    y=iris.feature_names[1],
    color="species",
    title="Distribución de las especies"
)

st.plotly_chart(fig2)

# =========================
# PREDICCIÓN MANUAL
# =========================
st.subheader("🧪 Probar Nueva Muestra")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length", 4.0, 8.0, 5.1)
    sepal_width = st.number_input("Sepal Width", 2.0, 4.5, 3.5)

with col2:
    petal_length = st.number_input("Petal Length", 1.0, 7.0, 1.4)
    petal_width = st.number_input("Petal Width", 0.1, 2.5, 0.2)

if st.button("Predecir"):
    input_data = np.array([
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]).reshape(1, -1)

    prediction = model.predict(input_data)
    predicted_species = iris.target_names[prediction][0]

    st.success(f"La especie predicha es: **{predicted_species.upper()}**")