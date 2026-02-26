import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from streamlit_drawable_canvas import st_canvas
from PIL import Image

# =========================
# CONFIGURACIÓN
# =========================
st.set_page_config(page_title="Clasificador MNIST", layout="wide")
st.title("🧠 Clasificador de Dígitos Escritos a Mano - MNIST")

# =========================
# CARGAR DATASET
# =========================
@st.cache_data
def load_data():
    mnist = fetch_openml("mnist_784", version=1)
    X = mnist.data.astype("float32") / 255.0
    y = mnist.target.astype("int")
    return X, y

X, y = load_data()

# Reducimos tamaño para que la app no sea muy pesada
X_small, _, y_small, _ = train_test_split(X, y, train_size=10000, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X_small, y_small, test_size=0.2, random_state=42
)

# =========================
# SIDEBAR CONFIGURACIÓN
# =========================
st.sidebar.header("⚙️ Configuración")

model_option = st.sidebar.selectbox(
    "Seleccione el modelo",
    ("KNN", "SVM", "Random Forest", "Logistic Regression")
)

# =========================
# SELECCIÓN DEL MODELO
# =========================
if model_option == "KNN":
    model = KNeighborsClassifier(n_neighbors=3)

elif model_option == "SVM":
    model = SVC()

elif model_option == "Random Forest":
    model = RandomForestClassifier(n_estimators=100)

elif model_option == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)

# =========================
# ENTRENAMIENTO
# =========================
with st.spinner("Entrenando modelo..."):
    model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# =========================
# MÉTRICAS
# =========================
st.subheader("📊 Métricas de Desempeño")

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
sns.heatmap(cm, cmap="Blues", ax=ax)
st.pyplot(fig)

# =========================
# VALIDACIÓN CON IMAGEN DEL DATASET
# =========================
st.subheader("🖼 Probar Imagen del Dataset")

index = st.slider("Seleccione índice de prueba", 0, len(X_test)-1, 0)

image = X_test.iloc[index].values.reshape(28, 28)

st.image(image, width=200, caption="Imagen seleccionada")

if st.button("Predecir Imagen Dataset"):
    prediction = model.predict([X_test.iloc[index]])
    st.success(f"Predicción: {prediction[0]}")

# =========================
# DIBUJAR Y PREDECIR
# =========================
st.subheader("✍️ Dibuja un Dígito")

canvas_result = st_canvas(
    fill_color="white",
    stroke_width=15,
    stroke_color="black",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predecir Dibujo"):

    if canvas_result.image_data is not None:
        img = Image.fromarray(
            (canvas_result.image_data[:, :, 0]).astype("uint8")
        )

        img = img.resize((28, 28))
        img_array = np.array(img)
        img_array = img_array.reshape(1, 784) / 255.0

        st.image(img, width=150, caption="Imagen procesada 28x28")

        prediction = model.predict(img_array)
        st.success(f"Predicción del modelo: {prediction[0]}")
