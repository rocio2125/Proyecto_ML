# ⚡️ Predicción del Precio Mayorista de la Electricidad (PVPC)

Este repositorio contiene un proyecto de Machine Learning centrado en la **predicción del precio mayorista de la electricidad (PVPC)**. El precio de la electricidad es una variable crucial en los mercados energéticos, y su volatilidad diaria está influenciada por factores como la demanda y la composición del mix energético (renovables, térmica, etc.).

El objetivo es desarrollar un modelo de regresión robusto capaz de pronosticar el precio diario promedio a partir de la composición del mix.

---

## 📂 Estructura del Repositorio

El proyecto se desarrolla a través de tres notebooks principales que cubren las fases de extracción de datos, preprocesamiento y modelado:

1.  **`01_Fuentes.ipynb`**: **Adquisición de Datos y Fuentes**
2.  **`02_LimpiezaEDA.ipynb`**: **Limpieza, Análisis Exploratorio de Datos (EDA) y Preprocesamiento**
3.  **`03_Entrenamiento_Evaluacion.ipynb`**: **Entrenamiento y Evaluación del Modelo**

STREAMLIT: Se ha creado una página para poder utilizar externamente el modelo predictivo: [enlace](https://predictor-precio-electricidad.streamlit.app/)
---

## 🛠️ Tecnologías y Librerías

El proyecto está desarrollado en **Python** y utiliza las siguientes librerías clave:

* **Manejo de Datos:** `pandas`, `numpy`
* **Adquisición de Datos:** `requests` (para interactuar con la API de ESIOS)
* **Preprocesamiento y Modelado:** `sklearn` (StandardScaler, PCA, SelectKBest, Regresiones Lineales, Random Forest Regressor, PCA, etc.)
* **Visualización:** `matplotlib`, `seaborn`
* **Otras Utilidades:** `scipy.stats` (Box-Cox)

---

## 📝 Fases del Proyecto

### 1. Adquisición de Datos (`01_Fuentes.ipynb`)

En esta fase se realiza la conexión a la **API de ESIOS** para obtener el precio mayorista de la electricidad.

* Se utiliza un **TOKEN** de acceso para lanzar múltiples consultas y recopilar los datos del periodo de estudio.
* Los datos horarios se agregan para obtener el **precio diario promedio**, que es la variable objetivo.

### 2. Limpieza y EDA (`02_LimpiezaEDA.ipynb`)

Aquí se prepara el conjunto de datos para el modelado:

* **Carga de Datos y Limpieza de Missings:** Se comprueban y gestionan los valores faltantes.
* **Separación Train/Test:** Los datos se dividen para asegurar una evaluación imparcial del modelo.
* **Análisis Univariante y Bivariante:** Se estudian las distribuciones de las variables y sus relaciones.
* **Tratamiento de Outliers:** Se utiliza el método del rango intercuartílico (IQR) para identificar y gestionar valores atípicos en la variable objetivo.

### 3. Entrenamiento y Evaluación (`03_Entrenamiento_Evaluacion.ipynb`)

Esta etapa se centra en la aplicación de técnicas de machine learning y la construcción del modelo predictivo:

* **Transformación de Variables:** Se utilizan técnicas como **Box-Cox** y **Normalización/Estandarización** (`StandardScaler`) para preparar las *features*.
* **Selección de Características:** Se exploran diversos métodos para reducir la dimensionalidad y mejorar el rendimiento:
    * Métodos de filtrado (`SelectKBest` con $mutual\_info\_regression$).
    * Métodos *Wrapper* (`SequentialFeatureSelector` - SFS).
    * Métodos *Embedded* (`SelectFromModel` con modelos lineales regularizados).
    * Reducción de Dimensionalidad (`PCA`).
* **Modelado y Optimización:** Se entrenan y evalúan múltiples modelos de regresión, como:
    * LinearRegression
    * LinearSMV
    * RandomForestRegression
    * XGBRegressor
    * LGMRegressor
* **Evaluación:** El rendimiento del modelo se mide utilizando métricas de regresión: **R2, MAE, RSME, MAPE**, y se visualizan las predicciones contra los valores reales. 

---

## 🚀 Cómo Empezar

Para replicar este análisis:

1.  **Clona el repositorio:**
    ```bash
    git clone [URL_DEL_REPOSITORIO]
    ```
2.  **Instala las dependencias de Python:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn requests
    ```
3.  **Obtén un Token de ESIOS** (necesario para el primer notebook).
4.  Ejecuta los notebooks en orden (`01_Fuentes.ipynb`, `02_LimpiezaEDA.ipynb`, `03_Entrenamiento_Evaluacion.ipynb`).