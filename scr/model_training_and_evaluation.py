# %%
# -----GENERAL-----
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter

# -----FEATURES SELECTION-----
from sklearn.feature_selection import (SelectKBest, SelectFromModel, RFE, SequentialFeatureSelector,
                                       mutual_info_regression)
from sklearn.feature_selection import SequentialFeatureSelector

# -----PREPROCESING-----
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import boxcox
from sklearn.decomposition import PCA

# -----PIPELINE-----
from sklearn.pipeline import Pipeline

# -----MODELOS-----
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

# -----ÁRBOLES-----
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# -----MÉTRICAS-----
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error,r2_score

# ----CROSSVALIDATION-----
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# -----GUARDAR MODELOS-----
import pickle

# -----EVALUACIÓN DE MODELOS-----
import matplotlib.pyplot as plt



# %% [markdown]
# # Entrenamiento y evaluación
# #### 1. [Featuring selection](#id1)
# #### 2. [Decisión de modelo](#id2)
# #### 3. [Elección de hiperparámetros](#id3)
# #### 4. [Evaluación del modelo](#id4)

# %% [markdown]
# Antes de empezar con el entrenamiento, vamos a cargar los datos de train y test. También aplicaremos ahora la transformación de Box-Cox en el target.

# %%
# Carga de datos
train_set = pd.read_csv('../data/train/train_set.csv', index_col=0) 
test_set = pd.read_csv('../data/test/test_set.csv', index_col=0)   

# Aplicación de Box-Cox al target

train_set['target_boxcox'], lambda_opt = boxcox(train_set['target'])

X_train = train_set.drop(columns=['target','target_boxcox'])
y_train = train_set['target_boxcox']

test_set['target_boxcox'] = boxcox(test_set['target'], lambda_opt)

X_test = test_set.drop(columns=['target', 'target_boxcox'])
y_test = test_set['target_boxcox']

# %% [markdown]
# <div id='id1' />
# 
# ## 1. Featuring selection
# Para la selección de variables del modelo vamos a seleccionar las más importantes utilizando diferentes estrategias:
# 1. Análisis visual combinado con filtrado por valores de correlación.
# 2. Selección de features numéricas mediante SelectKBest y ANOVA, selección de features categóricas mediante Mutual Information
# 3. Selección de las mejores features a través de un modelo intermedio (usando SelectFromModel)
# 4. Selección de las mejores features empleando RFE.
# 5. Selección de las mejores features empleando SFS.
# 6. Selección de las mejores features mediante un sistema de hard-voting aplicado a lo obtenido en los pasos 1 a 5 anteriores.
# 7. Selección de variables transformadas mediante PCA. (MACHINE LEARNING NO SUPERVISADO)

# %% [markdown]
# ### 2.1. Analisis por filtrado de valores de correlación
# Como vimos en el EDA, todas las variables son de tipo numéricas. Seleccionaremos las 5 con más correlación con el target.

# %%
# Variables ordenadas por correlación
features_visual = train_set.corr()['target_boxcox'].abs().sort_values(ascending=False)
features_visual

# %%
# Cogemos las variables con correlación >0.3
features_visual = list(features_visual[2:8].index)
features_visual

# %% [markdown]
# ### 2.2. Selección de features numéricas mediante SelectKBest y ANOVA, selección de features numéricas mediante mutual_info_regression

# %%
selector = SelectKBest(mutual_info_regression, k=6)
x_data_kbest = selector.fit_transform(X_train, y_train)
X_train_kbest = pd.DataFrame(x_data_kbest, columns = selector.get_feature_names_out())

# %%
features_filter = list(selector.get_feature_names_out())
features_filter

# %% [markdown]
# ### 2.3. Selección de las mejores features a través de un modelo intermedio (usando SelectFromModel)

# %%
features = X_train.columns
rf_selector = RandomForestRegressor(random_state= 42)
selector_modelo = SelectFromModel(estimator = rf_selector, threshold= "median") # Nos quedamos con la mitad
selector_modelo.fit(X_train, y_train)
features_modelo = list(selector_modelo.get_feature_names_out())
features_modelo

# %% [markdown]
# ### 2.4. Selección de las mejores features empleando RFE.

# %%
rf_RFE = RandomForestRegressor(random_state= 42)
rfe = RFE(estimator = rf_RFE,
        n_features_to_select= 6, # Iterará hasta quedarse con 6
        step = 1)
rfe.fit(train_set[features], y_train)

# %%
features_RFE = list(rfe.get_feature_names_out())
features_RFE

# %% [markdown]
# ### 2.5. Selección de las mejores features empleando SFS.

# %%
rf_SFS = RandomForestRegressor(random_state = 42)
sfs_forward = SequentialFeatureSelector(rf_SFS,
                                        n_features_to_select = 6,
                                        cv = 5,
                                        scoring = "r2",
                                        n_jobs=-1)
sfs_forward.fit(X_train, y_train)

# %%
features_SFS = list(sfs_forward.get_feature_names_out())
features_SFS

# %% [markdown]
# ### 2.6. Selección de las mejores features mediante un sistema de hard-voting aplicado a lo obtenido en los pasos 1 a 5 anteriores.

# %%
# Hard voting
lista_total = features_visual + features_filter + features_modelo + features_RFE + features_SFS
votaciones = Counter(lista_total)
escrutinio = pd.DataFrame(votaciones.values(), columns = ["Votos"], index = votaciones.keys()).sort_values("Votos", ascending = False)
escrutinio


# %%
features_hard_voting = escrutinio["Votos"].nlargest(6).index.to_list()
features_hard_voting

# %% [markdown]
# ### 2.7. PCA 
# Vamos a probar con 9 componentes.

# %%
pca = PCA(n_components=9)
X_pca = pd.DataFrame(pca.fit_transform(X_train))
features_pca = pca.explained_variance_ratio_
features_pca

# %% [markdown]
# Tras realizar la selección de variables con distintas estrategias, nos queda:
# - 6 listas de variables 
# - la transformación de variables PCA
# Probaremos los modelos seleccionados con estas variables y compararemos.

# %%
sel_features = [features_visual, features_filter, features_modelo, features_RFE, features_SFS, features_hard_voting, features_pca]
sel_features = pd.DataFrame(sel_features, index=["visual", "filter","modelo","rfe","sfs","voting", "pca"]).transpose()
sel_features.to_csv('../data/processed/selected_features.csv')
sel_features

# %% [markdown]
# <div id='id2' />
# 
# ## 2. Selección de modelo
# Dado que en nuestro caso lo que queremos predecir es un precio, usaremos modelos de regresión. Estos son los 5 seleccionados:
# 1. LinearRegression
# 2. LinearSMV
# 3. RandomForestRegression
# 4. XGBRegressor
# 5. LGMRegressor
# 
# Vamos a utilizar parámetros básicos para entrenar los modelos con las distintas selecciones de parametros  y ver cuales nos dan mejores resultados.   
# Pero antes, estandarizaremos los datos porque nos vendrá bien en los modelos lineales:

# %%
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns
)

# %% [markdown]
# Preparamos un diccionario con el método y las columnas seleccionadas:

# %%
features_set_names = ["visual", "filter","modelo","rfe","sfs","voting"]

X_train_dict = {}
X_test_dict = {}

for nombre, feature_list in zip(features_set_names, [features_visual, features_filter, features_modelo, features_RFE, features_SFS, features_hard_voting]):
    X_train_dict[nombre] = X_train_scaled[feature_list]
    X_test_dict[nombre] = X_test_scaled[feature_list]

# %% [markdown]
# Preparamos los modelos:

# %%
ln_reg = LinearRegression()
SVR_reg = SVR(kernel='poly')
rf_reg = RandomForestRegressor(max_depth = 10, random_state= 42)
xgb_reg = XGBRegressor(max_depth = 10, random_state = 42)
lgb_reg = LGBMRegressor(max_depth = 10, random_state = 42, verbose = -100)


modelos = {
    'Lineal': ln_reg,
    'SVR': SVR_reg,
    "Random Forest": rf_reg,
    "Lightgbm": lgb_reg,
    "XGBoost": xgb_reg
}

# %% [markdown]
# Realizamos un primer entrenamiento de los modelos para ver los valores de R2 que obtienen:

# %%
resultados_list = []

for feature_set, X_train in X_train_dict.items():
    print(f"Para el set {feature_set}:")

    for tipo, modelo in modelos.items():
        score_mean = np.mean(
            cross_val_score(modelo, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
            )
        print(f"{tipo}: {score_mean}")

        resultados_list.append({
            "Conjunto de Features": feature_set,
            "Modelo": tipo,
            "R2_Media_CV": score_mean
        })
    print("********")
for tipo, modelo in modelos.items():
    score_mean = np.mean(
        cross_val_score(modelo, X_pca, y_train, cv=5, scoring="r2", n_jobs=-1)
        )
    print(f"{tipo} (PCA): {score_mean}")
    resultados_list.append({
        "Conjunto de Features": "pca",
        "Modelo": tipo,
        "R2_Media_CV": score_mean
    })
print("********")

df_resultados = pd.DataFrame(resultados_list)
df_resultados = df_resultados.sort_values(by='R2_Media_CV', ascending=False)
df_resultados.to_csv('../data/processed/resultados_modelos_features.csv')
df_resultados

# %% [markdown]
# Los valores de r2 máximos están muy cercanos unos de otros. Nos quedaremos con la selección de variables **voting** que da los mejores resultados en los metodos **RandomForest** y en **Lightgbm**.
# Montaremos un pipeline para probar hiperparametros y escoger entre uno de los dos modelos.

# %%
X_train = train_set[features_hard_voting]
X_test = test_set[features_hard_voting]

# %%
pipe = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('regressor', rf_reg)
])

random_forest_params = {
    'scaler': [StandardScaler(), MinMaxScaler(), None],
    'regressor': [rf_reg],
    'regressor__n_estimators': [80, 100, 200, 500],
    'regressor__max_depth': [5,10,20, None],
    'regressor__min_samples_leaf':[1,2,10,20],
    "regressor__max_features": ["sqrt","log2",None]
}

lgb_param = {
    'regressor': [lgb_reg],
    "regressor__num_leaves": [31, 63, 127],
    "regressor__max_depth": [-1, 6, 10],
    "regressor__learning_rate": [0.01, 0.05, 0.1],
    "regressor__n_estimators": [200, 500, 1000],
    "regressor__min_data_in_leaf": [10, 20, 50],
    "regressor__boosting_type": ["gbdt"],
}

search_space = [
    random_forest_params,
    lgb_param
]

reg_grid = GridSearchCV(estimator = pipe,
                  param_grid = search_space,
                  cv = 5,
                  verbose=4,
                  n_jobs=-1,
                  scoring='r2')

reg_grid.fit(X_train, y_train)

# %% [markdown]
# El mejor modelo ha sido **RandomForest**. Será el que tomemos.  
# Vamos a ver que errores tiene:

# %%
# sin deshacer box-cox
y_pred = reg_grid.predict(X_test)
print("R2", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RSME:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAPE:", mean_absolute_percentage_error(y_test, y_pred))


# %%
# deshacemos boxcox
def inv_boxcox(y, lmbda):
    # inverse of scipy.stats.boxcox
    if np.isclose(lmbda, 0.0):
        return np.exp(y)
    
    return np.power(lmbda * y + 1.0, 1.0 / lmbda)

# predecir en espacio boxcox y deshacer la transformación
y_pred_boxcox = reg_grid.predict(X_test)
y_pred = inv_boxcox(y_pred_boxcox, lambda_opt)

# obtener target original del test
y_test_original = test_set['target']

# métricas en escala original
print("R2", r2_score(y_test_original, y_pred))
print("MAE:", mean_absolute_error(y_test_original, y_pred))
print("RSME:", np.sqrt(mean_squared_error(y_test_original, y_pred)))
print("MAPE:", mean_absolute_percentage_error(y_test_original, y_pred))

# %% [markdown]
# Los números son bastante buenos. Guardaremos el modelo.

# %%
model_1 = reg_grid.best_estimator_
reg_grid.best_estimator_
with open('../models/model_1.pkl', 'wb') as f:
    pickle.dump(model_1, f)

# %% [markdown]
# <div id='id3' />
# 
# ## 3. Elección de hiperparametros
# Vamos a realizar otro GridSearchCrossValidation con valores cercanos al modelo anterior para ver si lo podemos mejorar un poco:

# %%
reg_grid.best_estimator_

# %%
pipe_final = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('regressor', rf_reg)
])

random_forest_params = {
    'scaler': [StandardScaler(), None],
    'regressor': [rf_reg],
    'regressor__n_estimators': [500,700,800],
    'regressor__max_depth': [None],
    'regressor__min_samples_leaf':[1],
    "regressor__max_features": ["sqrt"]
}
reg_grid_final = GridSearchCV(estimator = pipe_final,
                  param_grid = random_forest_params,    
                  cv = 5,
                  verbose=2,
                  n_jobs=-1,
                  scoring='r2')

reg_grid_final.fit(X_train, y_train)

# %% [markdown]
# Parece que a partir de 700 estimadores el modelo no mejora más. Nos quedamos con esta configuración como modelo final. Guardamos el modelo.

# %%
model_final = reg_grid_final.best_estimator_
with open('../models/final_model.pkl', 'wb') as f:
    pickle.dump(model_final, f)

# %% [markdown]
# <div id='id3' />
# 
# ## 4. Evaluación del modelo
# En este apartado realizaremos métricas y sacaremos una visualización del error.
# 

# %% [markdown]
# Vamos a sacar las métricas de los errores.

# %%
# deshacemos boxcox
def inv_boxcox(y, lmbda):
    # inverse of scipy.stats.boxcox
    if np.isclose(lmbda, 0.0):
        return np.exp(y)
    
    return np.power(lmbda * y + 1.0, 1.0 / lmbda)

# predecir en espacio boxcox y deshacer la transformación
y_pred_boxcox = model_final.predict(X_test)
y_pred = inv_boxcox(y_pred_boxcox, lambda_opt)

# obtener target original del test
y_test_original = test_set['target']

# métricas en escala original
print("R2", r2_score(y_test_original, y_pred))
print("MAE:", mean_absolute_error(y_test_original, y_pred))
print("RSME:", np.sqrt(mean_squared_error(y_test_original, y_pred)))
print("MAPE:", mean_absolute_percentage_error(y_test_original, y_pred))

# %% [markdown]
# El r2 ha alcanzado un valor de 0.8867, lo que es bastante bueno y nos dará una buena predicción.  
# Veamos la representación de los errores:

# %%
def plot_predictions_vs_actual(y_real, y_pred):
    """
    Función para graficar los valores reales vs. los valores predichos en una regresión.

    Args:
    y_real (array-like): Valores reales de la variable objetivo.
    y_pred (array-like): Valores predichos de la variable objetivo.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, y_real, alpha=0.5)
    plt.xlabel("Valores Predichos")
    plt.ylabel("Valores Reales")

    # Línea y=x
    max_value = max(max(y_real), max(y_pred))
    min_value = min(min(y_real), min(y_pred))
    plt.plot([min_value, max_value], [min_value, max_value], 'r')

    plt.title("Comparación de Valores Reales vs. Predichos")
    plt.show()


# %%
plot_predictions_vs_actual(y_test_original, y_pred)

# %% [markdown]
# La recta se ajusta bastante a los valores reales, salvo por algunos outliers que se alejan. Consideramos que es una buena predición y pasamos el modelo a streamlit para que un usuario externo pueda realizar predicciones.


