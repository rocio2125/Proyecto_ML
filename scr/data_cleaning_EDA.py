# %% [markdown]
# ![Cabecera](img/img1.jpg)
# # Predicción del Precio de la Electricidad
# #### 1. [Introducción](#id1)
# #### 2. [Carga de datos y limpieza de missings](#id2)
# #### 3. [Train y Test](#id3)
# #### 4. [Análisis Univariante](#id4)
# #### 5. [Análisis Bivariante](#id5)
# #### 6. [Tratamiento de Outliers](#id6)
# 

# %% [markdown]
# <div id='id1' />
# 
# ## 1.Introducción

# %% [markdown]
# El precio mayorista de la electricidad es una variable clave dentro de los mercados energéticos. Su evolución diaria depende de 
# múltiples factores, entre los que destaca la composición del mix energético utilizado para cubrir la demanda: tecnologías renovables 
# (eólica, solar, hidráulica), generación térmica, ciclos combinados, nuclear, entre otras. Cada tecnología presenta costes marginales 
# distintos, lo que influye directamente en el precio final de casación en el mercado eléctrico.
# 
# En este contexto, contar con herramientas capaces de predecir el precio de la electricidad se vuelve esencial para agentes del sector 
# energético, comercializadoras, operadores de red y consumidores industriales.
# 
# **Objetivo del Proyecto**
# 
# El propósito de este proyecto es desarrollar un modelo de regresión basado en técnicas de Machine Learning que permita estimar el 
# precio diario de la electricidad a partir del mix energético disponible. Este modelo se entrenará utilizando datos históricos que 
# incluyan:
# - La contribución de cada tecnología al mix energético diario.
# - Precio resultante en el mercado mayorista.
# 
# El objetivo final es construir un modelo capaz de anticipar el comportamiento del precio, facilitando la toma de decisiones 
# estratégicas y mejorando la gestión operativa dentro del mercado.
# 
# **Enfoque Metodológico**
# 
# Para ello, el proyecto comprende las siguientes fases:
# 
# - *Recopilación y limpieza de datos*
# Reunir los datos históricos del mercado eléctrico y preprocesarlos para eliminar inconsistencias y valores atípicos.
# 
# - *Análisis exploratorio (EDA)*
# Examinar la distribución del mix energético, la variabilidad de los precios y las relaciones entre variables.
# 
# - *Selección y construcción del modelo*
# Probar diferentes modelos de regresión y optimizar hiperparámetros para mejorar el rendimiento.
# 
# - *Evaluación del rendimiento*
# Utilizar métricas como R², MAE o RMSE  para valorar la capacidad predictiva del modelo.
# 
# - *Interpretación y conclusiones*
# Identificar qué tecnologías o factores del mix energético impactan más en el precio.
# 
# **Resultados Esperados**
# 
# Se espera obtener un modelo robusto que permita:
# 
# - Predecir el precio de la electricidad con un margen de error reducido.
# - Detectar patrones entre el mix energético y la formación de precios.
# - Servir como herramienta de apoyo para planificación energética, negociación y análisis de escenarios.

# %% [markdown]
# <div id='id2' />
# 
# ## 2. Carga de datos y limpieza de missings
# Cargaremos los datos a partir de los archivos obtenidos de la API de ESIOS que obtuvimos en el notebook [01_Fuentes]('01_Fuentes.ipynb').

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import bootcampviztools as bt
import plotly.express as px
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.stats import boxcox

# %% [markdown]
# Cargamos los datos de generación energética:

# %%
# Cargamos el mix energético
mix_energetico = pd.read_csv('../data/raw/mix_generacion.csv')
mix_energetico['datetime'] = mix_energetico['datetime'].astype('string')
mix_energetico['date'] = mix_energetico['datetime'].str[:10]
# Visualizamos
display(mix_energetico)

# %% [markdown]
# Cargamos los precios diarios:

# %%
# Cargamos el precio diario
precio = pd.read_csv('../data/raw/precio.csv')

# Visualizamos
display(precio)

# %% [markdown]
# Cruzamos los dos dataframes:
# 

# %%
# Juntamos los dos datasets en uno solo
df = mix_energetico.merge(precio, on="date", how="left")

df.to_csv('../data/processed/df_final.csv', index=False)

# Cambio el nombre del precio
df.rename(columns={'value': 'target'}, inplace=True)

# visualizamos
display(df)

# %%
df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
df.info()

# %%
df.describe().to_csv('../data/processed/descripcion_datos.csv')

# %% [markdown]
# Vemos que hay varios missings, vamos a tratarlos:

# %%
filas_con_missings = df.isnull().any(axis=1)
df_missings = df[filas_con_missings]
df_missings

# %% [markdown]
# Se trata de las ultimas entradas, no tenemos datos de generación pero sí de las exportaciones e importaciones. Puede ser que los datos no estén actualizados o disponibles todavía.   
# Procedemos a deshacernos de ellos ya que no nos servirán para entrenar modelos y tenemos suficientes datos.

# %%
df.dropna(inplace=True)
display(df)

# %% [markdown]
# Vamos a ver cómo evoluciona en precio para detectar anomalías:

# %%
plt.figure(figsize=(18, 6))

sns.lineplot(
    data=df,
    x= df['datetime'],
    y='target')

plt.title("Evolución del precio a lo largo del tiempo", fontsize=16)
plt.xlabel("Fecha", fontsize=12)
plt.ylabel("Precio (target)", fontsize=12)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# Vemos que hay un pico en el año 2022:

# %%
plt.figure(figsize=(18, 6))

sns.lineplot(
    data=df,
    x= df['datetime'][df['datetime'].dt.year == 2022],
    y='target')

plt.title("Evolución del precio a lo largo del tiempo", fontsize=16)
plt.xlabel("Fecha", fontsize=12)
plt.ylabel("Precio (target)", fontsize=12)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# El pico se produce en marzo, en este periodo el precio del gas natural subió inusualmente debido a la inestabilidad por la [invasión rusa a Ucrania](https://www.rtve.es/noticias/20220330/marzo-mes-mas-caro-historia-precio-electricidad-luz/2325589.shtml). ![grafico](img/pico_marzo22.png)  
# Vamos a eliminar el mes de marzo completo de los datos que es cuando se acentúa el pico. Asumiremos que nuestro modelo predictivo no es compatible con las politicas expansionistas.

# %%
df = df[~((df['datetime'].dt.year == 2022) & (df['datetime'].dt.month == 3))]
df.reset_index(drop=True, inplace=True)

# %% [markdown]
# Lo que queremos no es un análisis estacional, queremos prececir el precio a partir del mix energético. Por esto, no volveremos a usar la columna 'datetime' y vamos a deshacernos de ella.

# %%
# Descartamos la columna datetime
df = df.drop(columns=['datetime','date'])

# Guardamos el dataframe 
df.to_csv('../data/processed/data.csv')

# %% [markdown]
# <div id='id3' />
# 
# ## 3. Train y test
# Separamos los datos aleatoriamente en dos sets: Train y Test.
# 

# %%
# Separamos en train y test
train_set, test_set = train_test_split(df, test_size = 0.2, random_state=42)
train_set.to_csv('../data/train/train_set.csv')
test_set.to_csv('../data/test/test_set.csv')

# %% [markdown]
# <div id='id4' />
# 
# ## 4. Análisis univariante:
# Vamoms a analizar las variables incluidas en los datos.
# 

# %% [markdown]
# ### 4.1 Target: Precio mercado SPOT Diario (€/MWh)
# El Mercado Diario es un mercado mayorista en el que se establecen transacciones de energía eléctrica para el día siguiente, mediante la presentación de ofertas de venta y adquisición de energía eléctrica por parte de los participantes en el mercado. 
# Como resultado del mismo se determina de forma simultánea el precio del mercado diario en cada zona de oferta, los programas de toma y entrega de energía, y los programas de intercambio entre zonas de oferta.

# %%
# Veamos como se distribuye el target
bt.plot_combined_graphs(train_set, ['target'])

# %%
train_set['target'].describe()

# %% [markdown]
# La distribución del precio tiende a campana con un valor medio de 103.43 €/MWh. Sólo vemos dos outlayers por la derecha.
# Los datos tienen de min 0.35 y de máximo 467.45. Veamos qué mix energético teníamos esos días:

# %%
# Max y min de target

fig, (ax1, ax2) = plt.subplots(
    nrows=1, 
    ncols=2, 
    figsize=(14, 6))

sns.barplot(train_set.loc[train_set['target'] == train_set['target'].min()], ax=ax1)
ax1.set_xticklabels(
    ax1.get_xticklabels(), 
    rotation=45,          
    ha='right')   
ax1.set_title('Minimo')        

sns.barplot(train_set.loc[train_set['target'] == train_set['target'].max()], ax=ax2)
ax2.set_xticklabels(
    ax2.get_xticklabels(), 
    rotation=45,          
    ha='right',           
    )
ax2.set_title('Máximo')

plt.tight_layout() 
plt.show()

# %% [markdown]
# Observamos que cuando el precio fue minimo la generación de las energías renovables fue más alta, mientras que el precio máximo se fijó cuando la energía mayoritaria fue el ciclo combinado.
# 
# Veamos si con una transformación logarítmica se ve mejor la campana:

# %%
# Veamos como se distribuye el target con transformación logaritmica
train_set_copy = train_set.copy()
train_set_copy['log_target'] = np.log1p(train_set['target'])
bt.plot_combined_graphs(train_set_copy, ['log_target'])

# %% [markdown]
# No ha mejorado, probemos con una transformaciones de raiz cuadrada y cúbica:

# %%
# Veamos como se distribuye el target con transformación cuadrática
train_set_copy['sq_target'] = np.square(train_set['target'])
bt.plot_combined_graphs(train_set_copy, ['sq_target'])

# %%
# Veamos como se distribuye el target con transformación cúbica
train_set_copy['cb_target'] = (train_set['target'])**(1/3)
bt.plot_combined_graphs(train_set_copy, ['cb_target'])

# %% [markdown]
# Probamos por último la transformación de Box-Cox:

# %%
# Por último probamos con box-cox
train_set_copy['target_boxcox'], lambda_opt = boxcox(train_set_copy['target'])
bt.plot_combined_graphs(train_set_copy, ['target_boxcox'])

# %%
# Comparamos Box-Cox con el original:

fig, (ax1, ax2) = plt.subplots(
    nrows=1, 
    ncols=2, 
    figsize=(14, 6))

sns.histplot(train_set['target'], kde=True, ax=ax1) 
sns.histplot(train_set_copy['target_boxcox'], kde=True, ax=ax2) 
ax1.set_title('Distribución Target')
ax2.set_title('Distribución Target Transformado')
plt.tight_layout() 
plt.show()

# %% [markdown]
# #### Parece que la distribución de los datos mejora con una transformación Box-Cox. Será la que escojamos.

# %% [markdown]
# ### 4.2 Resto de variables:
# El resto de variables se refieren a los MWh generados por cada tecnología el día y a las importaciones/exportaciones con los paises 
# vecinos:
# Energía eolica
# Energía nuclear
# Energía carbón
# Energía ciclo combinado
# Energía hidraulica
# Energía solar fotovoltaica
# Energía solar termica
# Energía cogeneración
# Exportacion Andorra
# Exportacion Marruecos
# Exportacion Portugal
# Exportacion Francia
# Importacion Francia
# Importacion Portugal
# Importacion Marruecos
# Importacion Andorra
# 
# Vamos a generar gráficos univariables para ver la distribución que tienen.

# %%
# Graficos para el resto de variables
bt.plot_combined_graphs(train_set, [col for col in train_set.columns if col != 'target'])

# %% [markdown]
# Todas las variables son numéricas continuas.

# %% [markdown]
# <div id='id5' />
# 
# # 5. Análisis bivariable
# Vamos a realizar una matriz de correlación para ver las variables que más influyen en el target.

# %%
# Matriz de correlación
plt.figure(figsize=(15,6))
sns.heatmap(train_set.corr(), annot=True, cmap='coolwarm')

# %%
# Variables ordenadas por correlación
train_set.corr()['target'].sort_values(ascending=False).round(2)

# %% [markdown]
# Tenemos variables que influyen incrementando el precio:
# - Carbón
# - Cogeneración
# - Ciclo combinado
# - Exportaciones e importaciones a Marruecos
# Otras que influyen en el precio disminuyendolo:
# - Solar fotovoltaica
# - Hidráulica
# 
# La colclusión que podemos extraer de esto es que **cuando se recurre a energías no renovables, el precio sube**.
# 
# Vamos a pasar los números a valores absolutos para ver el orden de influencia, independientemente de que esta sea positiva o negativa:

# %%
# Variables ordenadas por correlación v.abs
train_set.corr()['target'].abs().sort_values(ascending=False)

# %% [markdown]
# <div id='id6' />
# 
# ## 6. Tratamiento de Outliers
# En la grafica de boxplots del target pudimos ver que existen outliers a la derecha. Para mejorar nuestro modelo vamos a descartar outliers ya que no queremos aprender de ellos. 
# No queremos aprender de días puntuales en los que el precio fué inusualmente alto por alguna circunstancia, queremos que funcione en el día a día.  
# Quitaremos los que superen 2 veces la distancia intercuartil por la derecha.

# %%
Q1 = train_set['target'].quantile(0.25)
Q3 = train_set['target'].quantile(0.75)
IQR = Q3 - Q1

upper = Q3+2*IQR

# ¿Cuantos outliers hay?
print( "Número de outliers:", len(train_set[(train_set['target'] >= upper)]) )
train_set[(train_set['target'] >= upper)]

# %%
# Los quitamos y guardamos el dataset limpio 
train_set = train_set[(train_set['target'] <= upper)]
train_set.to_csv('../data/train/train_set.csv')


