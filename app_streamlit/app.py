import streamlit as st
import pickle
import numpy as np

st.title('Predicción del precio de la electricidad')

# Carga modelo
with open("../models/final_model.pkl", "rb") as f:
    
    modelo = pickle.load(f)

# Crear pestañas
tab1, tab2 = st.tabs(["Predicción", "Sobre el modelo"])

with tab1:
    texto_1 = """
<p style="text-align: justify;">
El precio de la electricidad varía continuamente en función del mix energético disponible en cada momento. 
Cada tipo de generación —carbón, cogeneración, ciclo combinado, hidráulica, solar fotovoltaica o nuclear— contribuye de forma diferente al coste final del mercado. 
\n En esta herramienta podrás simular la cantidad generada en un día en cada una de las 6 fuentes que más influyen mediante los deslizadores, y el modelo predictivo estimará
el precio de la electricidad en €/MWh según la combinación elegida.
</p>
"""
    st.markdown(texto_1, unsafe_allow_html=True)
    st.header("Introduce la generación en un día (MWh)")

    col1, col2, col3 = st.columns([6, 6, 6])

    with col1:
        carbon = st.slider("Carbón (MWh)", 0.0, 1770.0, 350.0, 10.0)
        cogeneracion = st.slider("Cogeneración (MWh)", 730.0, 3300.0, 1950.0, 10.0)
        ciclo_combinado = st.slider("Ciclo combinado (MWh)", 710.0, 15380.0, 4040.0, 10.0)
            
    with col2:
        hidraulica = st.slider("Hidráulica (MWh)", 520.0, 8150.0, 2840.0, 10.0)
        solar_fotovoltaica = st.slider("Solar fotovoltaica (MWh)", 0.0, 11010.0, 4940.0, 10.0)
        nuclear = st.slider("Nuclear (MWh)", 10.0, 7140.0, 6280.0, 10.0)

    with col3:
        if st.button("Predecir precio €/MWh"):
            input_data = np.array([[carbon, cogeneracion, ciclo_combinado, hidraulica, solar_fotovoltaica, nuclear]])
            pred = modelo.predict(input_data)
            st.metric("Precio estimado (€/MWh)", f"{pred[0]:.2f}")
    texto_2 = """
    <p style="text-align: justify;">
    Para el mejor funcionamiento del predictor, el valor por defecto de cada deslizador corresponde a la media de generación diaria de cada fuente en el conjunto de datos
    utilizado para entrenar el modelo. 
    \nLos valores máximos y mínimos corresponden con la generación máxima y mínima registrada en un día para cada fuente.
    </p>
    """            
    st.markdown(texto_2, unsafe_allow_html=True)

with tab2:
    st.header("Objetivo del proyecto")

    texto_3 = """
    <p style="text-align: justify;">
    El propósito de este proyecto es desarrollar un modelo de regresión basado en técnicas de Machine Learning que permita estimar el 
    precio diario de la electricidad a partir del mix energético disponible. Este modelo se entrenará utilizando datos históricos que
    incluyan:<br><br>
    - La contribución de cada tecnología al mix energético diario.<br>
    - Precio resultante en el mercado mayorista.<br><br>
    El objetivo final es construir un modelo capaz de anticipar el comportamiento del precio, facilitando la toma de decisiones
    estratégicas y mejorando la gestión operativa dentro del mercado.
    </p>
    """
    st.markdown(texto_3, unsafe_allow_html=True)

    st.header("Datos utilizados")
    texto_4 = """
    <p style="text-align: justify;">
    Los datos utilizados en este proyecto provienen de fuentes oficiales. En concreto, de Red Eléctrica de España. Los datos de han descargado desde
    la API de REE, que proporciona información detallada sobre la generación eléctrica por fuente, demanda y precios.
    Se han recopilado datos diarios desde noviembre de 2021 hasta noviembre de 2025 para asegurar una adecuada
    representatividad del modelo.<br><br>   
    """
    st.markdown(texto_4, unsafe_allow_html=True)

    st.header("Sobre el modelo")
    texto_5 = """
    <div style="line-height:1.4;">
      <p style="text-align: justify; margin-bottom: 0.6rem;">
        Para la construcción del predictor se ha utilizado un modelo de 
        <span style="text-decoration: underline; font-weight:600;">Random Forest Regressor</span>, seleccionado por su capacidad 
        para capturar relaciones no lineales y su buen desempeño en problemas de predicción complejos. 
        El modelo ha sido evaluado sobre un conjunto de validación de un 20% de los datos, obteniendo las siguientes métricas.
      </p>

      <p style="text-align: justify; margin-top:0.2rem;">
        Las métricas principales son: 
        <span style="text-decoration: underline; font-weight:600;">R²</span>, 
        <span style="text-decoration: underline; font-weight:600;">MAE</span>, 
        <span style="text-decoration: underline; font-weight:600;">RMSE</span> y 
        <span style="text-decoration: underline; font-weight:600;">MAPE</span>.
      </p>

      <div style="margin-top:1rem; display:flex; justify-content:center;">
        <table style="border-collapse: collapse; width: 340px;">
          <thead>
            <tr>
              <th style="text-align: left; padding: 6px 10px; border-bottom: 2px solid #ddd;">Métrica</th>
              <th style="text-align: right; padding: 6px 10px; border-bottom: 2px solid #ddd;">Valor</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style="padding: 6px 10px; border-bottom: 1px solid #f0f0f0;">R²</td>
              <td style="padding: 6px 10px; text-align: right; border-bottom: 1px solid #f0f0f0;">0.88673</td>
            </tr>
            <tr>
              <td style="padding: 6px 10px; border-bottom: 1px solid #f0f0f0;">MAE (€/MWh)</td>
              <td style="padding: 6px 10px; text-align: right; border-bottom: 1px solid #f0f0f0;">13.05375</td>
            </tr>
            <tr>
              <td style="padding: 6px 10px; border-bottom: 1px solid #f0f0f0;">RMSE (€/MWh)</td>
              <td style="padding: 6px 10px; text-align: right; border-bottom: 1px solid #f0f0f0;">20.98089</td>
            </tr>
            <tr>
              <td style="padding: 6px 10px;">MAPE</td>
              <td style="padding: 6px 10px; text-align: right;">35.978%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
    """

    st.markdown(texto_5, unsafe_allow_html=True)