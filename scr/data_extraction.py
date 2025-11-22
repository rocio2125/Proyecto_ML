# %% [markdown]
# ## Fuentes de datos:
# Para realizar el estudio y entrenar los modelos de machine learning se solicita acceso a la API de ESIOS. 
# Una vez obtenido un token se lanzan las siguientes consultas:

# %%
# Carga de librerias
import os
import requests
import pandas as pd
from datetime import datetime, timedelta

# %%
# Definimos token:
TOKEN = "90ff7e4af7adfafbe79ddc8a15901f15a7f6252d5976a2d9bac2646a50d697c3"

# Fechas de estudio:
start_str_1 = '2024-11-01'
end_str_1 = '2025-04-30'

start_str_2 = '2025-05-01'
end_str_2 = '2025-11-01'

start_str_3 = '2023-11-01'
end_str_3 = '2024-04-30'

start_str_4 = '2024-05-01'
end_str_4 = '2024-11-01'

start_str_5 = '2022-11-01'
end_str_5 = '2023-04-30'

start_str_6 = '2023-05-01'
end_str_6 = '2023-11-01'

start_str_7 = '2021-11-01'
end_str_7 = '2022-04-30'

start_str_8 = '2022-05-01'
end_str_8 = '2022-11-01'

# %%
# Sacamos todos los indicadores disponibles

headers = {
    "Accept": "application/json; application/vnd.esios-api-v1+json",
    "x-api-key": TOKEN
}

url = "https://api.esios.ree.es/indicators"

resp = requests.get(url, headers=headers)
resp.raise_for_status()

data = resp.json()

# Extraer la lista de indicadores
indicators = data["indicators"]

df_indicators = pd.DataFrame(indicators)[["id", "name", "short_name", "description"]]

print(df_indicators.head())
print(f"Total indicadores: {len(df_indicators)}")

df_indicators.to_csv("../data/raw/indicadores.csv", index=False)


# %% [markdown]
# Con todos los indicadores, realizamos una primera selección de los que nos interesan. 
# El objetivo es sacar una estimación del precio de la electricidad según el mix energético diario. En este caso nos interesan los indicadores de Generación Medida, nos quedaremos con las mayoritarias.
# 

# %%
# Llamamos a la API y descargamos los indicadores que nos interesan

headers = {
    "Accept": "application/json; application/vnd.esios-api-v1+json",
    "x-api-key": TOKEN
}

INDICATORS = {
    "eolica":10037,
     "nuclear":1153,
     "carbon":10036,
     "ciclo_combinado":1156,
     "hidraulica":10035,
     "solar_fotovoltaica":1161,
     "solar_termica":1162,
     "termica_renovable":1756,
     "turbina_de_vapor":1158,
     "cogeneracion":10039,
     "exportacion_andorra":1181,
     "exportacion_marruecos":1180,
     "exportacion_portugal":1179,
     "exportacion_francia":1178,
     "importacion_francia":1174,
     "importacion_portugal":1175,
     "importacion_marruecos":1176,
     "importacion_andorra":1177,
  }

def normalize_df(values):
    if not values:
        return pd.DataFrame()
    df = pd.DataFrame(values)
    
    for c in ["datetime_utc", "datetime", "ts", "timestamp", "hora"]:
        if c in df.columns:
            time_col = c
            break
    else:
        raise ValueError(f"No se encontró columna temporal. Columnas: {df.columns}")

    for c in ["value", "valor", "values", "magnitud", "data"]:
        if c in df.columns:
            value_col = c
            break
    else:
        raise ValueError(f"No se encontró columna de valores. Columnas: {df.columns}")

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df_renamed = df.rename(columns={time_col: "datetime", value_col: "value"})
    df_result = df_renamed[["datetime", "value"]]
    if df_result.columns.duplicated().any():
        df_result = df_result.loc[:, ~df_result.columns.duplicated()]

    return df_result

def get_indicator(indicator_id):
    def fetch_data(start_date, end_date):
        url = f"https://api.esios.ree.es/indicators/{indicator_id}"
        params = {
            "start_date": f"{start_date}T00:00:00Z",
            "end_date": f"{end_date}T23:59:59Z",
            "time_trunc": "day",
            "time_agg": "avg",
            "geo_agg": "sum",
            "locale": "es"
        }
        r = requests.get(url, headers=headers, params=params)
        r.raise_for_status()
        values = r.json()["indicator"]["values"]
        return normalize_df(values)

    # Descarga primer rango
    df1 = fetch_data(start_str_1, end_str_1)
    # Descarga segundo rango
    df2 = fetch_data(start_str_2, end_str_2)
    # Descarga tercer rango
    df3 = fetch_data(start_str_3, end_str_3)
    # Descarga cuarto rango
    df4 = fetch_data(start_str_4, end_str_4)
    # Descarga quinto rango
    df5 = fetch_data(start_str_5, end_str_5)
    # Descarga sexto rango
    df6 = fetch_data(start_str_6, end_str_6)
    # Descarga septimo rango
    df7 = fetch_data(start_str_7, end_str_7)
    # Descarga octavo rango
    df8 = fetch_data(start_str_8, end_str_8)
    
    # Combina y elimina duplicados si los hubiera
    df_full = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8]).drop_duplicates(subset="datetime").reset_index(drop=True)
    return df_full

dfs = []

for name, ind in INDICATORS.items():
    print(f"Descargando {name} (ID {ind})…")
    try:
        df = get_indicator(ind)
        if df.empty:
            print(f"⚠️  {name} no devolvió datos.")
            continue
        df = df.rename(columns={"value": name})
        dfs.append(df)
    except Exception as e:
        print(f"Error descargando {name}: {e}")

if not dfs:
    raise RuntimeError("Ningún indicador devolvió datos.")

df_final = dfs[0]
for df in dfs[1:]:
    df_final = df_final.merge(df, on="datetime", how="outer")

df_final = df_final.sort_values("datetime").reset_index(drop=True)

# Guardamos el archivo

df_final.to_csv("../data/raw/mix_generacion.csv", index=False)

print("\nArchivo generado: mix_generacion.csv")
print(df_final.head())

# %% [markdown]
# Ahora descargamos los precios marginales de mercado diario para el periodo de estudio.

# %%
# Volvemos a llamar a la API y sacamos el indicador de precio marginal de mercado diario 
headers = {
    "Accept": "application/json; application/vnd.esios-api-v1+json",
    "x-api-key": TOKEN
}

# Indicador precio marginal mercado diario
INDICATOR_ID = 600

def get_indicator_data(indicator_id, start_str, end_str):
    url = f"https://api.esios.ree.es/indicators/{indicator_id}"
    params = {
        "start_date": start_str,
        "end_date": end_str,
        "time_trunc": "hour",   # precio horario
        "time_agg": "avg",
        "geo_agg": "sum",
        "locale": "es"
    }
    r = requests.get(url, headers=headers, params=params)
    r.raise_for_status()
    data = r.json()["indicator"]["values"]
    df = pd.DataFrame(data)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    return df[["datetime_utc", "value"]]

# Descargar datos
df1 = get_indicator_data(INDICATOR_ID, start_str_1, end_str_1)
df2 = get_indicator_data(INDICATOR_ID, start_str_2, end_str_2)
df3 = get_indicator_data(INDICATOR_ID, start_str_3, end_str_3)
df4 = get_indicator_data(INDICATOR_ID, start_str_4, end_str_4)
df5 = get_indicator_data(INDICATOR_ID, start_str_5, end_str_5)
df6 = get_indicator_data(INDICATOR_ID, start_str_6, end_str_6)
df7 = get_indicator_data(INDICATOR_ID, start_str_7, end_str_7)
df8 = get_indicator_data(INDICATOR_ID, start_str_8, end_str_8)
df = (
    pd.concat([df1, df2, df3, df4, df5, df6, df7, df8])
      .drop_duplicates(subset="datetime_utc")
      .reset_index(drop=True)
)

# Agregar a precio diario promedio
df["date"] = df["datetime_utc"].dt.date
df_daily = df.groupby("date")["value"].mean().reset_index()

print(df_daily.head())
print(df_daily.tail())

# Exportar CSV
df_daily.to_csv("../data/raw/precio.csv", index=False)



