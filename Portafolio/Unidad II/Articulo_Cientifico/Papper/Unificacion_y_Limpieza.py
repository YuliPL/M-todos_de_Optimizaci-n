import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

# Carpeta base del script
base_dir = os.path.dirname(__file__)

# Función para cargar, limpiar y estandarizar un dataset individual
def limpiar_dataset(nombre_archivo, nombre_variable):
    ruta = os.path.join(base_dir, nombre_archivo + ".csv")
    df = pd.read_csv(ruta, encoding='latin1')
    df = df.drop(index=0)  # eliminar fila de encabezado duplicado si la hay
    df.columns = ['fecha', nombre_variable]
    df['fecha'] = pd.to_datetime(df['fecha'], format='%b%y', errors='coerce')
    df[nombre_variable] = pd.to_numeric(df[nombre_variable], errors='coerce')
    df = df.drop_duplicates(subset='fecha')
    return df

# Cargar y limpiar todos los archivos
ipc = limpiar_dataset("ipc", "IPC")
pbi = limpiar_dataset("pbi", "PBI")
rin = limpiar_dataset("rin", "RIN")
tasa = limpiar_dataset("tasa_interbancaria", "Tasa_Interbancaria")
tipo = limpiar_dataset("tipo_cambio", "Tipo_Cambio")
bvl = limpiar_dataset("bvl", "BVL")

# Unificar por fecha
df_final = ipc.merge(pbi, on='fecha') \
              .merge(rin, on='fecha') \
              .merge(tasa, on='fecha') \
              .merge(tipo, on='fecha') \
              .merge(bvl, on='fecha')

# Eliminar valores nulos
df_final = df_final.dropna()

# Ordenar y convertir fecha a string mensual (ej. 2020-01)
df_final['fecha'] = df_final['fecha'].dt.to_period('M').astype(str)
df_final = df_final.sort_values('fecha').reset_index(drop=True)

# Guardar dataset limpio
df_final.to_csv(os.path.join(base_dir, "base_unificada.csv"), index=False)
print("✅ Archivo base_unificada.csv generado con éxito.")

# Normalizar variables (excepto 'fecha')
scaler = MinMaxScaler()
df_norm = df_final.copy()
df_norm[df_final.columns[1:]] = scaler.fit_transform(df_final[df_final.columns[1:]])

# Guardar dataset normalizado
df_norm.to_csv(os.path.join(base_dir, "base_normalizada.csv"), index=False)
print("✅ Archivo base_normalizada.csv generado con éxito (normalizado para RL).")
