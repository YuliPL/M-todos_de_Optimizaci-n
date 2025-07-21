import pandas as pd
import os

# Ruta absoluta del dataset fuente
ruta_datos = r"c:\INGENIERIA ESTADISTICA E INFORMATICA - UNAP\QUINTO SEMESTRE\METODOS DE OPTIMIZACION\PORTAFOLIO\Articulo Cientifico\datos_bcrp\base_unificada.csv"

# Leer datos
df = pd.read_csv(ruta_datos, parse_dates=["fecha"])

# Eliminar columnas no numéricas para estadísticas
df_numerico = df.select_dtypes(include="number")

# Eliminar filas vacías
df_limpio = df_numerico.dropna()

# Estadísticas descriptivas
estadisticas = df_limpio.describe(percentiles=[.25, .5, .75]).T
estadisticas = estadisticas.rename(columns={
    "count": "Cantidad",
    "mean": "Media",
    "std": "Desviación Std",
    "min": "Mínimo",
    "25%": "Percentil 25%",
    "50%": "Mediana",
    "75%": "Percentil 75%",
    "max": "Máximo"
})
estadisticas = estadisticas.round(3)

# Guardar como CSV en la misma carpeta del script
ruta_script = os.path.dirname(os.path.abspath(__file__))
ruta_salida = os.path.join(ruta_script, "estadisticas_descriptivas.csv")
estadisticas.to_csv(ruta_salida)

print(f"\n✅ Estadísticas descriptivas guardadas en:\n{ruta_salida}")
