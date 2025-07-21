import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# === CONFIGURACIÓN GENERAL ===
sns.set(style="whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})

# Rutas necesarias
ruta_base = r"c:\INGENIERIA ESTADISTICA E INFORMATICA - UNAP\QUINTO SEMESTRE\METODOS DE OPTIMIZACION\PORTAFOLIO\Articulo Cientifico\datos_bcrp\base_normalizada.csv"
ruta_base_unificada = r"c:\INGENIERIA ESTADISTICA E INFORMATICA - UNAP\QUINTO SEMESTRE\METODOS DE OPTIMIZACION\PORTAFOLIO\Articulo Cientifico\datos_bcrp\base_unificada.csv"
carpeta_salida = r"c:\INGENIERIA ESTADISTICA E INFORMATICA - UNAP\QUINTO SEMESTRE\METODOS DE OPTIMIZACION\PORTAFOLIO\Articulo Cientifico\resultados"
os.makedirs(carpeta_salida, exist_ok=True)

# Tasa libre de riesgo mensual (aproximación: 3% anual = 0.25% mensual)
RISK_FREE_RATE_MONTHLY = 0.0025
RISK_FREE_RATE_ANNUAL = 0.03

# Carga de datos unificada
df_explora = pd.read_csv(ruta_base_unificada, parse_dates=["fecha"]).dropna()

# 1. Serie temporal
plt.figure(figsize=(12, 6))
for col in ['IPC', 'PBI', 'RIN', 'Tasa_Interbancaria', 'Tipo_Cambio', 'BVL']:
    plt.plot(df_explora['fecha'], df_explora[col], label=col)
plt.title('Evolución de Indicadores Macroeconómicos (2015–2024)')
plt.xlabel('Fecha'); plt.ylabel('Valor')
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, "serie_tiempo_macroeconomica.png"))

# 2. Correlación
plt.figure(figsize=(8, 6))
sns.heatmap(df_explora.drop(columns='fecha').corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlación de Variables Macroeconómicas')
plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, "correlacion_macroeconomica.png"))

# 3. Histogramas
fig, axs = plt.subplots(2, 3, figsize=(15, 8))
for i, col in enumerate(['IPC', 'PBI', 'RIN', 'Tasa_Interbancaria', 'Tipo_Cambio', 'BVL']):
    sns.histplot(df_explora[col], kde=True, ax=axs.flatten()[i])
    axs.flatten()[i].set_title(f'Distribución: {col}')
fig.tight_layout()
plt.savefig(os.path.join(carpeta_salida, "histogramas_macroeconomicos.png"))

# === PARTE 2: MODELO Q-LEARNING CORREGIDO ===
df = pd.read_csv(ruta_base).dropna()
df['fecha'] = pd.to_datetime(df['fecha'])
variables = ['IPC', 'PBI', 'Tipo_Cambio', 'Tasa_Interbancaria', 'RIN', 'BVL']
estados = df[variables].values

# Definir acciones con pesos de riesgo más realistas
acciones = {0: 'Conservador', 1: 'Moderado', 2: 'Agresivo', 3: 'Mantener'}
pesos_riesgo = {0: 0.3, 1: 0.6, 2: 1.0, 3: 0.6}  # Conservador menos riesgo

# Inicializar Q-table
Q = defaultdict(lambda: np.zeros(len(acciones)))

# Hiperparámetros optimizados
alpha, gamma, epsilon = 0.1, 0.95, 1.0
epsilon_min, decay, episodios = 0.01, 0.995, 500

# Función de recompensa corregida
def recompensa(bvl_act, bvl_sig, accion):
    delta_bvl = bvl_sig - bvl_act
    peso = pesos_riesgo[accion]
    return peso * delta_bvl

# Entrenamiento Q-Learning
recompensas_episodio = []
for ep in range(episodios):
    recompensa_total = 0
    for t in range(len(estados)-1):
        estado_actual = tuple(np.round(estados[t], 3))  # Redondear para discretizar
        estado_siguiente = tuple(np.round(estados[t+1], 3))
        
        # Selección de acción (ε-greedy)
        if np.random.rand() < epsilon:
            accion = np.random.choice(list(acciones.keys()))
        else:
            accion = np.argmax(Q[estado_actual])
        
        # Calcular recompensa
        r = recompensa(estados[t][5], estados[t+1][5], accion)
        
        # Actualizar Q-value
        Q[estado_actual][accion] += alpha * (r + gamma * np.max(Q[estado_siguiente]) - Q[estado_actual][accion])
        
        recompensa_total += r
    
    # Decaimiento de epsilon
    epsilon = max(epsilon * decay, epsilon_min)
    recompensas_episodio.append(recompensa_total)
    
    if ep % 100 == 0:
        print(f"Episodio {ep}: Recompensa = {recompensa_total:.4f}, Epsilon = {epsilon:.4f}")

print("✅ Entrenamiento completado")

# Curva de aprendizaje
plt.figure(figsize=(10, 4))
plt.plot(recompensas_episodio)
plt.title("Curva de Aprendizaje del Agente RL")
plt.xlabel("Episodio"); plt.ylabel("Recompensa Total")
plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, "curva_aprendizaje_rl.png"))

# === SIMULACIÓN CON ESTRATEGIAS CORREGIDAS ===

# Preparar datos para simulación
df_sim = df.copy()
df_sim["accion_rl"] = ""
df_sim["valor_RL"] = 1.0

# Simulación Q-Learning
for i in range(1, len(estados)):
    estado_actual = tuple(np.round(estados[i-1], 3))
    accion = np.argmax(Q[estado_actual])
    df_sim.loc[i, "accion_rl"] = acciones[accion]
    
    # Calcular rendimiento basado en BVL real (no normalizado)
    bvl_actual = df_explora.iloc[i-1]['BVL']
    bvl_siguiente = df_explora.iloc[i]['BVL']
    rendimiento_bvl = (bvl_siguiente - bvl_actual) / bvl_actual
    
    peso = pesos_riesgo[accion]
    rendimiento_ajustado = peso * rendimiento_bvl
    
    df_sim.loc[i, "valor_RL"] = df_sim.loc[i-1, "valor_RL"] * (1 + rendimiento_ajustado)

# === BENCHMARKS CORREGIDOS ===

# 1. Buy & Hold CORREGIDO (simplemente el índice BVL)
bvl_inicial = df_explora.iloc[0]['BVL']
df_sim["BuyHold"] = df_explora['BVL'] / bvl_inicial

# 2. Agente Aleatorio
np.random.seed(42)  # Para reproducibilidad
df_sim["Aleatorio"] = 1.0
for i in range(1, len(df_sim)):
    accion_aleatoria = np.random.choice(list(acciones.keys()))
    bvl_actual = df_explora.iloc[i-1]['BVL']
    bvl_siguiente = df_explora.iloc[i]['BVL']
    rendimiento_bvl = (bvl_siguiente - bvl_actual) / bvl_actual
    peso = pesos_riesgo[accion_aleatoria]
    rendimiento_ajustado = peso * rendimiento_bvl
    df_sim.loc[i, "Aleatorio"] = df_sim.loc[i-1, "Aleatorio"] * (1 + rendimiento_ajustado)

# 3. Estrategia Markowitz MEJORADA (pesos optimizados simples)
# Usando pesos basados en volatilidad inversa como proxy de Markowitz
rendimientos_bvl = df_explora['BVL'].pct_change().dropna()
volatilidad_bvl = rendimientos_bvl.std()

# Markowitz simplificado: 60% BVL, 40% tasa libre de riesgo
df_sim["Markowitz"] = 1.0
for i in range(1, len(df_sim)):
    bvl_actual = df_explora.iloc[i-1]['BVL']
    bvl_siguiente = df_explora.iloc[i]['BVL']
    rendimiento_bvl = (bvl_siguiente - bvl_actual) / bvl_actual
    
    # Portfolio: 60% BVL + 40% risk-free
    rendimiento_portfolio = 0.6 * rendimiento_bvl + 0.4 * RISK_FREE_RATE_MONTHLY
    df_sim.loc[i, "Markowitz"] = df_sim.loc[i-1, "Markowitz"] * (1 + rendimiento_portfolio)

# === FUNCIÓN DE MÉTRICAS CORREGIDA ===
def calcular_metricas(serie, nombre):
    """Calcula métricas financieras corregidas"""
    # Convertir a numpy array y limpiar
    valores = np.array(serie.dropna())
    if len(valores) < 2:
        return {"Estrategia": nombre, "Error": "Datos insuficientes"}
    
    # Calcular rendimientos
    rendimientos = np.diff(valores) / valores[:-1]
    rendimientos = rendimientos[np.isfinite(rendimientos)]
    
    if len(rendimientos) == 0:
        return {"Estrategia": nombre, "Error": "No hay rendimientos válidos"}
    
    # Métricas anualizadas
    rendimiento_promedio_mensual = np.mean(rendimientos)
    rendimiento_anual = (1 + rendimiento_promedio_mensual) ** 12 - 1
    
    volatilidad_mensual = np.std(rendimientos)
    volatilidad_anual = volatilidad_mensual * np.sqrt(12)
    
    # Sharpe ratio corregido (con tasa libre de riesgo)
    excess_return = rendimiento_anual - RISK_FREE_RATE_ANNUAL
    sharpe_ratio = excess_return / volatilidad_anual if volatilidad_anual > 0 else 0
    
    # Drawdown máximo
    valores_acum = np.cumprod(1 + rendimientos)
    peak = np.maximum.accumulate(valores_acum)
    drawdown = (valores_acum - peak) / peak
    max_drawdown = np.min(drawdown)
    
    # Retorno total
    retorno_total = (valores[-1] / valores[0]) - 1
    
    return {
        "Estrategia": nombre,
        "Rendimiento Anual": rendimiento_anual,
        "Volatilidad": volatilidad_anual,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Retorno Total": retorno_total
    }

# Calcular métricas para todas las estrategias
estrategias = ["valor_RL", "BuyHold", "Markowitz", "Aleatorio"]
nombres = ["Q-Learning", "Buy & Hold", "Markowitz", "Agente Aleatorio"]

tabla_metricas = []
for estrategia, nombre in zip(estrategias, nombres):
    metricas = calcular_metricas(df_sim[estrategia], nombre)
    tabla_metricas.append(metricas)

# Crear DataFrame con resultados
df_resultados = pd.DataFrame(tabla_metricas)
print("\n=== MÉTRICAS CORREGIDAS ===")
print(df_resultados.round(4))

# === VISUALIZACIONES MEJORADAS ===

# Gráfico comparativo
plt.figure(figsize=(12, 6))
fechas = df_explora['fecha'][:len(df_sim)]
plt.plot(fechas, df_sim["valor_RL"], label="Q-Learning", linewidth=2)
plt.plot(fechas, df_sim["BuyHold"], label="Buy & Hold", linewidth=2)
plt.plot(fechas, df_sim["Markowitz"], label="Markowitz", linewidth=2)
plt.plot(fechas, df_sim["Aleatorio"], label="Agente Aleatorio", linestyle="--", alpha=0.8)
plt.title("Evolución del Portafolio por Estrategia (Corregido)")
plt.xlabel("Fecha"); plt.ylabel("Valor Acumulado")
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, "evolucion_qlearning_corregido.png"))

# Drawdown mejorado
plt.figure(figsize=(12, 6))
for estrategia, nombre in zip(estrategias, nombres):
    serie = df_sim[estrategia]
    valores = np.array(serie)
    peak = np.maximum.accumulate(valores)
    drawdown = (valores - peak) / peak
    plt.plot(fechas, drawdown * 100, label=nombre, linewidth=2)

plt.title("Drawdown Comparado por Estrategia (%)")
plt.xlabel("Fecha"); plt.ylabel("Drawdown (%)")
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, "drawdown_qlearning_corregido.png"))

# Frecuencia de decisiones
conteo_acciones = df_sim["accion_rl"].value_counts()
plt.figure(figsize=(8, 6))
conteo_acciones.plot(kind='bar', color=['skyblue', 'lightgreen', 'salmon', 'gold'])
plt.title("Frecuencia de Estrategias RL")
plt.xlabel("Estrategia"); plt.ylabel("Frecuencia")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, "frecuencia_acciones_rl_corregido.png"))

# Guardar resultados
df_resultados.to_csv(os.path.join(carpeta_salida, "metricas_corregidas.csv"), index=False)
df_sim.to_csv(os.path.join(carpeta_salida, "simulacion_completa.csv"), index=False)

print("✅ Análisis completo y guardado")