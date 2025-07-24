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

# === PARTE 2: MODELO Q-LEARNING ===
df = pd.read_csv(ruta_base).dropna()
df['fecha'] = pd.to_datetime(df['fecha'])
variables = ['IPC', 'PBI', 'Tipo_Cambio', 'Tasa_Interbancaria', 'RIN', 'BVL']
estados = df[variables].values
acciones = {0: 'Conservador', 1: 'Moderado', 2: 'Agresivo', 3: 'Mantener'}
Q = defaultdict(lambda: np.zeros(len(acciones)))
alpha, gamma, epsilon = 0.1, 0.95, 1.0
epsilon_min, decay, episodios = 0.01, 0.995, 500

def recompensa(bvl_act, bvl_sig, a): return {0:0.2,1:0.5,2:0.8,3:0.5}[a]*(bvl_sig - bvl_act)

recompensas = []
for ep in range(episodios):
    total = 0
    for t in range(len(estados)-1):
        s, s1 = tuple(estados[t]), tuple(estados[t+1])
        a = np.random.choice(list(acciones)) if np.random.rand()<epsilon else np.argmax(Q[s])
        r = recompensa(estados[t][5], estados[t+1][5], a)
        Q[s][a] += alpha * (r + gamma*np.max(Q[s1]) - Q[s][a])
        total += r
    epsilon = max(epsilon * decay, epsilon_min)
    recompensas.append(total)

# Curva de aprendizaje
plt.figure(figsize=(10, 4))
plt.plot(recompensas)
plt.title("Curva de Aprendizaje del Agente RL")
plt.xlabel("Episodio"); plt.ylabel("Recompensa Total")
plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, "curva_aprendizaje_rl.png"))

# Simulación Q-Learning
df["accion"] = ""; df["valor_RL"] = 1.0
estado_actual = tuple(estados[0])
for i in range(1, len(estados)):
    a = np.argmax(Q[estado_actual])
    df.loc[i, "accion"] = acciones[a]
    peso = {0:0.2, 1:0.5, 2:0.8, 3:0.5}[a]
    delta = estados[i][5] - estados[i-1][5]
    df.loc[i, "valor_RL"] = float(df.loc[i-1, "valor_RL"]) * (1 + peso*delta)
    estado_actual = tuple(estados[i])

# Agente aleatorio
df["random"] = 1.0
for i in range(1, len(estados)):
    ar = np.random.choice(list(acciones))
    peso = {0:0.2, 1:0.5, 2:0.8, 3:0.5}[ar]
    delta = estados[i][5] - estados[i-1][5]
    df.loc[i, "random"] = float(df.loc[i-1, "random"]) * (1 + peso*delta)

# === CORRECCIÓN 1: BUY & HOLD ARREGLADO ===
# Calcular rendimientos mensuales del BVL
bvl_real = pd.read_csv(ruta_base_unificada)
bvl_real["fecha"] = pd.to_datetime(bvl_real["fecha"])
bvl_real = bvl_real.set_index("fecha").sort_index()

# Calcular rendimiento mensual limitando pérdidas extremas
bvl_rendimientos = bvl_real["BVL"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
# Limitar pérdidas extremas para evitar colapso total
bvl_rendimientos = bvl_rendimientos.clip(lower=-0.3, upper=0.5)  # Máximo 30% pérdida mensual

# Buy & Hold corregido
df["BuyHold"] = 1.0
for i in range(1, len(df)):
    df.loc[i, "BuyHold"] = float(df.loc[i-1, "BuyHold"]) * (1 + bvl_rendimientos.iloc[i])

# Estrategia Markowitz (50% renta variable, 50% renta fija)
df["Markowitz"] = 1.0
for i in range(1, len(df)):
    rv = bvl_rendimientos.iloc[i]  # Renta variable (BVL)
    rf = 0.003  # Renta fija (0.3% mensual ≈ 3.6% anual)
    df.loc[i, "Markowitz"] = float(df.loc[i-1, "Markowitz"]) * (1 + 0.5*rv + 0.5*rf)

# Métricas
def max_drawdown(serie): return ((serie - serie.cummax()) / serie.cummax()).min()
def metricas(serie, nombre):
    ra = (serie.iloc[-1]) ** (12 / len(serie)) - 1
    vol = np.std(np.diff(np.log(serie + 1e-8))) * np.sqrt(12)
    sharpe = ra / vol if vol != 0 else 0
    return {"Estrategia": nombre, "Rendimiento Anual": ra, "Volatilidad": vol,
            "Sharpe Ratio": sharpe, "Max Drawdown": max_drawdown(serie)}

tabla = pd.DataFrame([
    metricas(df["valor_RL"], "Q-Learning"),
    metricas(df["BuyHold"], "Buy & Hold"),
    metricas(df["Markowitz"], "Markowitz"),
    metricas(df["random"], "Agente Aleatorio")
])

# Gráfico comparativo
plt.figure(figsize=(10, 5))
plt.plot(df["fecha"], df["valor_RL"], label="Q-Learning", linewidth=2)
plt.plot(df["fecha"], df["BuyHold"], label="Buy & Hold", linewidth=2)
plt.plot(df["fecha"], df["Markowitz"], label="Markowitz", linewidth=2)
plt.plot(df["fecha"], df["random"], label="Agente Aleatorio", linestyle="--")
plt.title("Evolución del Portafolio por Estrategia")
plt.xlabel("Fecha"); plt.ylabel("Valor Acumulado")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, "evolucion_qlearning_real.png"))

# Drawdown
plt.figure(figsize=(10, 5))
for col, label in [("valor_RL", "Q-Learning"), ("BuyHold", "Buy & Hold"), 
                   ("Markowitz", "Markowitz"), ("random", "Agente Aleatorio")]:
    dd = (df[col] - df[col].cummax()) / df[col].cummax()
    plt.plot(df["fecha"], dd, label=label)
plt.title("Drawdown Comparado por Estrategia")
plt.xlabel("Fecha"); plt.ylabel("Drawdown")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, "drawdown_qlearning_real.png"))

# Frecuencia de decisiones
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="accion", hue="accion",
              order=["Conservador", "Moderado", "Agresivo", "Mantener"],
              palette="pastel", legend=False)
plt.title("Frecuencia de Estrategias RL"); plt.xlabel("Estrategia"); plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, "frecuencia_acciones_rl.png"))

# Evolución de decisiones RL
mapa = {"Conservador":0,"Moderado":1,"Agresivo":2,"Mantener":3}
df["Estrategia_Num"] = df["accion"].map(mapa)
plt.figure(figsize=(10, 4))
plt.plot(df["fecha"], df["Estrategia_Num"], marker='o', color='black')
plt.yticks([0,1,2,3], ["Conservador", "Moderado", "Agresivo", "Mantener"])
plt.title("Estrategias Elegidas por el Agente RL en el Tiempo")
plt.xlabel("Fecha"); plt.ylabel("Estrategia"); plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, "evolucion_estrategias_rl.png"))

# Heatmap final
plt.figure(figsize=(8, 6))
sns.heatmap(df[variables + ["valor_RL"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlación Variables vs Portafolio RL")
plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, "heatmap_correlaciones.png"))

# Guardar tabla
tabla.round(4).to_csv(os.path.join(carpeta_salida, "tabla_metricas_qlearning.csv"), index=False)

# Mostrar tabla de resultados
print("\n" + "="*60)
print("RESULTADOS FINALES - MÉTRICAS COMPARATIVAS")
print("="*60)
for _, row in tabla.iterrows():
    print(f"\n{row['Estrategia'].upper()}:")
    print(f"  Rendimiento Anual: {row['Rendimiento Anual']:.2%}")
    print(f"  Volatilidad:       {row['Volatilidad']:.2%}")
    print(f"  Sharpe Ratio:      {row['Sharpe Ratio']:.2f}")
    print(f"  Max Drawdown:      {row['Max Drawdown']:.2%}")

print("\n✅ Todos los análisis y gráficos fueron generados con éxito.")
