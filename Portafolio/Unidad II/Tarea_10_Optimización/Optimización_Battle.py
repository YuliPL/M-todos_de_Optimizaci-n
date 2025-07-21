import pandas as pd
import numpy as np
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
from itertools import combinations
warnings.filterwarnings('ignore')

class EliteOptimaBattleOptimizer:
    """
    Optimizador Élite para OptimaBattle Arena
    Combina múltiples estrategias avanzadas para maximizar el puntaje del torneo
    """
    
    def __init__(self, dataset_path="Ronda1.xlsx"):
        """
        Inicializa el optimizador élite con parámetros del torneo
        """
        self.dataset_path = dataset_path
        self.data = None
        self.elite_portfolio = None
        self.start_time = None
        self.end_time = None
        
        # Parámetros del torneo
        self.PRESUPUESTO = 1_000_000
        self.LAMBDA_RIESGO = 0.5
        self.MAX_SECTOR_WEIGHT = 0.30
        self.MIN_ASSETS = 5
        self.MAX_BETA = 1.2
        self.SECTORES = {1: 'Tech', 2: 'Salud', 3: 'Energía', 4: 'Financiero', 5: 'Consumo'}
        
        # Configuración avanzada
        self.AGRESIVIDAD = 3  # 1-4, donde 4 es ultra agresivo
        self.USE_ADVANCED_OPTIMIZATION = True
        self.MAX_ITERATIONS = 200
        
        print("🏆 OPTIMABATTLE ELITE OPTIMIZER v3.0 🏆")
        print("="*60)
        
    def cargar_y_procesar_datos(self):
        """Carga y procesa los datos con análisis avanzado"""
        try:
            # Intentar cargar desde directorio actual
            if not os.path.exists(self.dataset_path):
                current_dir = os.path.dirname(os.path.abspath(__file__))
                self.dataset_path = os.path.join(current_dir, "Ronda1.xlsx")
                
            self.data = pd.read_excel(self.dataset_path)
            print(f"✅ Datos cargados: {len(self.data)} activos desde {self.dataset_path}")
            
        except Exception as e:
            print(f"⚠️  Error cargando datos: {e}")
            print("📊 Generando datos de ejemplo optimizados...")
            self._generar_datos_ejemplo()
            
        # Procesar y enriquecer datos
        self._procesar_y_enriquecer_datos()
        print(f"✅ Procesamiento completado: {len(self.data)} activos listos")
        
    def _generar_datos_ejemplo(self):
        """Genera datos de ejemplo realistas basados en el PDF"""
        np.random.seed(42)
        n_assets = 100
        
        # Generar datos con distribuciones realistas
        retornos_base = np.random.normal(11, 3, n_assets)  # Media 11%, desviación 3%
        retornos_base = np.clip(retornos_base, 5, 18)  # Limitar al rango especificado
        
        volatilidades_base = np.random.gamma(2, 5, n_assets)  # Distribución gamma para volatilidad
        volatilidades_base = np.clip(volatilidades_base, 7, 30)
        
        # Correlación inversa entre retorno y volatilidad (más realista)
        correlation_factor = 0.3
        volatilidades = volatilidades_base - correlation_factor * (retornos_base - 11)
        volatilidades = np.clip(volatilidades, 7, 30)
        
        sample_data = {
            'activo_id': [f'A{str(i+1).zfill(3)}' for i in range(n_assets)],
            'retorno_esperado': retornos_base,
            'volatilidad': volatilidades,
            'beta': np.random.triangular(0.5, 1.0, 1.7, n_assets),  # Distribución triangular
            'liquidez_score': np.random.choice(range(1, 11), n_assets, p=[0.05]*3 + [0.15]*4 + [0.2]*3),
            'sector': np.random.choice(range(1, 6), n_assets, p=[0.25, 0.20, 0.15, 0.25, 0.15]),
            'precio_accion': np.random.lognormal(4.5, 0.5, n_assets) * 10,  # Log-normal para precios
            'min_inversion': np.random.triangular(2000, 4000, 10500, n_assets)
        }
        
        # Asegurar rangos correctos
        sample_data['precio_accion'] = np.clip(sample_data['precio_accion'], 50, 350)
        
        self.data = pd.DataFrame(sample_data)
        
    def _procesar_y_enriquecer_datos(self):
        """Procesa y enriquece los datos con métricas avanzadas"""
        # Convertir a decimales si están en porcentaje
        if self.data['retorno_esperado'].max() > 2:
            self.data['retorno_esperado'] = self.data['retorno_esperado'] / 100
            
        if self.data['volatilidad'].max() > 2:
            self.data['volatilidad'] = self.data['volatilidad'] / 100
        
        # Calcular métricas avanzadas
        self.data['sharpe_ratio'] = self.data['retorno_esperado'] / self.data['volatilidad']
        self.data['efficiency_score'] = self.data['retorno_esperado'] / (self.data['volatilidad'] ** 2)
        self.data['utility_individual'] = self.data['retorno_esperado'] - self.LAMBDA_RIESGO * (self.data['volatilidad'] ** 2)
        self.data['return_per_dollar'] = self.data['retorno_esperado'] / self.data['precio_accion'] * 1000
        self.data['risk_adjusted_return'] = self.data['retorno_esperado'] / (self.data['beta'] * self.data['volatilidad'])
        
        # Score compuesto élite
        self.data['elite_score'] = (
            self.data['utility_individual'] * 0.30 +
            self.data['sharpe_ratio'] * 0.25 +
            self.data['efficiency_score'] * 0.20 +
            (self.data['liquidez_score'] / 10) * 0.15 +
            self.data['risk_adjusted_return'] * 0.10
        )
        
        # Normalizar scores
        self.data['elite_score_norm'] = (self.data['elite_score'] - self.data['elite_score'].min()) / (self.data['elite_score'].max() - self.data['elite_score'].min())
        
        # Agregar información sectorial
        self.data['sector_nombre'] = self.data['sector'].map(self.SECTORES)
        
    def estrategia_elite_multiobjetivo(self):
        """
        Estrategia élite que combina múltiples enfoques de optimización
        """
        print(f"\n🚀 INICIANDO ESTRATEGIA ÉLITE MULTI-OBJETIVO 🚀")
        print("="*65)
        
        self.start_time = datetime.now()
        
        # Fase 1: Pre-filtrado inteligente
        candidatos = self._pre_filtrado_avanzado()
        print(f"🔍 Fase 1: {len(candidatos)} candidatos élite seleccionados")
        
        # Fase 2: Generar múltiples portafolios con diferentes estrategias
        portfolios_candidatos = []
        
        # Estrategia 1: Máximo Sharpe Ratio
        portfolio_sharpe = self._construir_portfolio_objetivo(candidatos, 'sharpe_ratio')
        if portfolio_sharpe and len(portfolio_sharpe) >= self.MIN_ASSETS:
            portfolios_candidatos.append(('Máximo Sharpe', portfolio_sharpe))
        
        # Estrategia 2: Máxima utilidad
        portfolio_utility = self._construir_portfolio_objetivo(candidatos, 'utility_individual')
        if portfolio_utility and len(portfolio_utility) >= self.MIN_ASSETS:
            portfolios_candidatos.append(('Máxima Utilidad', portfolio_utility))
        
        # Estrategia 3: Elite Score
        portfolio_elite = self._construir_portfolio_objetivo(candidatos, 'elite_score')
        if portfolio_elite and len(portfolio_elite) >= self.MIN_ASSETS:
            portfolios_candidatos.append(('Elite Score', portfolio_elite))
        
        # Estrategia 4: Balanceado sectorial
        portfolio_balanced = self._construir_portfolio_balanceado(candidatos)
        if portfolio_balanced and len(portfolio_balanced) >= self.MIN_ASSETS:
            portfolios_candidatos.append(('Balanceado Sectorial', portfolio_balanced))
        
        # Estrategia 5: Alto retorno controlado
        portfolio_high_return = self._construir_portfolio_alto_retorno(candidatos)
        if portfolio_high_return and len(portfolio_high_return) >= self.MIN_ASSETS:
            portfolios_candidatos.append(('Alto Retorno', portfolio_high_return))
        
        print(f"🎯 Fase 2: {len(portfolios_candidatos)} portafolios candidatos generados")
        
        # Fase 3: Evaluación y selección del mejor
        mejor_portfolio = None
        mejor_score = -999999
        mejor_estrategia = ""
        
        for estrategia, portfolio in portfolios_candidatos:
            score = self._calcular_score_torneo(portfolio)
            print(f"   📊 {estrategia}: Score = {score:.0f}")
            
            if score > mejor_score:
                mejor_score = score
                mejor_portfolio = portfolio
                mejor_estrategia = estrategia
        
        print(f"\n🏆 MEJOR ESTRATEGIA: {mejor_estrategia} (Score: {mejor_score:.0f})")
        
        # Fase 4: Optimización local avanzada
        if mejor_portfolio and self.USE_ADVANCED_OPTIMIZATION:
            print("🔧 Fase 4: Aplicando optimización local avanzada...")
            mejor_portfolio = self._optimizacion_local_avanzada(mejor_portfolio)
            score_final = self._calcular_score_torneo(mejor_portfolio)
            print(f"   ✨ Score después de optimización: {score_final:.0f}")
        
        self.elite_portfolio = mejor_portfolio
        self.end_time = datetime.now()
        
        return mejor_portfolio
    
    def _pre_filtrado_avanzado(self):
        """Pre-filtrado inteligente de activos prometedores"""
        # Criterios base más estrictos
        criterios_base = (
            (self.data['utility_individual'] > self.data['utility_individual'].quantile(0.2)) &  # Top 80%
            (self.data['beta'] <= 1.6) &  # Beta razonable
            (self.data['liquidez_score'] >= 5) &  # Liquidez mínima
            (self.data['retorno_esperado'] > 0.06) &  # Retorno mínimo decente
            (self.data['volatilidad'] < 0.35)  # Volatilidad no extrema
        )
        
        candidatos_base = self.data[criterios_base].copy()
        
        # Seleccionar top performers por sector
        candidatos_finales = []
        
        for sector in range(1, 6):
            sector_data = candidatos_base[candidatos_base['sector'] == sector]
            if not sector_data.empty:
                # Top 30% por elite_score en cada sector
                n_top = max(2, int(len(sector_data) * 0.3))
                top_sector = sector_data.nlargest(n_top, 'elite_score')
                candidatos_finales.append(top_sector)
        
        if candidatos_finales:
            resultado = pd.concat(candidatos_finales).drop_duplicates()
        else:
            # Fallback: mejores en general
            resultado = candidatos_base.nlargest(20, 'elite_score')
        
        return resultado
    
    def _construir_portfolio_objetivo(self, assets, objetivo_columna):
        """Construye portfolio optimizando una métrica específica"""
        portfolio = []
        assets_sorted = assets.sort_values(objetivo_columna, ascending=False)
        
        total_investment = 0
        sector_investments = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        
        for _, asset in assets_sorted.iterrows():
            if len(portfolio) >= 20:  # Límite práctico
                break
                
            # Verificar restricción sectorial
            max_sector_investment = self.PRESUPUESTO * self.MAX_SECTOR_WEIGHT
            available_sector_budget = max_sector_investment - sector_investments[asset['sector']]
            
            if available_sector_budget < asset['min_inversion']:
                continue
            
            # Calcular inversión agresiva
            multiplier = 2 + self.AGRESIVIDAD * 1.5
            target_investment = min(
                available_sector_budget,
                self.PRESUPUESTO - total_investment,
                asset['min_inversion'] * multiplier
            )
            
            shares = int(target_investment / asset['precio_accion'])
            actual_investment = shares * asset['precio_accion']
            
            if actual_investment >= asset['min_inversion']:
                portfolio.append({
                    'activo_id': asset['activo_id'],
                    'shares': shares,
                    'investment': actual_investment,
                    'retorno_esperado': asset['retorno_esperado'],
                    'volatilidad': asset['volatilidad'],
                    'beta': asset['beta'],
                    'sector': asset['sector'],
                    'precio_accion': asset['precio_accion'],
                    'min_inversion': asset['min_inversion'],
                    'liquidez_score': asset['liquidez_score'],
                    'elite_score': asset['elite_score']
                })
                
                total_investment += actual_investment
                sector_investments[asset['sector']] += actual_investment
                
                if total_investment >= self.PRESUPUESTO * 0.95:
                    break
        
        return portfolio if len(portfolio) >= self.MIN_ASSETS else None
    
    def _construir_portfolio_balanceado(self, assets):
        """Construye portfolio balanceado por sectores"""
        portfolio = []
        target_per_sector = self.PRESUPUESTO * 0.20  # 20% por sector
        
        for sector in range(1, 6):
            sector_assets = assets[assets['sector'] == sector].sort_values('elite_score', ascending=False)
            sector_budget = target_per_sector
            sector_investment = 0
            
            for _, asset in sector_assets.iterrows():
                if sector_investment >= sector_budget:
                    break
                
                remaining_budget = sector_budget - sector_investment
                target_investment = min(remaining_budget, asset['min_inversion'] * 4)
                
                shares = int(target_investment / asset['precio_accion'])
                actual_investment = shares * asset['precio_accion']
                
                if actual_investment >= asset['min_inversion']:
                    portfolio.append({
                        'activo_id': asset['activo_id'],
                        'shares': shares,
                        'investment': actual_investment,
                        'retorno_esperado': asset['retorno_esperado'],
                        'volatilidad': asset['volatilidad'],
                        'beta': asset['beta'],
                        'sector': asset['sector'],
                        'precio_accion': asset['precio_accion'],
                        'min_inversion': asset['min_inversion'],
                        'liquidez_score': asset['liquidez_score'],
                        'elite_score': asset['elite_score']
                    })
                    
                    sector_investment += actual_investment
        
        return portfolio if len(portfolio) >= self.MIN_ASSETS else None
    
    def _construir_portfolio_alto_retorno(self, assets):
        """Construye portfolio priorizando alto retorno con riesgo controlado"""
        # Filtrar activos de alto retorno
        high_return_assets = assets[
            (assets['retorno_esperado'] > assets['retorno_esperado'].quantile(0.7)) &
            (assets['volatilidad'] < assets['volatilidad'].quantile(0.8)) &
            (assets['beta'] <= 1.4)
        ].sort_values('retorno_esperado', ascending=False)
        
        return self._construir_portfolio_objetivo(high_return_assets, 'retorno_esperado')
    
    def _optimizacion_local_avanzada(self, portfolio_inicial):
        """Optimización local avanzada usando múltiples técnicas"""
        if not portfolio_inicial:
            return portfolio_inicial
        
        mejor_portfolio = portfolio_inicial.copy()
        mejor_score = self._calcular_score_torneo(mejor_portfolio)
        
        print(f"   🔄 Score inicial: {mejor_score:.0f}")
        
        # Técnica 1: Hill Climbing con múltiples reinicios
        for restart in range(3):
            portfolio_actual = self._hill_climbing_optimization(mejor_portfolio.copy())
            score_actual = self._calcular_score_torneo(portfolio_actual)
            
            if score_actual > mejor_score:
                mejor_score = score_actual
                mejor_portfolio = portfolio_actual
                print(f"   📈 Mejora encontrada (restart {restart+1}): {score_actual:.0f}")
        
        # Técnica 2: Rebalanceo inteligente de pesos
        portfolio_rebalanceado = self._rebalancear_pesos(mejor_portfolio.copy())
        score_rebalanceado = self._calcular_score_torneo(portfolio_rebalanceado)
        
        if score_rebalanceado > mejor_score:
            mejor_score = score_rebalanceado
            mejor_portfolio = portfolio_rebalanceado
            print(f"   ⚖️ Mejora por rebalanceo: {score_rebalanceado:.0f}")
        
        # Técnica 3: Intercambio de activos
        portfolio_intercambiado = self._intercambio_activos(mejor_portfolio.copy())
        score_intercambiado = self._calcular_score_torneo(portfolio_intercambiado)
        
        if score_intercambiado > mejor_score:
            mejor_score = score_intercambiado
            mejor_portfolio = portfolio_intercambiado
            print(f"   🔄 Mejora por intercambio: {score_intercambiado:.0f}")
        
        return mejor_portfolio
    
    def _hill_climbing_optimization(self, portfolio):
        """Hill climbing para optimización local"""
        max_iterations = 50
        iteration = 0
        improved = True
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            current_score = self._calcular_score_torneo(portfolio)
            
            # Intentar incrementar inversión en cada activo
            for i, asset in enumerate(portfolio):
                if asset['investment'] >= self.PRESUPUESTO * 0.15:  # No más del 15% en un activo
                    continue
                
                # Calcular incremento posible
                increment_budget = min(50000, self.PRESUPUESTO * 0.02)  # 2% del presupuesto
                additional_shares = int(increment_budget / asset['precio_accion'])
                
                if additional_shares > 0:
                    # Guardar estado original
                    original_shares = asset['shares']
                    original_investment = asset['investment']
                    
                    # Aplicar incremento
                    portfolio[i]['shares'] += additional_shares
                    portfolio[i]['investment'] += additional_shares * asset['precio_accion']
                    
                    # Verificar si es válido y mejor
                    if (self._es_portfolio_valido(portfolio) and 
                        self._calcular_score_torneo(portfolio) > current_score):
                        improved = True
                        break
                    else:
                        # Revertir cambios
                        portfolio[i]['shares'] = original_shares
                        portfolio[i]['investment'] = original_investment
        
        return portfolio
    
    def _rebalancear_pesos(self, portfolio):
        """Rebalancea los pesos del portfolio para optimizar score"""
        total_investment = sum(asset['investment'] for asset in portfolio)
        
        # Calcular pesos óptimos usando optimización matemática
        n_assets = len(portfolio)
        
        # Función objetivo para scipy.optimize
        def objective(weights):
            # Actualizar inversiones según pesos
            portfolio_temp = portfolio.copy()
            total_budget_used = 0
            
            for i, w in enumerate(weights):
                target_investment = w * self.PRESUPUESTO
                shares = int(target_investment / portfolio_temp[i]['precio_accion'])
                actual_investment = shares * portfolio_temp[i]['precio_accion']
                
                portfolio_temp[i]['shares'] = shares
                portfolio_temp[i]['investment'] = actual_investment
                total_budget_used += actual_investment
            
            # Penalizar si excede presupuesto o no cumple restricciones
            if not self._es_portfolio_valido(portfolio_temp):
                return 1e6
            
            return -self._calcular_score_torneo(portfolio_temp)  # Negativo porque minimize busca mínimo
        
        # Restricciones
        def constraint_sum_weights(weights):
            return 1.0 - sum(weights)
        
        def constraint_min_investment(weights):
            violations = 0
            for i, w in enumerate(weights):
                target_investment = w * self.PRESUPUESTO
                if target_investment > 0 and target_investment < portfolio[i]['min_inversion']:
                    violations += 1
            return -violations  # Negativo para que sea <= 0
        
        # Configurar optimización
        initial_weights = np.array([asset['investment'] for asset in portfolio])
        initial_weights = initial_weights / sum(initial_weights)
        
        bounds = [(0, 0.3) for _ in range(n_assets)]  # Max 30% por activo
        
        constraints = [
            {'type': 'eq', 'fun': constraint_sum_weights},
            {'type': 'ineq', 'fun': constraint_min_investment}
        ]
        
        try:
            result = minimize(objective, initial_weights, method='SLSQP', 
                            bounds=bounds, constraints=constraints, 
                            options={'maxiter': 100})
            
            if result.success:
                # Aplicar pesos optimizados
                for i, w in enumerate(result.x):
                    target_investment = w * self.PRESUPUESTO
                    shares = int(target_investment / portfolio[i]['precio_accion'])
                    portfolio[i]['shares'] = shares
                    portfolio[i]['investment'] = shares * portfolio[i]['precio_accion']
        except:
            pass  # Si falla la optimización, mantener portfolio original
        
        return portfolio
    
    def _intercambio_activos(self, portfolio):
        """Intenta intercambiar activos del portfolio por mejores opciones"""
        # Obtener activos no utilizados con buen score
        activos_en_portfolio = {asset['activo_id'] for asset in portfolio}
        activos_disponibles = self.data[
            ~self.data['activo_id'].isin(activos_en_portfolio) &
            (self.data['elite_score'] > self.data['elite_score'].quantile(0.6))
        ].sort_values('elite_score', ascending=False)
        
        mejor_portfolio = portfolio.copy()
        mejor_score = self._calcular_score_torneo(portfolio)
        
        # Intentar intercambiar cada activo del portfolio
        for i, asset_actual in enumerate(portfolio):
            for _, asset_candidato in activos_disponibles.iterrows():
                if asset_candidato['sector'] != asset_actual['sector']:
                    continue  # Mantener distribución sectorial
                
                # Crear portfolio temporal con intercambio
                portfolio_temp = portfolio.copy()
                
                # Calcular nueva inversión
                budget_disponible = asset_actual['investment']
                shares_nuevas = int(budget_disponible / asset_candidato['precio_accion'])
                inversion_nueva = shares_nuevas * asset_candidato['precio_accion']
                
                if inversion_nueva >= asset_candidato['min_inversion']:
                    portfolio_temp[i] = {
                        'activo_id': asset_candidato['activo_id'],
                        'shares': shares_nuevas,
                        'investment': inversion_nueva,
                        'retorno_esperado': asset_candidato['retorno_esperado'],
                        'volatilidad': asset_candidato['volatilidad'],
                        'beta': asset_candidato['beta'],
                        'sector': asset_candidato['sector'],
                        'precio_accion': asset_candidato['precio_accion'],
                        'min_inversion': asset_candidato['min_inversion'],
                        'liquidez_score': asset_candidato['liquidez_score'],
                        'elite_score': asset_candidato['elite_score']
                    }
                    
                    if (self._es_portfolio_valido(portfolio_temp)):
                        score_temp = self._calcular_score_torneo(portfolio_temp)
                        if score_temp > mejor_score:
                            mejor_score = score_temp
                            mejor_portfolio = portfolio_temp.copy()
        
        return mejor_portfolio
    
    def _es_portfolio_valido(self, portfolio):
        """Verifica si el portfolio cumple todas las restricciones"""
        if not portfolio or len(portfolio) < self.MIN_ASSETS:
            return False
        
        total_investment = sum(asset['investment'] for asset in portfolio)
        
        # Verificar presupuesto
        if total_investment > self.PRESUPUESTO:
            return False
        
        # Verificar restricciones sectoriales
        sector_investments = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for asset in portfolio:
            sector_investments[asset['sector']] += asset['investment']
        
        for sector, investment in sector_investments.items():
            if investment / total_investment > self.MAX_SECTOR_WEIGHT:
                return False
        
        # Verificar beta
        weighted_beta = sum(asset['beta'] * asset['investment'] for asset in portfolio) / total_investment
        if weighted_beta > self.MAX_BETA:
            return False
        
        # Verificar inversión mínima
        for asset in portfolio:
            if asset['investment'] < asset['min_inversion']:
                return False
        
        return True
    
    def _calcular_score_torneo(self, portfolio):
        """Calcula el score del torneo según la fórmula oficial"""
        if not portfolio:
            return -999999
        
        total_investment = sum(asset['investment'] for asset in portfolio)
        if total_investment == 0:
            return -999999
        
        # Calcular métricas del portfolio
        weights = [asset['investment'] / total_investment for asset in portfolio]
        
        portfolio_return = sum(asset['retorno_esperado'] * weight 
                             for asset, weight in zip(portfolio, weights))
        
        portfolio_variance = sum(asset['volatilidad']**2 * weight**2 
                               for asset, weight in zip(portfolio, weights))
        
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Verificar restricciones para factor Fr
        constraints_met = self._verificar_restricciones_detalladas(portfolio)
        violations = sum(1 for constraint in constraints_met.values() if not constraint)
        
        if violations == 0:
            Fr = 1.0
        elif violations == 1:
            Fr = 0.8
        else:
            Fr = 0.6
        
        # Factor de tiempo (asumimos ejecución rápida)
        Ft = 1.5
        
        # Calcular utilidad y score final
        utilidad = portfolio_return - self.LAMBDA_RIESGO * portfolio_volatility
        
        # Bonificaciones adicionales
        diversification_bonus = 1.0 + (len(portfolio) - 5) * 0.02  # 2% por activo adicional
        quality_bonus = 1.0 + (sum(asset['liquidez_score'] for asset in portfolio) / len(portfolio) / 10) * 0.1
        
        score = 1000 * utilidad * Fr * Ft * diversification_bonus * quality_bonus
        
        return score
    
    def _verificar_restricciones_detalladas(self, portfolio):
        """Verifica restricciones y retorna estado detallado"""
        total_investment = sum(asset['investment'] for asset in portfolio)
        
        restrictions = {}
        
        # Presupuesto
        restrictions['presupuesto'] = total_investment <= self.PRESUPUESTO
        
        # Diversificación sectorial
        sector_investments = {i: 0 for i in range(1, 6)}
        for asset in portfolio:
            sector_investments[asset['sector']] += asset['investment']
        
        for sector in range(1, 6):
            sector_weight = sector_investments[sector] / total_investment if total_investment > 0 else 0
            restrictions[f'sector_{sector}'] = sector_weight <= self.MAX_SECTOR_WEIGHT
        
        # Número mínimo de activos
        restrictions['min_activos'] = len(portfolio) >= self.MIN_ASSETS
        
        # Beta máximo
        weighted_beta = sum(asset['beta'] * asset['investment'] for asset in portfolio) / total_investment
        restrictions['beta_maximo'] = weighted_beta <= self.MAX_BETA
        
        # Inversión mínima
        restrictions['inversion_minima'] = all(asset['investment'] >= asset['min_inversion'] for asset in portfolio)
        
        return restrictions
    
    def generar_reporte_campeon(self):
        """Genera reporte detallado del portfolio campeón"""
        if not self.elite_portfolio:
            print("❌ No hay portfolio élite para reportar")
            return
        
        execution_time = (self.end_time - self.start_time).total_seconds() / 60
        
        # Calcular métricas
        total_investment = sum(asset['investment'] for asset in self.elite_portfolio)
        weights = [asset['investment'] / total_investment for asset in self.elite_portfolio]
        
        portfolio_return = sum(asset['retorno_esperado'] * weight 
                             for asset, weight in zip(self.elite_portfolio, weights))
        
        portfolio_variance = sum(asset['volatilidad']**2 * weight**2 
                               for asset, weight in zip(self.elite_portfolio, weights))
        
        portfolio_volatility = np.sqrt(portfolio_variance)
        portfolio_beta = sum(asset['beta'] * weight 
                           for asset, weight in zip(self.elite_portfolio, weights))
        
        score_final = self._calcular_score_torneo(self.elite_portfolio)
        
        print("\n" + "🏆" * 30)
        print("🚀 REPORTE DEL PORTFOLIO CAMPEÓN 🚀")
        print("🏆" * 30)
        
        # Métricas principales
        print(f"\n🎯 PUNTAJE FINAL DEL TORNEO: {score_final:.0f}")
        print(f"📈 RETORNO ESPERADO: {portfolio_return*100:.2f}%")
        print(f"📊 VOLATILIDAD: {portfolio_volatility*100:.2f}%")
        print(f"⚖️  BETA DEL PORTFOLIO: {portfolio_beta:.3f}")
        print(f"💎 UTILIDAD: {portfolio_return - self.LAMBDA_RIESGO * portfolio_volatility:.4f}")
        print(f"🔥 RATIO SHARPE: {portfolio_return/portfolio_volatility:.3f}")
        
        # Información de ejecución
        print(f"\n⏱️  TIEMPO DE EJECUCIÓN: {execution_time:.2f} minutos")
        print(f"💰 PRESUPUESTO UTILIZADO: S/.{total_investment:,.0f} ({total_investment/self.PRESUPUESTO*100:.1f}%)")
        print(f"💵 PRESUPUESTO RESTANTE: S/.{self.PRESUPUESTO - total_investment:,.0f}")
        print(f"🏗️  ACTIVOS EN PORTFOLIO: {len(self.elite_portfolio)}")
        
        # Distribución sectorial
        print(f"\n🏭 DISTRIBUCIÓN SECTORIAL:")
        sector_investments = {i: 0 for i in range(1, 6)}
        for asset in self.elite_portfolio:
            sector_investments[asset['sector']] += asset['investment']
        
        for sector in range(1, 6):
            if sector_investments[sector] > 0:
                weight = sector_investments[sector] / total_investment * 100
                status = "✅" if weight <= 30 else "❌"
                print(f"   {status} {self.SECTORES[sector]}: {weight:.1f}% (S/.{sector_investments[sector]:,.0f})")
        
        # Verificación de restricciones
        restrictions = self._verificar_restricciones_detalladas(self.elite_portfolio)
        print(f"\n✅ VERIFICACIÓN DE RESTRICCIONES:")
        restriction_names = {
            'presupuesto': 'Presupuesto Total',
            'min_activos': 'Mínimo de Activos',
            'beta_maximo': 'Beta Máximo',
            'inversion_minima': 'Inversión Mínima',
        }
        
        for key, value in restrictions.items():
            if key in restriction_names:
                status = "✅" if value else "❌"
                print(f"   {status} {restriction_names[key]}")
        
        # Portfolio detallado
        print(f"\n📈 PORTFOLIO ÉLITE DETALLADO:")
        print("-" * 120)
        print(f"{'Activo':<8} {'Sector':<12} {'Acciones':<10} {'Inversión':<15} {'Peso':<8} {'Retorno':<8} {'Volat.':<8} {'Beta':<6} {'Score':<8}")
        print("-" * 120)
        
        # Ordenar por inversión
        sorted_portfolio = sorted(self.elite_portfolio, key=lambda x: x['investment'], reverse=True)
        
        for asset in sorted_portfolio:
            weight = asset['investment'] / total_investment * 100
            sector_name = self.SECTORES[asset['sector']]
            elite_score = asset.get('elite_score', 0)
            
            print(f"{asset['activo_id']:<8} {sector_name:<12} {asset['shares']:<10,} "
                  f"S/.{asset['investment']:<12,.0f} {weight:<6.1f}% "
                  f"{asset['retorno_esperado']*100:<6.1f}% {asset['volatilidad']*100:<6.1f}% "
                  f"{asset['beta']:<6.2f} {elite_score:<8.3f}")
        
        print("-" * 120)
        print(f"{'TOTAL':<21} {sum(a['shares'] for a in sorted_portfolio):<10,} "
              f"S/.{total_investment:<12,.0f} {'100.0%':<6} "
              f"{portfolio_return*100:<6.1f}% {portfolio_volatility*100:<6.1f}% {portfolio_beta:<6.2f}")
        
        # Recomendaciones finales
        print(f"\n💡 RECOMENDACIONES FINALES:")
        if score_final > 5000:
            print("🔥 ¡PORTAFOLIO EXCEPCIONAL! Muy alta probabilidad de ganar")
        elif score_final > 4000:
            print("🚀 ¡EXCELENTE PORTAFOLIO! Alta probabilidad de ganar")
        elif score_final > 3000:
            print("⭐ Buen portafolio competitivo")
        else:
            print("📈 Portafolio mejorable - considera ajustar parámetros")
        
        print(f"\n🎉 ¡PORTFOLIO LISTO PARA DOMINAR EL TORNEO! 🎉")
    
    def crear_visualizaciones(self):
        """Crea visualizaciones del portfolio élite"""
        if not self.elite_portfolio:
            print("❌ No hay portfolio para visualizar")
            return
        
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🏆 ANÁLISIS DEL PORTFOLIO ÉLITE - OPTIMABATTLE ARENA', fontsize=16, fontweight='bold')
        
        # Preparar datos
        portfolio_df = pd.DataFrame(self.elite_portfolio)
        total_investment = portfolio_df['investment'].sum()
        portfolio_df['weight'] = portfolio_df['investment'] / total_investment
        
        # 1. Distribución de pesos por activo
        colors = plt.cm.Set3(np.linspace(0, 1, len(portfolio_df)))
        wedges, texts, autotexts = ax1.pie(portfolio_df['weight'], labels=portfolio_df['activo_id'], 
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Distribución de Pesos por Activo', fontweight='bold')
        
        # 2. Inversión por sector
        sector_investment = portfolio_df.groupby('sector')['investment'].sum()
        sector_names = [self.SECTORES[s] for s in sector_investment.index]
        bars = ax2.bar(sector_names, sector_investment.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        ax2.set_title('Inversión por Sector', fontweight='bold')
        ax2.set_ylabel('Inversión (S/.)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Agregar valores en las barras
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'S/.{height:,.0f}', ha='center', va='bottom')
        
        # 3. Retorno vs Volatilidad (tamaño = peso)
        scatter = ax3.scatter(portfolio_df['volatilidad']*100, portfolio_df['retorno_esperado']*100,
                             s=portfolio_df['weight']*2000, alpha=0.7, c=portfolio_df['beta'], 
                             cmap='viridis', edgecolors='black', linewidth=1)
        ax3.set_xlabel('Volatilidad (%)')
        ax3.set_ylabel('Retorno Esperado (%)')
        ax3.set_title('Retorno vs Volatilidad (tamaño=peso, color=beta)', fontweight='bold')
        plt.colorbar(scatter, ax=ax3, label='Beta')
        
        # Agregar etiquetas
        for i, row in portfolio_df.iterrows():
            ax3.annotate(row['activo_id'], 
                        (row['volatilidad']*100, row['retorno_esperado']*100),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 4. Comparación de métricas clave
        metrics = ['Retorno', 'Volatilidad', 'Beta', 'Liquidez']
        portfolio_metrics = [
            portfolio_df['retorno_esperado'].mean() * 100,
            portfolio_df['volatilidad'].mean() * 100,
            portfolio_df['beta'].mean(),
            portfolio_df['liquidez_score'].mean()
        ]
        
        # Benchmarks (valores típicos del mercado)
        benchmark_metrics = [10, 15, 1.0, 7]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, portfolio_metrics, width, label='Portfolio Élite', 
                       color='#FF6B6B', alpha=0.8)
        bars2 = ax4.bar(x + width/2, benchmark_metrics, width, label='Benchmark Mercado', 
                       color='#4ECDC4', alpha=0.8)
        
        ax4.set_xlabel('Métricas')
        ax4.set_ylabel('Valores')
        ax4.set_title('Portfolio Élite vs Benchmark', fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics)
        ax4.legend()
        
        # Agregar valores en las barras
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def ejecutar_optimizacion_completa(self):
        """Ejecuta el proceso completo de optimización élite"""
        print("🏆" * 25)
        print("🚀 ELITE OPTIMABATTLE ARENA OPTIMIZER 🚀")
        print("🏆" * 25)
        
        # Paso 1: Cargar y procesar datos
        self.cargar_y_procesar_datos()
        
        # Paso 2: Mostrar resumen de datos
        self._mostrar_resumen_datos()
        
        # Paso 3: Ejecutar estrategia élite
        portfolio = self.estrategia_elite_multiobjetivo()
        
        if portfolio:
            # Paso 4: Generar reporte completo
            self.generar_reporte_campeon()
            
            # Paso 5: Crear visualizaciones
            crear_visualizaciones = input("\n¿Crear visualizaciones? (s/n): ").lower().strip()
            if crear_visualizaciones == 's':
                self.crear_visualizaciones()
            
            return portfolio
        else:
            print("❌ No se pudo generar un portfolio válido")
            return None
    
    def _mostrar_resumen_datos(self):
        """Muestra resumen estadístico de los datos"""
        print(f"\n📊 ANÁLISIS DE DATOS:")
        print("-" * 50)
        print(f"Total de activos: {len(self.data)}")
        print(f"Retorno promedio: {self.data['retorno_esperado'].mean()*100:.2f}%")
        print(f"Volatilidad promedio: {self.data['volatilidad'].mean()*100:.2f}%")
        print(f"Beta promedio: {self.data['beta'].mean():.2f}")
        print(f"Elite score promedio: {self.data['elite_score'].mean():.3f}")
        
        print(f"\n🏭 Distribución sectorial:")
        sector_dist = self.data['sector_nombre'].value_counts().sort_index()
        for sector, count in sector_dist.items():
            print(f"   {sector}: {count} activos")


def main():
    """Función principal para ejecutar el optimizador élite"""
    try:
        # Crear instancia del optimizador élite
        optimizer = EliteOptimaBattleOptimizer("Ronda1.xlsx")
        
        # Configurar parámetros avanzados
        print(f"\n⚙️  CONFIGURACIÓN AVANZADA:")
        
        agresividad = input("Nivel de agresividad (1-4, default=3): ").strip()
        if agresividad and agresividad.isdigit():
            optimizer.AGRESIVIDAD = int(agresividad)
        
        usar_optimizacion = input("¿Usar optimización avanzada? (s/n, default=s): ").strip().lower()
        if usar_optimizacion == 'n':
            optimizer.USE_ADVANCED_OPTIMIZATION = False
        
        # Ejecutar optimización completa
        portfolio = optimizer.ejecutar_optimizacion_completa()
        
        if portfolio:
            print(f"\n🎊 ¡OPTIMIZACIÓN COMPLETADA EXITOSAMENTE! 🎊")
            print(f"Portfolio con {len(portfolio)} activos generado")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Optimización interrumpida por el usuario")
    except Exception as e:
        print(f"❌ Error durante la optimización: {str(e)}")
        print("Verifica que tengas instaladas las librerías necesarias:")
        print("pip install pandas numpy scipy matplotlib seaborn openpyxl")


if __name__ == "__main__":
    main()