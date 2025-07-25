from manim import *
import numpy as np
from scipy.ndimage import gaussian_filter1d

# Configuración global
config.background_color = "#0f0f23"

class IntroTitle(Scene):
    """Escena 1: Introducción del estudio con animaciones y elementos visuales"""

    def construct(self):
        # Fondo con gradiente sutil
        background = Rectangle(
            width=config.frame_width,
            height=config.frame_height,
            fill_color=[DARK_BLUE, BLUE],
            fill_opacity=0.1
        )
        self.add(background)
        
        # Crear elementos decorativos de fondo
        self.create_background_elements()
        
        # Título principal con efecto elegante
        title = Text("Reinforcement Learning", font_size=40, color=BLUE, weight=BOLD)
        title.set_stroke(WHITE, width=1)
        
        subtitle = Text("como Método de Optimización", font_size=32, color=TEAL)
        subtitle2 = Text("en la Gestión de Políticas de Inversión", font_size=28, color=GREEN)
        subtitle3 = Text("en Datos Macroeconómicos del Perú (2015–2024)", font_size=24, color=GRAY)
        
        # Línea decorativa
        line = Line(LEFT * 3, RIGHT * 3, color=GOLD, stroke_width=3)
        
        # Autora con estilo elegante
        author = Text("Autora: Etzel Yuliza Peralta Lopez", font_size=20, color=GRAY)
        author_frame = SurroundingRectangle(author, color=GOLD, buff=0.3, corner_radius=0.1)
        author_group = VGroup(author_frame, author)
        
        # Agrupar elementos del título
        title_group = VGroup(title, subtitle, subtitle2, subtitle3, line, author_group).arrange(DOWN, buff=0.4)
        
        # Animaciones del título con efectos suaves
        self.play(
            DrawBorderThenFill(title, run_time=1.5),
            rate_func=smooth
        )
        self.wait(0.5)
        
        self.play(
            FadeIn(subtitle, shift=DOWN * 0.5),
            run_time=1
        )
        self.wait(0.3)
        
        self.play(
            FadeIn(subtitle2, shift=DOWN * 0.5),
            run_time=1
        )
        self.wait(0.3)
        
        self.play(
            FadeIn(subtitle3, shift=DOWN * 0.5),
            run_time=1
        )
        self.wait(0.3)
        
        self.play(
            Create(line),
            run_time=1
        )
        self.wait(0.3)
        
        self.play(
            DrawBorderThenFill(author_group),
            run_time=1
        )
        self.wait(2)
        
        # Transición elegante
        self.play(
            FadeOut(title_group, shift=UP * 2),
            run_time=1.5
        )
        self.wait(0.5)
        
        # Crear objetivo con elementos visuales
        self.create_objective_section()
        
    def create_background_elements(self):
        """Crear elementos decorativos de fondo"""
        # Círculos decorativos
        circles = []
        for i in range(8):
            circle = Circle(
                radius=0.1 + i * 0.05,
                color=BLUE,
                fill_opacity=0.1 - i * 0.01,
                stroke_width=1
            )
            circle.move_to(
                RIGHT * (3 - i * 0.8) + UP * (2 - i * 0.5)
            )
            circles.append(circle)
        
        circle_group = VGroup(*circles)
        self.add(circle_group)
        
        # Líneas decorativas
        lines = []
        for i in range(3):
            line = Line(
                LEFT * 6 + UP * (1 - i),
                RIGHT * 6 + UP * (1 - i),
                color=GRAY,
                stroke_width=0.5,
                stroke_opacity=0.3
            )
            lines.append(line)
        
        line_group = VGroup(*lines)
        self.add(line_group)
        
    def create_objective_section(self):
        """Crear sección de objetivos con elementos visuales"""
        # Icono de objetivo (diana)
        target = Circle(radius=0.3, color=YELLOW, fill_opacity=0.2)
        target_center = Circle(radius=0.15, color=YELLOW, fill_opacity=0.5)
        target_inner = Circle(radius=0.05, color=RED, fill_opacity=0.8)
        target_icon = VGroup(target, target_center, target_inner)
        
        # Título del objetivo
        objetivo_title = Text("Objetivo del Estudio", font_size=36, color=YELLOW, weight=BOLD)
        objetivo_title.set_stroke(WHITE, width=0.5)
        
        # Crear header con icono
        header = VGroup(target_icon, objetivo_title).arrange(RIGHT, buff=0.5)
        header.to_edge(UP, buff=1)
        
        # Textos del objetivo con mejor formato
        objetivo_texts = [
            "• Proponer el uso de algoritmos de Aprendizaje por Refuerzo (Q-Learning)",
            "• Optimizar decisiones de inversión financiera basadas en variables macroeconómicas",
            "• Analizar datos del Perú en el período 2015-2024",
            "• Comparar desempeño con estrategias tradicionales (Buy & Hold y Markowitz)"
        ]
        
        # Crear elementos visuales para cada punto
        objective_items = []
        for i, text in enumerate(objetivo_texts):
            # Bullet point personalizado
            bullet = RegularPolygon(n=6, radius=0.1, color=TEAL, fill_opacity=0.8)
            
            # Texto
            text_obj = Text(text[2:], font_size=22, color=WHITE)  # Quitar el "• " inicial
            
            # Agrupar bullet y texto
            item = VGroup(bullet, text_obj).arrange(RIGHT, buff=0.3)
            objective_items.append(item)
        
        # Agrupar todos los elementos
        objective_group = VGroup(*objective_items).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        objective_group.next_to(header, DOWN, buff=0.8)
        
        # Crear cuadro de fondo para los objetivos
        background_rect = SurroundingRectangle(
            objective_group,
            color=BLUE,
            fill_opacity=0.1,
            stroke_width=2,
            corner_radius=0.2,
            buff=0.5
        )
        
        # Animaciones del objetivo
        self.play(
            DrawBorderThenFill(header),
            run_time=1.5
        )
        self.wait(0.5)
        
        self.play(
            Create(background_rect),
            run_time=1
        )
        self.wait(0.3)
        
        # Animar cada punto objetivo
        for i, item in enumerate(objective_items):
            self.play(
                FadeIn(item, shift=RIGHT * 0.5),
                run_time=0.8
            )
            self.wait(0.4)
        
        self.wait(2)
        
        # Crear elementos adicionales: gráficos decorativos
        self.create_decorative_charts()
        
        # Fade out final con efecto
        all_elements = VGroup(header, background_rect, objective_group)
        self.play(
            FadeOut(all_elements, shift=DOWN * 2),
            run_time=2
        )
        self.wait(1)
        
    def create_decorative_charts(self):
        """Crear gráficos decorativos para el contexto financiero"""
        # Gráfico de barras pequeño
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 10, 2],
            x_length=2,
            y_length=1.5,
            axis_config={"stroke_width": 1, "stroke_color": GRAY}
        ).to_corner(DR)
        
        # Barras del gráfico
        bars = []
        heights = [3, 7, 5, 9, 6]
        for i, height in enumerate(heights):
            bar = Rectangle(
                width=0.3,
                height=height * 0.15,
                color=BLUE,
                fill_opacity=0.7
            )
            bar.next_to(axes.c2p(i + 0.5, 0), UP, buff=0)
            bars.append(bar)
        
        chart = VGroup(axes, *bars)
        
        # Línea de tendencia
        trend_line = Line(
            LEFT * 2 + UP * 0.5,
            RIGHT * 2 + UP * 1.5,
            color=GREEN,
            stroke_width=2
        ).to_corner(DL)
        
        # Puntos de datos
        points = []
        for i in range(5):
            point = Dot(
                point=trend_line.point_from_proportion(i / 4),
                color=YELLOW,
                radius=0.05
            )
            points.append(point)
        
        trend_chart = VGroup(trend_line, *points)
        
        # Animar gráficos decorativos
        self.play(
            Create(chart),
            Create(trend_chart),
            run_time=1.5
        )
        self.wait(1)
        
        # Remover gráficos decorativos
        self.play(
            FadeOut(chart),
            FadeOut(trend_chart),
            run_time=1
        )

class ProblemaScene(Scene):
    """Escena que muestra la comparación entre métodos tradicionales y Reinforcement Learning"""
    
    def construct(self):
        # Fondo elegante
        self.create_background()
        
        # Título principal
        title = Text("El Problema", font_size=48, color=BLUE, weight=BOLD)
        title.set_stroke(WHITE, width=1)
        title.to_edge(UP, buff=1)
        
        # Animación del título
        self.play(
            DrawBorderThenFill(title),
            run_time=1.5
        )
        self.wait(1)
        
        # Crear las dos secciones
        self.create_traditional_methods()
        self.create_vs_arrow()
        self.create_reinforcement_learning()
        
        # Fade out final
        self.wait(2)
        self.play(FadeOut(Group(*self.mobjects)), run_time=2)
    
    def create_background(self):
        """Crear fondo con elementos decorativos similar a la introducción"""
        # Fondo con gradiente elegante
        background = Rectangle(
            width=config.frame_width,
            height=config.frame_height,
            fill_color=[DARK_BLUE, BLUE],
            fill_opacity=0.1
        )
        self.add(background)
        
        # Círculos decorativos de fondo (como en la introducción)
        circles = []
        for i in range(12):
            circle = Circle(
                radius=0.08 + i * 0.03,
                color=BLUE,
                fill_opacity=0.08 - i * 0.005,
                stroke_width=0.5,
                stroke_opacity=0.3
            )
            circle.move_to(
                RIGHT * (4 - i * 0.7) + UP * (1.5 - i * 0.3)
            )
            circles.append(circle)
        
        # Círculos en el lado opuesto
        for i in range(8):
            circle = Circle(
                radius=0.06 + i * 0.02,
                color=TEAL,
                fill_opacity=0.06 - i * 0.004,
                stroke_width=0.5,
                stroke_opacity=0.2
            )
            circle.move_to(
                LEFT * (3 - i * 0.5) + DOWN * (1 - i * 0.2)
            )
            circles.append(circle)
        
        circle_group = VGroup(*circles)
        self.add(circle_group)
        
        # Líneas decorativas sutiles
        lines = []
        for i in range(4):
            line = Line(
                LEFT * 7 + UP * (1.5 - i * 0.8),
                RIGHT * 7 + UP * (1.5 - i * 0.8),
                color=GRAY,
                stroke_width=0.3,
                stroke_opacity=0.15
            )
            lines.append(line)
        
        line_group = VGroup(*lines)
        self.add(line_group)
    
    def create_traditional_methods(self):
        """Crear sección de métodos tradicionales"""
        # Título de la sección
        traditional_title = Text(
            "Métodos Tradicionales", 
            font_size=30, 
            color=ORANGE,
            weight=BOLD
        )
        traditional_title.set_stroke(WHITE, width=0.5)
        traditional_title.move_to(LEFT * 3.5 + UP * 1.8)
        
        # Crear ejes para el gráfico tradicional
        axes_traditional = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 6, 2],
            x_length=2.5,
            y_length=2,
            axis_config={
                "stroke_width": 2,
                "stroke_color": GRAY,
                "include_tip": True,
                "tip_length": 0.15
            }
        )
        axes_traditional.move_to(LEFT * 3.5 + UP * 0.3)
        
        # Crear curva volátil (Buy & Hold)
        def volatile_function(x):
            return 1.5 + 1.2 * np.sin(x) + 0.4 * np.sin(3*x) + 0.2 * np.sin(5*x) + 0.15 * x
        
        volatile_curve = axes_traditional.plot(
            volatile_function,
            x_range=[0, 10],
            color=ORANGE,
            stroke_width=3
        )
        
        # Etiqueta Buy & Hold
        buy_hold_label = Text("Buy & Hold", font_size=18, color=ORANGE)
        buy_hold_label.move_to(LEFT * 3.5 + DOWN * 1.2)
        
        # Crear marco decorativo para la sección tradicional
        traditional_frame = SurroundingRectangle(
            VGroup(traditional_title, axes_traditional, buy_hold_label),
            color=ORANGE,
            stroke_width=2,
            corner_radius=0.15,
            buff=0.25
        )
        
        # Animaciones de métodos tradicionales
        self.play(
            Write(traditional_title),
            run_time=1
        )
        self.wait(0.5)
        
        self.play(
            Create(traditional_frame),
            run_time=1
        )
        self.wait(0.3)
        
        self.play(
            Create(axes_traditional),
            run_time=1
        )
        self.wait(0.5)
        
        self.play(
            Create(volatile_curve),
            run_time=2,
            rate_func=smooth
        )
        self.wait(0.3)
        
        self.play(
            FadeIn(buy_hold_label, shift=UP * 0.3),
            run_time=0.8
        )
        self.wait(1)
        
        # Guardar elementos para uso posterior
        self.traditional_group = VGroup(
            traditional_title, 
            axes_traditional, 
            volatile_curve, 
            buy_hold_label, 
            traditional_frame
        )
    
    def create_vs_arrow(self):
        """Crear flecha VS central"""
        # Texto VS
        vs_text = Text("VS", font_size=32, color=BLUE, weight=BOLD)
        vs_text.set_stroke(WHITE, width=1)
        vs_text.move_to(ORIGIN + DOWN * 0.2)
        
        # Flecha decorativa
        arrow = Arrow(
            start=LEFT * 1.2,
            end=RIGHT * 1.2,
            color=BLUE,
            stroke_width=3,
            max_tip_length_to_length_ratio=0.25
        )
        arrow.move_to(ORIGIN + UP * 0.3)
        
        # Círculo decorativo alrededor del VS
        vs_circle = Circle(
            radius=0.6,
            color=BLUE,
            stroke_width=2,
            fill_opacity=0.05
        )
        vs_circle.move_to(ORIGIN)
        
        # Animación del VS
        self.play(
            DrawBorderThenFill(vs_circle),
            run_time=1
        )
        self.wait(0.2)
        
        self.play(
            Write(vs_text),
            run_time=0.8
        )
        self.wait(0.3)
        
        self.play(
            Create(arrow),
            run_time=1
        )
        self.wait(0.5)
        
        # Efecto de pulso en el VS
        self.play(
            vs_circle.animate.scale(1.15),
            vs_text.animate.scale(1.1),
            run_time=0.4
        )
        self.play(
            vs_circle.animate.scale(1/1.15),
            vs_text.animate.scale(1/1.1),
            run_time=0.4
        )
        
        self.vs_group = VGroup(vs_circle, vs_text, arrow)
    
    def create_reinforcement_learning(self):
        """Crear sección de Reinforcement Learning"""
        # Título de la sección
        rl_title = Text(
            "Reinforcement Learning", 
            font_size=30, 
            color=GREEN,
            weight=BOLD
        )
        rl_title.set_stroke(WHITE, width=0.5)
        rl_title.move_to(RIGHT * 3.5 + UP * 1.8)
        
        # Crear ejes para el gráfico RL
        axes_rl = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 6, 2],
            x_length=2.5,
            y_length=2,
            axis_config={
                "stroke_width": 2,
                "stroke_color": GRAY,
                "include_tip": True,
                "tip_length": 0.15
            }
        )
        axes_rl.move_to(RIGHT * 3.5 + UP * 0.3)
        
        # Crear curva ascendente optimizada (Q-Learning)
        def optimized_function(x):
            return 0.8 + 0.6 * x + 0.08 * np.sin(2*x)
        
        optimized_curve = axes_rl.plot(
            optimized_function,
            x_range=[0, 10],
            color=GREEN,
            stroke_width=4
        )
        
        # Puntos de decisión en la curva
        decision_points = []
        for i in range(0, 11, 2):
            point = Dot(
                point=axes_rl.c2p(i, optimized_function(i)),
                color=YELLOW,
                radius=0.06
            )
            decision_points.append(point)
        
        decision_group = VGroup(*decision_points)
        
        # Etiqueta Q-Learning
        qlearning_label = Text("Q-Learning", font_size=18, color=GREEN)
        qlearning_label.move_to(RIGHT * 3.5 + DOWN * 1.2)
        
        # Crear marco decorativo para la sección RL
        rl_frame = SurroundingRectangle(
            VGroup(rl_title, axes_rl, qlearning_label),
            color=GREEN,
            stroke_width=2,
            corner_radius=0.15,
            buff=0.25
        )
        
        # Animaciones de Reinforcement Learning
        self.play(
            Write(rl_title),
            run_time=1
        )
        self.wait(0.5)
        
        self.play(
            Create(rl_frame),
            run_time=1
        )
        self.wait(0.3)
        
        self.play(
            Create(axes_rl),
            run_time=1
        )
        self.wait(0.5)
        
        self.play(
            Create(optimized_curve),
            run_time=2,
            rate_func=smooth
        )
        self.wait(0.3)
        
        # Animar puntos de decisión uno por uno
        for point in decision_points:
            self.play(
                FadeIn(point, scale=0.5),
                run_time=0.15
            )
        self.wait(0.5)
        
        self.play(
            FadeIn(qlearning_label, shift=UP * 0.3),
            run_time=0.8
        )
        self.wait(1)
        
        # Efecto de brillo en la curva optimizada
        self.play(
            optimized_curve.animate.set_stroke(width=5),
            decision_group.animate.set_color(WHITE),
            run_time=0.6
        )
        self.play(
            optimized_curve.animate.set_stroke(width=4),
            decision_group.animate.set_color(YELLOW),
            run_time=0.6
        )
        
        # Guardar elementos para uso posterior
        self.rl_group = VGroup(
            rl_title, 
            axes_rl, 
            optimized_curve, 
            decision_group,
            qlearning_label, 
            rl_frame
        )
        
        # Animación final de comparación
        self.create_comparison_effects()
    
    def create_comparison_effects(self):
        """Crear efectos finales de comparación"""
        # Crear flecha sutil que muestre la diferencia
        performance_arrow = Arrow(
            start=LEFT * 1.5 + DOWN * 1.8,
            end=RIGHT * 1.5 + DOWN * 1.8,
            color=YELLOW,
            stroke_width=2,
            max_tip_length_to_length_ratio=0.15
        )
        
        improvement_text = Text(
            "Optimización", 
            font_size=16, 
            color=YELLOW,
            weight=BOLD
        )
        improvement_text.next_to(performance_arrow, UP, buff=0.1)
        
        # Animación de mejora
        self.play(
            Create(performance_arrow),
            FadeIn(improvement_text, shift=DOWN * 0.2),
            run_time=1.2
        )
        self.wait(0.8)
        
        # Efecto de pulso final más sutil
        self.play(
            self.rl_group.animate.scale(1.03),
            run_time=0.4
        )
        self.play(
            self.rl_group.animate.scale(1/1.03),
            run_time=0.4
        )
        
        # Guardar elementos finales
        self.comparison_group = VGroup(performance_arrow, improvement_text)

class VariablesMacroeconomicas(Scene):
    """Escena 3: Variables macroeconómicas mejorada"""
    
    def construct(self):
        # Crear fondo elegante
        self.create_background()
        
        # Título principal con estilo mejorado
        title = Text(
            "Variables Macroeconómicas del Perú", 
            font_size=42, 
            color=BLUE,
            weight=BOLD
        )
        title.to_edge(UP, buff=0.5)
        
        # Subtítulo
        subtitle = Text(
            "Indicadores Clave de la Economía Peruana",
            font_size=24,
            color=GRAY,
            weight=NORMAL
        )
        subtitle.next_to(title, DOWN, buff=0.3)
        
        # Animación del título
        self.play(
            Write(title),
            run_time=1.5
        )
        self.play(
            FadeIn(subtitle, shift=UP),
            run_time=1
        )
        
        # Crear las 6 variables con colores más elegantes
        variables = [
            ("IPC", "Índice de Precios\nal Consumidor", "#FF6B6B"),
            ("PBI", "Producto Bruto\nInterno", "#4ECDC4"),
            ("TC", "Tipo de\nCambio", "#45B7D1"),
            ("TI", "Tasa\nInterbancaria", "#96CEB4"),
            ("RIN", "Reservas\nInternacionales", "#FFEAA7"),
            ("BVL", "Bolsa de Valores\nde Lima", "#DDA0DD")
        ]
        
        # Crear círculos en hexágono con diseño mejorado
        circles = VGroup()
        labels = VGroup()
        icons = VGroup()
        
        for i, (abbr, desc, color) in enumerate(variables):
            angle = i * PI / 3
            pos = 2.5 * np.array([np.cos(angle), np.sin(angle), 0])  # Aumenté la distancia
            
            # Círculo principal con borde y sombra
            circle = Circle(
                radius=0.6,  # Reducido el tamaño del círculo
                color=color, 
                fill_opacity=0.2,
                stroke_width=3,
                stroke_color=color
            )
            circle.move_to(pos)
            
            # Círculo interior para efecto de profundidad
            inner_circle = Circle(
                radius=0.5,  # Reducido proporcionalmente
                color=color,
                fill_opacity=0.1,
                stroke_width=1,
                stroke_color=color,
                stroke_opacity=0.5
            )
            inner_circle.move_to(pos)
            
            # Etiqueta principal (abreviatura) - DENTRO del círculo
            abbr_label = Text(
                abbr, 
                font_size=20,  # Reducido el tamaño de fuente
                weight=BOLD,
                color=color
            )
            abbr_label.move_to(pos)
            
            # Calcular posición del texto descriptivo - MÁS LEJOS del círculo
            text_distance = 1.8  # Aumenté la distancia del texto
            text_pos = pos + text_distance * np.array([np.cos(angle), np.sin(angle), 0])
            
            # Descripción - FUERA del círculo, bien separada
            desc_label = Text(
                desc, 
                font_size=14,  # Reducido el tamaño
                color=WHITE,
                weight=NORMAL
            )
            desc_label.move_to(text_pos)
            
            # Línea conectora sutil entre círculo y texto
            connector = Line(
                pos + 0.6 * np.array([np.cos(angle), np.sin(angle), 0]),  # Desde el borde del círculo
                text_pos - 0.4 * np.array([np.cos(angle), np.sin(angle), 0]),  # Hasta cerca del texto
                stroke_width=1,
                stroke_opacity=0.4,
                color=color
            )
            
            # Pequeño indicador de datos
            data_dot = Dot(
                radius=0.03,  # Más pequeño
                color=color,
                fill_opacity=0.8
            )
            data_dot.move_to(pos + 0.4 * UP)  # Ajustado a la nueva posición
            
            circles.add(VGroup(circle, inner_circle))
            labels.add(VGroup(abbr_label, desc_label, connector))
            icons.add(data_dot)
        
        # Animación de aparición con efecto cascada
        self.play(
            LaggedStart(
                *[Create(circle) for circle in circles], 
                lag_ratio=0.15,
                run_time=2
            )
        )
        
        self.play(
            LaggedStart(
                *[Write(label) for label in labels], 
                lag_ratio=0.15,
                run_time=1.5
            )
        )
        
        self.play(
            LaggedStart(
                *[FadeIn(icon, scale=0.5) for icon in icons], 
                lag_ratio=0.1,
                run_time=1
            )
        )
        
        # Conectar círculos con líneas elegantes
        connections = VGroup()
        for i in range(len(circles)):
            next_i = (i + 1) % len(circles)
            line = Line(
                circles[i].get_center(), 
                circles[next_i].get_center(),
                stroke_width=1.5, 
                stroke_opacity=0.2,
                color=BLUE
            )
            connections.add(line)
        
        self.play(
            Create(connections),
            run_time=1.5
        )
        
        # Texto explicativo
        explanation = Text(
            "Estas variables están interconectadas y afectan la economía peruana",
            font_size=18,
            color=GRAY
        )
        explanation.to_edge(DOWN, buff=0.5)
        
        self.play(
            Write(explanation),
            run_time=1.5
        )
        
        # Animación de datos fluyendo mejorada
        for cycle in range(3):
            # Crear partículas de datos con diferentes colores
            particles = VGroup()
            for i in range(6):
                particle = Circle(
                    radius=0.06,
                    color=variables[i][2],
                    fill_opacity=0.8,
                    stroke_width=2,
                    stroke_color=WHITE
                )
                particle.move_to(circles[i].get_center())
                particles.add(particle)
            
            # Animación de entrada
            self.play(
                *[FadeIn(particle, scale=0.3) for particle in particles],
                run_time=0.5
            )
            
            # Movimiento hacia el centro con rotación
            self.play(
                *[particle.animate.move_to(ORIGIN).rotate(PI/2) for particle in particles],
                run_time=1
            )
            
            # Efecto de fusión en el centro
            center_burst = Circle(
                radius=0.3,
                color=YELLOW,
                fill_opacity=0.3,
                stroke_color=YELLOW,
                stroke_width=2
            )
            center_burst.move_to(ORIGIN)
            
            self.play(
                FadeIn(center_burst, scale=0.1),
                FadeOut(particles),
                run_time=0.5
            )
            
            self.play(
                FadeOut(center_burst, scale=2),
                run_time=0.5
            )
            
            # Pequeña pausa entre ciclos
            if cycle < 2:
                self.wait(0.3)
        
        # Animación final de pulsación
        self.play(
            *[circle.animate.scale(1.1).set_stroke(width=4) for circle in circles],
            run_time=0.5
        )
        self.play(
            *[circle.animate.scale(1/1.1).set_stroke(width=3) for circle in circles],
            run_time=0.5
        )
        
        # Texto final
        final_text = Text(
            "Datos actualizados en tiempo real",
            font_size=16,
            color=GREEN,
            weight=BOLD
        )
        final_text.move_to(DOWN * 2.5)
        
        self.play(
            Write(final_text),
            run_time=1
        )
        
        self.wait(2)
    
    def create_background(self):
        """Crear fondo elegante similar al código de variables macroeconómicas"""
        # Fondo con gradiente elegante
        background = Rectangle(
            width=config.frame_width,
            height=config.frame_height,
            fill_color=[DARK_BLUE, BLUE],
            fill_opacity=0.1
        )
        self.add(background)
        
        # Círculos decorativos de fondo
        circles = []
        for i in range(12):
            circle = Circle(
                radius=0.08 + i * 0.03,
                color=BLUE,
                fill_opacity=0.08 - i * 0.005,
                stroke_width=0.5,
                stroke_opacity=0.3
            )
            circle.move_to(
                RIGHT * (4 - i * 0.7) + UP * (1.5 - i * 0.3)
            )
            circles.append(circle)
        
        # Círculos en el lado opuesto
        for i in range(8):
            circle = Circle(
                radius=0.06 + i * 0.02,
                color=TEAL,
                fill_opacity=0.06 - i * 0.004,
                stroke_width=0.5,
                stroke_opacity=0.2
            )
            circle.move_to(
                LEFT * (3 - i * 0.5) + DOWN * (1 - i * 0.2)
            )
            circles.append(circle)
        
        circle_group = VGroup(*circles)
        self.add(circle_group)
        
        # Líneas decorativas sutiles
        lines = []
        for i in range(4):
            line = Line(
                LEFT * 7 + UP * (1.5 - i * 0.8),
                RIGHT * 7 + UP * (1.5 - i * 0.8),
                color=GRAY,
                stroke_width=0.3,
                stroke_opacity=0.15
            )
            lines.append(line)
        
        line_group = VGroup(*lines)
        self.add(line_group)

class QLearningFormulation(Scene):
    """Escena 4: Formulación de Q-Learning con ecuaciones matemáticas"""
    
    def construct(self):
        # Crear fondo elegante
        self.create_background()
        
        # Título principal
        title = Text("Formulación Q-Learning", font_size=40, color=BLUE, weight=BOLD)
        title.to_edge(UP, buff=0.5)
        
        self.play(
            DrawBorderThenFill(title),
            run_time=1.5
        )
        self.wait(0.5)
        
        # MDP Components
        mdp_title = Text("Proceso de Decisión de Markov (MDP)", font_size=28, color=TEAL, weight=BOLD)
        mdp_title.move_to(UP * 2.5)
        
        self.play(Write(mdp_title), run_time=1)
        self.wait(0.3)
        
        # Estados, Acciones, Recompensas
        components = VGroup(
            self.create_component("Estados (S)", "Variables macroeconómicas:\nIPC, PBI, TC, TI, RIN, BVL", "#FF6B6B"),
            self.create_component("Acciones (A)", "Estrategias de inversión:\nConservadora, Moderada,\nAgresiva, Mantener", "#4ECDC4"),
            self.create_component("Recompensa (R)", "Variación del portafolio:\nrt = (Vt+1 - Vt) / Vt", "#96CEB4")
        ).arrange(RIGHT, buff=0.8)
        
        components.move_to(UP * 1.2)
        
        # Animar componentes
        for comp in components:
            self.play(
                FadeIn(comp, shift=UP * 0.3),
                run_time=1
            )
            self.wait(0.3)
        
        # Ecuación de Q-Learning
        eq_title = Text("Ecuación de Actualización Q-Learning", font_size=24, color=YELLOW, weight=BOLD)
        eq_title.move_to(DOWN * 0.8)
        
        self.play(Write(eq_title), run_time=1)
        self.wait(0.3)
        
        # Ecuación matemática
        equation = MathTex(
            r"Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]",
            font_size=28,
            color=WHITE
        )
        equation.move_to(DOWN * 1.8)
        
        self.play(
            Write(equation),
            run_time=2.5
        )
        self.wait(0.5)
        
        # Explicación de parámetros
        params = VGroup(
            Text("α = 0.1 (tasa de aprendizaje)", font_size=16, color=GRAY),
            Text("γ = 0.95 (factor de descuento)", font_size=16, color=GRAY),
            Text("ε = 0.1 (exploración)", font_size=16, color=GRAY)
        ).arrange(DOWN, buff=0.2)
        params.move_to(DOWN * 3)
        
        self.play(
            FadeIn(params),
            run_time=1.5
        )
        
        self.wait(3)
    
    def create_background(self):
        """Crear fondo elegante"""
        background = Rectangle(
            width=config.frame_width,
            height=config.frame_height,
            fill_color=[DARK_BLUE, BLUE],
            fill_opacity=0.1
        )
        self.add(background)
    
    def create_component(self, title, description, color):
        """Crear componente del MDP"""
        # Marco
        frame = RoundedRectangle(
            width=3.5,
            height=2,
            corner_radius=0.15,
            color=color,
            fill_opacity=0.1,
            stroke_width=2
        )
        
        # Título
        comp_title = Text(title, font_size=20, color=color, weight=BOLD)
        comp_title.move_to(frame.get_top() + DOWN * 0.3)
        
        # Descripción
        comp_desc = Text(description, font_size=14, color=WHITE)
        comp_desc.move_to(frame.get_center() + DOWN * 0.2)
        
        return VGroup(frame, comp_title, comp_desc)

class ResultadosComparacion(Scene):
    """Escena 5: Comparación de resultados con métricas clave"""
    
    def construct(self):
        # Crear fondo elegante
        self.create_background()
        
        # Título principal
        title = Text("Resultados de Rendimiento", font_size=40, color=BLUE, weight=BOLD)
        title.to_edge(UP, buff=0.5)
        
        self.play(
            DrawBorderThenFill(title),
            run_time=1.5
        )
        self.wait(0.5)
        
        # Crear gráfico de barras comparativo
        self.create_performance_bars()
        
        # Crear tabla de métricas
        self.create_metrics_table()
        
        # Mensaje clave
        key_message = Text(
            "Q-Learning supera a todas las estrategias tradicionales",
            font_size=22,
            color=GOLD,
            weight=BOLD
        )
        key_message.move_to(DOWN * 3.2)
        
        self.play(
            Write(key_message),
            run_time=2
        )
        
        # Efecto de brillo
        self.play(
            key_message.animate.set_color(YELLOW),
            run_time=0.5
        )
        self.play(
            key_message.animate.set_color(GOLD),
            run_time=0.5
        )
        
        self.wait(3)
    
    def create_background(self):
        """Crear fondo elegante"""
        background = Rectangle(
            width=config.frame_width,
            height=config.frame_height,
            fill_color=[DARK_BLUE, BLUE],
            fill_opacity=0.1
        )
        self.add(background)
    
    def create_performance_bars(self):
        """Crear gráfico de barras de rendimiento"""
        # Datos de rendimiento
        strategies = ["Q-Learning", "Buy & Hold", "Markowitz", "Aleatorio"]
        returns = [25.34, 22.18, 18.94, 17.09]
        colors = [GREEN, RED, ORANGE, GRAY]
        
        # Crear ejes con más espacio
        axes = Axes(
            x_range=[0, 5, 1],  # Más espacio entre barras
            y_range=[0, 30, 5],
            x_length=10,  # Gráfico más ancho
            y_length=3,
            axis_config={
                "stroke_width": 2,
                "stroke_color": WHITE
            }
        )
        axes.move_to(UP * 0.5)
        
        # Etiquetas del eje Y
        y_label = Text("Rendimiento Anual (%)", font_size=18, color=WHITE)
        y_label.next_to(axes.y_axis, LEFT, buff=0.3)
        y_label.rotate(PI/2)
        
        self.play(
            Create(axes),
            Write(y_label),
            run_time=1.5
        )
        
        # Crear barras
        bars = VGroup()
        labels = VGroup()
        
        for i, (strategy, return_val, color) in enumerate(zip(strategies, returns, colors)):
            # Barra más ancha y con mejor posicionamiento
            bar_height = return_val / 30 * 3  # Escalar a la altura del eje
            bar = Rectangle(
                width=0.7,  # Ancho óptimo
                height=bar_height,
                color=color,
                fill_opacity=0.8,
                stroke_width=2,
                stroke_color=color
            )
            bar.move_to(axes.c2p(i + 0.7, return_val/2))  # Mejor posicionamiento
            
            # Etiqueta de estrategia - SIN rotación para mejor legibilidad
            strategy_label = Text(strategy, font_size=14, color=WHITE, weight=BOLD)
            strategy_label.next_to(axes.c2p(i + 0.7, 0), DOWN, buff=0.4)
            
            # Valor encima de la barra - MÁS GRANDE y VISIBLE
            value_label = Text(f"{return_val}%", font_size=18, color=color, weight=BOLD)
            value_label.next_to(bar, UP, buff=0.15)
            
            bars.add(bar)
            labels.add(VGroup(strategy_label, value_label))
        
        # Animar barras
        for i, (bar, label) in enumerate(zip(bars, labels)):
            self.play(
                GrowFromEdge(bar, DOWN),
                run_time=1
            )
            self.play(
                Write(label),
                run_time=0.5
            )
            
            # Destacar Q-Learning
            if i == 0:
                highlight = SurroundingRectangle(
                    VGroup(bar, label[1]),
                    color=YELLOW,
                    stroke_width=3,
                    corner_radius=0.1
                )
                self.play(
                    Create(highlight),
                    run_time=0.8
                )
                self.play(
                    FadeOut(highlight),
                    run_time=0.5
                )
            
            self.wait(0.3)
    
    def create_metrics_table(self):
        """Crear tabla de métricas comparativas"""
        # Título de la tabla
        table_title = Text("Métricas Comparativas", font_size=24, color=TEAL, weight=BOLD)
        table_title.move_to(DOWN * 1.5)
        
        self.play(Write(table_title), run_time=1)
        self.wait(0.3)
        
        # Datos de la tabla
        data = [
            ["Estrategia", "Rend. Anual", "Sharpe", "Drawdown"],
            ["Q-Learning", "25.34%", "1.89", "-6.29%"],
            ["Buy & Hold", "22.18%", "1.18", "-24.85%"],
            ["Markowitz", "18.94%", "0.85", "-15.72%"]
        ]
        
        colors = [WHITE, GREEN, RED, ORANGE]
        
        # Crear tabla
        table_rows = VGroup()
        
        for i, row in enumerate(data):
            row_group = VGroup()
            for j, cell in enumerate(row):
                cell_text = Text(
                    cell, 
                    font_size=12 if i == 0 else 11,
                    color=colors[i] if i > 0 else WHITE,
                    weight=BOLD if i == 0 else NORMAL
                )
                row_group.add(cell_text)
            
            row_group.arrange(RIGHT, buff=1.2)  # Más espacio entre columnas
            table_rows.add(row_group)
        
        table_rows.arrange(DOWN, buff=0.3)
        table_rows.move_to(DOWN * 2.3)
        
        # Marco de la tabla
        table_frame = SurroundingRectangle(
            table_rows,
            color=WHITE,
            stroke_width=1,
            corner_radius=0.1,
            buff=0.3,
            fill_opacity=0.05
        )
        
        self.play(Create(table_frame), run_time=1)
        
        # Animar filas
        for row in table_rows:
            self.play(
                Write(row),
                run_time=0.8
            )
            self.wait(0.2)

class CurvaAprendizaje(Scene):
    """Escena 6: Curva de aprendizaje del agente Q-Learning"""
    
    def construct(self):
        # Crear fondo elegante
        self.create_background()
        
        # Título principal
        title = Text("Curva de Aprendizaje Q-Learning", font_size=40, color=BLUE, weight=BOLD)
        title.to_edge(UP, buff=0.5)
        
        self.play(
            DrawBorderThenFill(title),
            run_time=1.5
        )
        self.wait(0.5)
        
        # Crear gráfico de convergencia
        self.create_learning_curve()
        
        # Mensaje de convergencia
        convergence_msg = Text(
            "El agente converge en ~300 episodios",
            font_size=20,
            color=GREEN,
            weight=BOLD
        )
        convergence_msg.move_to(DOWN * 3)
        
        self.play(
            Write(convergence_msg),
            run_time=1.5
        )
        
        self.wait(3)
    
    def create_background(self):
        """Crear fondo elegante"""
        background = Rectangle(
            width=config.frame_width,
            height=config.frame_height,
            fill_color=[DARK_BLUE, BLUE],
            fill_opacity=0.1
        )
        self.add(background)
    
    def create_learning_curve(self):
        """Crear curva de aprendizaje animada"""
        # Ejes con valores visibles
        axes = Axes(
            x_range=[0, 500, 100],
            y_range=[-0.5, 2, 0.5],
            x_length=10,
            y_length=4,
            axis_config={
                "stroke_width": 2,
                "stroke_color": WHITE,
                "include_tip": True,
                "include_numbers": True,  # Mostrar números en los ejes
                "font_size": 16
            }
        )
        axes.move_to(ORIGIN)
        
        # Etiquetas de los ejes más visibles
        x_label = Text("Episodios", font_size=20, color=WHITE, weight=BOLD)
        x_label.next_to(axes.x_axis, DOWN, buff=0.5)
        
        y_label = Text("Recompensa Acumulada", font_size=20, color=WHITE, weight=BOLD)
        y_label.next_to(axes.y_axis, LEFT, buff=0.5)
        y_label.rotate(PI/2)
        
        self.play(
            Create(axes),
            Write(x_label),
            Write(y_label),
            run_time=2
        )
        
        # Función de aprendizaje (sigmoide suave) con valores más realistas
        def learning_function(x):
            # Función que simula aprendizaje gradual con convergencia
            base = 1.8 * (1 / (1 + np.exp(-0.015 * (x - 250))))  # Sigmoide suave
            noise = 0.08 * np.sin(x/25) * np.exp(-x/400)  # Ruido decreciente
            return base - 0.3 + noise
        
        # Crear curva completa
        learning_curve = axes.plot(
            learning_function,
            x_range=[0, 500],
            color=GREEN,
            stroke_width=4
        )
        
        # Animar la curva progresivamente (como si fuera entrenamiento en tiempo real)
        segments = 20
        segment_length = 500 / segments
        
        for i in range(segments):
            start_x = i * segment_length
            end_x = (i + 1) * segment_length
            
            segment = axes.plot(
                learning_function,
                x_range=[start_x, end_x],
                color=GREEN,
                stroke_width=4
            )
            
            self.play(
                Create(segment),
                run_time=0.3
            )
            
            # Mostrar valores clave en puntos importantes
            if i in [0, 5, 10, 15, 19]:  # Episodios 0, 125, 250, 375, 475
                x_val = end_x
                y_val = learning_function(x_val)
                
                # Punto destacado
                point = Dot(
                    point=axes.c2p(x_val, y_val),
                    color=YELLOW,
                    radius=0.08
                )
                
                # Etiqueta con valor
                value_label = Text(
                    f"({int(x_val)}, {y_val:.2f})",
                    font_size=12, 
                    color=YELLOW,
                    weight=BOLD
                )
                value_label.next_to(point, UP + RIGHT, buff=0.1)
                
                self.play(
                    FadeIn(point, scale=0.5),
                    Write(value_label),
                    run_time=0.4
                )
        
        # Punto final de convergencia más destacado
        final_x = 500
        final_y = learning_function(final_x)
        final_dot = Dot(
            point=axes.c2p(final_x, final_y),
            color=RED,
            radius=0.12
        )
        
        convergence_label = Text("Convergencia", font_size=18, color=RED, weight=BOLD)
        convergence_label.next_to(final_dot, UP + LEFT, buff=0.3)
        
        final_value_label = Text(
            f"Recompensa final: {final_y:.2f}",
            font_size=14,
            color=RED,
            weight=BOLD
        )
        final_value_label.next_to(convergence_label, DOWN, buff=0.1)
        
        arrow_to_convergence = Arrow(
            start=convergence_label.get_bottom(),
            end=final_dot.get_center(),
            color=RED,
            stroke_width=3,
            max_tip_length_to_length_ratio=0.2
        )
        
        self.play(
            FadeIn(final_dot, scale=0.5),
            Write(convergence_label),
            Write(final_value_label),
            Create(arrow_to_convergence),
            run_time=2
        )
        
        # Zona de estabilidad mejorada
        stable_start = 350
        stable_end = 500
        stable_y_center = learning_function(425)
        
        convergence_zone = Rectangle(
            width=3,
            height=0.3,
            color=GREEN,
            fill_opacity=0.2,
            stroke_width=2,
            stroke_color=GREEN,
            stroke_opacity=0.6
        )
        convergence_zone.move_to(axes.c2p(425, stable_y_center))
        
        zone_label = Text("Zona Estable", font_size=14, color=GREEN, weight=BOLD)
        zone_label.next_to(convergence_zone, DOWN, buff=0.2)
        
        stability_range = Text(
            f"Episodios {stable_start}-{stable_end}",
            font_size=12,
            color=GREEN
        )
        stability_range.next_to(zone_label, DOWN, buff=0.1)
        
        self.play(
            FadeIn(convergence_zone),
            Write(zone_label),
            Write(stability_range),
            run_time=1.5
        )

class TrabajoFuturo(Scene):
    """Escena 7: Trabajo futuro y extensiones"""
    
    def construct(self):
        # Crear fondo elegante
        self.create_background()
        
        # Título principal
        title = Text("Trabajo Futuro", font_size=44, color=BLUE, weight=BOLD)
        title.to_edge(UP, buff=0.5)
        
        self.play(
            DrawBorderThenFill(title),
            run_time=1.5
        )
        self.wait(0.5)
        
        # Lista de extensiones futuras
        extensiones = [
            ("🚀", "Algoritmos más avanzados", "Deep Q-Networks (DQN)\nProximal Policy Optimization (PPO)", GREEN),
            ("🌍", "Variables internacionales", "Incorporar señales globales\ny datos de mercados externos", BLUE),
            ("⚖️", "Penalizaciones de riesgo", "Funciones de recompensa\nmás sofisticadas", ORANGE),
            ("📊", "Validación en tiempo real", "Implementación con datos\nen vivo del mercado", PURPLE),
            ("🏛️", "Aplicación institucional", "Bancos centrales y fondos\nde pensiones públicos", TEAL)
        ]
        
        # Crear elementos visuales para cada extensión
        extension_items = VGroup()
        
        for emoji, titulo, descripcion, color in extensiones:
            # Ícono emoji
            icon = Text(emoji, font_size=32)
            
            # Título
            title_text = Text(titulo, font_size=20, color=color, weight=BOLD)
            
            # Descripción
            desc_text = Text(descripcion, font_size=14, color=WHITE)
            
            # Marco decorativo
            frame = SurroundingRectangle(
                VGroup(title_text, desc_text),
                color=color,
                stroke_width=2,
                corner_radius=0.15,
                buff=0.3,
                fill_opacity=0.05
            )
            
            # Agrupar elementos
            item = VGroup(
                frame,
                icon.move_to(frame.get_left() + RIGHT * 0.4),
                VGroup(title_text, desc_text).arrange(DOWN, buff=0.2).move_to(frame.get_center() + RIGHT * 0.3)
            )
            
            extension_items.add(item)
        
        # Organizar en cuadrícula (3 filas x 2 columnas para 5 elementos)
        extension_items.arrange_in_grid(
            rows=3,
            cols=2,
            buff=(1.2, 0.8)
        )
        extension_items.move_to(ORIGIN + DOWN * 0.5)
        
        # Animar cada extensión
        for i, item in enumerate(extension_items):
            self.play(
                FadeIn(item, shift=UP * 0.3),
                run_time=1
            )
            
            # Efecto de brillo para destacar
            highlight = item[0].copy().set_stroke(width=4, opacity=0.8)
            self.play(
                Create(highlight),
                run_time=0.3
            )
            self.play(
                FadeOut(highlight),
                run_time=0.3
            )
            
            self.wait(0.4)
        
        # Mensaje final inspirador
        final_message = Text(
            "El futuro del RL en finanzas es prometedor",
            font_size=22,
            color=GOLD,
            weight=BOLD
        )
        final_message.move_to(DOWN * 3.2)
        
        self.play(
            Write(final_message),
            run_time=2
        )
        
        # Efecto de brillo final
        self.play(
            final_message.animate.set_color(YELLOW).scale(1.05),
            run_time=0.5
        )
        self.play(
            final_message.animate.set_color(GOLD).scale(1/1.05),
            run_time=0.5
        )
        
        self.wait(3)
    
    def create_background(self):
        """Crear fondo elegante"""
        background = Rectangle(
            width=config.frame_width,
            height=config.frame_height,
            fill_color=[DARK_BLUE, BLUE],
            fill_opacity=0.1
        )
        self.add(background)

class Conclusiones(Scene):
    """Escena 5: Conclusiones del estudio"""
    
    def construct(self):
        # Fondo elegante
        self.create_background()
        
        # Título principal
        title = Text("Conclusiones", font_size=44, color=BLUE, weight=BOLD)
        title.to_edge(UP, buff=0.5)
        
        self.play(
            DrawBorderThenFill(title),
            run_time=1.5
        )
        self.wait(0.5)
        
        # Lista de conclusiones principales
        conclusiones = [
            "✓ Q-Learning supera significativamente a métodos tradicionales",
            "✓ Rendimiento anual del 25.34% con índice de Sharpe de 1.89",
            "✓ Superior control de riesgo (drawdown máximo: -6.29%)",
            "✓ Adaptabilidad efectiva durante crisis económicas",
            "✓ Decisiones interpretables y aplicables en la práctica",
            "✓ Marco reproducible para mercados emergentes"
        ]
        
        # Crear elementos visuales para cada conclusión
        conclusion_items = VGroup()
        
        for i, texto in enumerate(conclusiones):
            # Ícono de check mejorado
            check_circle = Circle(radius=0.15, color=GREEN, fill_opacity=0.3, stroke_width=2)
            check_mark = Text("✓", font_size=16, color=GREEN, weight=BOLD)
            check_mark.move_to(check_circle.get_center())
            icon = VGroup(check_circle, check_mark)
            
            # Texto de la conclusión
            text_obj = Text(texto[2:], font_size=20, color=WHITE)  # Quitar el "✓ " inicial
            
            # Crear item completo
            item = VGroup(icon, text_obj).arrange(RIGHT, buff=0.4, aligned_edge=LEFT)
            conclusion_items.add(item)
        
        # Organizar conclusiones verticalmente
        conclusion_items.arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        conclusion_items.move_to(ORIGIN)
        
        # Marco decorativo
        frame = SurroundingRectangle(
            conclusion_items,
            color=BLUE,
            stroke_width=2,
            corner_radius=0.2,
            buff=0.5,
            fill_opacity=0.05
        )
        
        # Animar marco
        self.play(
            Create(frame),
            run_time=1.5
        )
        self.wait(0.3)
        
        # Animar cada conclusión secuencialmente
        for i, item in enumerate(conclusion_items):
            self.play(
                FadeIn(item, shift=RIGHT * 0.5),
                run_time=0.8
            )
            
            # Efecto de highlight
            highlight = SurroundingRectangle(
                item,
                color=YELLOW,
                stroke_width=2,
                stroke_opacity=0.6,
                corner_radius=0.1
            )
            
            self.play(
                Create(highlight),
                run_time=0.3
            )
            self.play(
                FadeOut(highlight),
                run_time=0.3
            )
            
            self.wait(0.4)
        
        # Mensaje final impactante
        final_message = Text(
            "El futuro de la inversión inteligente está aquí",
            font_size=24,
            color=GOLD,
            weight=BOLD
        )
        final_message.move_to(DOWN * 3)
        
        self.play(
            Write(final_message),
            run_time=2
        )
        
        # Efecto de brillo final
        self.play(
            final_message.animate.set_color(YELLOW).scale(1.1),
            run_time=0.5
        )
        self.play(
            final_message.animate.set_color(GOLD).scale(1/1.1),
            run_time=0.5
        )
        
        self.wait(3)
    
    def create_background(self):
        """Crear fondo elegante"""
        background = Rectangle(
            width=config.frame_width,
            height=config.frame_height,
            fill_color=[DARK_BLUE, BLUE],
            fill_opacity=0.1
        )
        self.add(background)

# Función para renderizar todas las escenas
def render_all_scenes():
    """Renderiza todas las escenas en secuencia"""
    scenes = [
        IntroTitle,
        ProblemaScene,
        VariablesMacroeconomicas,
        QLearningFormulation,
        ResultadosComparacion,
        CurvaAprendizaje,
        Conclusiones,
        TrabajoFuturo
    ]

# manim -pql rl_paper_video.py IntroTitle ProblemaScene VariablesMacroeconomicas QLearningFormulation ResultadosComparacion CurvaAprendizaje Conclusiones TrabajoFuturo