# 📊 Automated-General-Report-for-Portfolio-Risk-Analysis

Este repositorio proporciona un **Informe de Riesgo Cuantitativo** avanzado enfocado en evaluar la estabilidad y eficiencia de una cartera de inversión diversificada. El análisis utiliza métricas de riesgo de alta frecuencia y pruebas estadísticas para validar la resiliencia de la cartera frente a shocks de mercado y riesgo sistemático.

🎯 **Objetivo:** Implementar un marco sólido de gestión de riesgos utilizando métricas de volatilidad condicional y riesgo de cola (tail risk) para optimizar la preservación de capital.

---

## 📖 Resumen Extendido
Este proyecto contiene una evaluación cuantitativa integral de una cartera multiactivo durante un período de alta volatilidad (**agosto de 2025 a noviembre de 2025**). El análisis integra la atribución de desempeño y el modelado de riesgos avanzado —como **Backtesting de VaR** y **Curtosis Móvil**— para identificar cambios de régimen en el riesgo de mercado.

### 🎯 Objetivos clave del análisis
* **Medición de Riesgo Dinámico:** Monitoreo de la volatilidad móvil de 21 días y la curtosis móvil para detectar "agrupamiento de volatilidad" (volatility clustering) y cambios leptocúrticos (riesgo de cola).
* **Riesgo de Cola y Déficit Esperado:** Cálculo del **Valor en Riesgo (VaR)** y **VaR Condicional (CVaR)** en niveles del 95% y 99% para cuantificar la severidad de escenarios de pérdida extrema.
* **Validación de Modelos:** Implementación de Backtesting de VaR con una **Tasa de Falla Empírica (4.35%)** para validar la calibración estadística de los modelos de riesgo.
* **Eficiencia Ajustada por Riesgo:** Evaluación a través de los ratios de **Sharpe (1.68)**, **Sortino (2.53)** y **Calmar (4.69)** para garantizar un rendimiento superior por unidad de riesgo a la baja.
* **Atribución de Desempeño:** Análisis semanal de contribución de activos y cálculo del **Alfa de Jensen** para aislar los resultados de la gestión activa.

---

## 🔍 Activos Analizados
La cartera emplea una mezcla estratégica de instrumentos de crecimiento y defensivos:

* **🚀 Acciones Individuales (Generadores de Alfa):** Apple (AAPL), Amazon (AMZN), JPMorgan (JPM), Goldman Sachs (GS).
* **🛡️ Renta Fija (Estabilizador):** Vanguard Total Bond Market ETF (BND), actuando como cobertura de riesgo principal con correlación baja/negativa respecto a las acciones.
* **📊 Benchmarks:** SPDR S&P 500 (SPY) y Dow Jones Industrial Average (DIA).

---

## 📈 Resultados Clave de la Cartera
* **Desempeño Acumulado:** Retorno total del **6.66% (16.67% YTD)**, alcanzando un máximo histórico (ATH) durante la fase de recuperación de tendencia.
* **Riesgo Sistemático:** **Beta de 0.91** y **$R^2$ de 77.51%**, lo que indica una menor sensibilidad a la volatilidad del mercado que el benchmark SPY.
* **Generación de Alfa:** **Alfa de Jensen del 7.36%**, confirmando una creación de valor significativa por encima del benchmark.
* **Gestión de Drawdown:** Máximo Drawdown contenido en **-5.34%** con una recuperación rápida, demostrando alta liquidez y un posicionamiento defensivo eficaz.

---

## 🛠️ Estructura y Lógica del Código

### 1. Volatilidad y Detección de Regímenes 🔍
* Utiliza ventanas móviles para identificar cambios de régimen (p. ej., transiciones en la volatilidad anualizada).

### 2. Motor Estadístico 🧬
* Realiza **pruebas de Jarque-Bera** y análisis de distribución (Asimetría: -0.27, Curtosis: 0.47) para evaluar la viabilidad de los modelos de riesgo Gaussianos frente a los t-Student.

### 3. Correlación y Diversificación 🤝
* Genera una matriz de correlación de activos para monitorear el efecto **"amortiguador del BND"** y las dependencias específicas de cada sector.

---

## 🚀 Tecnologías y Conceptos Utilizados
* **Finanzas Cuantitativas:** Teoría Moderna de Portafolio (MPT), Presupuestación de Riesgos y Atribución de Desempeño.
* **Modelado de Riesgos:** VaR/CVaR móvil, Backtesting y análisis de agrupamiento de volatilidad.
* **Stack de Python:** Pandas y NumPy (operaciones matriciales), Matplotlib y Seaborn (mapas de calor y gráficos móviles).
* **Contexto Económico:** Integración de eventos macro (recortes de la Fed, aranceles comerciales) en el análisis cuantitativo de la acción
